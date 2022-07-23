# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.cnn import ConvModule

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy

import clip # installation: https://github.com/openai/CLIP
import numpy as np
import torch.nn.functional as F


class CLIP_CLS_SEG(nn.Module):
    def __init__(self, channels, norm_cfg, dropout):
        super(CLIP_CLS_SEG, self).__init__()
        
        self.clip_dim = 512 # hardcoded for CLIP-Vit-B/32
        temperature = torch.tensor(np.log(1 / 0.07)) # follw CLIP
        self.logit_scale = nn.Parameter(temperature, requires_grad=True)
        self.clip_model, _ = clip.load("ViT-B/32", device='cuda', jit=False)
        for param in self.clip_model.parameters(): # freeze CLIP
            param.requires_grad = False
        
        # channels: 256 -> 512
        self.channel_transform = ConvModule(
                                    channels,
                                    self.clip_dim,
                                    kernel_size=3,
                                    padding=1,
                                    norm_cfg=norm_cfg)
        self.dropout = dropout
        self.channel_transform.init_weights()

    def forward(self, x, class_labels):
        text = clip.tokenize(class_labels).to(x.device)
        text_features = self.clip_model.encode_text(text)
        
        image_features = self.channel_transform(x)
        if self.dropout is not None:
            image_features = self.dropout(image_features)
        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.clip_dim)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        cls_logits = self.logit_scale.exp() * image_features.half() @ text_features.t()
        cls_logits = cls_logits.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        return cls_logits


class BaseDecodeHead_ADE_Part(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead_ADE_Part.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 num_classes_tol,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 with_others=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=None)):
        super(BaseDecodeHead_ADE_Part, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.with_others = with_others
        self.num_classes_tol = num_classes_tol

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None
        assert self.sampler is None

        # self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        # input texts
        if num_classes_tol == 19:
            self.class_labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        elif num_classes_tol == 28: # citys_part
            self.class_labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'torso', 'head', 'arm', 'leg', 'window', 'wheel', 'light', 'license_plate', 'chassis']
        elif num_classes_tol == 150:
            self.class_labels = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']
        elif num_classes_tol == 233: # ade20k_part
            self.class_labels = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag', 'frame', 'cistern', 'front', 'stem', 'mirror', 'aperture', 'stove', 'apron', 'monitor', 'arm', 'stretcher', 'gaze', 'column', 'right arm', 'right foot', 'right hand', 'right leg', 'mouse', 'computer case', 'back', 'back pillow', 'muntin', 'backplate', 'cord', 'neck', 'corner pocket', 'base', 'taillight', 'tap', 'opening', 'handle', 'oven', 'sash', 'head', 'headboard', 'headlight', 'screen', 'blade', 'pane', 'highlight', 'diffusor', 'hinge', 'seat', 'seat base', 'seat cushion', 'top', 'housing', 'bowl', 'torso', 'door', 'shade', 'door frame', 'drawer', 'shelf', 'bulb', 'bumper', 'earmuffs', 'side', 'tube', 'side rail', 'side pocket', 'button panel', 'keyboard', 'upper sash', 'knob', 'skirt', 'face', 'landing gear', 'faucet', 'canopy', 'left arm', 'left foot', 'left hand', 'left leg', 'leg', 'license plate', 'lid', 'light source', 'lower sash', 'wheel', 'window', 'windshield', 'footboard']
        else:
            print('Not support.')
            self.class_labels = []
        
        # modifed: replace conv_seg with CLIP_CLS_SEG for prediction
        self.CLIP_CLS_SEG = CLIP_CLS_SEG(channels, norm_cfg, self.dropout)

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, gt_part_seg, gt_ins_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.used_class_labels = self.class_labels
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, gt_part_seg, gt_ins_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        if self.num_classes < self.num_classes_tol:
            self.used_class_labels = self.class_labels[:self.num_classes]
        return self.forward(inputs)

#     def cls_seg(self, feat):
#         """Classify each pixel."""
#         if self.dropout is not None:
#             feat = self.dropout(feat)
#         output = self.conv_seg(feat)
#         return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.CLIP_CLS_SEG(feat, self.used_class_labels)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, seg_part_label, seg_ins_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        seg_part_label = seg_part_label.squeeze(1)
        seg_logit, seg_part_logit = seg_logit[:, :self.num_classes], seg_logit[:, self.num_classes:]

        # modify seg_logits based on seg_ins_label
        n_images = seg_logit.shape[0]
        ignored_mask = torch.zeros_like(seg_part_logit, dtype=torch.bool).to(seg_part_logit.device)
        ignored_mask = ignored_mask.permute(0, 2, 3, 1) # [N, H, W, C]
        for img_idx in range(n_images): # for each image
            cur_ins_map = seg_ins_label[img_idx][0]
            ins_idxs = cur_ins_map.unique()
            for ins_idx in ins_idxs: # for each instance
                if ins_idx == 255: continue
                part_idxs = (seg_part_label[img_idx][cur_ins_map == ins_idx]).unique()
                part_idxs = part_idxs[part_idxs < 255]
                if part_idxs.size(0) == 0: continue
                ignored_vector = torch.zeros(seg_part_logit.shape[1], dtype=torch.bool).to(seg_part_logit.device)
                ignored_vector[part_idxs] = True
                ignored_mask[img_idx, cur_ins_map == ins_idx] = ignored_vector

        part_loss_avg_factor = torch.sum(ignored_mask.sum(dim=-1).bool()).item()
        #part_loss_avg_factor = torch.sum(seg_ins_label != self.ignore_index).item()
        ignored_mask = ignored_mask.permute(0, 3, 1, 2) # [N, C, H, W]
        seg_part_logit[ignored_mask == False] = np.log(1e-40) # float('-inf') leads to nan loss

        loss['acc_seg'] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)
        loss['acc_seg_part'] = accuracy(seg_part_logit, seg_part_label, ignore_index=self.ignore_index)
        
        seg_label[seg_label==self.ignore_index] = -1
        seg_label += 1
        seg_part_label[seg_part_label==self.ignore_index] = -1
        seg_part_label += 1
        seg_part_label[seg_ins_label.squeeze(1) == self.ignore_index] = 0
        seg_logit, seg_part_logit = F.pad(seg_logit, (0, 0, 0, 0, 1, 0)), F.pad(seg_part_logit, (0, 0, 0, 0, 1, 0))

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name == 'loss_ce':
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    ignore_index=self.ignore_index)
            elif loss_decode.loss_name == 'loss_ce_part':
                loss[loss_decode.loss_name] = loss_decode(
                    seg_part_logit,
                    seg_part_label,
                    avg_factor=part_loss_avg_factor,
                    ignore_index=self.ignore_index)
            else:
                print('not support.')

        return loss