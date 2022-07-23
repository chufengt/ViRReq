_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

# smaller loss weight for others class
sem_class_weight = [0.1] + [1.0] * 150
part_class_weight = [0.0001] + [1.0] * 83

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsADE20KPart', reduce_zero_label=True), # custom
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_part_seg', 'gt_ins_seg']), # custom
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_ADEPart',
    pretrained='/your_pretrain_path/mit_b0.pth', # https://github.com/open-mmlab/mmsegmentation/tree/v0.24.0/configs/segformer#usage
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=32, # b0
        num_stages=4,
        num_layers=[2, 2, 2, 2], # b0
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead_ADE_Part', # custom
        in_channels=[32, 64, 160, 256], # b0
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=150, # ADE semantic classes only
        num_classes_tol=233, # CPP semantic + part classes
        norm_cfg=norm_cfg,
        align_corners=False,
        with_others=True, # extra 'other' class
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', avg_non_ignore=True, use_sigmoid=False, loss_weight=1.0, class_weight=sem_class_weight),
            dict(type='CrossEntropyLoss', loss_name='loss_ce_part', avg_non_ignore=True, use_sigmoid=False, loss_weight=1.0, class_weight=part_class_weight)
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(train=dict(pipeline=train_pipeline), samples_per_gpu=2, workers_per_gpu=2)