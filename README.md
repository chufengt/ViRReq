# Visual-Recognition-by-Request

Code for the paper "Visual Recognition by Request" [[arXiv]](https://arxiv.org/coming_soon).

Contact: chufeng.t@foxmail.com or tcf18@mails.tsinghua.edu.cn

**NOTE:** This release is currently a preliminary version, which could help you understand how the proposed algorithm works. We will release the complete version as well as the checkpoints in the near future.

## Installation

This project is built upon several open-source toolboxes, follow the default instruction to install:

[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) [[INSTALL](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)]: for whole-to-part semantic segmentation (Type-I requests).

- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) [[INSTALL](https://github.com/aim-uofa/AdelaiDet#installation)]: for instance segmentation (Type-II requests).

- [CLIP](https://github.com/openai/CLIP) [[INSTALL](https://github.com/openai/CLIP#usage)]: for text features.

Other requirements:

```
pip install cityscapesscripts
pip install panoptic_parts
```

## Data Preparation

- Cityscapes-Panoptic-Parts (CPP): [introduction](https://arxiv.org/abs/2004.07944), [download](https://www.cityscapes-dataset.com/downloads/)

- ADE20K (with Parts): [introduction](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

## Reference

If this project is useful to your research, please consider cite:

```
@article{tang2022request,
  title={Visual Recognition by Request},
  author={Tang, Chufeng and Xie, Lingxi and Zhang, Xiaopeng and Hu, Xiaolin and Tian, Qi},
  journal={arXiv preprint},
  year={2022}
}
```