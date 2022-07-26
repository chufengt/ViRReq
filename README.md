# Visual Recognition by Request

Code for the paper "Visual Recognition by Request" [[arXiv]](https://arxiv.org/abs/2207.14227).

Contact: chufeng.t@foxmail.com

**NOTE:** This release is currently a preliminary version, which could help you understand how the proposed algorithm works. We will release the complete version as well as the checkpoints in the near future.

## Installation

This project is built upon several open-source toolboxes, follow the default instruction to install:

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for whole-to-part semantic segmentation (Type-I requests): follow [INSTALL.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) to install the required packages and build the project locally (under the folder `whole-to-part-semantic-segmentation`).

- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for instance segmentation (Type-II requests): follow [INSTALL.md](https://github.com/aim-uofa/AdelaiDet#installation) to install the required packages and build the project locally (under the folder `instance-segmentation`).

- [CLIP](https://github.com/openai/CLIP) for text features: [INSTALL.md](https://github.com/openai/CLIP#usage).

Other requirements:

```
pip install cityscapesscripts
pip install panoptic_parts
```

## Data Preparation

- [Cityscapes-Panoptic-Parts (CPP)](https://arxiv.org/abs/2004.07944): [Download](https://www.cityscapes-dataset.com/downloads/)

- [ADE20K (with Parts)](https://groups.csail.mit.edu/vision/datasets/ADE20K/): [Download](http://sceneparsing.csail.mit.edu/) (images, semantic and instance annotations)

Code for data processing will be coming soon.

## Training and Inference

- Whole-to-part semantic segmentation (Type-I requests): follow [train.md](https://github.com/chufengt/Visual-Recognition-by-Request/blob/main/whole-to-part-semantic-segmentation/docs/en/train.md) and [inference.md](https://github.com/chufengt/Visual-Recognition-by-Request/blob/main/whole-to-part-semantic-segmentation/docs/en/inference.md). See available configs (`whole-to-part-semantic-segmentation/configs/segmentation-by-request/`).

- Instance segmentation (Type-II requests): follow [Quick-Start.md](https://github.com/aim-uofa/AdelaiDet#quick-start). See available configs (`instance-segmentation/configs/segmentation-by-request/`).

Checkpoints will be coming soon.


## Evaluation

Code for evaluation (e.g., HPQ computation) will be coming soon.

## Reference

If this project is useful to your research, please consider cite:

```
@article{tang2022request,
  title={Visual Recognition by Request},
  author={Tang, Chufeng and Xie, Lingxi and Zhang, Xiaopeng and Hu, Xiaolin and Tian, Qi},
  journal={arXiv preprint arXiv:2207.14227},
  year={2022}
}
```