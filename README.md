# VICRegORpt

### Author

Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

Copyright (c) Facebook, Inc. and its affiliates.

### Description

Variance-Invariance-Covariance Regularization Object Recognition (VICRegOR) for PyTorch - experimental

An implementation of;
Bardes, A., Ponce, J., & LeCun, Y. (2021). Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906.
https://arxiv.org/abs/2105.04906

Derived from;
https://github.com/facebookresearch/vicreg

### License

MIT License

### Installation
```
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics

https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
```

### Execution
```
source activate pytorchsenv
python ANNpt_main.py

#Single-node local training
python VICRegORpt/VICRegORpt_main_vicreg.py --data-dir datasets/imagenette2-320 --exp-dir out/exp --arch resnet50 --epochs 10 --batch-size 64 --base-lr 0.3

#Linear evaluation
python VICRegORpt/VICRegORpt_evaluate.py --data-dir datasets/imagenette2-320 --pretrained out/exp/resnet50.pth --exp-dir ./out/exp --epochs 10 --batch-size 64 --lr-head 0.02
```
