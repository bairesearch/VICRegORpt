"""VICRegORpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n pytorchsenv
source activate pytorchsenv
conda install python
pip install datasets
pip install torch
pip install lovely-tensors
pip install torchmetrics

https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz

# Usage:

#Train backbone using VICReg
source activate pytorchsenv
python VICRegORpt/VICRegORpt_main_vicreg.py --data-dir datasets/imagenette2-320 --exp-dir out/exp --arch resnet50 --epochs 10 --batch-size 64 --base-lr 0.3

#Train head + evaluation
source activate pytorchsenv
python VICRegORpt/VICRegORpt_evaluate.py --data-dir datasets/imagenette2-320 --pretrained out/exp/resnet50.pth --exp-dir ./out/exp --epochs 10 --batch-size 64 --lr-head 0.02

#Train backbone using Backprop + head + evaluation
source activate pytorchsenv
python VICRegORpt/VICRegORpt_main_vicreg.py --data-dir datasets/imagenette2-320 --exp-dir out/exp --arch resnet50 --epochs 0 #[generate model]
python VICRegORpt/VICRegORpt_evaluate.py --data-dir datasets/imagenette2-320 --pretrained out/exp/resnet50.pth --exp-dir out/exp --batch-size 64 --weights finetune --epochs 100 --lr-backbone 0.005 --lr-head 0.05 --weight-decay 0

---
Dataset selection;
--data-dir datasets/imagenette2-320
--data-dir datasets/tiny-imagenet-200
--data-dir /media/user/datasets/imagenet


	##Test out-of-distribution class classification after Backprop vs VICReg model backbone training;
	- expbp: Resnet-50 top 1 performance trained on 50% of images with Backprop, evaluate top-1 performance when only training final layer [ie linear classifier] on 100% images
	- expvr: Resnet-50 top 1 performance trained on 50% of images with VICREG, evaluate top-1 performance when only training final layer [ie linear classifier] on 100% images

	###expbp (backprop 50% training, 100% linear eval);

	#Backprop training with 50% classes
	python VICRegORpt/VICRegORpt_main_vicreg.py --data-dir datasets/imagenette2-320-half --exp-dir out/expbp --arch resnet50 --epochs 0	#[generate model]
	python VICRegORpt/VICRegORpt_evaluate.py --data-dir datasets/imagenette2-320-half --pretrained out/expbp/resnet50.pth --exp-dir out/expbp --batch-size 64 --weights finetune --epochs 100 --lr-backbone 0.3 --weight-decay 0
	rm expbp/checkpoint.pth

	#Linear evaluation with 100% classes
	python VICRegORpt/VICRegORpt_evaluate.py --data-dir datasets/imagenette2-320 --pretrained out/expbp/resnet50eval.pth --exp-dir out/expbp --batch-size 64 --weights freeze --lr-head 0.02

	###expvr (vicreg 50% training, 100% linear eval);

	#VICReg hidden layer training with 50% classes
	python VICRegORpt/VICRegORpt_main_vicreg.py --data-dir datasets/imagenette2-320-half --exp-dir out/expvr --batch-size 64 --arch resnet50 --epochs 100 --base-lr 0.3

	#Linear evaluation with 100% classes
	python VICRegORpt/VICRegORpt_evaluate.py --data-dir datasets/imagenette2-320 --pretrained out/expvr/resnet50.pth --exp-dir out/expvr --batch-size 64 --weights freeze --lr-head 0.02

# Description:
vicreg biological globalDefs

"""

import torch as pt
import torch.nn as nn

vicregBiologicalMods = True

#initialise (dependent vars);
trainLocal = False
usePositiveWeights = False
normaliseActivationSparsityLayer = False
smallInputImageSize = False
trainLocalConvLocationIndependenceAllPixels = False	
trainLocalConvLocationIndependenceAllPixelsSequential = False
normaliseActivationSparsityLayerSkip = False
normaliseActivationSparsityBatch = True
activationFunctionType = "relu"

if(vicregBiologicalMods):
	usePositiveWeights = False
	if(usePositiveWeights):
		activationFunctionType = "softmax"
		#activationFunctionType = "none"
		if(activationFunctionType == "softmax"):
			activationFunctionTypeSoftmaxIndependentChannels = True	#optional
		usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required
		normaliseActivationSparsityLayer = True	#perform layer norm instead of batch norm (default resnet model setting)
		if(normaliseActivationSparsityLayer):
			normaliseActivationSparsityLayerFunctionLayerNorm = False	#normalise all activations	#not currently supported; requires upgrade of codebase to specify shape of activation/image size
			normaliseActivationSparsityLayerFunctionInstanceNorm2d = True	#normalise across conv2d channels independently	#orig
			normaliseActivationSparsityLayerFunctionGroupNorm = False	#normalise across conv2d channels independently
			normaliseActivationSparsityBatch = False	#perform standard specification batch norm as well as layer norm
		normaliseActivationSparsityLayerSkip = False	#normalise activation across skip connections layer
	trainLocal = True
	if(trainLocal):
		applyIndependentLearningForDownsample = False	#optional	#apply independent learning for downsample (conv1x1)	#not downsample is only used for first block in l1
		learningRateLocal = 0.005
		debugTrainLocal = False
		trainLocalConvLocationIndependence = True 	#local VICReg Conv2D implementation independent of location
		if(trainLocalConvLocationIndependence):
			trainLocalConvLocationIndependenceAveragedPixels = True	#average pixel values across each Conv2D filter
			trainLocalConvLocationIndependenceAllPixels = False	#use data from all Conv2D filter pixels for VICReg calculations
			trainLocalConvLocationIndependenceSinglePixelRandom = False	#reduces number of comparisons by randomising matched pixel pairs in image
			smallInputImageSize = False
			if(trainLocalConvLocationIndependenceAllPixels):
				trainLocalConvLocationIndependenceAllPixelsSequential = False	#optional	#execute VICReg on pixels sequentially
				trainLocalConvLocationIndependenceAllPixelsCombinations = True	#calculate VICReg on all pixel pair combinations sequentially
				trainLocalConvLocationIndependenceAllPixelsMatched = False	#calculate VICReg on each pixel pair sequentially
				if(trainLocalConvLocationIndependenceAllPixelsCombinations):
					smallInputImageSize = True
		else:
			smallInputImageSize = True	#required for vicregBiologicalMods:trainLocal:!trainLocalConvLocationIndependence due to cov_x = (x.T @ x) operation on large image size
	else:
		smallInputImageSize = False	#optional (not required); can be used to ensure that experiments are equivalent [to trainLocal]

smallInputImageSize = True	#temporary override for debugging only!

if(smallInputImageSize):
	imageWidth = 32	#needs to be sufficiently high such that convolutions do not prematurely converge	#cannot be too high else will run out of RAM
	#not required: use datasetTinyImagenet, as imagenet/imagenette images will be downsampled
else:
	imageWidth = 224	#use imagenet [1000 classes]/imagenette [10 class]


def printe(str):
	print(str)
	exit()
	
useLovelyTensors = True
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	pt.set_printoptions(profile="full")
	pt.set_printoptions(sci_mode=False)

class sequentialMultiInput(nn.Sequential):
	def forward(self, *inputs):
		for module in self._modules.values():
			if type(inputs) == tuple:
				inputs = module(*inputs)
			else:
				inputs = module(inputs)
		return inputs

'''
class sequentialMultiInput(nn.Sequential):
	def forward(self, *input):
		for module in self._modules.values():
			input = module(*input)
		return input
'''		
