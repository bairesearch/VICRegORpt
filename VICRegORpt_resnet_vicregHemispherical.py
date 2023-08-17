"""VICRegORpt_resnet_vicregHemispherical.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see VICRegORpt_globalDefs.py

# Usage:
see VICRegORpt_globalDefs.py

# Description:
vicreg biological resnet hemispherical

"""

from VICRegORpt_globalDefs import *
import torch
import torch.nn as nn
import statistics
import torch.nn.functional as F
import VICRegORpt_resnet_vicregLocal

def setArgs(argsNew):
	VICRegORpt_resnet_vicregLocal.setArgs(argsNew)

def InputForwardVicregLocal(blockPair, x, lossSum, lossIndex, trainOrTest, optim):
	out = x

	if(debugTrainLocal):
		print("InputForwardVicregLocal")
		print("l1 = ", blockPair[0].l1)
		print("l2 = ", blockPair[0].l2)
		print("x = ", x)

	for p in range(numberOfHemispheres):
		#print("out[p].shape = ", out[p].shape)
		out[p] = blockPair[p].conv1(out[p])
		out[p] = blockPair[p].bn1(out[p])
		out[p] = blockPair[p].relu(out[p])
	
	if(trainLocal and trainOrTest):
		out, lossAvg0 = VICRegORpt_resnet_vicregLocal.trainLayerLocal(out, trainOrTest, optim, blockPair[0].l1, blockPair[0].l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	for p in range(numberOfHemispheres):
		out[p] = blockPair[p].maxpool(out[p])

	if(trainLocal):
		return out, lossSum, lossIndex, trainOrTest, optim
	else:
		return out
		
def BasicBlockForwardVicregLocal(blockPair, x, lossSum, lossIndex, trainOrTest, optim):
	out = [None]*2
	for p in range(numberOfHemispheres):
		out[p] = torch.clone(x[p])

	if(debugTrainLocal):
		print("BasicBlockForwardVicregLocal")
		print("l1 = ", blockPair[0].l1)
		print("l2 = ", blockPair[0].l2)
		print("x = ", x)

	for p in range(numberOfHemispheres):
		out[p] = blockPair[p].conv1(out[p])
		out[p] = blockPair[p].bn1(out[p])
		out[p] = blockPair[p].relu(out[p])

	if(trainLocal and trainOrTest):
		out, lossAvg0 = VICRegORpt_resnet_vicregLocal.trainLayerLocal(out, trainOrTest, optim, blockPair[0].l1, blockPair[0].l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	for p in range(numberOfHemispheres):
		out[p] = blockPair[p].conv2(out[p])
		out[p] = blockPair[p].bn2(out[p])

	if(trainLocal and trainOrTest):
		if(applyIndependentLearningForDownsample):
			out, lossAvg1 = VICRegORpt_resnet_vicregLocal.trainLayerLocal(out, trainOrTest, optim, blockPair[0].l1, blockPair[0].l2, l3=1)
			lossIndex+=1
			lossSum+=lossAvg1

	for p in range(numberOfHemispheres):
		if blockPair[p].downsample is not None:
			identity1 = blockPair[p].downsample(x[p])
		else:
			identity1 = x[p]
		out[p] += identity1
		if(normaliseActivationSparsityLayerSkip):
			out[p] = blockPair[p].bnSkip(out[p])
		out[p] = blockPair[p].last_activation(out[p])

	if(trainLocal and trainOrTest):
		out, lossAvg2 = VICRegORpt_resnet_vicregLocal.trainLayerLocal(out, trainOrTest, optim, blockPair[0].l1, blockPair[0].l2, l3=1+int(applyIndependentLearningForDownsample))
		lossIndex+=1
		lossSum+=lossAvg2

	if(trainLocal):
		return out, lossSum, lossIndex, trainOrTest, optim
	else:
		return out

def BottleneckForwardVicregLocal(blockPair, x, lossSum, lossIndex, trainOrTest, optim):
	out = [None]*2
	for p in range(numberOfHemispheres):
		out[p] = torch.clone(x[p])

	if(debugTrainLocal):
		print("BottleneckForwardVicregLocal")
		print("l1 = ", blockPair[0].l1)
		print("l2 = ", blockPair[0].l2)
		print("x = ", x)

	for p in range(numberOfHemispheres):
		out[p] = blockPair[p].conv1(out[p])
		out[p] = blockPair[p].bn1(out[p])
		out[p] = blockPair[p].relu(out[p])

	if(trainLocal and trainOrTest):
		out, lossAvg0 = VICRegORpt_resnet_vicregLocal.trainLayerLocal(out, trainOrTest, optim, blockPair[0].l1, blockPair[0].l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	for p in range(numberOfHemispheres):
		out[p] = blockPair[p].conv2(out[p])
		out[p] = blockPair[p].bn2(out[p])
		out[p] = blockPair[p].relu(out[p])

	if(trainLocal and trainOrTest):
		out, lossAvg1 = VICRegORpt_resnet_vicregLocal.trainLayerLocal(out, trainOrTest, optim, blockPair[0].l1, blockPair[0].l2, l3=1)
		lossIndex+=1
		lossSum+=lossAvg1

	for p in range(numberOfHemispheres):
		out[p] = blockPair[p].conv3(out[p])
		out[p] = blockPair[p].bn3(out[p])

	if(trainLocal and trainOrTest):
		if(applyIndependentLearningForDownsample):
			out, lossAvg2 = VICRegORpt_resnet_vicregLocal.trainLayerLocal(out, trainOrTest, optim, blockPair[0].l1, blockPair[0].l2, l3=2)
			lossIndex+=1
			lossSum+=lossAvg2

	for p in range(numberOfHemispheres):
		if blockPair[p].downsample is not None:
			identity1 = blockPair[p].downsample(x[p])
		else:
			identity1 = x[p]
		out[p] += identity1
		if(normaliseActivationSparsityLayerSkip):
			out[p] = blockPair[p].bnSkip(out[p])
		out[p] = blockPair[p].last_activation(out[p])
	
	if(trainLocal and trainOrTest):
		out, lossAvg3 = VICRegORpt_resnet_vicregLocal.trainLayerLocal(out, trainOrTest, optim, blockPair[0].l1, blockPair[0].l2, l3=2+int(applyIndependentLearningForDownsample))
		lossIndex+=1
		lossSum+=lossAvg3		

	if(trainLocal):
		return out, lossSum, lossIndex, trainOrTest, optim
	else:
		return out

def propagateNetwork(x, backbone1, backbone2, trainOrTest, optim, blockType):

	backbone = [backbone1, backbone2]
	
	for p in range(numberOfHemispheres):
		x[p] = backbone[p].padding(x[p])
		
	lossAvg = 0.0
	
	lossSum = 0.0
	lossIndex = 0
	if(smallInputImageSize):
		layersList1 = [backbone1.layer0, backbone1.layer1, backbone1.layer2, backbone1.layer3]
		layersList2 = [backbone2.layer0, backbone2.layer1, backbone2.layer2, backbone2.layer3]
	else:
		layersList1 = [backbone1.layer0, backbone1.layer1, backbone1.layer2, backbone1.layer3, backbone1.layer4]
		layersList2 = [backbone2.layer0, backbone2.layer1, backbone2.layer2, backbone2.layer3, backbone2.layer4]

	for l in range(len(layersList1)):
		blockTypeLayer = blockType
		layer1 = layersList1[l]
		layer2 = layersList2[l]
		if(l == 0):
			blockTypeLayer = blockTypeInput
			numBlocks = 1
		else:
			numBlocks = len(layer1)	#number of modules in nn.Sequential
		for b in range(numBlocks):
			if(l == 0):
				b1 = layer1
				b2 = layer2
			else:
				b1 = layer1[b]
				b2 = layer2[b]
			blockPair = [b1, b2]
			if(blockTypeLayer == blockTypeInput):
				x, lossSum, lossIndex, trainOrTest, optim = InputForwardVicregLocal(blockPair, x, lossSum, lossIndex, trainOrTest, optim)
			if(blockTypeLayer == blockTypeBasic):
				x, lossSum, lossIndex, trainOrTest, optim = BasicBlockForwardVicregLocal(blockPair, x, lossSum, lossIndex, trainOrTest, optim)
			elif(blockTypeLayer == blockTypeBottleneck):
				x, lossSum, lossIndex, trainOrTest, optim = BottleneckForwardVicregLocal(blockPair, x, lossSum, lossIndex, trainOrTest, optim)

	for p in range(numberOfHemispheres):
		x[p] = backbone[p].avgpool(x[p])
		x[p] = torch.flatten(x[p], 1)

	if(trainLocal): 
		if(trainOrTest):
			lossAvg = lossSum/lossIndex
		
	return x, lossAvg
	
