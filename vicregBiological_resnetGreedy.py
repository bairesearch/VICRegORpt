"""vicregBiological_resnetGreedy.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see vicregBiological_globalDefs.py

# Usage:
see vicregBiological_globalDefs.py

# Description:
vicreg biological resnet greedy

"""

from vicregBiological_globalDefs import *
import torch
import torch.nn as nn
import statistics
import torch.nn.functional as F

def setArgs(argsNew):
	global args
	global trainOrTest
	args = argsNew
	if(trainGreedyIndependentBatchNorm):
		trainOrTest = args.trainOrTest
		
def InputForwardVicregGreedy(self, x, lossSum, lossIndex, trainOrTest, optim):
	out = x

	if(debugTrainGreedy):
		print("InputForwardVicregGreedy")
		print("l1 = ", self.l1)
		print("l2 = ", self.l2)
		print("x = ", x)

	out = self.conv1(out)
	out = self.bn1(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg0 = trainLayerLocal(self, out, trainOrTest, optim, self.l1, self.l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	out = self.maxpool(out)

	return out, lossSum, lossIndex, trainOrTest, optim

def BasicBlockForwardVicregGreedy(self, x, lossSum, lossIndex, trainOrTest, optim):
	identity1 = x
	out = x

	if(debugTrainGreedy):
		print("BasicBlockForwardVicregGreedy")
		print("l1 = ", self.l1)
		print("l2 = ", self.l2)
		print("x = ", x)

	out = self.conv1(out)
	out = self.bn1(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg0 = trainLayerLocal(self, out, trainOrTest, optim, self.l1, self.l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	out = self.conv2(out)
	out = self.bn2(out)

	if(trainOrTest):
		if(applyIndependentLearningForDownsample):
			out, lossAvg1 = trainLayerLocal(self, out, trainOrTest, optim, self.l1, self.l2, l3=1)
			lossIndex+=1
			lossSum+=lossAvg1

	if self.downsample is not None:
		identity1 = self.downsample(x)
	out += identity1
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg2 = trainLayerLocal(self, out, trainOrTest, optim, self.l1, self.l2, l3=1+int(applyIndependentLearningForDownsample))
		lossIndex+=1
		lossSum+=lossAvg2

	return out, lossSum, lossIndex, trainOrTest, optim

def BottleneckForwardVicregGreedy(self, x, lossSum, lossIndex, trainOrTest, optim):
	identity1 = x
	out = x

	if(debugTrainGreedy):
		print("BottleneckForwardVicregGreedy")
		print("l1 = ", self.l1)
		print("l2 = ", self.l2)
		print("x = ", x)

	out = self.conv1(out)
	out = self.bn1(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg0 = trainLayerLocal(self, out, trainOrTest, optim, self.l1, self.l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	out = self.conv2(out)
	out = self.bn2(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg1 = trainLayerLocal(self, out, trainOrTest, optim, self.l1, self.l2, l3=1)
		lossIndex+=1
		lossSum+=lossAvg1

	out = self.conv3(out)
	out = self.bn3(out)

	if(trainOrTest):
		if(applyIndependentLearningForDownsample):
			out, lossAvg2 = trainLayerLocal(self, out, trainOrTest, optim, self.l1, self.l2, l3=2)
			lossIndex+=1
			lossSum+=lossAvg2

	if self.downsample is not None:
		identity1 = self.downsample(x)
	out += identity1
	out = self.last_activation(out)

	if(trainOrTest):
		out, lossAvg3 = trainLayerLocal(self, out, trainOrTest, optim, self.l1, self.l2, l3=2+int(applyIndependentLearningForDownsample))
		lossIndex+=1
		lossSum+=lossAvg3		

	return out, lossSum, lossIndex, trainOrTest, optim

def ResNetForwardVicregGreedy(self, x, trainOrTest, optim):
	x = self.padding(x)

	lossSum = 0.0
	lossIndex = 0
	x, lossSum, lossIndex, trainOrTest, optim = self.layer0(x, lossSum, lossIndex, trainOrTest, optim)
	x, lossSum, lossIndex, trainOrTest, optim = self.layer1(x, lossSum, lossIndex, trainOrTest, optim)
	x, lossSum, lossIndex, trainOrTest, optim = self.layer2(x, lossSum, lossIndex, trainOrTest, optim)
	x, lossSum, lossIndex, trainOrTest, optim = self.layer3(x, lossSum, lossIndex, trainOrTest, optim)
	if(not smallInputImageSize):
		x, lossSum, lossIndex, trainOrTest, optim = self.layer4(x, lossSum, lossIndex, trainOrTest, optim)

	x = self.avgpool(x)
	x = torch.flatten(x, 1)

	if(trainOrTest):
		lossAvg = lossSum/lossIndex
	else:
		lossAvg = 0.0
		
	return x, lossAvg

def trainLayerLocal(self, x, trainOrTest, optim, l1, l2, l3):
	
	if(trainOrTest):
		batchSize = x.shape[0]
		x1, x2 = torch.split(x, batchSize//2, dim=0)
		
		if(debugTrainGreedy):
			print("trainLayerLocal: l1 = ", l1, ", l2 = ", l2, ", l3 = ", l3)
			
		loss = None
		accuracy = 0.0

		opt = optim[l1][l2][l3]
		opt.zero_grad()

		loss = calculateLossVICregLocal(self, x1, x2)

		loss.backward()
		opt.step()

		x = x.detach()
	else:
		print("trainLayerLocal warning: currently requires trainOrTest=True")
		pass

	return x, loss

def calculateLossVICregLocal(self, x, y):
	x = torch.flatten(x, 1)	#convert to linear for VICreg
	y = torch.flatten(y, 1)	#convert to linear for VICreg
	num_features = x.shape[1]
	batch_size = x.shape[0]
	if(debugTrainGreedy):
		print("num_features = ", num_features)
		print("batch_size = ", batch_size)
		print("x = ", x)
		print("y = ", y)

	repr_loss = F.mse_loss(x, y)

	#if(distributedExecution):	#not currently supported
	#	x = torch.cat(FullGatherLayer.apply(x), dim=0)
	#	y = torch.cat(FullGatherLayer.apply(y), dim=0)
	x = x - x.mean(dim=0)
	y = y - y.mean(dim=0)

	std_x = torch.sqrt(x.var(dim=0) + 0.0001)
	std_y = torch.sqrt(y.var(dim=0) + 0.0001)
	std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

	cov_x = (x.T @ x) / (batch_size - 1)
	cov_y = (y.T @ y) / (batch_size - 1)

	cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

	loss = (
		args.sim_coeff * repr_loss
		+ args.std_coeff * std_loss
		+ args.cov_coeff * cov_loss
	)
	return loss

def off_diagonal(x):
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def createLocalOptimisers(model):
	l1Len = len(model.backbone.layers)
	l1optim = []
	for l1 in range(l1Len):
		l2optim = []
		l2Len = model.backbone.layers[l1]
		for l2 in range(l2Len):
			l3optim = []
			if(l1 == 0):
				l3Len = 1
			else:
				if(model.backbone.block.name == "Bottleneck"):
					l3Len = 3
				elif(model.backbone.block.name == "BasicBlock"):
					l3Len = 2
				else:
					printe("error: model.backbone.block.name not found")
				if(applyIndependentLearningForDownsample):
					l3Len += 1
			for l3 in range(l1Len):
				opt = torch.optim.Adam(model.backbone.parameters(), lr=learningRateLocal)	#LARS
				l3optim.append(opt)
			l2optim.append(l3optim)
		l1optim.append(l2optim)
	optim = l1optim
	return optim

if(trainGreedyIndependentBatchNorm):
	class BatchNormLayerVICregLocal(nn.Module):
		def __init__(self, num_out_filters):
			super(BatchNormLayerVICregLocal, self).__init__()
			self.normFunction = nn.BatchNorm2d(num_out_filters)

		def forward(self, x):
			if(trainOrTest):
				batchSize = x.shape[0]
				x1, x2 = torch.split(x, batchSize//2, dim=0)
				x1 = self.normFunction(x1)
				x2 = self.normFunction(x2)
				x = torch.cat((x1, x2), dim=0)
			else:
				x = self.normFunction(x)
			return x
			
