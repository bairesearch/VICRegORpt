"""VICRegORpt_resnet_vicregLocal.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see VICRegORpt_globalDefs.py

# Usage:
see VICRegORpt_globalDefs.py

# Description:
vicreg biological resnet local

"""

from VICRegORpt_globalDefs import *
import torch
import torch.nn as nn
import statistics
import torch.nn.functional as F

def setArgs(argsNew):
	global args
	args = argsNew
		
def InputForwardVicregLocal(self, x, lossSum, lossIndex, trainOrTest, optim):
	out = x

	if(debugTrainLocal):
		print("InputForwardVicregLocal")
		print("l1 = ", self.l1)
		print("l2 = ", self.l2)
		print("x = ", x)

	out = self.conv1(out)
	out = self.bn1(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg0 = trainLayerLocal(out, trainOrTest, optim, self.l1, self.l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	out = self.maxpool(out)

	return out, lossSum, lossIndex, trainOrTest, optim

def BasicBlockForwardVicregLocal(self, x, lossSum, lossIndex, trainOrTest, optim):
	identity1 = x
	out = x

	if(debugTrainLocal):
		print("BasicBlockForwardVicregLocal")
		print("l1 = ", self.l1)
		print("l2 = ", self.l2)
		print("x = ", x)

	out = self.conv1(out)
	out = self.bn1(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg0 = trainLayerLocal(out, trainOrTest, optim, self.l1, self.l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	out = self.conv2(out)
	out = self.bn2(out)

	if(trainOrTest):
		if(applyIndependentLearningForDownsample):
			out, lossAvg1 = trainLayerLocal(out, trainOrTest, optim, self.l1, self.l2, l3=1)
			lossIndex+=1
			lossSum+=lossAvg1

	if self.downsample is not None:
		identity1 = self.downsample(x)
	out += identity1
	if(normaliseActivationSparsityLayerSkip):
		out = self.bnSkip(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg2 = trainLayerLocal(out, trainOrTest, optim, self.l1, self.l2, l3=1+int(applyIndependentLearningForDownsample))
		lossIndex+=1
		lossSum+=lossAvg2

	return out, lossSum, lossIndex, trainOrTest, optim

def BottleneckForwardVicregLocal(self, x, lossSum, lossIndex, trainOrTest, optim):
	identity1 = x
	out = x

	if(debugTrainLocal):
		print("BottleneckForwardVicregLocal")
		print("l1 = ", self.l1)
		print("l2 = ", self.l2)
		print("x = ", x)

	out = self.conv1(out)
	out = self.bn1(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg0 = trainLayerLocal(out, trainOrTest, optim, self.l1, self.l2, l3=0)
		lossIndex+=1
		lossSum+=lossAvg0

	out = self.conv2(out)
	out = self.bn2(out)
	out = self.relu(out)

	if(trainOrTest):
		out, lossAvg1 = trainLayerLocal(out, trainOrTest, optim, self.l1, self.l2, l3=1)
		lossIndex+=1
		lossSum+=lossAvg1

	out = self.conv3(out)
	out = self.bn3(out)

	if(trainOrTest):
		if(applyIndependentLearningForDownsample):
			out, lossAvg2 = trainLayerLocal(out, trainOrTest, optim, self.l1, self.l2, l3=2)
			lossIndex+=1
			lossSum+=lossAvg2

	if self.downsample is not None:
		identity1 = self.downsample(x)
	out += identity1
	if(normaliseActivationSparsityLayerSkip):
		out = self.bnSkip(out)
	out = self.last_activation(out)

	if(trainOrTest):
		out, lossAvg3 = trainLayerLocal(out, trainOrTest, optim, self.l1, self.l2, l3=2+int(applyIndependentLearningForDownsample))
		lossIndex+=1
		lossSum+=lossAvg3		

	return out, lossSum, lossIndex, trainOrTest, optim

def ResNetForwardVicregLocal(self, x, trainOrTest, optim):
	x = self.padding(x)

	lossSum = 0.0
	lossIndex = 0
	
	if(smallInputImageSize):
		layersList = [self.layer0, self.layer1, self.layer2, self.layer3]
	else:
		layersList = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
	for l in range(len(layersList)):
		x, lossSum, lossIndex, trainOrTest, optim = layersList[l](x, lossSum, lossIndex, trainOrTest, optim)

	x = self.avgpool(x)
	x = torch.flatten(x, 1)

	if(trainOrTest):
		lossAvg = lossSum/lossIndex
	else:
		lossAvg = 0.0
		
	return x, lossAvg

def trainLayerLocal(x, trainOrTest, optim, l1, l2, l3):
	
	if(trainOrTest):
		if(networkHemispherical):
			(x1, x2) = (x[0], x[1])
		else:
			batchSize = x.shape[0]
			x1, x2 = torch.split(x, batchSize//2, dim=0)

		if(debugTrainLocal):
			print("trainLayerLocal: l1 = ", l1, ", l2 = ", l2, ", l3 = ", l3)
			
		loss = None
		accuracy = 0.0

		opt = optim[l1][l2][l3]
		opt.zero_grad()

		loss = calculateLossVICregLocal(x1, x2)

		loss.backward()
		opt.step()

		if(networkHemispherical):
			x1 = x1.detach()
			x2 = x2.detach()
			x = [x1, x2]
		else:
			x = x.detach()
	else:
		printe("trainLayerLocal error: currently requires trainOrTest=True")

	return x, loss

#distributedExecution not currently supported
def calculateLossVICregLocal(x1, x2):

	#convert to linear for VICreg
	imageSize = x1.shape[2]*x1.shape[3]
	if(trainLocalConvLocationIndependence):
		if(trainLocalConvLocationIndependenceAllPixels):
			x1 = pt.reshape(x1, (x1.shape[0], imageSize, x1.shape[1])) 
			x2 = pt.reshape(x2, (x2.shape[0], imageSize, x2.shape[1]))
			if(trainLocalConvLocationIndependenceAllPixelsCombinations):
				x1 = x1.repeat(1, imageSize, 1)
				x2 = torch.repeat_interleave(x2, imageSize, dim=1)
		elif(trainLocalConvLocationIndependenceSinglePixelRandom):
			randPixelX = pt.randint(0, x1.shape[2], (1,))
			randPixelY = pt.randint(0, x1.shape[3], (1,))
			x1 = x1[:, :, randPixelX[0], randPixelY[0]]
			x2 = x2[:, :, randPixelX[0], randPixelY[0]]	
		elif(trainLocalConvLocationIndependenceAveragedPixels):
			x1 = torch.mean(x1, dim=(2, 3), keepdim=False)
			x2 = torch.mean(x2, dim=(2, 3), keepdim=False)
	else:
		x1 = torch.flatten(x1, 1)
		x2 = torch.flatten(x2, 1)
	
	num_features = x1.shape[1]
	batch_size = x1.shape[0]
	if(debugTrainLocal):
		print("num_features = ", num_features)
		print("batch_size = ", batch_size)
		print("x1 = ", x1)
		print("x2 = ", x2)

	if(trainLocalConvLocationIndependenceAllPixelsSequential):
		numberOfPixelsIter = imageSize
		lossPixelList = []
	else:
		numberOfPixelsIter = 1
		if(trainLocalConvLocationIndependenceAllPixels):
			x1Sim = pt.reshape(x1, (x1.shape[0]*x1.shape[1], x1.shape[2])) 	#merge pixel dim with batch dim
			x2Sim = pt.reshape(x2, (x2.shape[0]*x2.shape[1], x2.shape[2])) 	#merge pixel dim with batch dim
		else:
			x1Sim = x1
			x2Sim = x2
		#if(distributedExecution):	#not currently supported
		#	x1Var = torch.cat(FullGatherLayer.apply(x1Var), dim=0)
		#	x2Var = torch.cat(FullGatherLayer.apply(x2Var), dim=0)
		x1Var = x1 - x1.mean(dim=0)	#maintain batch dim
		x2Var = x2 - x2.mean(dim=0)	#maintain batch dim
		if(trainLocalConvLocationIndependenceAllPixels):
			x1Cov = pt.reshape(x1Var, (x1Var.shape[1], x1Var.shape[0], x1Var.shape[2]))	#set pixel dim as batch dim
			x2Cov = pt.reshape(x2Var, (x2Var.shape[1], x2Var.shape[0], x2Var.shape[2]))	#set pixel dim as batch dim
		else:
			x1Cov = x1Var
			x2Cov = x2Var
		
	for i in range(numberOfPixelsIter):
		if(trainLocalConvLocationIndependenceAllPixelsSequential):
			#print("i = ", i)
			x1Sim = x1[:, i]
			x2Sim = x2[:, i]
			x1Var = x1[:, i]
			x2Var = x2[:, i]
			#if(distributedExecution):	#not currently supported
			#	x1Var = torch.cat(FullGatherLayer.apply(x1), dim=0)
			#	x2Var = torch.cat(FullGatherLayer.apply(x2), dim=0)
			x1Var = x1Var - x1Var.mean(dim=0)
			x2Var = x2Var - x2Var.mean(dim=0)
			x1Cov = x1Var
			x2Cov = x2Var

		repr_loss = F.mse_loss(x1Sim, x2Sim)

		std_x1 = torch.sqrt(x1Var.var(dim=0) + 0.0001)
		std_x2 = torch.sqrt(x1Var.var(dim=0) + 0.0001)
		std_loss = torch.mean(F.relu(1 - std_x1)) / 2 + torch.mean(F.relu(1 - std_x2)) / 2

		if(trainLocalConvLocationIndependenceAllPixels):
			x1CovT = x1Cov.permute(0, 2, 1)
			x2CovT = x2Cov.permute(0, 2, 1)
		else:
			x1CovT = x1Cov.T
			x2CovT = x2Cov.T
		cov_x = (x1CovT @ x1Cov) / (batch_size - 1)
		cov_y = (x2CovT @ x2Cov) / (batch_size - 1)
		if(trainLocalConvLocationIndependenceAllPixels):
			cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features*imageSize)
		else:
			cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(cov_y).pow_(2).sum().div(num_features)
		
		loss = (args.sim_coeff * repr_loss + args.std_coeff * std_loss + args.cov_coeff * cov_loss)
		if(trainLocalConvLocationIndependenceAllPixelsSequential):
			lossPixelList.append(loss) 
		
	if(trainLocalConvLocationIndependenceAllPixelsSequential):
		loss = torch.mean(torch.stack(lossPixelList))
	
	return loss

def off_diagonal(x):
	if(trainLocalConvLocationIndependenceAllPixels):
		b, n, m = x.shape
		assert n == m
		off = x.flatten(start_dim=1)[:, :-1].view(b, n - 1, n + 1)[:, :, 1:].flatten()
	else:
		n, m = x.shape
		assert n == m
		off = x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
	return off

