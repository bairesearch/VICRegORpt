"""VICRegORpt_resnet_positiveWeights.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see VICRegORpt_globalDefs.py

# Usage:
see VICRegORpt_globalDefs.py

# Description:
vicreg biological resnet positive weights

"""

from VICRegORpt_globalDefs import *
import torch
import torch.nn as nn
import VICRegORpt_resnet_vicregLocal
			
class BatchNormLayer(nn.Module):
	def __init__(self, num_out_filters):
		super(BatchNormLayer, self).__init__()
		if(normaliseActivationSparsityBatch):
			batchNormFunction = nn.BatchNorm2d(num_out_filters)
		if(normaliseActivationSparsityLayer):
			if(normaliseActivationSparsityLayerFunctionInstanceNorm2d):
				layerNormFunction = nn.InstanceNorm2d(num_out_filters)	#nn.GroupNorm(1, num_out_filters)
			elif(normaliseActivationSparsityLayerFunctionLayerNorm):
				layerNormFunction = nn.LayerNorm(num_out_filters)
			elif(normaliseActivationSparsityLayerFunctionGroupNorm):
				layerNormFunction = nn.GroupNorm(1, num_out_filters)
		if(trainLocal):
			if(normaliseActivationSparsityBatch):
				self.batchNormFunction = VICRegORpt_resnet_vicregLocal.ArbitraryLayerVICregLocal(batchNormFunction)
			if(normaliseActivationSparsityLayer):
				self.layerNormFunction = VICRegORpt_resnet_vicregLocal.ArbitraryLayerVICregLocal(layerNormFunction)
		else:
			if(normaliseActivationSparsityBatch):
				self.batchNormFunction = batchNormFunction
			if(normaliseActivationSparsityLayer):
				self.layerNormFunction = layerNormFunction

	def forward(self, x):
		#TODO: check order of norm function execution
		if(normaliseActivationSparsityBatch):
			x = self.batchNormFunction(x)
		if(normaliseActivationSparsityLayer):
			x = self.layerNormFunction(x)
		return x

class LayerNormLayer(nn.Module):
	def __init__(self, num_out_filters):
		super(LayerNormLayer, self).__init__()
		layerNormFunction = nn.InstanceNorm2d(num_out_filters)	#nn.GroupNorm(1, num_out_filters)
		if(trainLocal):
			self.layerNormFunction = VICRegORpt_resnet_vicregLocal.ArbitraryLayerVICregLocal(layerNormFunction)
		else:
			self.layerNormFunction = layerNormFunction

	def forward(self, x):
		x = self.layerNormFunction(x)
		return x
		
			
def createBatchNormLayer():
	if(normaliseActivationSparsityLayer):
		norm_layer = BatchNormLayer
	else:
		if(trainLocal):
			norm_layer = BatchNormLayer
		else:
			norm_layer = nn.BatchNorm2d
	return norm_layer

def createLayerNormLayer():
	if(normaliseActivationSparsityLayerSkip):
		if(normaliseActivationSparsityLayerFunctionInstanceNorm2d):
			layerNormFunction = nn.InstanceNorm2d
		elif(normaliseActivationSparsityLayerFunctionLayerNorm):
			layerNormFunction = nn.LayerNorm
		elif(normaliseActivationSparsityLayerFunctionGroupNorm):
			layerNormFunction = nn.GroupNorm
		if(trainLocal):
			lnorm_layer = LayerNormLayer
		else:
			lnorm_layer = layerNormFunction
	else:
		printe("createLayerNormLayer requires normaliseActivationSparsityLayerSkip")

	return lnorm_layer

class SoftmaxLayer(nn.Module):
	def __init__(self, ):
		super(SoftmaxLayer, self).__init__()
		softmaxFunction = SoftmaxConv2d()
		if(trainLocal):
			self.softmaxFunction = VICRegORpt_resnet_vicregLocal.ArbitraryLayerVICregLocal(softmaxFunction)
		else:
			self.softmaxFunction = softmaxFunction

	def forward(self, x):
		x = self.softmaxFunction(x)
		return x

class SoftmaxConv2d(nn.Module):
	def __init__(self, ):
		super(SoftmaxConv2d, self).__init__()
		self.softmaxFunctionConv2d = nn.Softmax(dim=1)
	def forward(self, x):
		numberOfSamples = x.shape[0]
		numberOfSublayers = x.shape[1]
		numberOfPixelsX = x.shape[2]
		numberOfPixelsY = x.shape[2]
		if(activationFunctionTypeSoftmaxIndependentChannels):
			#from executeActivationLayer;
			x = pt.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
			x = self.softmaxFunctionConv2d(x)
			x = pt.reshape(x, (numberOfSamples, numberOfSublayers, x.shape[1], x.shape[2]))
		else:
			x = self.softmaxFunction(x)
		return x

def generateActivationFunction():
	if(activationFunctionType=="relu"):
		activation = nn.ReLU(inplace=True)
	elif(activationFunctionType=="none"):
		activation = None
	elif(activationFunctionType=="softmax"):
		activation = SoftmaxLayer()
	return activation
