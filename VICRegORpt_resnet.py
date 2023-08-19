# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from VICRegORpt_globalDefs import *
if(trainLocal):
	import VICRegORpt_resnet_vicregLocal
	if(networkHemispherical):
		import VICRegORpt_resnet_vicregHemispherical
	from VICRegORpt_resnet_vicregLocal import sequentialMultiInput
#if(vicregBiologicalMods):
import VICRegORpt_resnet_positiveWeights


def setArgs(argsNew):
	if(trainLocal):
		VICRegORpt_resnet_vicregLocal.setArgs(argsNew)
	if(networkHemispherical):
		VICRegORpt_resnet_vicregHemispherical.setArgs(argsNew)
	VICRegORpt_resnet_positiveWeights.setArgs(argsNew)
	
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(
		in_planes,
		out_planes,
		kernel_size=3,
		stride=stride,
		padding=dilation,
		groups=groups,
		bias=False,
		dilation=dilation,
	)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def createBatchNormLayer(norm_layer):
	if norm_layer is None:
		if(vicregBiologicalMods):
			norm_layer = VICRegORpt_resnet_positiveWeights.createBatchNormLayer()
		else:
			norm_layer = nn.BatchNorm2d
	return norm_layer
				
class Input(nn.Module):
	name = "Input"
	def __init__(
		self, num_channels, num_out_filters, norm_layer, l1, l2
	):
		super(Input, self).__init__()
		self.conv1 = nn.Conv2d(
			num_channels,
			num_out_filters,
			kernel_size=7,
			stride=2,
			padding=2,
			bias=False,
		)
		norm_layer = createBatchNormLayer(norm_layer)
		self.bn1 = norm_layer(num_out_filters)
		self.relu = VICRegORpt_resnet_positiveWeights.generateActivationFunction()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.l1 = l1
		self.l2 = l2

	if(trainLocal):
		def forward(self, x, lossSum, lossIndex, trainOrTest, optim):
			return VICRegORpt_resnet_vicregLocal.InputForwardVicregLocal(self, x, lossSum, lossIndex, trainOrTest, optim)
	else:
		def forward(self, x):
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
			x = self.maxpool(x)
			return x
			
class BasicBlock(nn.Module):
	expansion = 1
	__constants__ = ["downsample"]
	name = "BasicBlock"
	
	def __init__(
		self,
		inplanes,
		planes,
		stride=1,
		downsample=None,
		groups=1,
		base_width=64,
		dilation=1,
		norm_layer=None,
		last_activation="relu",
		l1=None,
		l2=None,
	):
		super(BasicBlock, self).__init__()
		norm_layer = createBatchNormLayer(norm_layer)
		if groups != 1 or base_width != 64:
			raise ValueError("BasicBlock only supports groups=1 and base_width=64")
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = VICRegORpt_resnet_positiveWeights.generateActivationFunction()
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride
		self.l1 = l1
		self.l2 = l2
		if(normaliseActivationSparsityLayerSkip):
			expansion = 1
			lnorm_layer = VICRegORpt_resnet_positiveWeights.createLayerNormLayer()
			self.lnSkip = lnorm_layer(planes*expansion)

	if(trainLocal):
		def forward(self, x, lossSum, lossIndex, trainOrTest, optim):
			return VICRegORpt_resnet_vicregLocal.BasicBlockForwardVicregLocal(self, x, lossSum, lossIndex, trainOrTest, optim)
	else:
		def forward(self, x):
			identity = x

			out = self.conv1(x)
			out = self.bn1(out)
			out = self.relu(out)

			out = self.conv2(out)
			out = self.bn2(out)

			if self.downsample is not None:
				identity = self.downsample(x)

			out += identity
			if(normaliseActivationSparsityLayerSkip):
				out = self.lnSkip(out)
			out = self.relu(out)

			return out


class Bottleneck(nn.Module):
	expansion = 4
	__constants__ = ["downsample"]
	name = "Bottleneck"

	def __init__(
		self,
		inplanes,
		planes,
		stride=1,
		downsample=None,
		groups=1,
		base_width=64,
		dilation=1,
		norm_layer=None,
		last_activation="relu",
		l1=None,
		l2=None,
	):
		super(Bottleneck, self).__init__()
		norm_layer = createBatchNormLayer(norm_layer)
		width = int(planes * (base_width / 64.0)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = VICRegORpt_resnet_positiveWeights.generateActivationFunction()
		self.downsample = downsample
		self.stride = stride
		self.l1 = l1
		self.l2 = l2
		if(normaliseActivationSparsityLayerSkip):
			expansion = 4
			lnorm_layer = VICRegORpt_resnet_positiveWeights.createLayerNormLayer()
			self.lnSkip = lnorm_layer(planes*expansion)

		if last_activation == "relu":
			self.last_activation = VICRegORpt_resnet_positiveWeights.generateActivationFunction()
		elif last_activation == "none":
			self.last_activation = nn.Identity()
		elif last_activation == "sigmoid":
			self.last_activation = nn.Sigmoid()

	if(trainLocal):
		def forward(self, x, lossSum, lossIndex, trainOrTest, optim):
			return VICRegORpt_resnet_vicregLocal.BottleneckForwardVicregLocal(self, x, lossSum, lossIndex, trainOrTest, optim)
	else:
		def forward(self, x):
			identity = x

			out = self.conv1(x)
			out = self.bn1(out)
			out = self.relu(out)

			out = self.conv2(out)
			out = self.bn2(out)
			out = self.relu(out)

			out = self.conv3(out)
			out = self.bn3(out)

			if self.downsample is not None:
				identity = self.downsample(x)

			out += identity
			if(normaliseActivationSparsityLayerSkip):
				out = self.lnSkip(out)
			out = self.last_activation(out)

			return out


class ResNet(nn.Module):
	def __init__(
		self,
		block,
		layers,
		num_channels=3,
		zero_init_residual=False,
		groups=1,
		widen=1,
		width_per_group=64,
		replace_stride_with_dilation=None,
		norm_layer=None,
		last_activation="relu",
	):
		super(ResNet, self).__init__()
		norm_layer = createBatchNormLayer(norm_layer)
		self._norm_layer = norm_layer
		# self._last_activation = last_activation

		if(trainLocal):
			self.block = block
			self.layers = layers
		
		self.padding = nn.ConstantPad2d(1, 0.0)

		self.inplanes = width_per_group * widen
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError(
				"replace_stride_with_dilation should be None "
				"or a 3-element tuple, got {}".format(replace_stride_with_dilation)
			)
		self.groups = groups
		self.base_width = width_per_group

		# change padding 3 -> 2 compared to original torchvision code because added a padding layer
		num_out_filters = width_per_group * widen
		self.layer0 = Input(num_channels, num_out_filters, norm_layer, l1=0, l2=0)
		self.layer1 = self._make_layer(block, num_out_filters, layers[1], l1=1)
		num_out_filters *= 2
		self.layer2 = self._make_layer(
			block,
			num_out_filters,
			layers[2],
			stride=2,
			dilate=replace_stride_with_dilation[0],
			l1=2,
		)
		num_out_filters *= 2
		self.layer3 = self._make_layer(
			block,
			num_out_filters,
			layers[3],
			stride=2,
			dilate=replace_stride_with_dilation[1],
			l1=3,
		)
		num_out_filters *= 2
		self.layer4 = self._make_layer(
			block,
			num_out_filters,
			layers[4],
			stride=2,
			dilate=replace_stride_with_dilation[2],
			last_activation=last_activation,
			l1=4,
		)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		if(not normaliseActivationSparsityLayer and not trainLocal):
			# Zero-initialize the last BN in each residual branch,
			# so that the residual branch starts with zeros, and each residual block behaves like an identity.
			# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
			if zero_init_residual:
				for m in self.modules():
					if isinstance(m, Bottleneck):
						nn.init.constant_(m.bn3.weight, 0)
					elif isinstance(m, BasicBlock):
						nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(
		self, block, planes, blocks, stride=1, dilate=False, last_activation="relu", l1=None
	):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(
			block(
				self.inplanes,
				planes,
				stride,
				downsample,
				self.groups,
				self.base_width,
				previous_dilation,
				norm_layer,
				last_activation=(last_activation if blocks == 1 else "relu"),
				l1=l1,
				l2=0,
			)
		)
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(
				block(
					self.inplanes,
					planes,
					groups=self.groups,
					base_width=self.base_width,
					dilation=self.dilation,
					norm_layer=norm_layer,
					last_activation=(last_activation if i == blocks - 1 else "relu"),
					l1=l1,
					l2=i,
				)
			)

		if(trainLocal):
			sequential = sequentialMultiInput(*layers)
		else:
			sequential = nn.Sequential(*layers)
		
		return sequential

	if(trainLocal):
		def forward(self, x, trainOrTest=False, optim=None):
			x, lossAvg = VICRegORpt_resnet_vicregLocal.ResNetForwardVicregLocal(self, x, trainOrTest, optim)
			if(trainOrTest):
				return x, lossAvg
			else:
				#print("x = ", x)
				return x
	else:
		def forward(self, x):
			#print("x = ", x)
			x = self.padding(x)
	
			if(smallInputImageSize):
				layersList = [self.layer0, self.layer1, self.layer2, self.layer3]
			else:
				layersList = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
			for l in range(len(layersList)):
				x = layersList[l](x)
		
			x = self.avgpool(x)
			x = torch.flatten(x, 1)
	
			return x

def resnet34(**kwargs):
	return ResNet(BasicBlock, [1, 3, 4, 6, 3], **kwargs), 512, blockTypeBasic


def resnet50(**kwargs):
	if(smallInputImageSize):
		return ResNet(Bottleneck, [1, 3, 4, 6, 3], **kwargs), 1024, blockTypeBottleneck
	else:
		return ResNet(Bottleneck, [1, 3, 4, 6, 3], **kwargs), 2048, blockTypeBottleneck


def resnet101(**kwargs):
	return ResNet(Bottleneck, [1, 3, 4, 23, 3], **kwargs), 2048, blockTypeBottleneck


def resnet50x2(**kwargs):
	return ResNet(Bottleneck, [1, 3, 4, 6, 3], widen=2, **kwargs), 4096, blockTypeBottleneck


def resnet50x4(**kwargs):
	return ResNet(Bottleneck, [1, 3, 4, 6, 3], widen=4, **kwargs), 8192, blockTypeBottleneck


def resnet50x5(**kwargs):
	return ResNet(Bottleneck, [1, 3, 4, 6, 3], widen=5, **kwargs), 10240, blockTypeBottleneck


def resnet200x2(**kwargs):
	return ResNet(Bottleneck, [1, 3, 24, 36, 3], widen=2, **kwargs), 4096, blockTypeBottleneck
