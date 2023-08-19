# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from VICRegORpt_globalDefs import *


class GaussianBlur(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if np.random.rand() < self.p:
			sigma = np.random.rand() * 1.9 + 0.1
			return img.filter(ImageFilter.GaussianBlur(sigma))
		else:
			return img


class Solarization(object):
	def __init__(self, p):
		self.p = p

	def __call__(self, img):
		if np.random.rand() < self.p:
			return ImageOps.solarize(img)
		else:
			return img

class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std
		self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

	def __call__(self, img):
		if(usePositiveWeights):
			return img
		else:
			return self.normalize(img)
		
class TrainTransform(object):
	def __init__(self, imageWidth):
		self.transform = transforms.Compose(
			[
				transforms.RandomResizedCrop(
					imageWidth, interpolation=InterpolationMode.BICUBIC
				),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.RandomApply(
					[
						transforms.ColorJitter(
							brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
						)
					],
					p=0.8,
				),
				transforms.RandomGrayscale(p=0.2),
				GaussianBlur(p=1.0),
				Solarization(p=0.0),
				transforms.ToTensor(),
				Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)
		self.transform_prime = transforms.Compose(
			[
				transforms.RandomResizedCrop(
					imageWidth, interpolation=InterpolationMode.BICUBIC
				),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.RandomApply(
					[
						transforms.ColorJitter(
							brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
						)
					],
					p=0.8,
				),
				transforms.RandomGrayscale(p=0.2),
				GaussianBlur(p=0.1),
				Solarization(p=0.2),
				transforms.ToTensor(),
				Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			]
		)

	def __call__(self, sample):
		if(networkHemisphericalStereoInput):
			x = self.transform(sample)
			x1 = transforms.functional.crop(x,0,0,imageWidth,imageWidth//2)
			x2 = transforms.functional.crop(x,0,imageWidth//2,imageWidth,imageWidth//2)
		else:
			x1 = self.transform(sample)
			x2 = self.transform_prime(sample)
		return x1, x2
