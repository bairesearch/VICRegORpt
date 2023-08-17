"""VICRegORpt_operations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see VICRegORpt_globalDefs.py

# Usage:
see VICRegORpt_globalDefs.py

# Description:
vicreg biological operations

"""

import torch
from VICRegORpt_globalDefs import *

def weightsSetPositiveModel(self):
	if(usePositiveWeights):
		if(usePositiveWeightsClampModel):
			for p in self.parameters():
				p.data.clamp_(0)

def createLocalOptimisers(model):
	if(networkHemispherical):
		backbone = model.backbone1
	else:
		backbone = model.backbone
		
	l1Len = len(backbone.layers)
	l1optim = []
	for l1 in range(l1Len):
		l2optim = []
		l2Len = backbone.layers[l1]
		for l2 in range(l2Len):
			l3optim = []
			if(l1 == 0):
				l3Len = 1
			else:
				if(backbone.block.name == "Bottleneck"):
					l3Len = 3
				elif(backbone.block.name == "BasicBlock"):
					l3Len = 2
				else:
					printe("error: backbone.block.name not found")
				if(applyIndependentLearningForDownsample):
					l3Len += 1
			for l3 in range(l1Len):
				opt = torch.optim.Adam(backbone.parameters(), lr=learningRateLocal)	#LARS
				l3optim.append(opt)
			l2optim.append(l3optim)
		l1optim.append(l2optim)
	optim = l1optim
	return optim
