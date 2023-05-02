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

from VICRegORpt_globalDefs import *

def weightsSetPositiveModel(self):
	if(usePositiveWeights):
		if(usePositiveWeightsClampModel):
			for p in self.parameters():
				p.data.clamp_(0)
