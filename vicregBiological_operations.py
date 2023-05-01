"""vicregBiological_operations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022-2023 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see vicregBiological_globalDefs.py

# Usage:
see vicregBiological_globalDefs.py

# Description:
vicreg biological operations

"""

from vicregBiological_globalDefs import *

def weightsSetPositiveModel(self):
	if(usePositiveWeights):
		if(usePositiveWeightsClampModel):
			for p in self.parameters():
				p.data.clamp_(0)
