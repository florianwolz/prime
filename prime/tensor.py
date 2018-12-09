#   Copyright 2018 The Prime Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import sympy

class Tensor:
    def __init__(self, tensor, indices):
        # Convert to a numpy array
        if type(tensor) is list:
            tensor = np.array(tensor)

        # Check the indices
        if len(tensor.shape) != len(indices):
            raise Exception("Not enough index information. Given {} for tensor of shape {}".format(indices, tensor.shape))

        # Store
        self.components = tensor
        self.indices = indices

    def __getitem__(self, pos):
        return self.components[pos]

    def __str__(self):
        return str(self.components)

    def __repr__(self):
        return str(self)
