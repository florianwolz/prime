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
from prime.tensor import Tensor


"""
Class for the kinematical coefficient

This represents the components of the p^.. coefficient.
"""
class Kinematical(Tensor):
    def __init__(self, parametrization, components, degP):
        self.parametrization = parametrization
        self.degP = degP

        if type(components) is list:
            components = np.array(components)

        # Check the dimensions of the tensor
        if len(components.shape) != 2 or np.any(np.array(components.shape) != 3):
            raise Exception("The kinematical coefficient expects 2 spatial indices")

        # Initialize
        Tensor.__init__(self, components, [(3,1), (3,1)])


"""
Syntactic sugar for coefficients in functional form
"""
class kinematical_coefficient:
    def __init__(self, parametrization, degP):
        self.parametrization = parametrization
        self.degP = degP

    def __call__(self, fn):
        return Kinematical(
            parametrization=self.parametrization,
            components=[[fn(a,b) for b in range(3)] for a in range(3)],
            degP=self.degP
        )
