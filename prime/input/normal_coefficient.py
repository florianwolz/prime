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
from prime.input.intertwiners import InverseIntertwiner
from prime.tensor import Tensor
from prime.utils import to_tensor

class NormalCoefficient:
    def __init__(self, parametrization):
        self.parametrization = parametrization
        self.coeffs = [None for i in range(len(parametrization.fields))]
        parametrization._M = self

    def forField(self, field, components):
        i = self.parametrization.fields.index(field)

        # Assign the tensor
        self.coeffs[i] = Tensor(components, field.indices + [(3, +1)])

        # Check the index dimensions
        if self.coeffs[i].components.shape != field.components.shape + (3,):
            raise Exception("The normal coefficient does not have the correct shape.")


class normal_coefficient:
    def __init__(self, parametrization, forField):
        if parametrization._M is None:
            parametrization._M = NormalCoefficient(parametrization)
        self.parametrization = parametrization
        self.forField = forField

    def __call__(self, fn):
        comps = to_tensor(shape=self.forField.components.shape + (3,))(fn)
        self.parametrization._M.forField(self.forField,
            components=comps
        )
        return comps
