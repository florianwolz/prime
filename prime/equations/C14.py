#   Copyright 2019 The Prime Authors
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
from prime.equations.equation import SequenceEquation, FN, SpatialN, MaxOrder
from prime.input.parametrization import spatial_diff
from prime.utils import symmetrize, binomial

class C14(SequenceEquation):
    shape = (FN-1, 3,3)
    Nmax = MaxOrder
    componenWise = False

    def __init__(self, parametrization, Cs, E, F, M, p):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p)

    def allComponents(self, N):
        # Calculate the coefficient in brackets
        tmp = np.tensordot(self.diff(self.M, order=0), self.M, axes=(2, 0))
        tmp = tmp + (self.degP -1) * np.tensordot(self.F, self.p, axes=(1, 0))
        tmp = tmp - tmp.transpose((0,2,1))

        return np.tensordot(self.Cs[N], tmp, axes=(0,0))