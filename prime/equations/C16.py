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
from prime.equations.equation import SequenceEquation, F, FN, SpatialN, MaxOrder
from prime.input.parametrization import spatial_diff
from prime.utils import symmetrize

class C16(SequenceEquation):
    shape = (FN, 3, 3)
    Nmax = MaxOrder-1
    componenWise = False

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)

    def allComponents(self, N):
        # First term
        result = N * (N+1) * (self.degP-1) * np.tensordot(self.Cs[N+1], np.tensordot(self.F, self.p, axes=(1, 0)), axes=(0, 1))

        # Second and third term
        tmp = self.sumCoefficientDerivativeContraction(self.M, freeIndices=1, combinatorial='K+1')
        if tmp is not None:
            result += N * tmp
        
        # Last term
        if N > 2:
            Cd = self.diff(self.Cs[N-1], order=2)
            result += (N-2) * Cd

        x = len(result.shape)
        return symmetrize(result, [x-2, x-1])