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
from prime.equations.equation import SequenceEquation, F, SpatialN, MaxOrder
from prime.input.parametrization import spatial_diff
from prime.utils import symmetrize

class C17(SequenceEquation):
    shape = (FN+1, 3)
    Nmax = MaxOrder-1
    componenWise = False

    def __init__(self, parametrization, Cs, E, F, M, p):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p)

    def allComponents(self, N):
        V = np.tensordot(self.E, self.p, axes=(1,1)) - np.tensordot(self.F, spatial_diff(self.p, order=1), axes=((2,0), (1,1)))
        result = (N+2)* (self.degP-1) * np.tensordot(self.Cs[N+2], V, axes=(0,0))

        # First M term
        tmp = np.tensordot(self.Cs[N+1], self.diff(self.M, order=0).swapaxes(1,2), axes=(0, 0))
        result -= (N+1) * symmetrize(tmp, list(range(N+1)))

        # The other M terms
        tmp = self.sumCoefficientDerivativeContraction(self.M, N=N+1, freeIndices=0)
        if tmp is not None:
            result -= tmp
        
        # Multiply everything we have so far by (N+1)
        result = (N+1) * result

        # The sum with the index swapping
        Cd = self.diff(self.Cs[N], order=1)
        for K in range(0,N+1):
            result -= Cd.swapaxes(K, len(self.Cs.shape)-2)
        
        # Last term
        result += 2 * spatial_diff(self.diff(self.Cs[N], order=2), order=1).trace(0, N+2)

        return result