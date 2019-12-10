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
from prime.equations.equation import SequenceEquation, FN, MaxOrder
from prime.utils import symmetrize

class C10(SequenceEquation):
    shape = (FN, 3, 3)
    Nmax = MaxOrder-1
    componentWise = False
    name = "C10_N"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)

    def allComponents(self, N):
        result = - np.tensordot(self.Cs[N], np.eye(3), axes=0)

        # Subtract the second term
        tmp = np.tensordot(self.Cs[N], self.diff(self.F, order=0).transpose(0,3,2,1), axes=(0,0))
        tmp = symmetrize(tmp, list(range(N)))
        result -= N * tmp

        # E terms
        tmp = self.sumCoefficientDerivativeContraction(self.E, N=N, freeIndices=1, combinatorial='K+1')
        if tmp is not None:
            result += tmp
        
        # F terms
        tmp = self.sumCoefficientDerivativeContraction(self.F.swapaxes(1,2))
        if tmp is not None:
            result -= tmp
        
        return result