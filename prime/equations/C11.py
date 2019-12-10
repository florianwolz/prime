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

class C11(SequenceEquation):
    shape = (FN, 3, 3, 3)
    Nmax = MaxOrder-1
    componentWise = False
    name = "C11_N"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)

    def allComponents(self, N):
        # E terms
        result = 0
        tmp = self.coefficientDerivativeContraction(self.E, N=N, derivOrder=0, freeIndices=2)
        if tmp is not None:
            result = tmp

        # F terms
        tmp = self.sumCoefficientDerivativeContraction(self.F.swapaxes(1,2), N=N, freeIndices=1, combinatorial='K+1')
        if tmp is not None:
            tmp = symmetrize(tmp, [len(tmp.shape)-3, len(tmp.shape)-2])
            result -= tmp

        return result