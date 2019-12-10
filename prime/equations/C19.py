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
from prime.equations.equation import SequenceEquation, F, SpatialN, Collapse
from prime.input.parametrization import spatial_diff
from prime.utils import symmetrize, binomial

class C19(SequenceEquation):
    shape = (F, SpatialN, 3)
    Nmax = Collapse
    componentWise = False
    name = "C19_N"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)

    def allComponents(self, N):
        # The C_A term
        tmpA = self.sumCoefficientDerivativeContraction(self.M, N=1, freeIndices=N, combinatorial='binomial(K+N,K)')
        if tmpA is not None:
            tmpA = symmetrize(tmpA, list(range(len(tmpA.shape)-N, len(tmpA.shape))))

        # The C term
        tmpB = self.sumCoefficientDerivativeTrace(N=0, freeIndices=N+1, combinatorial='binomial(K+N-1,K)', alternatingSign=True)
        if tmpB is not None:
            tmpB = (-1)**N * tmpB

        # Return the result
        if tmpA is not None and tmpB is not None:
            return tmpA + tmpB
        elif tmpA is not None:
            return tmpA
        elif tmpB is not None:
            return tmpB
        else:
            return np.array([0])