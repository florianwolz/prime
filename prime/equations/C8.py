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
from prime.equations.equation import SequenceEquation, SpatialN, Collapse
from prime.utils import symmetrize

class C8(SequenceEquation):
    shape = (SpatialN, 3)
    Nmax = Collapse + 1
    componentWise = False
    name = "C8_N"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)

    def allComponents(self, N):
        result = None

        # First term
        tmp = self.sumCoefficientDerivativeContraction(self.E, N=0, freeIndices=N, combinatorial='binomial(K+N,K)')
        if tmp is not None:
            result = tmp
                
        # Second term
        tmp = self.sumCoefficientDerivativeContraction(self.F.swapaxes(1,2), N=0, freeIndices=N-1, combinatorial='binomial(K+N-1,K)')

        if tmp is not None:
            # Symmetrize in the N free indices
            tmp = symmetrize(tmp, list(range(0, N)))

            if result is None: result = -tmp
            else: result -= tmp
                
        return result