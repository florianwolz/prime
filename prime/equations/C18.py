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

class C18(SequenceEquation):
    shape = (F, F, SpatialN)
    Nmax = Collapse
    componentWise = False

    def __init__(self, parametrization, Cs, E, F, M, p):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p)

    def allComponents(self, N):
        result = self.diff(self.Cs[1], order=N)

        tmp = self.sumCoefficientDerivativeTrace(N=1, freeIndices=N, alternatingSign=True, combinatorial='binomial(K+N, K)')
        if tmp is not None:
            result = result - (-1)**N * tmp

        return result
