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

class C13(SequenceEquation):
    shape = (FN, 3, 3, 3)
    Nmax = MaxOrder-1
    componentWise = False

    def __init__(self, parametrization, Cs, E, F, M, p):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p)

    def allComponents(self, N):
        Cd = self.diff(expr=self.Cs[N], order=2)

        # Contract the coefficient with F
        result = np.tensordot(Cd, self.M, axes=(len(Cd.shape)-3,0))

        # Symmetrize in alpha, beta, gamma
        return symmetrize(result, list(range(len(result.shape)-3, len(result.shape))))