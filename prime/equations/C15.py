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
from prime.utils import symmetrize, binomial

class C15(SequenceEquation):
    shape = (FN+1, 3, 3)
    Nmax = MaxOrder-1
    componenWise = False
    name = "C15_N"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        SequenceEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)

    def allComponents(self, N):
        Cd = self.diff(self.Cs[N], order=2)
        x = len(Cd.shape)
        return Cd - Cd.swapaxes(x-4, x-3)