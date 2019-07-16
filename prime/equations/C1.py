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
from prime.input.parametrization import spatial_diff
from prime.equations.equation import ScalarEquation

class C1(ScalarEquation):
    shape = (3,3)
    componentWise = False
    name = "C1"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        # Initialize the scalar equation
        ScalarEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)

    def allComponents(self):
        # Prepare the result
        result = - self.Cs[0] * np.eye(3)

        # Add the E term
        tmp = self.sumCoefficientDerivativeContraction(self.E, N=0, freeIndices=1, combinatorial='K+1')
        if tmp is not None:
            result += tmp

        # Add the F term (need to swap gamma and mu first in F so that the index positions match)
        tmp = self.sumCoefficientDerivativeContraction(self.F.swapaxes(1,2), N=0, freeIndices=0)
        if tmp is not None:
            result -= tmp
        
        return result