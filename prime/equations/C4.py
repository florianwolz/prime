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
from prime.equations.equation import ScalarEquation, F
from prime.utils import symmetrize

class C4(ScalarEquation):
    shape = (F,3)
    componentWise = False
    name = "C4"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        # Initialize the scalar equation
        ScalarEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)
    
    """
    Index layout: (B, mu)
    """
    def allComponents(self):
        V = np.tensordot(self.p, self.E, axes=(1,1)) - np.tensordot(spatial_diff(self.p, order=1), self.F, axes=((0,2),(2,1)))

        # First summand. It is transposed into the shape (B,nu)
        result = 2 * (self.degP-1) * np.tensordot(self.Cs[2], V, axes=(0,1))
        
        # Second summand
        result -= np.tensordot(self.Cs[1], self.diff(self.M, order=0), axes=(0,0)).transpose()

        # Third summand
        tmp = self.sumCoefficientDerivativeContraction(self.M, N=1)
        if tmp is not None:
            result -= tmp
        
        # Fourth summand
        tmp = self.sumCoefficientDerivativeTrace(N=0, freeIndices=1, combinatorial='K+1', alternatingSign=True)
        if tmp is not None:
            result -= tmp
        
        return result