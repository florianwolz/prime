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

class C3(ScalarEquation):
    shape = (F,3,3)
    componentWise = False
    name = "C3"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        # Initialize the scalar equation
        ScalarEquation.__init__(self, parametrization, Cs[0:3], E, F, M, p, degP, *args, **kwargs)
    
    """
    Calculate all components

    Index layout: B mu nu 
    """
    def allComponents(self):
        # First summand. It is transposed into the shape (B,mu,nu)
        result = 2 * (self.degP-1) * np.tensordot(np.tensordot(self.p, self.F, axes=(1, 1)), self.Cs[2], axes=(1, 0)) \
                    .transpose((2,0,1))

        # Second summand
        tmp = self.sumCoefficientDerivativeContraction(self.M, N=1, freeIndices=1, combinatorial='K+1')
        if tmp is not None:
            result += tmp
        
        # Third summand
        # FIXME
        tmp = self.sumCoefficientDerivativeTrace(N=1, freeIndices=2, combinatorial='(K+2)*(K+1)/2', alternatingSign=True)
        if tmp is not None:
            result -= tmp

        return symmetrize(result, [1,2])