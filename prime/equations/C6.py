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
from prime.utils import symmetrize, binomial

class C6(ScalarEquation):
    shape = (F,F,3)
    componentWise = False

    def __init__(self, parametrization, Cs, E, F, M, p, *args, **kwargs):
        # Initialize the scalar equation
        ScalarEquation.__init__(self, parametrization, Cs[0:2], E, F, M, p, *args, **kwargs)
    
    # TODO: In the Cs also setup the first unconsidered one as O(1)
    
    def allComponents(self):
        # First term
        V = np.tensordot(self.p, self.E, axes=(1,1)) - np.tensordot(spatial_diff(expr=self.p, order=1), self.F, axes=((0,2),(2,1)))
        result = 6*(self.degP-1) * np.tensordot(self.Cs[3], V, axes=(0, 1))

        # Second term. In the M_: term we swap B2 and mu
        result -= 4 * symmetrize(np.tensordot(self.Cs[2], self.diff(expr=self.M, order=0).swapaxes(1,2), axes=(0,0)), [0,1])

        # Third to fifth term
        tmp = self.sumCoefficientDerivativeContraction(self.M, N=2)
        if tmp is not None:
            result -= 2 * tmp

        # Sixth summand. Need to swap the first and second axes to get into the B1 B2 order
        result -= self.diff(expr=self.Cs[1], order=1).swapaxes(0,1)

        # Last summand
        tmp = self.sumCoefficientDerivativeTrace(N=1, freeIndices=1, combinatorial='K+1', alternatingSign=True)
        if tmp is not None:
            result -= tmp
        
        return result
