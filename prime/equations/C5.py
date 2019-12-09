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

class C5(ScalarEquation):
    shape = (3,)
    componentWise = False

    def __init__(self, parametrization, Cs, E, F, M, p, *args, **kwargs):
        # Initialize the scalar equation
        ScalarEquation.__init__(self, parametrization, Cs[0:2], E, F, M, p, *args, **kwargs)
    
    def allComponents(self):
        # First summand
        tmp = np.tensordot(self.Cs[1], np.tensordot(self.diff(self.M, order=0), self.M, axes=(2,0)), axes=(0,0))
        tmp = tmp - tmp.transpose()
        result = spatial_diff(tmp, order=1).trace(axis1=0, axis2=1)

        # Second summand
        tmp = np.tensordot(self.Cs[1], self.E, axes=(0,0))
        tmp = tmp + spatial_diff(np.tensordot(self.Cs[1], self.F, axes=(0,0)), order=1).trace(0,2)
        tmp = np.tensordot(self.p, tmp, axes=(0,0))
        result = result - 2 * (self.degP-1) * tmp
        
        # Second summand
        Md = self.diff(self.M, order=0)
        result = result - np.tensordot(self.Cs[1], Md, axes=(0,0)).transpose()

        # Third summand
        # TODO: check this
        for K in range(self.collapse + 1):
            for J in range(0, K+1):
                Cd = self.diff(self.Cs[0], order=K)
                Md = spatial_diff(self.M, order=K-J)
                tmp = np.tensordot(Cd, Md, axes=(
                    tuple(range(1, K-J+1)) + (0,),
                    tuple(range(0, K-J+1))
                ))
                tmp = symmetrize(tmp, list(range(len(tmp.shape))))
                if J>0:
                    tmp = spatial_diff(tmp, order=J)
                for i in range(J):
                    tmp = tmp.trace(0, len(tmp.shape)-1)

                result = result + (-1)**J * binomial(K,J) * (J+1) * tmp

        # Last summand.        
        tmp = self.sumCoefficientDerivativeContraction(self.M, N=0, freeIndices=0)
        if tmp is not None:
            result += tmp

        return result 
