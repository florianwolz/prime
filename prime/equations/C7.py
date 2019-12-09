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

class C7(ScalarEquation):
    shape = ()
    componentWise = False

    def __init__(self, parametrization, Cs, E, F, M, p, *args, **kwargs):
        # Initialize the scalar equation
        ScalarEquation.__init__(self, parametrization, Cs, E, F, M, p, *args, **kwargs)
    
    def allComponents(self):
        result = 0

        # Third summand
        for K in range(2,self.collapse + 1):
            for J in range(2, K+1):
                Cd = self.diff(self.Cs[0], order=K)
                Md = spatial_diff(self.M, order=K-J)
                tmp = np.tensordot(Cd, Md, axes=(
                    tuple(range(1, K-J+1)) + (0,),
                    tuple(range(0, K-J+1))
                ))
                tmp = symmetrize(tmp, list(range(len(tmp.shape))))
                tmp = spatial_diff(tmp, order=J+1)
                for i in range(J+1):
                    tmp = tmp.trace(axis1=0, axis2=len(tmp.shape)-1)

                result = result + (-1)**J * binomial(K,J) * (J-1) * tmp

        return result