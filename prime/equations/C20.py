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
from prime.equations.equation import SequenceEquation, F, SpatialN, Collapse
from prime.utils import symmetrize, binomial

class C20(SequenceEquation):
    shape = (SpatialN)
    Nmax = Collapse
    onlyEven = True
    componentWise = False

    def __init__(self, parametrization, Cs, E, F, M, p, *args, **kwargs):
        # Initialize the scalar equation
        SequenceEquation.__init__(self, parametrization, [Cs[0]], E, F, M, p, *args, **kwargs)
    
    def allComponents(self, N):
        result = 0
        
        for K in range(N,self.collapse + 1):
            for J in range(N+1, K+2):
                Cd = self.diff(self.Cs[0], order=K)
                Md = spatial_diff(self.M, order=K-J+1)
                tmp = np.tensordot(Cd, Md, axes=(
                    tuple(range(1, K-J+2)) + (0,),
                    tuple(range(0, K-J+2))
                ))
                tmp = symmetrize(tmp, list(range(len(tmp.shape))))
                tmp = spatial_diff(tmp, order=J-N)
                for i in range(J-N+1):
                    tmp = tmp.trace(0, len(tmp.shape)-1)

                result = result + (-1)**J * binomial(K,J-1) * binomial(J,N) * tmp

        return result