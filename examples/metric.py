#   Copyright 2018 The Prime Authors
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

import prime
import sympy as sp

# Setup the phis
phis = prime.phis(6)

# Setup the fields
g = prime.Field([[-1 + phis[0], phis[1] / sp.sqrt(2), phis[2] / sp.sqrt(2)],
                 [phis[1]/sp.sqrt(2), -1 + phis[3], phis[4] / sp.sqrt(2)],
                 [phis[2]/sp.sqrt(2), phis[4] / sp.sqrt(2), -1 + phis[5]]], [+1,+1])

#g = prime.Field([[-1 + phis[0], phis[1], phis[2]],
#                 [phis[1], -1 + phis[3], phis[4]],
#                 [phis[2], phis[4], -1 + phis[5]]], [1, 1])

# Setup the parametrization
param = prime.Parametrization(fields=[g])

# Setup the kinematical coefficient
P = prime.Kinematical(param, components=g.components, degP=2)

# Solve
prime.solve(
    parametrization=param,
    kinematical_coefficient=P,

    # Linear equations of motion
    order=1
)
