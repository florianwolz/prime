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

from prime import Field, Parametrization, Intertwiner, InverseIntertwiner
import sympy
import numpy as np

phis = [sympy.Symbol("phi{}".format(i+1)) for i in range(6)]

# Setup the components
g = [
  [-1+phis[0], phis[1]/sympy.sqrt(2), phis[2]/sympy.sqrt(2)],
  [phis[1]/sympy.sqrt(2), -1+phis[3], phis[4]/sympy.sqrt(2)],
  [phis[2]/sympy.sqrt(2), phis[4]/sympy.sqrt(2), -1+phis[5]]
]

# Start the field
field = Field(g, [+1, +1])

# Calculate the F coefficient
F = field.tangential_coefficient()

# Setup the parametrization
param = Parametrization(field)

expr = F[0,0,0,0]

I = Intertwiner(param)
invI = InverseIntertwiner(I)

print(invI.inverse[0])

# Get the constant intertwiner and invert
constI = np.reshape(I.constant()[0], (9,6))
#constIinv = np.array(sympy.Matrix(constI).pinv(), dtype=np.float64)



#print(I.indices)

#print(I.order(0)[0].shape)
#print(I.order(1)[0].shape)
#print(I.order(2)[0].shape)

#print(param.dofs)
