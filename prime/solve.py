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

from prime.input.field import Field
from prime.input.parametrization import Parametrization
from prime.input.intertwiners import Intertwiner, InverseIntertwiner
from prime.input.kinematical import Kinematical
from prime.input.normal_coefficient import NormalCoefficient
from prime.utils import dropHigherOrder
from prime.output import all_coefficients

import numpy as np
from sympy import Symbol


"""
Start solving the gravitational closure equations

Args:
    parametrization             The parametrization of the theory
    kinematical_coefficient     The kinematical coefficient (p^..)
    normal_coefficient          The normal deformation coefficient (M^A\\gamma)
    order                       The order of the resulting equations of motion
"""
def solve(parametrization, kinematical_coefficient, normal_coefficient=None, order=1):
    print("Start calculating the remaining input coefficients ...")

    # Start by calculation of the intertwiners
    I = Intertwiner(parametrization)
    J = InverseIntertwiner(I, order=order)

    print("Constructed the intertwiners. Calculate F ...")

    # Get the real F coefficient
    F = J.contractWith([field.tangential_coefficient() for field in parametrization.fields])

    print("Finished F. Moving to M ...")

    # If there is no M coefficient, setup a trivial one
    if normal_coefficient is None: normal_coefficient = parametrization._M

    if normal_coefficient is None:
        normal_coefficient = NormalCoefficient(parametrization)

    # Get the real M coefficient
    M = np.zeros((len(parametrization.dofs), 3))
    for i in range(len(parametrization.fields)):
        # The field has a trivial M coefficient, continue
        if normal_coefficient.coeffs[i] is None:
            continue

        # Else, take the coefficient and contract with the inverse intertwiner
        M = M + np.tensordot(J.components[i], normal_coefficient.coeffs[i].components,
                axes=(tuple(range(1,len(J.components[i].shape))), tuple(range(len(J.components[i].shape)-1)))
            )

    print("Finished M. Drop higher order terms ...")

    # Drop all the higher order terms in the tensors, since we won't need them
    F = dropHigherOrder(F, parametrization.dofs, order=order)
    M = dropHigherOrder(M, parametrization.dofs, order=order+1)
    p = dropHigherOrder(kinematical_coefficient.components, parametrization.dofs, order=order+1)

    print("Dropped higher order terms. Calculate E ...")

    # Setup the E coefficient
    E = np.array([[Symbol("{}_{}".format(str(phi), chr(ord('x')+mu))) for mu in range(3)] for phi in parametrization.dofs])

    print("Finished E. Add the dirtbag terms to the coefficients ...")

    # Add the O(n) terms to the coefficients
    #o1 = parametrization.O(order+1)
    #o2 = parametrization.O(order+2)

    #F = F + o1
    #M = M + o2
    #p = p + o2
    #E = E + o1 # TODO: check

    print("Finished. Setting up all the closure equations ...")

    #print(p)
    #print(F)
    #print(M)

    # Generate the list of all the output coefficients
    coeffs = all_coefficients(parametrization, J, order=order+1, collapse=2)

    #coeffs[-1][0].generate()

    print("Done.")
