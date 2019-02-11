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
from dask.distributed import Client, get_client, secede, rejoin
import dask


"""
Start solving the gravitational closure equations

Args:
    parametrization             The parametrization of the theory
    kinematical_coefficient     The kinematical coefficient (p^..)
    normal_coefficient          The normal deformation coefficient (M^A\\gamma)
    order                       The order of the resulting equations of motion
"""
def solve(parametrization, kinematical_coefficient, normal_coefficient=None, order=1, collapse=2):
    # Setting up the cluster
    client = get_client()

    # Start by calculation of the intertwiner
    I = Intertwiner(parametrization)

    # Create a task for the calculation of the intertwiner
    J = client.submit(lambda : InverseIntertwiner(I, order=order))

    # Get the real F coefficient
    F = client.submit(lambda x, y : x.contractWith([field.tangential_coefficient() for field in y]), J, parametrization.fields)

    def calculateM(parametrization, normal_coefficient, J):
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
        return M
    M = client.submit(calculateM, parametrization, normal_coefficient, J)

    # Drop all the higher order terms in the tensors, since we won't need them
    F = client.submit(dropHigherOrder, F, parametrization.dofs, order=order)
    M = client.submit(dropHigherOrder, M, parametrization.dofs, order=order+1)
    p = client.submit(dropHigherOrder, kinematical_coefficient.components, parametrization.dofs, order=order+1)

    # Setup the E coefficient
    E = client.submit(lambda x : np.array([[Symbol("{}_{}".format(str(phi), chr(ord('x')+mu))) for mu in range(3)] for phi in x]), parametrization.dofs)

    # Add the O(n) terms to the coefficients
    o1 = client.submit(parametrization.O, order+1)
    o2 = client.submit(parametrization.O, order+2)

    F = client.submit(sum, F, o1)
    M = client.submit(sum, M, o2)
    p = client.submit(sum, p, o2)
    E = client.submit(sum, E, o1) # TODO: check

    # Generate the list of all the output coefficients
    coeffs = client.submit(all_coefficients, parametrization, J, order=order+1, collapse=collapse)

    # Try to generate the basis elements
    def generateCoeffs(coeffs):
        c = get_client()
        futures = [c.submit(coeff.generate) for coeff in coeffs]

        secede()
        c.gather(futures)
        rejoin()

        return coeffs
    coeffs = client.submit(generateCoeffs, coeffs)

    print(coeffs.result())

    #coeffs[-1][0].generate()

    print("Done.")
