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
from prime.equations import allEqns

from prime.reporter import Reporter, Status
#from prime.checkpoints import get_checkpoint

import numpy as np
from sympy import Symbol, O

import emoji
from yaspin import yaspin

def log(sp, s, the_emoji='zap'):
    lines = str(s).splitlines()
    
    sp.write(emoji.emojize("---- :{}:  {}".format(the_emoji, lines[0]), use_aliases=True))
    for i in range(1, len(lines)):
        sp.write("        {}".format(lines[i]))


"""
Start solving the gravitational closure equations

Args:
    parametrization             The parametrization of the theory
    kinematical_coefficient     The kinematical coefficient (p^..)
    normal_coefficient          The normal deformation coefficient (M^A\\gamma)
    order                       The order of the resulting equations of motion
"""
def solve(parametrization, kinematical_coefficient, normal_coefficient=None, order=1, collapse=2):
    with yaspin().cyan.point as sp:

        # Setup the reporter
        reporter = Reporter(order=order, silent=True)

        # Retrieve the checkpoints instance
        #cs = get_checkpoint()

        log(sp, "Start calculating.\nThis is gone take a while")

        # Start by calculation of the intertwiner
        log(sp, "Calculate the intertwiner")
        I = Intertwiner(parametrization)

        # Create a task for the calculation of the intertwiner
        log(sp, "Calculate the inverse interwiner")
        J = InverseIntertwiner(I, order=order)

        # Get the real F coefficient
        log(sp, "Calculate the tangential deformation coefficient F")
        F = J.contractWith([field.tangential_coefficient() for field in parametrization.fields])
        #F.add_done_callback(lambda x : reporter.update("F", Status.PREPARING))

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

        log(sp, "Calculate the normal deformation coefficient M")
        M = calculateM(parametrization, normal_coefficient, J)

        # Drop all the higher order terms in the tensors, since we won't need them
        log(sp, "Drop all higher order terms in the coefficients")
        F = dropHigherOrder(F, parametrization.dofs, order=order)
        M = dropHigherOrder(M, parametrization.dofs, order=order+1)
        p = dropHigherOrder(kinematical_coefficient.components, parametrization.dofs, order=order+1)

        ##F = client.submit(dropHigherOrder, F, parametrization.dofs, order=order)
        ##M = client.submit(dropHigherOrder, M, parametrization.dofs, order=order+1)
        ##p = client.submit(dropHigherOrder, kinematical_coefficient.components, parametrization.dofs, order=order+1)
        degP = kinematical_coefficient.degP
        ##p.add_done_callback(lambda x : reporter.update("p", Status.PREPARING))

        # Setup the E coefficient
        log(sp, "Calculate the E coefficient")
        E = np.array([[Symbol("{}_{}".format(str(phi), chr(ord('x')+mu))) for mu in range(3)] for phi in parametrization.dofs])
        #E = client.submit(lambda x : np.array([[Symbol("{}_{}".format(str(phi), chr(ord('x')+mu))) for mu in range(3)] for phi in x]), parametrization.dofs)
        #E.add_done_callback(lambda x : reporter.update("E", Status.PREPARING))

        # Add the O(n) terms to the coefficients
        #log("Add O({}) terms to the coefficients".format(order+1))
        #o1 = parametrization.O(order+1)
        #o2 = parametrization.O(order+2)

        ##o1 = client.submit(parametrization.O, order+1)
        ##o2 = client.submit(parametrization.O, order+2)

        #F = F + o1
        #M = M + o2
        #p = p + o2
        #E = E + o1

        ##F = client.submit(sum, F, o1)
        ##M = client.submit(sum, M, o2)
        ##p = client.submit(sum, p, o2)
        ##E = client.submit(sum, E, o1) 

        ### Add the callbacks to notify the reporter
        ##E.add_done_callback(lambda x : reporter.update("E", Status.FINISHED))
        ##F.add_done_callback(lambda x : reporter.update("F", Status.FINISHED))
        ##M.add_done_callback(lambda x : reporter.update("M", Status.FINISHED))
        ##p.add_done_callback(lambda x : reporter.update("p", Status.FINISHED))

        ## Wait for the input coefficients to finish
        ##F.result()
        ##M.result()
        ##p.result()
        ##E.result()

        log(sp, "Finished input coefficients.\nGenerate ansatz for the output coefficients ...")

        # Generate the list of all the output coefficients
        coeffs = all_coefficients(parametrization, J, order=order+1, collapse=collapse)
        log(sp, "Found {}".format(", ".join([coeff.name for coeff in coeffs])), the_emoji="tada")

        #coeffs = client.submit(all_coefficients, parametrization, J, order=order+1, collapse=collapse)

        # Try to generate the basis elements
        for coeff in coeffs:
            log(sp, "Generate coefficient {}".format(coeff.name))
            coeff.generate()
    
        totals = sum([len(coeff.variableMap.keys()) for coeff in coeffs], 0)

        log(sp, "Finished output coefficient ansatz. Total of {} gravitational constants.".format(totals))
    
    #def generateCoeffs(coeffs):
    #    c = get_client()
    #    futures = [c.submit(coeff.generate) for coeff in coeffs]
    #
    #    # Add the callback
    #    for i,f in enumerate(futures,0):
    #        name = "C_{}".format("".join([chr(ord('A')+j) for j in range(i)])) if i > 0 else "C"
    #        reporter.update(name, Status.CALCULATING)
    #        f.add_done_callback(lambda x : reporter.update(name, Status.FINISHED))
    #
    #    secede()
    #    c.gather(futures)
    #    rejoin()
    #
    #    return coeffs

    #coeffs = client.submit(generateCoeffs, coeffs)

    #print(coeffs.result())
    #print("Finished output coefficient ansatz.")

        def generateAllEqns(param, E, F, M, p, degP, Cs, order):
            coeffs = [C.components for C in Cs]

            # Add the next coefficient that only contains O(1) terms
            x = len(coeffs)
            shape = tuple([len(param.dofs) for i in range(x)])
            coeffs.append(np.full(shape, O(1)))

            return allEqns(param, coeffs, E, F, M, p, degP, order)
    
        # Collect all equations
        #eqns = client.submit(generateAllEqns, parametrization, E, F, M, p, degP, coeffs, order+1)
        eqns = generateAllEqns(parametrization, E, F, M, p, degP, coeffs, order+1)

        print(eqns)

        # Get all the relations
        relations = [relation for eq in eqns for relation in eq.relations(diagonalize=False)]

        print("Final step. Diagonalize ...")

        print(len(relations))

        return

    # Solve
    def solveEquation(equation):
        relations = equation.relations()
        return relations

    relations = [client.submit(solveEquation, eq) for eq in eqns]
    client.gather(relations)

    # TODO: Do another huge Gauss algorithm iteration to collect the overall results
    # TODO: Bring into the form c_i = ... (can copy from the apple)
    # TODO: Collect all the variables on the r.h.s. (i.e. the free gravitational constants)
    #       and give them proper labels, i.e. g_i

    # TODO: Turn the result into fancy output
    variables = { k: v for c in coeffs for k, v in c.variableMap.items() }

    # REMARK: for each variable we know from which coefficient this comes from,
    #         since this is encoded in the values of the dict. 
    #         As a result, we now can prepare an output that iterates over all
    #         constant coefficients in a C coefficient and looks up the
    #         corresponding pre factor in the solution and combines it with
    #         


    print("Done.")
