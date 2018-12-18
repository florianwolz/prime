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

import numpy as np
import sympy
from prime.output.indices import Indices 
from prime.input.parametrization import dPhis
from prime.input.field import Symmetry

from dask.distributed import get_client, secede, rejoin

# Symbols
symbols = ["lambda", "xi", "theta", "chi", "omega"]


"""
BasisElement

Represents on single element in the basis. It is constructed via
finding the transversal of the double coset of the label symmetries
and getting rid of the dimensional dependent identities.
"""
class BasisElement:
    def __init__(self, indices, variable):
        self.indices = indices
        self.variable = variable

        # TODO: Keep the information which intertwiner was used to generate the
        #       element. This allows to print the Lagrangian in a better way.


"""
ConstantOutputCoefficient

Represents one constant output coefficients
"""
class ConstantOutputCoefficient:
    def __init__(self, parametrization, J, order, derivs=None, symbol=None):
        # Store the variables
        self.parametrization = parametrization
        self.J = J
        self.order = order

        # Syntactic sugar for the spatial derivatives
        if derivs is None:
            self.derivs = []
        elif type(derivs) is int:
            self.derivs = [derivs]
        else:
            self.derivs = derivs

        # Assign the symbol
        self.symbol = symbol if symbol is not None else symbols[order]

        # Calculate the shape
        self.shape = tuple([len(parametrization.dofs) for i in range(order)])
        x = [(len(parametrization.dofs), [3 for i in range(d)]) for d in self.derivs]
        for d in x:
            self.shape = self.shape + (d[0],)
            for y in d[1]:
                self.shape = self.shape + (y,)

        # Also store the symmetry
        self.symmetric = []
        if order > 1:
            self.symmetric = self.symmetric + [tuple(range(order))]
        i = order
        for d in self.derivs:
            i = i+1
            if d <= 1:
                i = i + d
                continue
            self.symmetric = self.symmetric + [tuple(range(i, i + d))]
            i = i + d

        # ... and the block symmetries
        self.block_symmetric = []
        for i, d in enumerate(self.derivs):
            for j, e in enumerate(self.derivs):
                if d == e and i < j:
                    offset = self.order
                    for k in range(0,i):
                        offset = offset + 1 + self.derivs[k]
                    blockA = tuple(range(offset, offset + self.derivs[i] + 1))
                    offset = offset + self.derivs[i]+1

                    for k in range(i+1,j):
                        offset = offset + 1 + self.derivs[k]
                    blockB = tuple(range(offset, offset + self.derivs[j] + 1))

                    self.block_symmetric.append((blockA,blockB))

        # Setup the variable for the basis elements
        self.basis = []

        # TODO: Properly generate the components by generating the basis
        self.components = np.full(self.shape, 0 * sympy.Symbol("x"))

    def __str__(self):
        s = self.symbol

        def alpha(N, offset=0):
            return list(map(chr, range(ord('A')+offset, ord('A')+N+offset)))

        if self.order + len(self.derivs) > 0:
            s = s + "_{}".format("".join(alpha(self.order)))

        for offset, d in enumerate(self.derivs, self.order):
            s = s + "{}".format(alpha(1, offset)[0]) + "".join(["d" for i in range(d)])

        return s

    def __repr__(self):
        return "#<{}>".format(str(self))

    def generate(self):
        # All of those indices must be contracted with the J intertwiner
        from itertools import product
        from copy import deepcopy

        # Generate the possible intertwiner contractions for the K-kind indices first
        contrsK = list(product(*[list(range(len(self.J.components))) for i in range(self.order)]))
        contrsK = sorted(list(set([tuple(sorted(d)) for d in contrsK])))

        # Do the same for the derivative indices
        contrsP = list(product(*[list(range(len(self.J.components))) for i,d in enumerate(self.derivs)]))

        # Get rid of exchange symmetric blocks
        if len(self.derivs) > 1:
            for i, d in enumerate(self.derivs):
                for j, e in enumerate(self.derivs):
                    if d == e and i < j:
                        for x in contrsP:
                            # Exchange the i-th and the j-th entry
                            c = list(deepcopy(x))
                            tmp = c[i]
                            c[i] = c[j]
                            c[j] = tmp
                            c = tuple(c)

                            if x == c: continue

                            # Delete this one from the list
                            try:
                                id = contrsP.index(c)
                                del contrsP[id]
                            except:
                                continue

        contractions = list(product(contrsK, contrsP))

        # For each of the possible assignments, generate the tensor of that shape
        futures = []
        client = get_client()
        for contr in contractions:
            idsK, idsP = contr

            # Generate the shape of the background tensor
            kIndices = [(self.parametrization.fields[i].components.shape, self.parametrization.fields[i].symmetries) for i in idsK]
            pIndices = [(self.parametrization.fields[i].components.shape, self.parametrization.fields[i].symmetries, d) for i, d in zip(idsP, self.derivs)]

            shape = tuple()
            symmetries = []
            offset = 0
            for id in kIndices:
                shape = shape + id[0]

                for idx in id[1]:
                    sym = id[1][idx]
                    symmetries.append(Symmetry(sym.type, tuple([i + offset for i in idx])))
                    offset = offset + len(idx)
            for id in pIndices:
                shape = shape + id[0]

                for idx in id[1]:
                    sym = id[1][idx]
                    symmetries.append(Symmetry(sym.type, tuple([i + offset for i in idx])))
                    offset = offset + len(idx)
                    if id[2] > 1:
                        symmetries.append(Symmetry(Symmetry.SymmetryType.SYMMETRIC, tuple(range(offset, offset+id[2]))))
                    offset = offset + id[2]

                for x in range(id[2]):
                    shape = shape + (3,)

            # TODO: Also implement the exchange symmetries

            # Create a task to generate a index assignment with these kind of symmetries
            def generateIndex(contr, rank, symmetries):
                idx = Indices(rank)
                # TODO: also allow antisymmetric index blocks
                syms = [tuple(s.indices) for s in symmetries if s.type == Symmetry.SymmetryType.SYMMETRIC]
                idx.symmetrize(syms)

                # TODO: Contract the terms with the intertwiner

                # What are the things that can happen?
                #  - Two intertwiner indices could be contracted by two gamma indices => Einstein sum
                #  - An intertwiner index is made into a derivative index => transpose to the corresponding slot
                #  - 

                return contr, idx
            futures.append(client.submit(generateIndex, contr, len(shape), symmetries))

        # Wait for the indices to be generated
        secede()
        client.gather(futures)
        rejoin()

        # TODO: select the linear independent ones from the list

        # TODO: for each basis tensor take one variable

        print(self)
        for f in futures:
            print(f.result()[1].indices)



"""
OutputCoefficient

The real output coefficients. They are polynomial in phis
"""
class OutputCoefficient:
    def __init__(self, parametrization, J, order, maxOrder, collapse=2, dropCosmologicalConstants=True):
        # Store the variables
        self.parametrization = parametrization
        self.J = J
        self.order = order
        self.maxOrder = maxOrder
        self.collapse = collapse if order < 2 else 2

        # Calculate the list of all the possible derivative index assignments
        from itertools import product
        derivs = []
        for o_ in range(maxOrder-order+1):
            derivs_ = list(product(*[list(range(collapse+1)) for o in range(o_)]))
            derivs = derivs + sorted(list(set([tuple(sorted(list(d))) for d in derivs_])))

        # For C we know that the constant part and the linear part in phi will give
        # constant contributions to the e.o.m. which have to be dropped due to
        # consistency reasons anyway, so we can already drop them here...
        if order == 0 and dropCosmologicalConstants:
            derivs = derivs[2:]

        # Prepare all the constant output coefficients
        self.constCoeffs = [ConstantOutputCoefficient(self.parametrization, self.J, self.order, list(d)) for d in derivs]

        # Prepare the components
        self.components = np.zeros(tuple([len(self.parametrization.dofs) for i in range(order)]))

    """
    Generate the components of the output coefficient by contracting the
    constant coefficients with phis and its derivatives

    TODO: Still buggy, since the components are initialized with zeros.
          Should be working when the constant coefficients are properly
          calculated. If not, need to cast the numpy elements into
          proper sympy expressions (compare the intertwiners)
    """
    def generate(self):
        def generateConstCoeff(c, dofs, order):
            # Generate the coefficient
            c.generate()

            #tmp = c.components

            # Constant part of the coefficient?
            #if len(c.derivs) == 0:
            #    return tmp

            # Contract the indices from the phi expansions
            #for d in c.derivs:
            #    tmp = np.tensordot(tmp, dPhis(dofs, d), axes=(tuple(range(self.order, self.order + d + 1)), tuple(range(d + 1))))

            # Ignore zeros
            #return tmp

        c = get_client()
        futures = [c.submit(generateConstCoeff, coeff, self.parametrization.dofs, self.order) for coeff in self.constCoeffs]

        secede()
        c.gather(futures)
        rejoin()


    def __str__(self):
        s = "C"

        def alpha(N, offset=0):
            return list(map(chr, range(ord('A')+offset, ord('A')+N+offset)))

        if self.order > 0:
            s = s + "_{}".format("".join(alpha(self.order)))

        s = s + ": {}".format([self.constCoeffs])
        return s

    def __repr__(self):
        return str(self)


def all_coefficients(parametrization, J, order, collapse=2):
    return [OutputCoefficient(parametrization, J, o, order, collapse) for o in range(order+1)]
