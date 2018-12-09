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
from prime.output.indices import generateEvenRank, generateOddRank

# Symbols
symbols = ["lambda", "xi", "theta", "chi", "omega"]


class OutputCoefficient:
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
        # Some statistics first
        numKindices = self.order
        phiIndices = self.derivs

        # All of those indices must be contracted with the J intertwiner
        from itertools import product

        # Generate the possible intertwiner contractions for the K-kind indices first
        contrsK = list(product(*[list(range(len(self.J.components))) for i in range(self.order)]))
        contrsK = sorted(list(set([tuple(sorted(d)) for d in contrsK])))

        # For each of the possible assignments, generate the tensor of that shape




def all_coefficients_of_order(parametrization, J, order, maxOrder, collapse=2, symbol=None):
    # For C_AB and higher we have the collapse at 2
    if order >= 2: collapse = 2

    from itertools import product

    # Calculate the list of all the possible derivative index assignments
    derivs = []
    for o_ in range(maxOrder-order+1):
        derivs_ = list(product(*[list(range(collapse+1)) for o in range(o_)]))
        derivs = derivs + sorted(list(set([tuple(sorted(list(d))) for d in derivs_])))

    # Return the output coeffcieints
    return [OutputCoefficient(parametrization, J, order, list(d), symbol) for d in derivs]


def all_coefficients(parametrization, J, order, collapse=2):
    return [all_coefficients_of_order(parametrization, J, o, order, collapse) for o in range(order+1)]
