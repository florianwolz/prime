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


class Symmetry:
    class SymmetryType:
        SYMMETRIC = 1
        ANTISYMMETRIC = 2

    def __init__(self, type, indices, top=False):
        self.type = type
        self.indices = indices
        self.top = top

    def __str__(self):
        return "<{},{}>".format(self.type, self.indices)

    def __repr__(self):
        return str(self)


class Field:
    """
    Constructor of a field

    Args:
        components      (Numpy array) of the tensor components
        indexPositions  List with an indicator of the index positions
    """
    def __init__(self, components, indexPositions, weight=0):
        self.components = components

        # Turn into a numpy array if necessary
        if not type(self.components) is np.ndarray:
            self.components = np.array(self.components)

        # Enough index positions?
        shape = self.components.shape
        assert(len(indexPositions) == len(shape))

        # Save the index dimension and position
        self.indices = [(shape[i], indexPositions[i]) for i in range(len(indexPositions))]

        # Get a list of all the free variables
        self.dofs = sorted(list(set([phi for comp in self.components.flatten() for phi in comp.free_symbols ])), key=str)

        # The weight
        self.weight = weight

        # Application at zero
        self.onZero = np.vectorize(lambda x : x.subs([(dof,0) for dof in self.dofs]))

        # Find the symmetries of the field
        self.findSymmetries()

    """
    Calculates the tangential coefficient (F)
    """
    def tangential_coefficient(self):
        # Calculate the shape of the coefficient
        Fshape = self.components.shape + (3,3,)

        # Initialize the result
        F = np.zeros(Fshape)

        # Calculate the tensor product of the identity with g
        tmp = np.tensordot(self.components, np.identity(3), axes=0)
        mu = len(tmp.shape)-2
        gamma = mu+1

        for i, v in enumerate(self.indices):
            dim, pos = v

            # Ignore all indices that are not spatial
            if dim != 3:
                continue

            # Calculate the shape
            newShape = list(range(len(tmp.shape)))
            if pos > 0:
                newShape[gamma] = i
                newShape[i] = gamma
            else:
                newShape[mu] = i
                newShape[i] = mu

            # Add the components
            F = F + pos * tmp.transpose(newShape)

        if self.weight != 0:
            F = F + self.weight * tmp

        return F

    def __getitem__(self, pos):
        # Syntactic sugar for scalars
        if pos == 0 and self.components.shape == tuple(): return self.components

        return self.components[pos]

    def findSymmetries(self):
        self.symmetries = {}

        # If we do not have at least 2 indices, there is no possible symmetry
        if len(self.indices) < 2: return

        # Create a list of all the indices
        idx = list(range(len(self.indices)))

        import itertools

        # Generate all the two-index-combinations of the field
        combs = itertools.combinations(idx, 2)
        for c in combs:
            if np.array_equal(self.components,self.components.swapaxes(*c)):
                self.symmetries[c] = Symmetry(Symmetry.SymmetryType.SYMMETRIC, c, False)
            elif np.array_equal(self.components, -self.components.swapaxes(*c)):
                self.symmetries[c] = Symmetry(Symmetry.SymmetryType.ANTISYMMETRIC, c, False)


        # TODO: Generalize to higher order symmetries


class field:
    def __init__(self, indexPositions, weight=0):
        self.indexPositions = indexPositions
        self.weight = weight

    def __call__(self, fn):
        from prime.utils import to_tensor
        shape = tuple([f[0] for f in self.indexPositions])
        components = to_tensor(shape=shape)(fn)
        return Field(
            components=components,
            indexPositions=self.indexPositions,
            weight=self.weight
        )
