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

import sympy
import numpy as np

from prime.tensor import Tensor


class Intertwiner:
    def __init__(self, parametrization):
        """
        Constructor for the intertwiner

        Takes a parametrization and calculates the derivative by the
        degrees of freedom.
        """
        self.parametrization = parametrization

        # Initialize the tensor
        self.components = [self.parametrization.diff(field.components) for field in self.parametrization.fields]
        self.indices = [fields.indices + [(len(self.parametrization.dofs), -1)] for fields in self.parametrization.fields]

        #Tensor.__init__(self,
        #    tensor=[self.parametrization.diff(field.components) for field in self.parametrization.fields],
        #    indices=[fields.indices + [(len(self.parametrization.dofs), -1)] for fields in self.parametrization.fields]
        #)

    def constant(self):
        """
        Get the constant part of the intertwiner.
        Returns the same result as order(0).
        """
        return [self.parametrization.evaluate(field) for field in self.components]

    def order(self, k):
        """
        Gets the higher order part of the intertwiner by first
        calculating the derivative by phis and then evaluating at zero.
        """
        return [self.parametrization.order(field, k) for field in self.components]


class InverseIntertwiner:
    def __init__(self, intertwiners, order=3):
        self.intertwiners = intertwiners
        self.parametrization = intertwiners.parametrization
        self.order = order # TODO: check the order of the intertwiner

        # Get the constant part and reshape
        constInts = self.intertwiners.constant()
        constInts = [i.reshape((int(np.prod(i.shape[0:-1])), i.shape[-1])) for i in constInts]
        constIntsIdx = self.intertwiners.indices

        # TODO:
        #   So far, this method is more or less a heuristics that
        #   works for almost all the parametrizations but it is possible
        #   that strange examples exist where this is not the case.
        #   Then the exception will be raised. Try to come up with a
        #   more general way to solve this.

        # Calculate the indices of the inverse intertwiners
        constInvsIdx = [[(id[0], -id[1]) for id in idx] for idx in self.intertwiners.indices]
        for i in range(len(constInvsIdx)):
            constInvsIdx[i].insert(0, constInvsIdx[i].pop(-1))
        constInvsShapes = [tuple([id[0] for id in idx]) for idx in constInvsIdx]
        # For now, sympy does not support rank defficient matrices, so we cannot use it here.
        # There is already a fix merged into master, so it should be usable quite soon...
        #constInvs = [ np.reshape(np.array(sympy.Matrix(int).pinv()), shape) for int, shape in zip(constInts, constInvsShapes) ]
        constInvs = [np.reshape(np.linalg.pinv(int.astype(np.float64)), shape) for int, shape in zip(constInts, constInvsShapes)]

        # Check the parametrization
        reshapedInvs = [I.reshape((I.shape[0], int(np.prod(I.shape[1:])))) for I in constInvs]
        d = np.zeros((len(self.parametrization.dofs), len(self.parametrization.dofs)))
        for A,B in zip(constInts, reshapedInvs):
            d = d + np.matmul(B.astype(np.float64),A.astype(np.float64))
        valid = np.all((d - np.identity(len(self.parametrization.dofs))) < 1e-10)

        if not valid:
            raise Exception("The given parametrization is invalid. The inverted intertwiners and the constant ones do not contract to the identity.")

        # Import the method for symmetrization
        from prime.utils import symmetrize

        # Get the orders for the
        cacheConsts = [ self.intertwiners.order(o) for o in range(order) ]
        cacheInvs   = [ constInvs ]

        # Iteratively construct the inverse
        components = constInvs
        for N in range(1, order):
            # Take the first one
            tmp = sum([np.tensordot(constInvs[i], cacheConsts[N][i],
                    axes=(
                        tuple(range(1,len(constInvs[i].shape))),
                        tuple(range(len(constInvs[i].shape)-1))
                    )) for i in range(len(constInvs))])
            tmp = [-np.tensordot(tmp, constInvs[j], axes=((1,), (0,))) for j in range(len(constInvs))]

            # Add the others on top
            for k in range(1,N):
                tmp2 = sum([np.tensordot(cacheInvs[k][i], cacheConsts[N-k][i],
                    axes=(
                        tuple(range(1,len(constInvs[i].shape))),
                        tuple(range(len(constInvs[i].shape)-1))
                    )) for i in range(len(constInvs))])
                tmp2 = [np.tensordot(tmp2, constInvs[j], axes=((k+1,), (0,))) for j in range(len(constInvs))]
                tmp2 = [symmetrize(t, list(range(1, k+1)) + list(range(1, N-k+1))) for t in tmp2]

                tmp = [x-y for x,y in zip(tmp, tmp2)]

            # Transpose
            tmp = [t.transpose((0,) + tuple(range(N+1, len(t.shape))) + tuple(range(1,N+1))) for t in tmp]

            # Add to the list
            cacheInvs.append(tmp)

            # Contract with phis
            r = tmp
            for i in range(N):
                r = [np.dot(t, self.parametrization.dofs) for t in r]

            # Add to the components
            components = [x + y for x, y in zip(components, r)]

        # Set the result
        self.components = components
        self.indices = constInvsIdx
        #Tensor.__init__(self,
        #    tensor=components,
        #    indices=constInvsIdx
        #)

    def constant(self):
        """
        Get the constant part of the inverse intertwiner.
        Returns the same result as order(0).
        """
        return [self.parametrization.evaluate(field) for field in self.components]

    def order(self, k):
        """
        Gets the higher order part of the inverse intertwiner by first
        calculating the derivative by phis and then evaluating at zero.
        """

        # Check the order
        if k > self.order:
            raise Exception("The inverse intertwiner was only calculated to {}-th order. Cannot obtain the {}-th order.".format(self.order, k))

        return [self.parametrization.order(field, k) for field in self.components]

    def contractWith(self, T, indexOffset=0):
        if len(self.components) != len(T):
            raise Exception("The intertwiners do not belong to the same fields.")

        return sum([np.tensordot(self.components[i], T[i],
                axes=( tuple(range(1,len(self.components[i].shape))), tuple(range(indexOffset+len(self.components[i].shape)-1))  )
            ) for i in range(len(self.components))])
