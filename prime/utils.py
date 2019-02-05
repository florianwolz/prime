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


"""
Symmetrize a numpy tensor over some indices

Args:
    tensor      The tensor
    indices     A list with the indices
"""
def symmetrize(tensor, indices):
    # Make sure that we have enough indices
    if len(tensor.shape) <= max(indices) or min(indices) < 0:
        raise Exception("You need to give proper indices to symmetrize over")

    # Make sure that all the indices have the same dimension
    if len(list(set([tensor.shape[idx] for idx in indices]))) != 1:
        raise Exception("The indices all need to have the same dimensions")

    import itertools

    # Generate all the permutations of the indices
    perms = list(itertools.permutations(indices))

    # Turn them into index shapes
    shapes = []
    for perm in perms:
        shape = []
        pos = 0
        for i in range(len(tensor.shape)):
            if not i in indices:
                shape.append(i)
            else:
                shape.append(perm[pos])
                pos = pos + 1
        shapes.append(tuple(shape))

    # Add the different permutations of the tensor
    result = np.zeros(tensor.shape)
    for shape in shapes:
        result = result + tensor.transpose(shape)

    # Return the result divided by N!
    return result / len(perms)

"""
Syntactic sugar for generation of a list of dofs

Args:
    F       The number of dofs
    symbol  The symbol for the dofs (default: "phi")
"""
def phis(F, symbol="phi"):
    import sympy
    return [sympy.Symbol("{}{}".format(symbol, i+1)) for i in range(F)]


def dropHigherOrder(expr, phis, order):
    from sympy import Symbol, Expr

    # If the expression is a list
    if type(expr) is list:
        return [dropHigherOrder(e) for e in expr]

    # If the expression is a numpy array, apply the function to all the cells
    if type(expr) is np.ndarray:
        tmp = np.vectorize(lambda x : dropHigherOrder(x, phis, order))
        return tmp(expr)

    # Not a sympy expression?
    if not isinstance(expr, Expr):
        return expr

    # Else...
    t = Symbol("t")
    return expr.subs({ phi : t * phi for phi in phis }).series(t, 0, order+1).removeO().subs(t, 1)


def constantSymmetricIntertwiner():
    from sympy import sqrt

    return np.array([
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,1/sqrt(2),0],[1/sqrt(2),0,0],[0,0,0]],
        [[0,0,1/sqrt(2)],[0,0,0],[1/sqrt(2),0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,1/sqrt(2)],[0,1/sqrt(2),0]],
        [[0,0,0],[0,0,0],[0,0,1]]
    ])

def constantSymmetricTracelessIntertwiner():
    from sympy import sqrt

    return np.array([
        [[1/sqrt(2),0,0],[0,-1/sqrt(2),0],[0,0,0]],
        [[1/sqrt(6),0,0],[0,1/sqrt(6),0],[0,0,-2/sqrt(6)]],
        [[0,1/sqrt(2),0],[1/sqrt(2),0,0],[0,0,0]],
        [[0,0,0],[0,0,1/sqrt(2)],[0,1/sqrt(2),0]],
        [[0,0,1/sqrt(2)],[0,0,0],[1/sqrt(2),0,0]],
    ])


def tensorFromFn(fn, shape):
    import itertools
    a = np.array([fn(*ids) for ids in itertools.product(*[range(i) for i in shape])])
    return a.reshape(shape)

"""
Syntactic sugar to turn tensors given in functional form
into proper lists with components.
"""

class to_tensor:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, fn):
        return tensorFromFn(fn, self.shape)


@to_tensor(shape=(3,3))
def gamma(a,b):
    return -1 if a == b else 0


epsilon = np.array([
    [[0,0,0],[0,0,1],[0,-1,0]],
    [[0,0,-1],[0,0,0],[1,0,0]],
    [[0,1,0],[-1,0,0],[0,0,0]]
])


def dirt(phis, order=1):
    from sympy import Order

    result = [1]
    for i in range(order):
        result = [i*p for i in result for p in phis]
    return Order(sum(result))


def det(matrix):
    if type(matrix) is np.ndarray:
        return np.linalg.det(matrix)
    
    if not hasattr(matrix, "components"):
        raise Exception("Cannot calculate the determinant of object of class {}".format(type(matrix)))
    
    shape = matrix.components.shape
    if shape != (3,3):
        raise Exception("Cannot calculate the determinant of matrix of shape {}")
    
    M = matrix.components
    return M[0,0] * M[1,1] * M[2,2] - M[2,0] * M[1,1] * M[0,2] + \
           M[0,1] * M[1,2] * M[2,0] - M[2,1] * M[1,2] * M[0,0] + \
           M[0,2] * M[1,0] * M[2,1] - M[2,2] * M[1,0] * M[0,1]


def sqrt(expr):
    from sympy import sqrt
    return sqrt(expr)


def binomial(N, K):
    if K == 0: return 1
    if 2*K > N: K = N-K
    result = 1
    for i in range(1, K+1):
        result = result * (N-K+i)/i
    return int(result)


def factorial(N):
    if N==0: return 1
    elif N==1: return 1
    else: return N*factorial(N-1)