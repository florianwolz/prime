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

from sympy import Symbol, diff, Function, symbols, O
import numpy as np

from .field import Field


def jet_variables(phi, order):
    # Also applicable for lists of dofs
    if type(phi) is list:
        return [e for p in phi for e in jet_variables(p, order)]
    if type(order) is list:
        return [e for o in order for e in jet_variables(phi, o)]

    if order == 0: return [str(phi)]

    from itertools import product
    idx = list(sorted(list(product(*[list(range(0,3)) for _ in range(order)]))))
    return ["{}_{}".format(phi, "".join(reversed([chr(ord('x') + e) for e in i]))) for i in idx]


def sympy_derivative(term, dofs, replacements):
    vars = str(term).split("_")
    try:
        fn = replacements[dofs.index(Symbol(vars[0]))]
    except:
        return 0

    x,y,z = symbols("x y z")

    # No derivative indices, just return the function
    if len(vars) == 1: return fn(x,y,z)

    # Get the number of derivatives in that direction
    times = [(Symbol(s), vars[1].count(s)) for s in ["x","y","z"]]

    # Return the result
    return fn(x,y,z).diff(*times)


class Parametrization:
    """
    Constructor

    Constructs a parametrization from a list of fields in terms of the degrees
    of freedom
    """
    def __init__(self, fields):
        self.fields = fields

        # Syntactic sugar for only one field
        if type(self.fields) is Field:
            self.fields = [self.fields]

        # Type check
        for field in self.fields:
            assert(type(field) is Field)

        import re
        def sortkey_natural(s):
            s = str(s)
            return tuple(int(part) if re.match(r'[0-9]+$', part) else part for part in re.split(r'([0-9]+)', s))

        # Merge the degrees of freedom
        self.dofs = sorted(set([dof for field in self.fields for dof in field.dofs]), key=sortkey_natural)

        # Add some methods to generally apply to expressions
        self._onZero = np.vectorize(lambda x : x.subs([(dof, 0) for dof in self.dofs]) )
        self._deriv = np.vectorize(lambda x : np.array([diff(x, dof) for dof in self.dofs ]), signature='()->(n)')

        # Create a cache for the normal deformation coefficient
        self._M = None

    def evaluate(self, expr):
        return self._onZero(expr)

    def diff(self, expr):
        return self._deriv(expr)

    def order(self, expr, k):
        e = expr

        for i in range(k):
            e = self.diff(e)

        # Evaluate the result on zero
        return self.evaluate(e)

    def spatial_derivative(self, expr):
        # Also allow for lists
        if type(expr) is list:
            return np.array([spatial_derivative(e) for e in expr]).transpose()

        # Also applicable for numpy arrays
        if type(expr) is np.ndarray:
            tmp = np.vectorize(lambda x : np.array(self.spatial_derivative(x)))
            res = tmp(expr)
            return res.transpose((len(res.shape)-1,) + tuple(range(len(res.shape)-1)))

        # If there is a big O present, treat it separately
        if expr.getO() is not None:
            expr1 = self.spatial_derivative(expr.removeO())
            expr2 = self.spatial_derivative(expr.getO().expr)
            return np.array([a+O(b) for a,b in zip(expr1, expr2)])

        # Get all the symbols
        syms = expr.free_symbols
        hs = [Function("htmp{}".format(i)) for i in range(len(self.dofs))]

        # Get the order
        order = 0
        try:
            order = max([len(str(e).split("_")[1]) for e in syms if len(str(e).split("_")) > 1])
        except:
            pass

        # Make up the substitutions
        subst = { s : sympy_derivative(s, self.dofs, hs) for s in syms }

        # Do the derivative
        x,y,z = symbols("x y z")
        exprs = [diff(expr.subs(subst), dir) for dir in [x,y,z]]

        # Go back to the jet variables
        jets = jet_variables(self.dofs, list(range(order+2)))
        subst = [(sympy_derivative(s, self.dofs, hs), s) for s in jets]

        # Substitute
        # Remark: Need to do it in this fashion, with the reversed direction
        # to make sure that the highest order derivatives get replaced first.
        res = exprs
        for s in reversed(subst):
            res = [e.subs(*s) for e in res]

        # Return the result
        return np.array(res)


    def O(self, order):
        res = self.dofs
        for i in range(order-1):
            res = [x*y for x in res for y in self.dofs]
        return O(res, *self.dofs)
