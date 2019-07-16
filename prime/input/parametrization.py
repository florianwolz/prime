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

from sympy import Symbol, diff, Function, symbols, O, sympify, Add
import numpy as np
from prime.utils import memoize

from .field import Field


def jet_variables(phi, order):
    # Also applicable for lists of dofs
    if type(phi) is list:
        return [e for p in phi for e in jet_variables(p, order)]
    if type(order) is list:
        return [e for o in order for e in jet_variables(phi, o)]

    if order == 0: return [str(phi)]

    from itertools import product
    idx = list(sorted(set(product(*[list(range(0,3)) for _ in range(order)]))))
    return list(set(["{}_{}".format(phi, "".join(sorted([chr(ord('x') + e) for e in i]))) for i in idx]))


def formal_derivative_of_symbol(symbol, direction, independentVariables=['x','y','z']):
    # Check if the direction checks out
    if not direction in independentVariables: 
        raise Exception("'{}' is not a independent variables. Expected one of {}".format(direction, independentVariables))

    # Make sure the derivative is just a string
    symbolStr = str(symbol)

    # Split the string
    splitted = symbolStr.split('_')

    # No partial derivative yet at the symbol name, add the direction
    if len(splitted) == 1: return Symbol("{}_{}".format(symbolStr, direction))

    # Build the multi index
    multiIndex = sorted([independentVariables.index(i) for i in splitted[1] + direction])
    result = [independentVariables[i] for i in multiIndex]
    return Symbol("{}_{}".format(splitted[0], "".join(result)))


def dPhis(phis, order):
    if order == 0: return np.array(phis)
    from itertools import product
    comps = product(list(range(len(phis))), *[(0,1,2) for i in range(order)])
    l = [Symbol("{}_{}".format(phis[c[0]], "".join(sorted([chr(ord('x') + e) for e in c[1:]])))) for c in comps]
    return np.reshape(l, (len(phis),) + tuple([3 for i in range(order)]))


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
        self._onZero = np.vectorize(lambda x : sympify(x).subs([(dof, 0) for dof in self.dofs]) )
        self._deriv = np.vectorize(lambda x : np.array([diff(x, dof) for dof in self.dofs ]), signature='()->(n)')

        # Create a cache for the normal deformation coefficient
        self._M = None

        self._OdiffMemo = {}

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
        if order==0: return O(1)
        res = self.dofs
        for i in range(order-1):
            res = [x*y for x in res for y in self.dofs]
        return O(Add(*res), *self.dofs)
    

    def Odiff(self, order, collapse=2):
        if (order, collapse) in self._OdiffMemo: return self._OdiffMemo[(order, collapse)]

        if order==0: return O(1)
        z = [Symbol(x) for i in range(0, collapse+1) for x in jet_variables(self.dofs, i)]
        res = z
        for i in range(order-1):
            res = [x*y for x in res for y in z]
        
        summed = Add(*res)

        result = O(summed, *z)
        #return summed

        self._OdiffMemo[(order, collapse)] = result

        #return Add(*res)
        return result


"""
Calculate the spatial derivative of an expression.

If we don't give a direction, the result is a numpy tensor with the derivatives
in all directions, with the first index being the derivative index.

Args:
    expr        The expression to derive
    direction   In which spatial direction should we take the derivative?
                If none, the result is a vector in all three spatial directions
"""
def spatial_diff(expr, direction=None, order=None):
    if order is not None and type(order) is int:
        if order == 0:
            return expr
        elif order == 1:
            return spatial_diff(expr=expr, direction=direction)
        else:
            return np.array(spatial_diff(expr=spatial_diff(expr, direction=direction).tolist(), direction=direction, order=order-1))

    # No direction? Return the vector x,y,z
    if direction is None:
        return np.array([spatial_diff(expr, direction=d).tolist() for d in ['x','y','z']])

    # Also allow for lists
    if type(expr) is list:
        return np.array([spatial_diff(e, direction).tolist() for e in expr])

    # Also applicable for numpy arrays
    if type(expr) is np.ndarray:
        tmp = np.vectorize(lambda x : np.array(spatial_diff(x, direction).tolist()))
        return tmp(expr)
    
    if np.isreal(expr):
        return np.array(0.0)

    # If there is a big O present, treat it separately
    if hasattr(expr, "getO") and expr.getO() is not None:
        expr1 = spatial_diff(expr.removeO(), direction).tolist()
        expr2 = spatial_diff(expr.getO().expr, direction).tolist()
        return np.array([a+O(b) for a,b in zip(expr1, expr2)])

    # Get all the symbols
    syms = [s for s in expr.free_symbols if s not in ['x', 'y', 'z']]

    # Handle explicit x, y, z dependency
    result = diff(expr, Symbol(direction))

    # Add all the chain rule terms
    for s in syms:
        result = result + diff(expr, s) * formal_derivative_of_symbol(s, direction)
        
    return np.array(result)


class jet_diff(object):
    def __init__(self, parametrization):
        if __debug__:
            if not type(parametrization) is Parametrization:
                raise Exception("Cannot create a jet derivative helper with a parametrization of wrong type.")
    
        # Only need the degrees of freedom
        self.dofs = parametrization.dofs
    
    def __call__(self, expr, direction=None, order=None):
        if direction is None and order is None:
            raise Exception("Cannot calculate the jet derivative if neither a direction or an order is given.")
        
        # If we are given a derivative order, calculate the tensor into all the direction
        if direction is None:
            # Make sure order is an int
            if type(order) is not int:
                raise Exception("The order `{}` is not an int.".format(order))
            
            # Get all the jet variables at that order
            vars = dPhis(self.dofs, order)
            shape = vars.shape

            # Apply the derivative into all directions
            # TODO: Get rid of duplicate calculations
            l = np.vectorize(lambda x : self(expr, direction=x, order=None).tolist())
            diffs = np.array(l(vars).tolist())

            # The derivative indices are now the first ones. Move them to the back.
            return diffs.transpose(tuple(range(len(shape), len(diffs.shape))) + tuple(range(len(shape))))

        # Check if direction is a proper jet variable
        if __debug__:
            v = str(direction).split("_")
            if Symbol(v[0]) not in self.dofs: raise Exception("Invalid direction.")
            if len(v) == 2:
                for d in v[1]:
                    if d not in ['x', 'y', 'z']:
                        raise Exception("Invalid direction.")

        # Also allow for lists
        if type(expr) is list:
            return np.array([self(e, direction=direction).tolist() for e in expr])

        # Also applicable for numpy arrays
        if type(expr) is np.ndarray:
            tmp = np.vectorize(lambda x : self(x, direction=direction).tolist())
            return tmp(expr)
        
        return np.array(diff(expr, direction))