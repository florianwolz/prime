from sympy import Symbol, diff
import numpy as np

from .field import Field

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

        # Merge the degrees of freedom
        self.dofs = sorted(set([dof for field in self.fields for dof in field.dofs]), key=str)

        # Add some methods to generally apply to expressions
        self._onZero = np.vectorize(lambda x : x.subs([(dof, 0) for dof in self.dofs]) )
        self._deriv = np.vectorize(lambda x : np.array([diff(x, dof) for dof in self.dofs ]), signature='()->(n)')

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
