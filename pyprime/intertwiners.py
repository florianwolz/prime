import sympy
import numpy as np

class Intertwiner:
    def __init__(self, parametrization):
        """
        Constructor for the intertwiner

        Takes a parametrization and calculates the derivative by the
        degrees of freedom.
        """
        self.parametrization = parametrization

        # Get the components
        self.components = [self.parametrization.diff(field.components) for field in self.parametrization.fields]
        self.indices = [fields.indices + [(len(self.parametrization.dofs), -1)] for fields in self.parametrization.fields]

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
    def __init__(self, intertwiners):
        self.intertwiners = intertwiners
        self.parametrization = intertwiners.parametrization

        # Get the constant part and reshape
        constInts = self.intertwiners.constant()
        constInts = [int.reshape((np.prod(int.shape[0:-1]), int.shape[-1])) for int in constInts]
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
        constInvs = [ np.reshape(np.array(sympy.Matrix(int).pinv()), shape) for int, shape in zip(constInts, constInvsShapes) ]

        # Check the parametrization
        reshapedInvs = [I.reshape((I.shape[0], np.prod(I.shape[1:]))) for I in constInvs]
        d = np.zeros((len(self.parametrization.dofs), len(self.parametrization.dofs)))
        for A,B in zip(constInts, reshapedInvs):
            d = d + np.matmul(B.astype(np.float64),A.astype(np.float64))
        valid = np.all((d - np.identity(len(self.parametrization.dofs))) < 1e-10)

        if not valid:
            raise Exception("The given parametrization is invalid. The inverted intertwiners and the constant ones do not contract to the identity.")

        # Go to the next order


        self.inverse = constInvs
