import unittest

from prime.input import Field, Parametrization, Intertwiner, InverseIntertwiner
from prime.utils import phis, to_tensor
from prime.output import ConstantOutputCoefficient
import sympy as sp
import numpy as np

class GenerateConstantOutputCoefficient(unittest.TestCase):
    # Setup
    phis = phis(6)

    g = Field([[-1 + phis[0], phis[1] / sp.sqrt(2), phis[2] / sp.sqrt(2)],
               [phis[1]/sp.sqrt(2), -1 + phis[3], phis[4] / sp.sqrt(2)],
               [phis[2]/sp.sqrt(2), phis[4] / sp.sqrt(2), -1 + phis[5]]], [+1,+1])

    # Setup the parametrization
    param = Parametrization(fields=[g])

    # Setup the intertwiners
    I = Intertwiner(param)
    J = InverseIntertwiner(I, order=1)

    def test_others(self):
        phis = phis(4)
        g = Field(1+phis[0], [])
        gg = Field([phis[1], phis[2], phis[3]])

        param = Parametrization(fields, [g])
        I = Intertwiner(param)
        J = InverseIntertwiner(I, order=1)

        coeff = ConstantOutputCoefficient(param, J, 0, [2,2])

        cJ = J.components[0]
        gamma = np.eye(3)



    def test_eights(self):
        coeff = ConstantOutputCoefficient(GenerateConstantOutputCoefficient.param, GenerateConstantOutputCoefficient.J, 0, [2,2])

        J = self.J.components[0]
        gamma = np.eye(3)

        def contracted(indices):
            x = [chr(ord('a') + i) for i in indices]
            s = "gamma[{},{}] * gamma[{},{}] * gamma[{},{}] * gamma[{},{}]".format(*x)

            @to_tensor(shape=(6,3,3,6,3,3))
            def fn(A,c,d,B,g,h):
                return sum([
                    J[A,a,b] * J[B,e,f] * eval(s, { "J": J, "gamma": gamma, "a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g, "h": h })
                    for a in range(3) for b in range(3) for e in range(3) for f in range(3)])
            
            return fn


        basisTensors = [[0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 6, 5, 7],
            [0, 1, 2, 4, 3, 5, 6, 7],
            [0, 1, 2, 4, 3, 6, 5, 7],
            [0, 1, 2, 6, 3, 7, 4, 5],
            [0, 2, 1, 3, 4, 5, 6, 7],
            [0, 2, 1, 3, 4, 6, 5, 7],
            [0, 2, 1, 4, 3, 5, 6, 7],
            [0, 2, 1, 4, 3, 6, 5, 7],
            [0, 2, 1, 6, 3, 4, 5, 7],
            [0, 2, 1, 6, 3, 7, 4, 5],
            [0, 4, 1, 5, 2, 3, 6, 7],
            [0, 4, 1, 5, 2, 6, 3, 7],
            [0, 4, 1, 6, 2, 3, 5, 7],
            [0, 4, 1, 6, 2, 5, 3, 7],
            [0, 6, 1, 7, 2, 3, 4, 5],
            [0, 6, 1, 7, 2, 4, 3, 5]]
        
        # Generate the correct result
        #results = [contracted(t) for t in basisTensors]
        #reshaped = [t.reshape(-1) for t in results]

        # Generate the contraction
        contraction = coeff.generateAllContractions()[0]
        
        # For each contraction generate the tensor shape
        tensorShape = coeff.generateTensorShape(contraction)
        # Generate the basis tensors
        basisTensor = coeff.generateBasisTensor(contraction, tensorShape)

        calculatedResults = [coeff.generateContractedBasisTensor(contraction, tensorShape, t) for t in basisTensor.indices]
        calculatedReshaped = [t.reshape(-1) for t in calculatedResults]
        
        #self.assertEqual(len(results), len(calculatedResults))

        # Check if all the results are the same
        #for a,b in zip(reshaped, calculatedReshaped):
        #    self.assertTrue(np.array_equal(a,b))

        # Symmetrize the tensors in (c d) and (g h)
        results = [(t.transpose((0,1,2,3,4,5)) + t.transpose((0,2,1,3,4,5)))/2 for t in calculatedResults]
        results = [(t + t.transpose((0,1,2,3,5,4)))/2 for t in results]
        results = [(t + t.transpose((3,4,5,0,1,2)))/2 for t in results]

        reshaped = [t.reshape(-1) for t in results]
        _, ids = sp.Matrix(reshaped).T.rref(simplify=True, iszerofunc=lambda x:abs(x)<1e-13)

        print(np.linalg.matrix_rank(reshaped))

        print(len(ids))
        print(ids)

        return



        

        