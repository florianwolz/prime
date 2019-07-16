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

import coloredlogs, logging

#from dask.distributed import get_client, secede, rejoin

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




def moveAxes(M, fromPos, toPos):
    order = list(range(len(M.shape)))
    order.remove(fromPos)
    order.insert(toPos, fromPos)

    return M.transpose(order), { v : k for k, v in zip(list(range(len(M.shape))), order) }


def randomName(length=3):
    import random
    return "".join([chr(ord('a') + random.randint(0, 25)) for i in range(length)])


greekAlphabet = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", 
                 "\\zeta", "\\eta", "\\theta", "\\iota", "\\kappa",
                 "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi", "\\rho",
                 "\\sigma", "\\tau", "\\varphi", "\\chi", "\\psi", "\\omega"]


def intertwinerToTeX(Findex, indices, field=0, fields=1, offsetF=0, offset=0):
    result = "\\mathcal{I}"
        
    if fields > 1:
        result += "^{(" + str(field+1) + ")}"
    
    result = result + "_" + \
        chr(ord('A')+ Findex + offsetF) + \
        "{}_{"

    for i in indices:
        result += greekAlphabet[i + offset]

    result += "}"
    return result


def epsilonGammaToTeX(indices, offset=0):
    if len(indices) == 0 or len(indices) == 1: return "0"
    elif len(indices) % 2 == 0:
        result = ""
        for i in range(0,len(indices),2):
            result += "\gamma^{" + \
                greekAlphabet[indices[i] + offset] + \
                greekAlphabet[indices[i+1] + offset] + \
                "}"
        return result
    else:
        result = "\\varepsilon^{" + \
                greekAlphabet[indices[0] + offset] + \
                greekAlphabet[indices[1] + offset] + \
                greekAlphabet[indices[2] + offset] + \
                "}"
        if len(indices) == 3: return result
        return result + epsilonGammaToTeX(indices[3:], offset)


def phisToLaTeX(index, derivIndices=[]):
    result = "\\varphi^{" + chr(ord('A') + index) + "}"
    if len(derivIndices) == 0: return result
    result += "{}_{,"
    for i in derivIndices:
        result += greekAlphabet[i]
    result += "}"
    return result

        
"""
VelocityContraction

Storage for a contraction of a velocity index with a certain intertwiner
"""
class VelocityContraction(object):
    def __init__(self, id, intertwiner, offset=0, symmetries={}):
        self.id = id
        self.intertwiner = intertwiner

        from copy import deepcopy
        self.symmetries = deepcopy(symmetries)

        self.offset = offset

    def shape(self):
        return self.intertwiner.shape[1:]
    
    def rank(self):
        return len(self.shape())
    
    def getIndices(self, ignoreOffset=False):
        offset = self.offset if not ignoreOffset else 0
        return tuple(range(offset, offset + self.rank()))
    
    def getSymmetries(self):
        return [Symmetry(sym.type, tuple([i + self.offset for i in sym.indices])) for _, sym in self.symmetries.items()]
    
    def __str__(self):
        return "<{}, {}, {}>".format(self.id, self.getIndices(), self.offset)
    
    def __repr__(self):
        return str(self)

"""
PhiContraction

"""
class PhiContraction(object):
    def __init__(self, id, intertwiner, derivs=0, offset=0, symmetries={}):
        self.id = id
        self.intertwiner = intertwiner
        self.derivs = derivs

        from copy import deepcopy
        self.symmetries = deepcopy(symmetries)

        self.offset = 0

        # Append the derivative symmetry if necessary
        if self.derivs > 1:
            inds = self.getDerivativeIndices()
            self.symmetries[inds] = Symmetry(Symmetry.SymmetryType.SYMMETRIC, self.getDerivativeIndices())
        
        # Apply the offset
        self.offset = offset
    
    def shape(self):
        return self.intertwiner.shape[1:]
    
    def derivativeShape(self):
        shape = tuple()
        for d in range(self.derivs):
            shape = shape + (3,)
        return shape
    
    def totalShape(self):
        return self.shape() + self.derivativeShape()

    def rank(self):
        return len(self.shape())
    
    def totalRank(self):
        return self.rank() + self.derivs
    
    def getIndices(self, ignoreOffset=False):
        offset = self.offset if not ignoreOffset else 0
        return tuple(range(offset, offset + self.rank()))
    
    def getDerivativeIndices(self, ignoreOffset=False):
        offset = self.offset if not ignoreOffset else 0
        return tuple(range(offset + self.rank(), offset + self.rank() + self.derivs))
    
    def getAllIndices(self, ignoreOffset=False):
        return self.getIndices(ignoreOffset) + self.getDerivativeIndices(ignoreOffset)
    
    def getAllIndicesAfterContraction(self, offset=0, ignoreOffset=False):
        if ignoreOffset: offset = 0
        return tuple(range(offset, offset + 1 + self.derivs))

    def getSymmetries(self):
        return [Symmetry(sym.type, tuple([i + self.offset for i in sym.indices])) for _, sym in self.symmetries.items()]
    
    def __str__(self):
        return "<{}, {}, {}, {}>".format(self.id, self.getAllIndices(), self.offset, self.derivs)
    
    def __repr__(self):
        return str(self)

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

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

        # Properly generate the components by generating the basis
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
    

    """
    Generate all the possible contractions with the constant intertwiners

    Generates all the possible contractions with the constant intertwiners.
    It makes use of the block symmetries of the indices in order to get rid
    of redundent contractions that would fall out of the Gauss elimination
    later anyways. 
    """
    def generateAllContractions(self):
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

        # Take the Cartesian product of all the K and Phi contractions
        contractions = list(product(contrsK, contrsP))

        # Turn them into proper contractions
        contractions = [([VelocityContraction(K, self.J.constant()[K], symmetries=self.parametrization.fields[K].symmetries) for K in Ks], [PhiContraction(Phi, self.J.constant()[Phi], derivs=d, symmetries=self.parametrization.fields[Phi].symmetries) for Phi, d in zip(Phis, self.derivs)]) for Ks, Phis in contractions]

        # Update the offsets
        for Ks, Phis in contractions:
            offset = 0
            for K in Ks:
                K.offset = offset
                offset = offset + K.rank()
            for Phi in Phis:
                Phi.offset = offset
                offset = offset + Phi.totalRank()
        
        return contractions

    
    """
    Generate the tensor shape and the symmetries to a given contraction

    Args:
        contraction     The contraction
    
    Returns:
        shape           The shape of the resulting tensor
        symmetries      The tensor symmetries
    """
    def generateTensorShape(self, contraction):
        # Unfold the contraction
        Ks, Phis = contraction

        # Calculate the shape
        shape = tuple(sum([K.shape() for K in Ks], tuple())) + tuple(sum([Phi.totalShape() for Phi in Phis], tuple()))

        # Generate a list of all the symmetries
        symmetries = [sym for K in Ks for sym in K.getSymmetries()]
        symmetries = symmetries + [sym for Phi in Phis for sym in Phi.getSymmetries()]

        return shape, symmetries
    
    
    """
    Generate the list of all the possible basis tensors for 
    a given contraction and tensor shape

    Args:
        contraction         The contraction
        tensorShape         The tensor shape
    
    Returns:
        Indices object contain all the possible tensor basis elements
    """
    def generateBasisTensor(self, contraction, tensorShape):
        # Unfold the tensor shape
        shape, symmetries = tensorShape

        # Generate the possible basis terms
        idx = Indices(len(shape))

        # Apply the symmetries
        # TODO: Also allow antisymmetric terms
        syms = [tuple(s.indices) for s in symmetries if s.type == Symmetry.SymmetryType.SYMMETRIC]
        idx.symmetrize(syms)

        return idx
    

    """

    """
    def generateContractedBasisTensorNaive(self, contraction, tensorShape, basisTensor):
        pass
    
    
    """
    Does all the heavy-lifting to turn the possible epsilon-gamma index
    assignments into a proper basis tensor element. To do this in an efficient
    fashion we use the following algorithm:

    First we identify all the traces in the index assignment, i.e. a gamma or
    epsilon is completely contracted with an intertwiner, and trace / "epsilon-trace"
    over the indices since this immediately reduces the rank of the involved tensors.
    For the resulting intertwiner terms we calculate the tensor product and get rid of 
    the remaining traces. The rest remaining epsilon / gamma indices are then tensor 
    transposes to get the index in the correct derivative slot(s) or have to be multiplied to
    the tensor and transposed to the correct slot. This is only for coefficients with
    second derivative order phi terms.

    Args:
        Js          The constant inverse intertwiners
        bars        Which slots are there?
        index       The index assignment for the epsilon-gamma term
        shape       The final shape
    """
    def generateContractedBasisTensor(self, contraction, tensorShape, basisTensor):
        # Take the basis tensor and split into epsilon-gamma blocks
        blocks = basisTensor.blocks()

        # Import epsilon
        from prime.utils import epsilon
        eps = epsilon

        # Unfold the contractions
        Ks, Phis = contraction

        # Create the components
        from copy import deepcopy
        comps = [deepcopy(K.intertwiner) for K in Ks] + [deepcopy(Phi.intertwiner) for Phi in Phis]

        # Create the index swapping memory
        idxMemory = [{ idx : idx - K.offset + 1 if idx in K.getIndices() else None for idx in range(basisTensor.rank()) } for K in Ks] + \
                    [{ idx : idx - Phi.offset + 1 if idx in Phi.getIndices() else None for idx in range(basisTensor.rank()) } for Phi in Phis]
        
        # Calculate the derivative indices
        derivs = {}
        outputMemory = {}
        offset = len(Ks)
        for i, Phi in enumerate(Phis):
            d = Phi.getDerivativeIndices()
            offset = offset + 1
            for x in d:
                derivs[x] = offset
                outputMemory[x] = None
                offset = offset + 1
        
        # Get rid of all blocks with traces
        newBlocks = []
        for block in blocks:
            isTrace = False
            traceI = 0
            traceIndices = None
            isGammaTrace = False

            for i, K in enumerate(Ks):
                idx = K.getIndices()

                if len([None for x in block if x in idx]) == len(block):
                    isTrace = True
                    isGammaTrace = len(block) == 2
                    traceI = i
                    traceIndices = tuple([idxMemory[i].get(j, None) for j in block])
                    break
            
            # If no trace in the K slots are found, look for ones in the phi slots
            if not isTrace:
                for i, Phi in enumerate(Phis):
                    idx = Phi.getIndices()

                    if len([None for x in block if x in idx]) == len(block):
                        isTrace = True
                        isGammaTrace = len(block) == 2
                        traceI = i + len(Ks)
                        traceIndices = tuple([idxMemory[traceI].get(j, None) for j in block])
                        break
            
            if not isTrace:
                newBlocks.append(block)
                continue
            
            if isGammaTrace:
                a, b = traceIndices

                if a is None or b is None: raise Exception("Try to take the trace over an index that is already gone.")

                # Take the trace
                comps[traceI] = np.trace(comps[traceI], axis1=a, axis2=b)
                
                # Update the index memory for the intertwiner
                for k in range(len(idxMemory)):
                    idxMemory[k][block[0]] = None
                    idxMemory[k][block[1]] = None
                
                for k in range(basisTensor.rank()):
                    c = idxMemory[traceI].get(k, None)
                    if c is None: continue
                    elif c > a and c < b:
                        idxMemory[traceI][k] = c-1
                    elif c > a and c > b:
                        idxMemory[traceI][k] = c-2
            else:
                a, b, c = traceIndices
                assert(a < b and b < c)

                if a is None or b is None or c is None: raise Exception("Try to take the trace over an index that is already gone.")
                
                # Contract with epsilon
                comps[traceI] = np.tensordot(comps[traceI], eps, axes=((a,b,c), (0,1,2)))

                # Update the index memory for the intertwiner
                for k in range(len(idxMemory)):
                    idxMemory[k][block[0]] = None
                    idxMemory[k][block[1]] = None
                    idxMemory[k][block[2]] = None
                
                for k in range(basisTensor.rank()):
                    d = idxMemory[traceI].get(k, None)
                    if d is None: continue
                    elif d > a and d < b and d < c:
                        idxMemory[traceI][k] = d-1
                    elif d > a and d > b and d < c:
                        idxMemory[traceI][k] = d-2
                    elif d > a and d > b and d > c:
                        idxMemory[traceI][k] = d-3
        
        # Get rid of all blocks we traced out
        blocks = newBlocks

        # Tensor multiply all the traced intertwiners together
        Js = comps[0]
        if len(comps) > 1:
            for k in range(1, len(comps)):
                Js = np.tensordot(Js, comps[k], axes=0)

        # If there are no more blocks left, can return the tensor
        if len(blocks) == 0:
            return Js
        
        # Update the index memory for the new tensor
        offset = 0
        for m in range(len(idxMemory)):
            for k in idxMemory[m]:
                if idxMemory[m][k] is not None:
                    idxMemory[m][k] = idxMemory[m][k] + offset
            offset = offset + len(comps[m].shape)
        firstOrNone = lambda x : x[0] if len(x) > 0 else None
        idxMemory = { k : firstOrNone([idxMemory[m].get(k, None) for m in range(len(idxMemory)) if idxMemory[m].get(k, None) is not None]) for  k in range(basisTensor.rank()) }

        # Get rid of the contractions inside of this tensor
        newBlocks = []
        for block in blocks:
            mapped = [ idxMemory.get(i) for i in block if idxMemory.get(i, None) is not None ]

            if len(mapped) != len(block):
                newBlocks.append(block)
                continue

            # Trace with gamma
            if len(mapped) == 2:
                # Take the trace
                Js = np.trace(Js, axis1=mapped[0], axis2=mapped[1])

                # Update the memory
                idxMemory[block[0]] = None
                idxMemory[block[1]] = None

                for k in range(basisTensor.rank()):
                    c = idxMemory.get(k, None)
                    if c is None: continue
                    elif c > mapped[0] and c < mapped[1]:
                        idxMemory[k] = c-1
                    elif c > mapped[0] and c > mapped[1]:
                        idxMemory[k] = c-2
            # Trace with epsilon
            elif len(mapped) == 3:
                # Contract with epsilon
                Js = np.tensordot(Js, eps, axes=((mapped[0],mapped[1],mapped[2]), (0,1,2)))

                # Update the memory
                idxMemory[block[0]] = None
                idxMemory[block[1]] = None
                idxMemory[block[2]] = None

                for k in range(basisTensor.rank()):
                    c = idxMemory.get(k, None)
                    if c is None: continue
                    elif c > mapped[0] and c < mapped[1] and c < mapped[2]:
                        idxMemory[k] = c-1
                    elif c > mapped[0] and c > mapped[1] and c < mapped[2]:
                        idxMemory[k] = c-2
                    elif c > mapped[0] and c > mapped[1] and c > mapped[2]:
                        idxMemory[k] = c-3
        
        # Update the blocks
        blocks = newBlocks

        # No more blocks left? Return the result
        if len(blocks) == 0:
            return Js

        # Multiply missing gammas/epsilons in
        newBlocks = []
        swap = []
        for block in blocks:
            mapped = [derivs[b] for b in block if b in derivs]
            if len(mapped) == 2 and len(block) == 2:
                for i in range(2):
                    outputMemory[block[i]] = len(Js.shape)+i

                # Tensorproduct with gamma
                Js = np.tensordot(Js, np.eye(3), axes=0)

            elif len(mapped) == 3 and len(block) == 3:
                for i in range(3):
                    outputMemory[block[i]] = len(Js.shape)+i

                # Multiply epsilon in
                Js = np.tensordot(Js, eps, axes=0)
            else:
                newBlocks.append(block)

        blocks = newBlocks

        # Pull indices up and to the correct position
        newBlocks = []
        for block in blocks:
            mapped = [ idxMemory.get(i) for i in block if idxMemory.get(i, None) is not None ]

            # Pulling with gamma?
            if len(mapped) == 1 and len(block) == 2:
                in_  = mapped[0]
                out_ = [x for x in block if idxMemory.get(x, None) != in_]
                assert(len(out_) == 1)

                outputMemory[out_[0]] = in_

            # Pulling one index with epsilon
            elif len(mapped) == 1 and len(block) == 3:
                out_    = [x for x in block if idxMemory.get(x, None) == mapped[0]]
                derivs_ = [x for x in block if idxMemory.get(x, None) != mapped[0]]
                assert(len(out_) == 1 and len(derivs_) == 2)
                Js = np.tensordot(Js, eps, axes=(mapped[0], block.index(out_[0])))

                # Update the index memory since we got rid of some indices
                for k in idxMemory:
                    if idxMemory[k] is None: continue
                    if idxMemory[k] > mapped[0]: idxMemory[k] = idxMemory[k]-1
                    elif idxMemory[k] == mapped[0]: idxMemory[k] = None

                # Update the derivative memory since we moved some axes due to the contraction
                for k in outputMemory:
                    if outputMemory[k] is None: continue
                    if outputMemory[k] > mapped[0]: outputMemory[k] = outputMemory[k]-1
            
                # Note that the derivative indices are in the end so that 
                # the swapping can do its magic
                outputMemory[derivs_[0]] = len(Js.shape)-2
                outputMemory[derivs_[1]] = len(Js.shape)-1

            # Pulling two indices with epsilon
            elif len(mapped) == 2 and len(block) == 3:
                out_    = [block.index(x) for x in block if idxMemory.get(x, None) in mapped]
                derivs_ = [x for x in block if idxMemory.get(x, None) not in mapped]
                assert(len(out_) == 2 and len(derivs_) == 1)

                Js = np.tensordot(Js, eps, axes=(tuple(mapped), tuple(out_)))

                a = min(mapped[0], mapped[1])
                b = max(mapped[0], mapped[1])
                for k in idxMemory:
                    if idxMemory[k] is None: continue
                    if idxMemory[k] == a or idxMemory[k] == b:
                        idxMemory[k] = None
                    elif idxMemory[k] > a and idxMemory[k] < b: 
                        idxMemory[k] = idxMemory[k]-1
                    elif idxMemory[k] > b: 
                        idxMemory[k] = idxMemory[k]-2

                for k in outputMemory:
                    if outputMemory[k] is None: continue
                    if outputMemory[k] == a or outputMemory[k] == b:
                        outputMemory[k] = None
                    elif outputMemory[k] > a and outputMemory[k] < b: 
                        outputMemory[k] = outputMemory[k]-1
                    elif outputMemory[k] > b: 
                        outputMemory[k] = outputMemory[k]-2

                # Note that the derivative indices are in the end so that 
                # the swapping can do its magic
                outputMemory[derivs_[0]] = len(Js.shape)-1

            else:
                newBlocks.append(block)

        blocks = newBlocks

        # Swap again if necessary
        while True:
            swaps = []
            swapped = False
            for x in derivs:
                if derivs[x] == outputMemory[x]: continue

                a = outputMemory[x]
                b = derivs[x]

                Js, _ = moveAxes(Js, a, b)

                for k in outputMemory:
                    c = outputMemory[k]
                    if c == a: outputMemory[k] = b
                    if a > b:
                        if c >= b and c < a: outputMemory[k] = c+1
                    elif b > a:
                        if c > a and c <= b: outputMemory[k] = c-1

                swapped = True
                break
            
            # No more swaps? Finally finished
            if not swapped: break
        
        # Make sure there are no more blocks in the assignment ...
        blocks = newBlocks
        if len(blocks) == 0:
            return Js
        
        raise Exception("What a cruel world ...")


    """
    Symmetrize the potential basis tensors in the derivative indices

    When generated, the potential basis tensors are not yet symmetric
    in the derivative indices since the symmetry was only implicitely used
    to get rid of other representatives of the double coset.
    """
    def symmetrizeDerivatives(self, contraction, tensor):
        # Import the symmetrization method
        from prime.utils import symmetrize

        # Unfold
        Ks, Phis = contraction

        # Prepare the result
        from copy import deepcopy
        result = deepcopy(tensor)

        offset = len(Ks) 
        for i, Phi in enumerate(Phis):
            # If there are more than one derivative indices
            if Phi.derivs > 1:
                indices = list(range(offset + 1, offset + 1 + Phi.derivs))
                result = symmetrize(result, indices)
            
            offset = offset + 1 + Phi.derivs
        
        return result
    

    def symmetrizeBlocks(self, contraction, tensor):
        # Import the symmetrization method
        from prime.utils import symmetrize

        # Unfold
        Ks, Phis = contraction

        # Prepare the result
        from copy import deepcopy
        result = deepcopy(tensor)

        # First the exchange symmetries in the velocity indices
        if len(Ks) > 1:
            result = symmetrize(result, list(range(len(Ks))))

        # Now the Phi part
        if len(Phis) > 1:
            for i, Phi in enumerate(Phis):
                for j in range(i+1, len(Phis)):
                    # If the number of derivative indices is different move to the next one
                    if Phi.derivs != Phis[j].derivs: continue

                    # Symmetrize in the i-th and j-th block
                    shape = tuple(range(len(Ks)))
                    for k, P in enumerate(Phis):
                        if k == i: shape = shape + Phis[j].getAllIndicesAfterContraction(offset=Phis[j].new_offset)
                        elif k == j: shape = shape + Phi.getAllIndicesAfterContraction(offset=Phi.new_offset)
                        else: shape = shape + P.getAllIndicesAfterContraction(offset=P.new_offset)

                    result = (result + result.transpose(shape)) / 2

        return result
    

    def toLaTeX(self, substitutions={}):
        lines = []

        # Fix all substitutions if necessary:
        for k in self.variableMap:
            if k not in substitutions:
                substitutions[k] = k

        for k, v in self.variableMap.items():
            epsilonGamma = epsilonGammaToTeX(v[1].indices)
            phiTerms = ""
            intTerms = ""

            # Get the substituted value
            key = substitutions[k]

            # Ignore all terms that are nulled by the substitions
            if key == 0: continue

            # Put the key in brackets, if necessary
            key = str(key)
            if "+" in key or "-" in key:
                key = "(" + key + ")"

            # The K terms
            Aoffset = 0
            if len(v[0][0]) > 0:
                for K in v[0][0]:
                    intTerms += intertwinerToTeX(Aoffset, K.getIndices(), field=K.id, fields=len(self.J.components))
                    Aoffset += 1
                intTerms += " "

            # The Phi terms
            if len(v[0][1]) > 0:
                phiTerms += " "
                for Phi in v[0][1]:
                    intTerms += intertwinerToTeX(Aoffset, Phi.getIndices(), field=Phi.id, fields=len(self.J.components))
                    phiTerms += phisToLaTeX(Aoffset, Phi.getDerivativeIndices())
                    Aoffset += 1
                intTerms += " "

            lines.append("{} * {}{}{}".format(key, intTerms, epsilonGamma, phiTerms))

        # No terms left after substitutions => return 0
        if len(lines) == 0: return "0"

        return " +\n".join(lines)


    def generate(self):
        # First generate all the contractions
        contractions = self.generateAllContractions()

        # For each contraction generate the tensor shape
        tensorShapes = [self.generateTensorShape(c) for c in contractions]

        # Generate the basis tensors
        basisTensors = [self.generateBasisTensor(*args) for args in zip(contractions, tensorShapes)]

        # Contract the tensors with the epsilon-gamma terms and flatten the list
        contractedBasisTensors = [(contractions[i], b, self.generateContractedBasisTensor(contractions[i], tensorShapes[i], b)) for i in range(len(contractions)) for b in basisTensors[i].indices]
        contractedBasisTensors = [x for x in contractedBasisTensors if x[2] is not None]

        # No contraction?
        if len(contractedBasisTensors) == 0:
            self.components = sympy.Symbol("x") * np.zeros(self.shape)
            self.variableMap = {}
            return

        # Make sure all the tensors have the same shape
        shapes = list(set([t.shape for _, _, t in contractedBasisTensors]))
        if len(shapes) != 1:
            raise Exception("The output coefficients don't all have the correct shape. Found {}".format(shapes))
        
        # Recalculate the offset of the contractions
        for k in range(len(contractedBasisTensors)):
            the_offset = 0
            for i in range(len(contractedBasisTensors[k][0][0])):
                contractedBasisTensors[k][0][0][i].new_offset = the_offset
                the_offset = the_offset + 1
            for i in range(len(contractedBasisTensors[k][0][1])):
                contractedBasisTensors[k][0][1][i].new_offset = the_offset
                the_offset = the_offset + 1 + contractedBasisTensors[k][0][1][i].derivs

        # Implement the derivative symmetries
        contractedBasisTensors = [(c, b, self.symmetrizeDerivatives(c,t)) for c, b, t in contractedBasisTensors]
        contractedBasisTensors = [(c, b, self.symmetrizeBlocks(c,t)) for c, b, t in contractedBasisTensors]

        # Gauss elimination to get rid of all linear dependent ones
        from sympy import Matrix
        _, linIndeps = Matrix([t.reshape(-1) for _, _, t in contractedBasisTensors]).T.rref(simplify=True, iszerofunc=lambda x:abs(x)<1e-13)
        basis = [contractedBasisTensors[i] for i in linIndeps]

        # After the Gauss elimination no tensors left?
        if len(basis) == 0:
            self.components = sympy.Symbol("x") * np.zeros(self.shape)
            self.variableMap = {}
            return

        # Calculate the components
        uniqueId = randomName(length=5)
        self.components = sum([sympy.Symbol("{}_{}".format(uniqueId, i)) * b[2] for i, b in enumerate(basis, 1)])
        self.variableMap = { sympy.Symbol("{}_{}".format(uniqueId, i)) : (b[0], b[1]) for i, b in enumerate(basis, 1) }

        #print(self)
        #print(self.toLaTeX())
        #print()


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
    """
    def generate(self):
        expandify = np.vectorize(lambda x : x.expand())

        def generateConstCoeff(c, dofs, order):
            # Generate the coefficient
            c.generate()

            tmp = c.components

            # Constant part of the coefficient?
            if len(c.derivs) == 0:
                return tmp

            # Contract the indices from the phi expansions
            for d in c.derivs:
                dphis = dPhis(dofs, d)
                a = tuple(range(self.order, self.order + d + 1))
                b = tuple(range(d + 1))

                tmp = np.tensordot(tmp, dphis, axes=(a, b))
            
            #print("       Finished {}".format(c))
            tmp = expandify(tmp)

            # Ignore zeros
            return tmp

        futures = [generateConstCoeff(coeff, self.parametrization.dofs, self.order) for coeff in self.constCoeffs]

        #c = get_client()
        #futures = [c.submit(generateConstCoeff, coeff, self.parametrization.dofs, self.order) for coeff in self.constCoeffs]

        #secede()
        #c.gather(futures)
        #rejoin()

        # Add them together
        if len(futures) == 0:
            return

        self.components = np.array(sum(futures))
        #self.components = futures[0]#.result()
        #for i in range(1, len(futures)):
        #    self.components = self.components + futures[i]#.result()
        
        #print(self.toLaTeX())
        
        # Add O(n+1) terms

        ## Add the O(n+1) terms but more quickly. Probably good idea to cache that
        ## for future reference

        #os = self.parametrization.Odiff(self.maxOrder - self.order + 1, collapse=self.collapse)
        #self.components = self.components + os

        # Merge all variable maps
        self.variableMap = { k: v for coeff in self.constCoeffs for k, v in coeff.variableMap.items() }
    
    def toLaTeX(self, substitions={}):
        lines = []
        for coeff in self.constCoeffs:
            s = coeff.toLaTeX(substitions)
            if s != "0": lines.append(s)
        return " +\n".join(lines)


    @property
    def name(self):
        s = "C"

        def alpha(N, offset=0):
            return list(map(chr, range(ord('A')+offset, ord('A')+N+offset)))

        if self.order > 0:
            s = s + "_{}".format("".join(alpha(self.order)))

        return s


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
    result = [OutputCoefficient(parametrization, J, o, order, collapse) for o in range(order+1)]
    return result