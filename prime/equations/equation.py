import numpy as np
from sympy import poly, diff, Matrix, expand
from prime.input.parametrization import jet_diff, spatial_diff, jet_variables
from prime.utils import to_tensor


def filter_non_jet_vars(expr, dofs):
    if type(expr) is np.ndarray:
        res = [var for vars in [filter_non_jet_vars(e, dofs) for e in expr.reshape(-1)] for var in vars]

        d = dict()
        for var in res:
            d[str(var)] = var

        return list(d.values())

    symbs = expr.free_symbols
    result = []
    for s in symbs:
        t = str(s).split("_")
        if not t[0] in [str(d) for d in dofs]: continue
        result.append(s)
    # TODO: Sort result
    return result


##
# Shape tokens for free indices
class ShapeToken:
    def __init__(self, name, offset=0):
        self.name = name
        self.offset = offset
    
    def __add__(self, x):
        return ShapeToken(self.name, self.offset+x)

    def __sub__(self, x):
        return ShapeToken(self.name, self.offset+x)

F = ShapeToken('F')
FN = ShapeToken('FN')
SpatialN = ShapeToken('SpatialN')
Collapse = ShapeToken('Collapse')
MaxOrder = ShapeToken('MaxOrder')


class ScalarEquation(object):
    shape = ()
    componentWise = True
    name = "ScalarEquation"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, order=1, *args, **kwargs):
        self.parametrization = parametrization
        self.Cs = Cs
        self.E = E
        self.F = F
        self.M = M
        self.p = p
        self.degP = degP

        self.collapse = kwargs.get('collapse', 2)
        self.order = order

        # Replace the F in shapes by the correct number of degrees of freedoms
        self.shape = [len(self.parametrization.dofs) if type(x) is ShapeToken and x.name == 'F' else x for x in self.shape]

        # Inspect the component method
        #if not hasattr(self, "component"): raise Exception("The equation does not calculate any components.")
        #from inspect import signature
        #if len(self.shape)+1 != len(signature(self.component).parameters):
        #    raise Exception("The method to calculate the components of the equation has the wrong signature.")

        # Setup the jet derivative operation
        self.diff = jet_diff(parametrization)

        self.calculated = False
    
    def component(self, *args, **kwargs):
        raise NotImplementedError()
    
    def allComponents(self, *args, **kwargs):
        raise NotImplementedError()

    def calculate(self):
        if self.calculated: return

        # Call the individual method to setup the components of the relation
        if self.componentWise:
            self.components = to_tensor(shape=self.shape)(self.component)
        else:
            self.components = self.allComponents()

        # If no components were setup, throw an exception
        if not hasattr(self, "components"):
            raise Exception("No components of the equations were calculated")
        
        # Turn the components into a numpy array
        if not type(self.components) is np.ndarray:
            self.components = np.array(self.components)

        # Get rid of all the dirt terms
        dropOs = np.vectorize(lambda x : x.removeO())
        self.components = dropOs(self.components)

        # Mark the equation as calculated
        self.calculated = True


    """
    Extracts all the relations on the arbitrary constants of the output coefficients
    """
    def relations(self, diagonalize=True):
        print(F"Solve equation {self.name} ...")

        if not self.calculated: self.calculate()

        # Extract all the coefficients from the components
        exprs = self.components.reshape(-1).tolist()
        #expr = [expr.expand() for expr in exprs]
        variables = [filter_non_jet_vars(expr, self.parametrization.dofs) for expr in exprs]
        
        # Merge the jet variables
        variables = [var for vars_ in variables for var in vars_]

        variables = list(({ str(v) : v for v in variables}).values())
#
#        variables = list(dict.fromkeys([str(v) for v in variables], variables).values())
        print("Build polynomials ...")

        polys = [poly(expand(expr), variables).coeffs() for expr in exprs]

        print("Derive relations ...")
    
        # Flatten the relations into a list
        relations = [y for x in polys for y in x]

        # Find all the free symbols in there
        syms = list(set([sym for relation in relations for sym in relation.free_symbols]))

        # Diagonalize if necessary
        if diagonalize:
            print("Diagonalize ...")
            M = Matrix([[diff(relation, var) for var in syms] for relation in relations])
            M, _ = M.rref()
            relations = [sum([M[i,j] * sym for j, sym in enumerate(syms)]) for i, _ in enumerate(relations)]
            relations = [relation for relation in relations if relation != 0]
        
        print(F"Finished equation {self.name}.")

        return relations, syms
    

    """
    Takes the spatial derivative of a 1st derivative of an output coefficient
    and then takes the trace over these derivative indices
    """
    def coefficientDerivativeTrace(self, N=0, derivOrder=0, freeIndices=0):
        Cd = self.diff(expr=self.Cs[N], order=derivOrder+freeIndices)
        result = spatial_diff(expr=Cd, order=derivOrder) if derivOrder > 0 else Cd
        for i in range(derivOrder):
            result = np.trace(result, axis1=0, axis2=len(result.shape)-1)
        return result
    

    def sumCoefficientDerivativeTrace(self, N=0, freeIndices=0, maxOrder=-1, combinatorial='1', alternatingSign=False, locals={}):
        if N>=2:
            maxOrder = 2 - freeIndices
        
        if maxOrder == -1:
            maxOrder = self.collapse - freeIndices
        
        if maxOrder < 0: return None
        
        from copy import deepcopy
        from prime.utils import binomial, factorial
        locals_ = deepcopy(locals)
        locals_["factorial"] = factorial
        locals_["binomial"] = binomial

        result = None

        for K in range(0, maxOrder+1):
            factor = (-1)**K if alternatingSign else 1
            locals_["K"] = K
            factor = factor * eval(combinatorial, locals_)

            if K == 0:
                result = factor * self.coefficientDerivativeTrace(N=N, derivOrder=K, freeIndices=freeIndices)
            else:
                result = result + factor * self.coefficientDerivativeTrace(N=N, derivOrder=K, freeIndices=freeIndices)
        
        return result
            

    """
    Method to quickly calculate the contraction of the 1st derivatives of an output coefficient
    with the spatial derivative of an input coefficient
    """
    def coefficientDerivativeContraction(self, coeff, N=0, derivOrder=0, Aposition=0, freeIndices=0):
        Cshape = self.Cs[N].shape
        Cd = self.diff(expr=self.Cs[N], order=derivOrder+freeIndices)
        coeffd = spatial_diff(expr=coeff, order=derivOrder) if derivOrder > 0 else coeff

        # Calculate the index positions for the contraction
        Caxes = (len(Cshape), ) + tuple(range(len(Cd.shape) - derivOrder, len(Cd.shape)))
        caxes = (Aposition + derivOrder, ) + tuple(range(derivOrder))

        result = np.tensordot(Cd, coeffd, axes=(Caxes, caxes))

        return result


    def sumCoefficientDerivativeContraction(self, coeff, N=0, freeIndices=0, maxOrder=-1, Aposition=0, combinatorial='1', alternatingSign=False, locals={}):
        if N>=2:
            maxOrder = 2 - freeIndices
        
        if maxOrder == -1:
            maxOrder = self.collapse - freeIndices
        
        if maxOrder < 0: return None
        
        from copy import deepcopy
        from prime.utils import binomial, factorial
        locals_ = deepcopy(locals)
        locals_["factorial"] = factorial
        locals_["binomial"] = binomial

        result = None

        for K in range(0, maxOrder+1):
            factor = (-1)**K if alternatingSign else 1
            locals_["K"] = K
            factor = factor * eval(combinatorial, locals_)

            if K == 0:
                result = factor * self.coefficientDerivativeContraction(coeff, N=N, derivOrder=K, freeIndices=freeIndices, Aposition=Aposition)
            else:
                result = result + factor * self.coefficientDerivativeContraction(coeff, N=N, derivOrder=K, freeIndices=freeIndices, Aposition=Aposition)
        
        return result


class SequenceEquation(object):
    shape = ()
    componentWise = True
    onlyEven = False
    onlyOdd = False
    Nmax = 0
    name = "SequenceEquation"

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        self.parametrization = parametrization
        self.Cs = Cs
        self.E = E
        self.F = F
        self.M = M
        self.p = p
        self.degP = degP

        # Setup the jet derivative operation
        self.diff = jet_diff(parametrization)

        self.order = kwargs.get('order', 2)
        self.collapse = kwargs.get('collapse', 2)

        self.calculated = False
    
    def component(self, N, *args, **kwargs):
        raise NotImplementedError()
    
    def allComponents(self, N, *args, **kwargs):
        raise NotImplementedError()

    def calculate(self):
        if self.calculated: return

        self.components = []

        # Calculate the maximal term
        Nmax = self.Nmax
        if type(Nmax) is ShapeToken and Nmax.name == 'MaxOrder': Nmax = self.order + Nmax.offset
        elif type(Nmax) is ShapeToken and Nmax.name == 'Collapse': Nmax = self.collapse + Nmax.offset

        # Allow to calculate only odd/even terms
        from_ = 3 if self.onlyOdd else 2
        steps = 2 if self.onlyOdd or self.onlyEven else 1

        # Iterate over the free indices
        for N in range(from_, Nmax+1, steps):
            if self.componentWise:

                from itertools import product

                # Prepare the indices in the shape 
                Bs = []
                mus = []
                As = []
                betas = []

                for i in self.shape:
                    if type(i) is ShapeToken and i.name == 'FN':
                        Bs.append(list(product(*[list(range(len(self.parametrization.dofs))) for p in range(N+i.offset)])))
                    elif type(i) is ShapeToken and i.name == 'SpatialN':
                        mus.append(list(product(*[[0,1,2] for p in range(N+i.offset)])))
                    elif type(i) is ShapeToken and i.name == 'F':
                        As.append(list(range(len(self.parametrization.dofs))))
                    else:
                        betas.append([0,1,2])

                # Prepare the function parameters
                args = product(Bs, mus, As, betas) 
                
                # Calculate
                self.components.append([self.component(N, *arg) for arg in args])
            else:
                # Calculate all components at once
                self.components.append(self.allComponents(N))
        
        # Remark: The tensor doesn't necessarily have the intended shape, i.e. when calculating component wise
        #         it is simply a vector. Since we will then iterate over the components anyway, we do not care
        #         about this.
        
        # Turn into a numpy array
        self.components = np.array(self.components)

        # Get rid of all the dirt terms
        dropOs = np.vectorize(lambda x : x.removeO())
        self.components = dropOs(self.components)
        
        # Mark the equation as calculated
        self.calculated = True

    """
    Extracts all the relations on the arbitrary constants of the output coefficients
    """
    def relations(self, diagonalize=True):
        print(F"Solve equation {self.name} ...")

        if not self.calculated: self.calculate()

        # Extract all the coefficients from the components
        polys = [poly(expr, filter_non_jet_vars(expr, self.parametrization.dofs)).coeffs() for expr in np.nditer(self.components)]

        print("Derive relations ...")
    
        # Flatten the relations into a list
        relations = [y for x in polys for y in x]

        # Find all the free symbols in there
        syms = list(set([sym for relation in relations for sym in relation.free_symbols]))

        # Diagonalize if necessary
        if diagonalize and len(relations) > 0:
            print("Diagonalize ...")
            M = Matrix([[diff(relation, var) for var in syms] for relation in relations])
            M, _ = M.rref()
            relations = [sum([M[i,j] * sym for j, sym in enumerate(syms)]) for i, _ in enumerate(relations)]
            relations = [relation for relation in relations if relation != 0]
        
        print(F"Done with {self.name}")

        return relations, syms
        
    """
    Takes the spatial derivative of a 1st derivative of an output coefficient
    and then takes the trace over these derivative indices
    """
    def coefficientDerivativeTrace(self, N=0, derivOrder=0, freeIndices=0):
        Cd = self.diff(expr=self.Cs[N], order=derivOrder+freeIndices)
        result = spatial_diff(expr=Cd, order=derivOrder) if derivOrder > 0 else Cd
        for i in range(derivOrder):
            result = np.trace(result, axis1=0, axis2=len(result.shape)-1)
        return result
    

    def sumCoefficientDerivativeTrace(self, N=0, freeIndices=0, maxOrder=-1, combinatorial='1', alternatingSign=False, locals={}):
        if N>=2:
            maxOrder = 2 - freeIndices
        
        if maxOrder == -1:
            maxOrder = self.collapse - freeIndices
        
        if maxOrder < 0: return None
        
        from copy import deepcopy
        from prime.utils import binomial, factorial
        locals_ = deepcopy(locals)
        locals_["factorial"] = factorial
        locals_["binomial"] = binomial

        result = None

        for K in range(0, maxOrder+1):
            factor = (-1)**K if alternatingSign else 1
            locals_["K"] = K
            factor = factor * eval(combinatorial, locals_)

            if K == 0:
                result = factor * self.coefficientDerivativeTrace(N=N, derivOrder=K, freeIndices=freeIndices)
            else:
                result = result + factor * self.coefficientDerivativeTrace(N=N, derivOrder=K, freeIndices=freeIndices)
        
        return result
            

    """
    Method to quickly calculate the contraction of the 1st derivatives of an output coefficient
    with the spatial derivative of an input coefficient
    """
    def coefficientDerivativeContraction(self, coeff, N=0, derivOrder=0, Aposition=0, freeIndices=0):
        Cshape = self.Cs[N].shape
        Cd = self.diff(expr=self.Cs[N], order=derivOrder+freeIndices)
        coeffd = spatial_diff(expr=coeff, order=derivOrder) if derivOrder == 0 else coeff

        # Calculate the index positions for the contraction
        Caxes = (len(Cshape), ) + tuple(range(len(Cd.shape) - derivOrder, len(Cd.shape)))
        caxes = (Aposition + derivOrder, ) + tuple(range(derivOrder))

        result = np.tensordot(Cd, coeffd, axes=(Caxes, caxes))

        return result


    def sumCoefficientDerivativeContraction(self, coeff, N=0, freeIndices=0, maxOrder=-1, Aposition=0, combinatorial='1', alternatingSign=False, locals={}):
        if N>=2:
            maxOrder = 2 - freeIndices
        
        if maxOrder == -1:
            maxOrder = self.collapse - freeIndices
        
        if maxOrder < 0: return None
        
        from copy import deepcopy
        from prime.utils import binomial, factorial
        locals_ = deepcopy(locals)
        locals_["factorial"] = factorial
        locals_["binomial"] = binomial

        result = None

        for K in range(0, maxOrder+1):
            factor = (-1)**K if alternatingSign else 1
            locals_["K"] = K
            factor = factor * eval(combinatorial, locals_)

            if K == 0:
                result = factor * self.coefficientDerivativeContraction(coeff, N=N, derivOrder=K, freeIndices=freeIndices, Aposition=Aposition)
            else:
                result = result + factor * self.coefficientDerivativeContraction(coeff, N=N, derivOrder=K, freeIndices=freeIndices, Aposition=Aposition)
        
        return result