
import prime
import numpy as np
import sympy as sp

# Setup the phis
phis = prime.phis(16)
n = [1,0,0,0,-1,0,0,-1,0,-1,-1,0,0,-1,0,-1]

Is = np.array(prime.constantSymmetricIntertwiner())

# Setup the fields
gs = prime.Field(1+phis[0], [])
gv = prime.Field([phis[1], phis[2], phis[3]], [+1])
gt = prime.Field([[-1 + phis[4], phis[5] / sp.sqrt(2), phis[6] / sp.sqrt(2)],
                  [phis[5]/sp.sqrt(2), -1 + phis[7], phis[8] / sp.sqrt(2)],
                  [phis[6]/sp.sqrt(2), phis[8] / sp.sqrt(2), -1 + phis[9]]], [+1,+1])
ht = prime.Field([[-1 + phis[10], phis[11] / np.sqrt(2), phis[12] / np.sqrt(2)],
                  [phis[11]/np.sqrt(2), -1 + phis[13], phis[14] / np.sqrt(2)],
                  [phis[12]/np.sqrt(2), phis[14] / np.sqrt(2), -1 + phis[15]]], [+1,+1])

# Setup the parametrization
param = prime.Parametrization(fields=[gs,gv,gt,ht])

# Calculate P^..
@prime.kinematical_coefficient(param, degP=4)
def P(a,b):
    return sp.Rational(1,6) * (1/gs[0] * gt[a,b] + gs[0] * ht[a,b]) - sp.Rational(2,3) * gv[a] * gv[b] / gs[0]**2

# Setup the normal deformation coefficients
@prime.normal_coefficient(param, forField=gs)
def M1(g):
    return -2 * gv[g]

@prime.normal_coefficient(param, forField=gv)
def M2(a,g):
    return -gt[a,g]/2 + gs[0]**2 * ht[a,g]/2 - 2 * gv[a] * gv[g] / gs[0]

@prime.normal_coefficient(param, forField=gt)
def M3(a,b,g):
    return 3 * gv[a] * P.components[b,g] + 3 * gv[b] * P.components[a,g]

@prime.normal_coefficient(param, forField=ht)
def M4(a,b,g):
    return - 1/gs[0]**2 * M3[a,b,g]

# Start the calculation
prime.solve(
    parametrization=param,
    kinematical_coefficient=P,

    # Linear equations of motion
    order=1
)
