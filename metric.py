import prime
import sympy as sp

# Setup the phis
phis = prime.phis(6)

# Setup the fields
g = prime.Field([[-1 + phis[0], phis[1] / sp.sqrt(2), phis[2] / sp.sqrt(2)],
                 [phis[1]/sp.sqrt(2), -1 + phis[3], phis[4] / sp.sqrt(2)],
                 [phis[2]/sp.sqrt(2), phis[4] / sp.sqrt(2), -1 + phis[5]]], [+1,+1])

#g = prime.Field([[-1 + phis[0], phis[1], phis[2]],
#                 [phis[1], -1 + phis[3], phis[4]],
#                 [phis[2], phis[4], -1 + phis[5]]], [1, 1])

# Setup the parametrization
param = prime.Parametrization(fields=[g])

# Setup the kinematical coefficient
P = prime.Kinematical(param, components=g.components, degP=2)

# Solve
prime.solve(
    parametrization=param,
    kinematical_coefficient=P,

    # Linear equations of motion
    order=1
)
