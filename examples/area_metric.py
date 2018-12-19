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

import prime
import sympy

# Set the order to which we want to evaluate
ORDER_EOM = 1

# Setup the degrees of freedom
phis = prime.phis(17)

# Start with the first and second field
gb   = prime.Field([
            [1 + phis[0], phis[1] / prime.sqrt(2), phis[2] / prime.sqrt(2)],
            [phis[1] / prime.sqrt(2), 1 + phis[3], phis[4] / prime.sqrt(2)],
            [phis[2] / prime.sqrt(2), phis[4] / prime.sqrt(2), 1 + phis[5]],
        ], [+1, +1])

gbb  = prime.Field([
            [1 + phis[6], phis[7] / prime.sqrt(2), phis[8] / prime.sqrt(2)],
            [phis[7] / prime.sqrt(2), 1 + phis[9], phis[10] / prime.sqrt(2)],
            [phis[8] / prime.sqrt(2), phis[10] / prime.sqrt(2), 1 + phis[11]],
        ], [-1, -1])

# Some helpers to construct the strange parametrization of the third field
Is  = prime.constantSymmetricIntertwiner()
Ist = prime.constantSymmetricTracelessIntertwiner()

@prime.to_tensor(shape=(3,3))
def p1(a,b):
    return sum([Is[A,a,b]*phis[A] for A in range(0,6)])

@prime.to_tensor(shape=(3,3))
def p3(a,b):
    return sum([Ist[A,a,b]*phis[A+12] for A in range(0,5)])

def anticomm(a,b):
    return sum([p1[a,m] * p3[m,b] - p1[m,b] * p3[a,m] for m in range(0,3)])

def brackets(a,b, order):
    if order==0:
        return anticomm(a,b)
    field = prime.to_tensor(shape=(3,3))(lambda x,y : brackets(x,y,order-1))
    return sum([p1[a,m] * field[m,b] + p1[m,b] * field[a,m] for m in range(0,3)])

def f(a,b):
    return sum([(-1)**N * sympy.Rational(1, 2**(N+1)) * brackets(a,b,N) for N in range(0, ORDER_EOM+1)])

# Now the endomorphism
@prime.field(indexPositions=((3,+1), (3,-1)))
def gbbb(a,b):
    return sum([Ist[M,a,b] * phis[M + 12] for M in range(0,5)]) + f(a,b)

# Setup the parametrization
param = prime.Parametrization([gb,gbb,gbbb])

# Some utility for the remaining coefficients
detgb = prime.sqrt(prime.det(gb)) 
epsilon = prime.epsilon

# Start with the kinematical coefficient P^..
@prime.kinematical_coefficient(param, degP=4)
def P(a,b):
    return sympy.Rational(1,6) * (sum([
        gb[a,m] * gb[b,n] * gbb[m,n] - gb[a,b] * gb[m,n] * gbb[m,n]
        -2 * gb[a,b] * gbbb[m,n] * gbbb[n,m] + 3 * gb[m,n] * gbbb[a,m] * gbbb[b,n]
        for m in range(0,3) for n in range(0,3)]))

# Next the normal deformation coefficient M
@prime.normal_coefficient(param, forField=gb)
def Mb(a,b,c):
    return detgb * sum([epsilon[s,c,a] * gbbb[b,s] + epsilon[s,c,b] * gbbb[a,s] for s in range(0,3)])

@prime.normal_coefficient(param, forField=gbb)
def Mbb(a,b,c):
    return 3 / detgb * sum([
        epsilon[s,t,a] * gbbb[t,b] * P[s,c] + epsilon[s,t,b] * gbbb[t,a] * P[s,c]
        for s in range(0,3) for t in range(0,3)])

@prime.normal_coefficient(param, forField=gbbb)
def Mbbb(a,b,c):
    return - detgb * sum([epsilon[c,a,s] * gbb[s,b] for s in range(0,3)]) + \
           3 / detgb * sum([epsilon[b,s,t] * gb[a,s] * P[t,c] for s in range(0,3) for t in range(0,3)])

# Start the calculation
prime.solve(
    parametrization=param,
    kinematical_coefficient=P,

    # Linear equations of motion
    order=ORDER_EOM
)