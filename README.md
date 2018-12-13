![logo](https://github.com/florianwolz/prime/raw/master/docs/images/header.png "Prime")

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Build Status](https://travis-ci.com/florianwolz/prime.svg?branch=master)](https://travis-ci.com/florianwolz/prime)
[![Coverage Status](https://coveralls.io/repos/github/florianwolz/prime/badge.svg?branch=master)](https://coveralls.io/github/florianwolz/prime?branch=master)

Prime is a python package that allows to derive the weak field Lagrangians of any
geometry by the gravitational closure mechanism.

# Overview

 - [ ] Easy to use setup for any input geometry
 - [ ] Complete automation from the geometry to the output
 - [ ] Modern technologies: Python 3 + Kubernetes for distributed computing

 **Note**: Prime is in VERY early development stage. Its API and design is still fluctuating quantum style and its results are as trustable as from string theory.

### Current Status

 - [x] Reading of the input files
 - [x] Calculation of the remaining input coefficients (E, F)
 - [x] Perturbative expansion of the input coefficients
 - [x] Generation of a list of all perturbative output coefficients
 - [ ] Generation of the basis terms of the perturbative output coefficients
 - [ ] Collecting all perturbative output coefficients into the polynomial in the degrees of freedom
 - [ ] Writing classes for all closure equations
 - [ ] Plugging all coefficients into the closure equations
 - [ ] Solving them
 - [ ] Fancy output of the whole Lagrangian

## Installation

```sh
$ git clone https://github.com/florianwolz/prime
$ cd prime
$ pip install -r requirements.txt
```

## Input scripts

Prime is configured with the help of a Python input script. Before explaining
the several steps, let's start with an example:

```python
import prime
from sympy import sqrt

# Setup the list of the six degrees of freedom
phis = prime.phis(6)

# Setup the (pulled-back) metric
g = prime.Field([[-1 + phis[0], phis[1], phis[2]],
                 [phis[1], -1 + phis[3], phis[4]],
                 [phis[2], phis[4], -1 + phis[5]]], [1, 1])

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
```

Executing this script will give the perturbative expansion of the Einstein-Hilbert
action to second order.

# Advanced topics

## Gravitational Closure

Gravitational closure is a theoretical mechanism that allows to calculate the Lagrangian
of a geometry by solving a system of linear homogeneous partial differential equations.
These equations encode the requirement that matter theories that is coupled to this geometry
can be formulated on common initial data surfaces -- i.e. allow a consistent predictive
physical theory.

![closure](https://github.com/florianwolz/prime/raw/master/docs/images/closure.png "Gravitational closure")

Solving this system is practically quite complicated -- and fortunately practically often not even required.
Instead one can already use a perturbative expansion around some constant background. This turns the
system of partial differential equations into a huge linear algebra problem.

Prime tackles this problem by completely automating the process between the input data and the
final Lagrangian.

# License

Prime is released under the Apache 2.0 licence. See [LICENSE](https://github.com/crazyphysicist/cobalt/blob/master/LICENSE.txt)

Crafted with :heart: and lots of :coffee: as part of my PhD thesis.
