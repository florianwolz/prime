![logo](https://github.com/florianwolz/prime/raw/master/docs/images/header.png "Prime")

[![Build Status](https://travis-ci.org/florianwolz/prime.png?branch=master)](https://travis-ci.org/florianwolz/prime)

Prime is a python package that allows to derive the weak field Lagrangians of any
geometry by the gravitational closure mechanism.

# Overview

 - [ ] Easy to use setup for any input geometry
 - [ ] Complete automation from the geometry to the output
 - [ ] Modern technologies: Python 3 + Kubernetes for distributed computing

## Installation

```sh
$ git clone https://github.com/florianwolz/prime
$ cd prime
$ pip install -f requirements.txt
```

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
