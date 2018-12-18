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

from prime.input import Field, field, Parametrization
from prime.input import Parametrization
from prime.input import Kinematical, kinematical_coefficient
from prime.input import NormalCoefficient, normal_coefficient

from prime.utils import phis, dirt, dropHigherOrder, constantSymmetricIntertwiner, to_tensor
from prime.utils import gamma, epsilon, symmetrize

from prime.solve import solve

import prime.cli
import prime.output

from prime.cli import main
