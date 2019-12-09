#   Copyright 2019 The Prime Authors
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

from prime.equations.equation import ScalarEquation

# Import the scalar equations
from prime.equations.C1 import C1
from prime.equations.C2 import C2
from prime.equations.C3 import C3
from prime.equations.C4 import C4
from prime.equations.C5 import C5
from prime.equations.C6 import C6
from prime.equations.C7 import C7

# Import the sequence equations
from prime.equations.C8 import C8
from prime.equations.C9 import C9
from prime.equations.C10 import C10
from prime.equations.C11 import C11
from prime.equations.C12 import C12
from prime.equations.C13 import C13
from prime.equations.C14 import C14
from prime.equations.C15 import C15
from prime.equations.C16 import C16
from prime.equations.C17 import C17
from prime.equations.C18 import C18
from prime.equations.C19 import C19
from prime.equations.C20 import C20
from prime.equations.C21 import C21


class TestEquation(ScalarEquation):
    shape = tuple()
    componentWise = False

    def __init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs):
        # Initialize the scalar equation
        ScalarEquation.__init__(self, parametrization, Cs, E, F, M, p, degP, *args, **kwargs)
    
    def allComponents(self):
        result = self.Cs[0]
        return result


# Load all the equations into a list
equations = [
    C1, C2, C3, C4, C5, C6, C7,
    C10, C11, C12, C13, C15, C16, C17, C18, C19, C20, C21

    # Missing: C8, C9, C14
] 

def allEqns(parametrization, Cs, E, F, M, p, degP, order):
    return [eq(parametrization, Cs=Cs, E=E, F=F, M=M, p=p, degP=degP, order=order) for eq in equations]