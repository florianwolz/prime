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


class Status:
    WAITING = 1
    PREPARING = 2
    CALCULATING = 3
    FINISHED = 4


class Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state

class Reporter(Borg):
    def __init__(self, order=1, silent=False):
        Borg.__init__(self)

        # Add status for the input coefficients
        self.input_coefficients = {
            "E": Status.WAITING,
            "F": Status.WAITING,
            "M": Status.WAITING,
            "p": Status.WAITING
        }

        # Depending on the order setup "C", "C_A", "C_AB", ...
        self.coefficients = {
            "C_{}".format("".join([chr(ord('A')+j) for j in range(i)])) if i > 0 else "C": Status.WAITING
            for i in range(order+2)
        }

        # Closure equations
        self.equations = {
            "C{}".format(i): Status.WAITING
            for i in range(1,22)
        }

        self.silent = silent

    
    def statusToString(self, status):
        if status == Status.WAITING: return "Waiting"
        elif status == Status.PREPARING: return "Preparing"
        elif status == Status.CALCULATING: return "Calculating"
        elif status == Status.FINISHED: return "Finished"
        else: return "Unknown"
        
    
    
    def update(self, key, value):
        if key in self.input_coefficients:
            self.input_coefficients[key] = value
        elif key in self.coefficients:
            self.coefficients[key] = value
        elif key in self.equations:
            self.equations[key] = value
        
        if not self.silent: print(str(self))


    def __str__(self):
        result = "<Reporter [\n"
        for key, value in self.input_coefficients.items():
            result += "    {} ({})\n".format(key, self.statusToString(value))
        for key, value in self.coefficients.items():
            result += "    {} ({})\n".format(key, self.statusToString(value))
        for key, value in self.equations.items():
            result += "    {} ({})\n".format(key, self.statusToString(value))
        result += ">"
        return result
    

    def __repr__(self):
        return str(self)