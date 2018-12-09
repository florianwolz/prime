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

gamma = np.array([[1,0,0],[0,1,0],[0,0,1]])
epsilon = np.array([
    [[0,0,0],[0,0,1],[0,-1,0]],
    [[0,0,-1],[0,0,0],[1,0,0]],
    [[0,1,0],[-1,0,0],[0,0,0]]
])

def generateEvenRank(indices):
    t = gamma
    for i in range(int(len(indices)/2)-1):
        t = np.tensordot(t, gamma, axes=0)
    print(t.shape)
    return t.transpose(tuple(indices))

def generate(indices):
    if len(indices) == 0:
        return np.array(0)

    if len(indices) % 2 == 0:
        return generateEvenRank(indices)
    else:
        raise Exception("Not implemented.")
