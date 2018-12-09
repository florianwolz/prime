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


def generateTwoPartition(indices):
    gammas = []
    remainings = []

    first = indices[0]
    for i, e in enumerate(indices[1:]):
        gammas.append([first, e])
        remainings.append([x for j, x in enumerate(indices[1:]) if i != j])

    return gammas, remainings


"""
Generates possible index assignments to terms only containing gammas

Args:
    indices         List with the indices
"""
def generateEvenRank(indices):
    gammas, remainings = generateTwoPartition(indices)

    if len(indices) == 2:
        return gammas

    result = []
    for gamma, remaining in zip(gammas, remainings):
        newComb = generateEvenRank(remaining)
        for v in newComb:
            result.append(gamma + v)
    return result


# TODO: Odd rank
def generateOddRank(indices):
    raise Expception("Unimplemented.")


def bringListIntoOrder(l, elements):
    # If elements is a list of tuples(!)
    if type(elements) is list:
        res = l
        for e in elements:
            res = bringListIntoOrder(res, e)
        return res

    # Else
    result = []
    i=0
    for x in l:
        if x in elements:
            result.append(elements[i])
            i=i+1
        else:
            result.append(x)
    return result


class Index:
    def __init__(self, indices, symmetries=None):
        self.indices = list(indices)
        self.symmetries = symmetries or []

    """
    Canonicalizes the index assignment

    Canonicalize the index assignment by ordering the indices on epsilon (if present),
    on each gamma term and then ordering the gamma blocks by first indices.
    """
    def canonicalize(self):
        num = len(self.indices)

        if num == 0: return 0
        if num == 1: return 0

        # Gamma-gamma terms
        if num % 2 == 0:
            splits = [list(sorted([self.indices[i], self.indices[i+1]])) for i in range(0,num,2)]
            splits = sorted(splits, key=lambda y : y[0])
            indices = [j for i in splits for j in i]
            return Index(indices=indices, symmetries=self.symmetries)
        # Epsilon-gamma terms
        else:
            a = list(sorted(self.indices[0:3]))
            b = Index(indices=self.indices[3:]).canonicalize()
            indices = a + b.indices
            return Index(indices=indices, symmetries=self.symmetries)

    """
    Exchange indices to bring the given indices into the desired order.

    Args:
        indices         The indices.
    """
    def bringIntoOrder(self, indices):
        # If elements is a list of tuples(!)
        if type(indices) is list:
            result = self
            for e in indices: result = result.bringIntoOrder(e)
            return result

        # Else
        result = []
        i=0
        for x in self.indices:
            if x in indices:
                result.append(indices[i])
                i=i+1
            else:
                result.append(x)

        return Index(result, self.symmetries)

    def rank(self):
        return len(self.indices)

    def __str__(self):
        return "".join([chr(ord('a') + i) for i in self.indices])

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)


class Indices:
    def __init__(self, rank):
        self.rank = rank
        self.indices = [Index(idx) for idx in generateEvenRank(list(range(rank)))]

    def symmetrize(self, symmetries, kind="symmetric"):
        # Antisymmetrization is not so well, tested so throw for now
        if kind == "antisymmetric":
            raise Exception("Antisymmetrization support is still shaky.")

        if symmetries is None:
            return indices

        if type(symmetries) is not list:
            symmetries = [symmetries]

        # Type check
        for sym in symmetries:
            if type(sym) is not tuple:
                raise Exception("Expected a tuple with the indices. Instead got {}".format(sym))

        # Prepare result
        from itertools import permutations, product

        for e in self.indices:
            # Generate all the possible index permutations of the symmetries
            perms = list(product(*[list(permutations(tuple(sym))) for sym in symmetries]))
            perms = [list(e) for e in perms]
            ordered = [e.bringIntoOrder(p) for p in perms]
            for x in ordered:
                # Canonicalize the terms
                y = x.canonicalize()

                # If the canonicalized entry is again the original one,
                # don't delete unless we antisymmetrized since this means that
                # T = -T.
                if str(y)==str(e) and kind == "symmetric":
                    continue

                try:
                    id = self.indices.index(y)
                except:
                    continue
                del self.indices[id]

        # Update the symmetry tags
        for e in self.indices:
            e.symmetries = e.symmetries + symmetries

    def exchangeSymmetrize(self, symmetries):
        # PROBLEM: Need proper "basis" since the algorithm does not
        #   detect terms that, when employing the symmetry freedom to move
        #   indices around, are the same, i.e. are in the same orbits.

        if symmetries is None:
            return indices

        if type(symmetries) is list:
            result = self
            for sym in symmetries: result = result.exchangeSymmetrize(sym)
            return result

        if not type(symmetries) is tuple or len(symmetries) != 2:
            raise Exception("The symmetry must be a tuple with two elements")

        originalIdx, newIdx = symmetries

        if len(originalIdx) != len(newIdx):
            raise Exception("The index assignments need to have the same length. Given {} and {}".format(originalIdx, newIdx))

        replacement = { a : b for a,b in zip(originalIdx, newIdx) }

        xs = []
        for e in self.indices:
            # Replace
            f = Index([ replacement[x] for x in e.indices ], e.symmetries)

            # Canonicalize f
            f = f.canonicalize()

            # Before we canonicalize, use the symmetries to bring into standard form
            #f = f.bringIntoOrder(f.symmetries)

            x = tuple(sorted([e,f], key=lambda x : x.indices)) if e != f else f
            xs.append(x)

            # canonicalized item is the same
            #if f == e: continue

            #try:
            #    id = self.indices.index(f)
            #except:
            #    continue
            #del self.indices[id]

        ys = []
        for x in xs:
            if x not in ys: ys.append(x)
        for i,y in enumerate(xs,1):
            print("{}: {}".format(i,y))
        print(len(ys))


"""
Symmetrize the index assignments in some indices

Symmetrization by bringing each element into canonical order under the symmetries.
Then all the duplicates are deleted.

Remark:
The algorithm does not recognize if

Args:
    indices         The list of all the
"""
def symmetrize(indices, symmetries=None, kind="symmetric"):
    # Antisymmetrization is not so well, tested so throw for now
    if kind == "antisymmetric":
        raise Exception("Antisymmetrization support is still shaky.")

    if symmetries is None:
        return indices

    if type(symmetries) is not list:
        symmetries = [symmetries]

    # Type check
    for sym in symmetries:
        if type(sym) is not tuple:
            raise Exception("Expected a tuple with the indices. Instead got {}".format(sym))

    # Prepare result
    from copy import deepcopy
    result = deepcopy(indices)
    from itertools import permutations, product

    for e in result:
        # Generate all the possible index permutations of the symmetries
        perms = list(product(*[list(permutations(tuple(sym))) for sym in symmetries]))
        perms = [list(e) for e in perms]
        ordered = [bringListIntoOrder(e, p) for p in perms]
        for x in ordered:
            # Canonicalize the terms
            y = canonicalize(x)

            # If the canonicalized entry is again the original one,
            # don't delete unless we antisymmetrized since this means that
            # T = -T.
            if y==e and kind == "symmetric":
                continue

            try:
                id = result.index(y)
            except:
                continue
            del result[id]

    return result


def antisymmetrize(indices, symmetries=None):
    return symmetrize(indices, symmetries, kind="antisymmetric")


"""
Implement block symmetries

DO NOT USE, NOT WORKING YET

Args:
    indices         The list of all the index assignments
    symmetries      A tuple where the first element is the original
                    index order and the second element the replacement.
"""
def blocksymmetrize(indices, symmetries=None):
    if symmetries is None:
        return indices

    if type(symmetries) is list:
        result = indices
        for sym in symmetries: result = blocksymmetrize(indices, sym)
        return result

    if not type(symmetries) is tuple or len(symmetries) != 2:
        raise Exception("The symmetry must be a tuple with two elements")

    originalIdx, newIdx = symmetries

    if len(originalIdx) != len(newIdx):
        raise Exception("The index assignments need to have the same length. Given {} and {}".format(originalIdx, newIdx))

    replacement = { a : b for a,b in zip(originalIdx, newIdx) }

    from copy import deepcopy
    result = deepcopy(indices)

    for e in result:
        # Replace
        f = [ replacement[x] for x in e ]

        # Canonicalize f
        f = canonicalize(f)

        # canonicalized item is the same
        if f == e: continue

        try:
            id = result.index(f)
        except:
            continue
        del result[id]

    return result


def canonicalize(indices):
    num = len(indices)

    if num == 0: return 0
    if num == 1: return 0

    # Gamma-gamma terms
    if num % 2 == 0:
        splits = [list(sorted([indices[i], indices[i+1]])) for i in range(0,len(indices),2)]
        splits = sorted(splits, key=lambda y : y[0])
        return [j for i in splits for j in i]
    # Epsilon-gamma terms
    else:
        a = list(sorted(indices[0:3]))
        b = canonicalize(indices[3:])
        return a + b


def indexToString(index):
    return "".join([chr(ord('a') + x) for x in index])
