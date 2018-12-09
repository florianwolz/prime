import unittest

from prime.output import generateEvenRank, symmetrize, indexToString

class Symmetrize(unittest.TestCase):
    twos   = ['ab']
    fours  = ['abcd', 'acbd']
    sixs   = ['abcdef', 'abcedf', 'acbdef', 'acbedf', 'aebfcd']
    eights = ['abcdefgh', 'abcdegfh', 'abcedfgh', 'abcedgfh', 'abcgdhef',
              'acbdefgh', 'acbdegfh', 'acbedfgh', 'acbedgfh', 'acbgdefh',
              'acbgdhef', 'aebfcdgh', 'aebfcgdh', 'aebgcdfh', 'aebgcfdh',
              'agbhcdef', 'agbhcedf']

    def test_two(self):
        idx = generateEvenRank(list(range(2)))
        symmetrized = [indexToString(a) for a in symmetrize(idx, [(i, i+1) for i in range(0,2,2)])]
        self.assertEqual(symmetrized, Symmetrize.twos)

    def test_fours(self):
        idx = generateEvenRank(list(range(4)))
        symmetrized = [indexToString(a) for a in symmetrize(idx, [(i, i+1) for i in range(0,4,2)])]
        self.assertEqual(symmetrized, Symmetrize.fours)

    def test_sixs(self):
        idx = generateEvenRank(list(range(6)))
        symmetrized = [indexToString(a) for a in symmetrize(idx, [(i, i+1) for i in range(0,6,2)])]
        self.assertEqual(symmetrized, Symmetrize.sixs)

    def test_eights(self):
        idx = generateEvenRank(list(range(8)))
        symmetrized = [indexToString(a) for a in symmetrize(idx, [(i, i+1) for i in range(0,8,2)])]
        self.assertEqual(symmetrized, Symmetrize.eights)

if __name__ == "__main__":
    unittest.main()
