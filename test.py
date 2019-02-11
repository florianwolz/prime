#from prime.output import Index, Indices
#import time

#idx = Indices(8)
#print(len(idx.indices))
#idx.symmetrize([(0,1),(2,3),(4,5),(6,7)])
#print(len(idx.indices))

#for id in idx.indices:
#    print(id)
#print()

#idx.exchangeSymmetrize([((0,1,2,3,4,5,6,7),(4,5,6,7,0,1,2,3))])

#for id in idx.indices:
#    print(id)
#print()

from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, bsgs_direct_product
