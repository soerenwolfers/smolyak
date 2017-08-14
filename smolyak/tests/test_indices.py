'''
Test smolyak.sparse.sparse_index
'''
import unittest
from smolyak.indices import MultiIndex,MultiIndexDict,cartesian_product
from numpy.random import rand, randint
import numpy as np
from smolyak import indices

class TestSparseIndex(unittest.TestCase):
    
    def test1(self):
        A = MultiIndex([1, 0, 0, 1, 2])
        self.assertEqual(A[0], 1)
        A[0] = 0
        self.assertEqual(A[2], 0)
        self.assertEqual(A[0], 0)
        self.assertEqual(A[4], 2)
        A[4] = 3
        A[5] = 4
        A[5] = 0
        self.assertEqual(A.full_tuple(), (0, 0, 0, 1, 3))
        A[4] = 2
        B = MultiIndex([0, 0, 0, 1, 4])
        self.assertFalse(A == B)
        self.assertTrue(A.equal_mod(B, lambda dim: dim == 4))
        self.assertTrue(A != B)
        T = MultiIndex((0, 5, 0))
        self.assertTrue(T == MultiIndex((0, 5, 0)))
        self.assertFalse(T != MultiIndex((0, 5, 0)))
        C = B - A
        self.assertEqual(C.full_tuple(), (0, 0, 0, 0, 2))
        D = MultiIndex(zip([0, 2, 3], [4, 2.4, 4]),sparse=True)
        self.assertEqual(D.sparse_tuple(), ((0, 4), (2, 2.4), (3, 4)))
        self.assertEqual(D.full_tuple(5), (4, 0, 2.4, 4, 0))
        
    def test_retract(self):
        AA = MultiIndex([0, 0, 1, 0])
        AAA = AA.retract(lambda dim: dim * 2)
        self.assertEqual(AAA.full_tuple(), (0, 1))
                
    def test_sparse_index_dict(self):
        B = MultiIndexDict()
        B[MultiIndex((1, 2))] = 1
        B[MultiIndex((1, 2))] += 3
        B[MultiIndex((2, 3))] = 5
        self.assertEqual(B[MultiIndex((1, 2))], 4)
        self.assertEqual([B[i] for i in B], [4, 5])
        with self.assertRaises(KeyError):
            B[MultiIndex((4, 120))]
        self.assertEqual([a.full_tuple() for a in B], [(1, 2), (2, 3)])
        
    def test_rectangle(self):
        self.assertItemsEqual(indices.rectangle(L=[4, 4], c_dim=2), [mi for mi in cartesian_product([range(4)] * 2)])
        self.assertItemsEqual(indices.rectangle(L=2, c_dim=2), [MultiIndex(), MultiIndex((1, 0)), MultiIndex((0, 1)), MultiIndex((1, 1))])
    
    def test_simplex(self):
        self.assertItemsEqual(indices.simplex(L=1, c_dim=2), [MultiIndex(), MultiIndex((1, 0)), MultiIndex((0, 1))])
        
    def test_cartesian_product(self):
        T = [[1, 2], [3, 4], [2]]
        TT = cartesian_product(T)
        self.assertEqual([mi.full_tuple() for mi in TT], [(1, 3, 2), (1, 4, 2), (2, 3, 2), (2, 4, 2)])
        TT = cartesian_product(T, (1, 3, 4))
        self.assertEqual([mi.full_tuple() for mi in TT], [(0, 1, 0, 3, 2), (0, 1, 0, 4, 2), (0, 2, 0, 3, 2), (0, 2, 0, 4, 2)])
    
    def test_hyperbolic_cross(self):
        A = indices.hyperbolic_cross(L=10, c_dim=2) 
        self.assertEqual(len(A), 23)
        
    def test_returnshapes(self):
        cRandomTests = 100
        for __ in range(cRandomTests):
            A = []
            cSets = randint(0, 5)
            cElements = []
            for j in range(cSets):
                cElements.append(randint(0, 5))
                A.append(rand(cElements[j]))
            C = cartesian_product(A)
            self.assertEqual(len(C), np.prod(cElements))
        
if __name__ == "__main__":
    unittest.main()
