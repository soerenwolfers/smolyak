'''
Test smolyak.sparse.dc_set
'''
import unittest
from smolyak.sparse.dc_set import DCSet
from smolyak.sparse.sparse_index import SparseIndex

class TestDCSet(unittest.TestCase):

    def test1(self):
        A = DCSet()
        self.assertSetEqual(A.add_mi(SparseIndex([0])), set())
        New = SparseIndex([0, 1, 0, 0])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in A.add_mi(New)], [((1, 2),)])
        B = DCSet()
        self.assertItemsEqual([candidate.sparse_tuple() for  candidate in B.add_mi(SparseIndex([]))], [])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add_mi(SparseIndex([0, 1, 0]))], [((1, 2),)])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add_mi(SparseIndex([0, 0, 1]))], [((1, 1), (2, 1)), ((2, 2),)])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add_mi(SparseIndex([0, 2, 0]))], [((1, 3),)])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add_mi(SparseIndex([1, 0, 0]))], [((0, 1), (2, 1)), ((0, 2),), ((0, 1), (1, 1))])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add_mi(SparseIndex([1, 1, 0]))], [((0, 1), (1, 2))])
        B.plot()

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
