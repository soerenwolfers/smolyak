'''
Test smolyak.sparse.dc_set
'''
import unittest
from smolyak.indices import MultiIndex, DCSet
from smolyak.misc import plots

class TestDCSet(unittest.TestCase):

    def test1(self):
        A = DCSet()
        self.assertSetEqual(A.add(MultiIndex([0])), set())
        New = MultiIndex([0, 1, 0, 0])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in A.add(New)], [((1, 2),)])
        B = DCSet()
        self.assertItemsEqual([candidate.sparse_tuple() for  candidate in B.add(MultiIndex([]))], [])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add(MultiIndex([0, 1, 0]))], [((1, 2),)])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add(MultiIndex([0, 0, 1]))], [((1, 1), (2, 1)), ((2, 2),)])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add(MultiIndex([0, 2, 0]))], [((1, 3),)])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add(MultiIndex([1, 0, 0]))], [((0, 1), (2, 1)), ((0, 2),), ((0, 1), (1, 1))])
        self.assertItemsEqual([candidate.sparse_tuple() for candidate in B.add(MultiIndex([1, 1, 0]))], [((0, 1), (1, 2))])
        plots.plot_indices(DCSet())

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
