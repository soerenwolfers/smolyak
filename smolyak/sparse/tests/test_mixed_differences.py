'''
Test smolyak.sparse.mixed_differences
'''
import unittest
from smolyak.sparse.mixed_differences import MixedDifferences
from smolyak.sparse.sparse_index import SparseIndex
from smolyak.sparse.sparse_approximator import SparseApproximator
            
class TestMixedDifferences(unittest.TestCase):

    def test1(self):
        f = lambda mi: 3 + 2.**(-mi[0]) * 2.**(-mi[1])
        md = MixedDifferences(f)
        s = 0
        for j in range(4):
            for i in range(4):
                s += md(SparseIndex((i, j)))
        self.assertAlmostEqual(s, 3, places=1)
        SA = SparseApproximator(decomposition=md, init_dims=2, is_md=True)
        SA.expand_adaptive(c_steps=100)
        self.assertAlmostEqual(SA.get_approximation(), 3, places=2)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
