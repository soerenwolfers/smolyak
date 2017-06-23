'''
Test smolyak.sparse.mixed_differences
'''
import unittest
from smolyak.indices import MixedDifferences
from smolyak.indices import MultiIndex
from smolyak.approximator import Approximator
from smolyak.decomposition import Decomposition
            
class TestMixedDifferences(unittest.TestCase):

    def test1(self):
        f = lambda mi: 3 + 2.**(-mi[0]) * 2.**(-mi[1])
        md = MixedDifferences(f)
        s = 0
        for j in range(4):
            for i in range(4):
                s += md(MultiIndex((i, j)))
        self.assertAlmostEqual(s, 3, places=1)
        problem = Decomposition(decomposition=md,n=3,is_md=True)
        SA = Approximator(problem)
        SA.expand_adaptive(c_steps=100)
        self.assertAlmostEqual(SA.get_approximation(), 3, places=2)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
