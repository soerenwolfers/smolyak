'''
Test smolyak.sparse.combination_rule
'''
import unittest
from smolyak.sparse.combination_rule import combination_rule
from smolyak.sparse.sparse_index import SparseIndex, \
    get_admissible_indices, cartesian_product
import pstats
import cProfile

class TestCombinationRule(unittest.TestCase):

    def setUp(self):
        self.pr = cProfile.Profile()
        self.pr.enable()
        print "\n<<<---"
        
    def tearDown(self):
        p = pstats.Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('cumulative').print_stats(20)
        print "\n--->>>"
        
    def test_simplices(self):
        d = 2
        L = 5
        def admissible(mi):
            T = [value for __, value in mi]
            return sum(T) <= L 
        mis = get_admissible_indices(admissible, dim=d)
        CR = combination_rule(mis)
        for mi in CR:
            self.assertIn(sum([value for __, value in mi]), [L, L - 1])
            self.assertEqual(CR[mi], (-1) ** (sum([value for __, value in mi]) - L))
   
    def test_rectangles(self):
        d = 5
        L = 5
        sparseindices = cartesian_product([range(L)] * d)
        CR = combination_rule(sparseindices)
        self.assertEqual([mi for mi in CR], [SparseIndex((L - 1,) * d)])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_combination_rule']
    unittest.main()
