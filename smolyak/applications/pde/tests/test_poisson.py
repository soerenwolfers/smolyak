'''
Test smolyak.pde.poisson
'''
import unittest
from smolyak.applications.pde.poisson import poisson_kink
from smolyak.misc.plots import plot_convergence
import numpy as np
import matplotlib.pyplot as plt
import timeit

class TestPoisson(unittest.TestCase):
        
    def foo(self, degree):
        runtimes = []
        grids = [64 * 2 ** (1.15 / 2 * i) for i in range(0, 3 if degree == 3 else 6)]
        V = []
        for N in grids:
            tic = timeit.default_timer()    
            v = poisson_kink(np.array([[1, 2], [4, 5]]), N, order=degree)
            self.assertEqual(v.shape[0], 2)
            V.append(v)
            runtimes.append(timeit.default_timer() - tic)  
            print(runtimes[-1])  
        orderRuntime = plot_convergence(runtimes, grids, expect_limit=0, expect_order='fit')
        orderConvergence = plot_convergence(runtimes, V, expect_order='fit')
        orderConvergence2 = plot_convergence(grids, V, expect_order='fit')
        plt.show()
        print('Fit', orderRuntime)
        print('Fit', orderConvergence)
        print('Fit', orderConvergence2)
        self.assertAlmostEqual(orderRuntime / 0.5, 1, delta=0.3)
        self.assertAlmostEqual(orderConvergence / -degree, 1, delta=0.3)
        
    def test_poisson_piecewise1(self):
        self.foo(1)
        
    def test_poisson_piecewise2(self):
        self.foo(2)
        
    def test_poisson_piecewise3(self):
        self.foo(3)
        
    def test_timing(self):
        import cProfile
        cProfile.run('from smolyak.applications.pde.poisson import poisson_kink;import numpy as np;'
         'poisson_kink(np.array([[1,2],[2,3],[4,3],[3,2],[3,2],[3,2]]).reshape(-1,2), 64)', 'restats')
        import pstats
        p = pstats.Stats('restats') 
        p.sort_stats('cumulative').print_stats(20)
               
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
