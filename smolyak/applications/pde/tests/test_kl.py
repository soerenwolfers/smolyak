'''
Test smolyak.pde.elliptic
'''
import unittest
from smolyak.applications.pde.kl import kl
from smolyak.aux.plots import plot_convergence
import numpy as np
import matplotlib.pyplot as plt
import timeit

class TestKL(unittest.TestCase):
      
    def kl(self):
        runtimes = []
        grids = [2 ** i for i in range(6, 11)]
        V = []
        for N in grids:
            tic = timeit.default_timer()    
            v = kl(np.array([[1, 2], [4, 5]]), N)
            self.assertEqual(v.shape[0], 2)
            V.append(v)
            runtimes.append(timeit.default_timer() - tic)    
            print(runtimes[-1])
        orderRuntime = plot_convergence(runtimes[1:], grids[1:], expect_limit=0, expect_order='fit')
        orderConvergence = plot_convergence(runtimes[2:], V[2:], expect_order='fit')
        # plt.show()
        self.assertAlmostEqual(orderRuntime / 0.5, 1, delta=0.25)
        self.assertAlmostEqual(orderConvergence / -1, 1, delta=0.25)

    
    def test(self):
#         import cProfile
#         cProfile.run('from smolyak.pde.kl import kl;import numpy as np;from smolyak.pde.kl import KLSolver;'
#         'klsolver=KLSolver(order=1);'
#         'klsolver(np.array([[1,2],[2,3],[4,3],[3,2],[3,2],[3,2]]).reshape(-1,2), 50)', 'restats')
#         import pstats
#         p = pstats.Stats('restats') 
#         p.sort_stats('cumulative').print_stats(20)
        self.kl()
        self.kl4()
        
    def kl4(self):
        runtimes = []
        grids = [2 ** i for i in range(6, 11)]
        V = []
        for N in grids:
            tic = timeit.default_timer()    
            v = kl(np.array([[1, 2, 0, -1], [4, 5, -2, 3]]), N)
            self.assertEqual(v.shape[0], 2)
            V.append(v)
            runtimes.append(timeit.default_timer() - tic)    
            print(runtimes[-1])
        orderRuntime = plot_convergence(runtimes[1:], grids[1:], expect_limit=0, expect_order='fit')
        orderConvergence = plot_convergence(runtimes[2:], V[2:], expect_order='fit')
        print(orderRuntime, orderConvergence)
        plt.show()
        self.assertAlmostEqual(orderRuntime / 0.5, 1, delta=0.25)
        self.assertAlmostEqual(orderConvergence / -1, 1, delta=0.25)   

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
