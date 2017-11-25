'''
Test smolyak.smolyak
'''
import unittest
from smolyak import SparseApproximation
import numpy as np
from smolyak import Decomposition
from swutil.time import snooze
from smolyak.indices import MultiIndex
from math import inf
from swutil.decorators import print_runtime

class TestSparseApproximation(unittest.TestCase):
    
    def test_dimensionadaptivity(self):
        ndim = 5
        def func(i):
            ii = [v for __, v in i.sparse_tuple()]
            value = np.exp(-sum(ii))
            return value + snooze(2 ** sum(ii))
        decomp = Decomposition(func=func, n=inf,next_dims=lambda dim: [dim+1] if dim<ndim -1 else [])
        SA = SparseApproximation(decomp)
        print_runtime(SA.expand_adaptive)(N=30)
        #print(SA.get_indices())
        self.assertGreaterEqual(max(max(mi.active_dims()) if mi!=MultiIndex() else 0 for mi in SA.get_indices()), 1)
        self.assertAlmostEqual(SA._get_contribution_exponent(0), 1, delta=0.1)
        
    def test_bundled(self):
        bundled_dims = [0, 1]
        def func(mis):
            value = sum([np.exp(-3 * mi[0]) * np.exp(-mi[1]) * np.exp(-mi[2]) for mi in mis])
            return value + snooze(sum([2 ** mi[0] * 2 ** mi[1] * 2 ** mi[2] for mi in mis]))
        decomp = Decomposition(func=func,n=3,bundled=True,bundled_dims=bundled_dims, work_multipliers=[2,2,2], contribution_multipliers=np.exp([-3,-1,-1]))
        SA = SparseApproximation(decomp)
        print_runtime(SA.expand_apriori)(L=15)
        T = SA.get_approximation()
        self.assertAlmostEqual(T, 1 / (1 - np.exp(-3)) * 1 / (1 - np.exp(-1)) * 1 / (1 - np.exp(-1)), places=2, msg='Limit incorrect')

    def test_notbundled(self):
        def func(i):
            value = 3.**(-i[0]) * 5.**(-2 * i[1]) * 7.**(-3 * i[2])
            return value + snooze(2 ** i[0] * 2 ** i[1] * 2 ** i[2])
        contribution_factor = lambda i: 3.**(-float(i[0])) * 5.**(-2 * float(i[1])) * 7.**(-3 * float(i[2]))
        decomp=Decomposition(func=func,n=3,contribution_function=contribution_factor)
        SA = SparseApproximation(decomp)
        print_runtime(SA.expand_adaptive)(N=75, T=2)
        self.assertAlmostEqual(SA.get_approximation(), 3. / 2 * 25. / 24 * 343. / 342, places=2)
        self.assertAlmostEqual(SA.get_runtime_multiplier(0), 2, delta=0.2)
        self.assertAlmostEqual(SA.get_runtime_multiplier(2), 2, delta=0.2)
        
    def test_notbundled2(self):
        def func(i):
            value = 3.**(-i[0]) * 5.**(-2 * i[1]) * 7.**(-3 * i[2])
            return value + snooze(2 ** i[0] * 2 ** i[1] * 2 ** i[2])
        decomp = Decomposition(func=func,n=3)
        SA = SparseApproximation(decomp)
        print_runtime(SA.expand_continuation)(L=9)
        self.assertAlmostEqual(SA.get_approximation(), 3. / 2 * 25. / 24 * 343. / 342, places=3)
        self.assertAlmostEqual(SA.get_runtime_multiplier(0), 2, delta=0.2)
        self.assertAlmostEqual(SA._get_runtime_exponent(2), np.log(2), delta=0.2)
        self.assertAlmostEqual(SA.get_contribution_multiplier(0), 1/3, delta=0.05)

if __name__ == "__main__":
    #suite = unittest.TestSuite()
    #suite.addTest(TestSparseApproximation('test_bundled'))
    #unittest.TextTestRunner().run(suite)
    unittest.main()
    
