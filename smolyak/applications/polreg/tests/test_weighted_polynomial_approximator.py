'''
Test smolyak.polreg.weighted_polynomial_approximator
'''
import unittest
import numpy as np
from smolyak.applications.polreg.weighted_polynomial_approximator import WeightedPolynomialApproximator
from smolyak.sparse.sparse_index import cartesian_product
from smolyak.sparse import sparse_index
from smolyak.applications.polreg.polynomial_subspace import UnivariatePolynomialSubspace, \
    MultivariatePolynomialSubspace

class TestWeightedPolynomialApproximator(unittest.TestCase):
        
    def setUp(self):
        import matplotlib.pyplot as plt
        plt.ion()
        
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close()
        
    def test_hermite(self):
        d = 2
        def function(X):
            return np.ones(shape=(X.shape[0], 1))  # np.prod(np.sin(3 * X), axis=1)
        ups = UnivariatePolynomialSubspace(measure='h')
        ps = MultivariatePolynomialSubspace(ups=ups, c_var=2, sampler='optimal')
        P = WeightedPolynomialApproximator(function=function, ps=ps)
        sparseindices = cartesian_product([range(2)] * d)
        P.expand(sparseindices)
        # P.plot_sampling_measure(N=200,L=10)
        P.plot_xy()
        P.plot()
        M = 100
        sparseindices = cartesian_product([range(3)] * d)
        P.expand(sparseindices)
        # P.plot_sampling_measure(N=200,L=10)
        P.plot_xy()
        P.plot(L=5)
        sparseindices = sparse_index.simplex(6, c_dim=d)
        P.expand(sparseindices)
        # P.plot_sampling_measure(N=200,L=20)
        P.plot_xy()
        P.plot(L=5)
        X = np.random.rand(M, d)
        Y = function(X.reshape((X.shape[0], d))).reshape((X.shape[0], 1))
        self.assertLess(np.linalg.norm(P(X) - Y) / np.sqrt(M), 0.1)
        
    # def test_adaptive(self):
    #    self.adaptive('u')
    # 
#     def test_uniform_1(self):
#         self.fixd('u', 1)
#         
#     def test_uniform_2(self):
#         self.fixd('u', 2)
#         
#     def test_chebyshev_1(self):
#         self.fixd('c', 1)
#         
#     def test_chebyshev_2(self):
#         self.fixd('c', 2)
        
    def fixd(self, measure, d):
        def function(X):
            return np.prod(np.sin(3 * X), axis=1)
        P = WeightedPolynomialApproximator(function, d, measure=measure, interval=(-2, 1))
        sparseindices = cartesian_product([range(2)] * d)
        P.expand(sparseindices)
        self.assertAlmostEqual(P.plot_sampling_measure(N=200), 1, delta=0.2)
        P.plot_xy()
        P.plot()
        M = 100
        sparseindices = cartesian_product([range(6)] * d)
        P.expand(sparseindices)
        self.assertAlmostEqual(P.plot_sampling_measure(N=200), 1, delta=0.2)
        P.plot_xy()
        P.plot()
        X = np.random.rand(M, d)
        Y = function(X.reshape((X.shape[0], d))).reshape((X.shape[0], 1))
        self.assertLess(np.linalg.norm(P(X) - Y) / np.sqrt(M), 0.01)
        
    def adaptive(self, measure):
        def function(X):
            return np.prod(np.sin(3 * X), axis=1)
        P = WeightedPolynomialApproximator(function, measure=measure)
        sparseindices = cartesian_product([range(4)] * 1)
        P.expand(sparseindices)
        self.assertAlmostEqual(P.plot_sampling_measure(N=200), 1, delta=0.2)
        P.plot_xy()
        M = 100
        sparseindices = cartesian_product([range(4)] * 2)
        P.expand(sparseindices)
        self.assertAlmostEqual(P.plot_sampling_measure(N=200), 1, delta=0.2)
        P.plot_xy()
        X = np.random.rand(M, 2)
        Y = function(X.reshape((X.shape[0], 2))).reshape((X.shape[0], 1))
        self.assertLess(np.linalg.norm(P(X) - Y) / np.sqrt(M), 0.01)
        
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
