'''
Test smolyak.polreg.weighted_polynomial_approximator
'''
import unittest
import numpy as np
from smolyak.applications.polynomials.weighted_polynomial_approximator import WeightedPolynomialApproximator
from smolyak.indices import cartesian_product
from smolyak import indices

from smolyak.applications.polynomials.probability_spaces import UnivariateProbabilitySpace
from smolyak.smolyak import Decomposition, SparseApproximation, WorkFunction
from swutil.time import snooze

class TestWeightedPolynomialApproximator(unittest.TestCase):
        
    def setUp(self):
        import matplotlib.pyplot as plt
        plt.ion()
        
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close()
    
    def foo(self, approximation_type):
        T=30
        def function(X, mi): 
            return np.sin(3 * X[:, 0] + 2 ** (-mi[0])) + 2 ** (-mi[0]) * 2 ** (-mi[1]) + snooze(2 ** mi[0] + 2 ** mi[1])
        n_acc = 2
        n_par = 1
        prob_space = UnivariateProbabilitySpace('u', (0,1))**n_par
        poly_space = TensorPolynomialSpace(probability_space=prob_space)
        MIWPA = WeightedPolynomialApproximator(function=function,n=n_acc,polynomial_space=poly_space)
        if approximation_type == 'expand_continuation':
            decomp=Decomposition(func=MIWPA.update_approximation,n=n_acc + n_par, bundled = True, bundled_dims=lambda dim: dim >= n_acc,
                              stores_approximation=True,returns_work=True,returns_contributions=True,reset=MIWPA.reset)
            SA = SparseApproximation(decomp)
            SA.expand_continuation(T=T)
        elif approximation_type == 'adaptive':
            decomp=Decomposition(func=MIWPA.update_approximation,
                             n =n_acc + n_par, bundled = True, bundled_dims=lambda dim: dim >= n_acc,
                              work_function = WorkFunction(func = MIWPA.estimated_work,dims = lambda dim: dim >= n_acc,bundled = True),
                              stores_approximation=True,reset = MIWPA.reset,returns_work=True,returns_contributions=True)
            SA = SparseApproximation(decomp)
            SA.expand_adaptive(T=T)  
        elif approximation_type == 'nonadaptive':
            decomp=Decomposition(func=MIWPA.update_approximation,
                              n=n_acc + n_par, bundled = True, bundled_dims = lambda dim: dim >= n_acc,
                              work_multipliers=[2, 2, 2],
                              contribution_multipliers=[1/2, 1/2,1/2],
                              stores_approximation=True,
                              returns_work=True,returns_contributions=True)
            SA = SparseApproximation(decomp)
            SA.expand_apriori(T=T)
        if not approximation_type == 'nonadaptive':
            print([SA._get_contribution_exponent(dim) for dim in range(n_acc + n_par)])
            print([SA._get_work_exponent(dim) for dim in [0, 1]])
            self.assertAlmostEqual(SA.get_contribution_multiplier(0), 1/2, delta=0.1) 
            self.assertAlmostEqual(SA.get_contribution_multiplier(1), 1/2, delta=0.1) 
        A = MIWPA.get_approximation()
        #print(approximation_type)
        #A.plot()
        #plt.show()
        #plt.figure(2)
        #SA.plot_indices(weights='contribution/runtime')
        #plt.show()
        self.assertAlmostEqual(A(np.array([0.5]).reshape(1, 1)), np.sin(1.5), delta=0.1)
        
    def test_continuation(self):
        self.foo('expand_continuation')
        
    def test_adaptive(self):
        self.foo('adaptive')
        
    def test_nonadaptive(self):
        self.foo('nonadaptive')
     
    def best_hermite(self):
        d = 2
        def function(X):
            return np.ones(shape=(X.shape[0], 1))  # np.prod(np.sin(3 * X), axis=1)
        ups = UnivariatePolynomialSpace(measure='h')
        ps = TensorPolynomialSpace(ups=ups, c_var=2, sampler='optimal')
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
        sparseindices = indices.simplex(6, n=d)
        P.expand(sparseindices)
        # P.plot_sampling_measure(N=200,L=20)
        P.plot_xy()
        P.plot(L=5)
        X = np.random.rand(M, d)
        Y = function(X.reshape((X.shape[0], d))).reshape((X.shape[0], 1))
        self.assertLess(np.linalg.norm(P(X) - Y) / np.sqrt(M), 0.1)
        
    def test_adaptive2(self):
        self.adaptive('u')
     
    #def test_uniform_1(self):
    #    self.fixd('u', 1)
         
    #def test_uniform_2(self):
    #    self.fixd('u', 2)
         
    #def test_chebyshev_1(self):
    #    self.fixd('c', 1)
         
    #def test_chebyshev_2(self):
    #    self.fixd('c', 2)
        
    def fixd(self, measure, d):
        def function(X):
            return np.prod(np.sin(3 * X), axis=1)
        ups = UnivariatePolynomialSpace(measure=measure,interval=(-2,1))
        ps = TensorPolynomialSpace(ups=ups, c_var=d, sampler='optimal')
        P = WeightedPolynomialApproximator(function, ps)
        sparseindices = cartesian_product([range(2)] * d)
        P.expand(sparseindices)
        self.assertAlmostEqual(ps.plot_optimal_distribution(N=200), 1, delta=0.2)
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
        ups = UnivariatePolynomialSpace(measure=measure)
        P = WeightedPolynomialApproximator(function, ps=ups)
        sparseindices = cartesian_product([range(4)] * 1)
        P.expand(sparseindices)
        ps = TensorPolynomialSpace(ups=ups,c_var=1)
        self.assertAlmostEqual(ps.plot_optimal_distribution(N=200), 1, delta=0.2)
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
