'''
Test smolyak.polreg.mi_weighted_polynomial_approximator
'''
import unittest
from smolyak import SparseApproximation
import numpy as np
from smolyak.applications.polynomials.mi_weighted_polynomial_approximator import MIWeightedPolynomialApproximator
from swutil.time import snooze
import cProfile
import pstats
from smolyak import Decomposition
from smolyak.smolyak import WorkFunction
from smolyak.applications.polynomials.polynomial_spaces import PolynomialSpace,\
    TensorPolynomialSpace
from smolyak.applications.polynomials.probability_spaces import ProbabilitySpace,\
    TensorProbabilitySpace, UnivariateProbabilitySpace
from matplotlib import pyplot as plt
class TestMIWeightedPolynomialApproximator(unittest.TestCase):

    def setUp(self):
        """init each validate_args"""
        self.pr = cProfile.Profile()
        self.pr.enable()
        print("\n<<<---")
        
    def tearDown(self):
        """finish any validate_args"""
        p = pstats.Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('cumulative').print_stats(20)
        print("\n--->>>")

    def foo(self, approximation_type):
        T=30
        def function(X, mi): 
            return np.sin(3 * X[:, 0] + 2 ** (-mi[0])) + 2 ** (-mi[0]) * 2 ** (-mi[1]) + snooze(2 ** mi[0] + 2 ** mi[1])
        n_acc = 2
        n_par = 1
        prob_space = TensorProbabilitySpace(univariate_probability_spaces=[UnivariateProbabilitySpace('u', (0,1))]*n_par)
        poly_space = TensorPolynomialSpace(probability_space=prob_space)
        MIWPA = MIWeightedPolynomialApproximator(function=function,n=n_acc,polynomial_space=poly_space)
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

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
