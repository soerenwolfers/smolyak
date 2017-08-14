'''
Test smolyak.polreg.mi_weighted_polynomial_approximator
'''
import unittest
from smolyak.approximator import Approximator
import numpy as np
from smolyak.applications.polynomials.mi_weighted_polynomial_approximator import MIWeightedPolynomialApproximator
from smolyak.aux.time import snooze
import cProfile
import pstats
from smolyak.decomposition import Decomposition

class TestMIWeightedPolynomialApproximator(unittest.TestCase):

    def setUp(self):
        """init each test"""
        self.pr = cProfile.Profile()
        self.pr.enable()
        print "\n<<<---"
        
    def tearDown(self):
        """finish any test"""
        p = pstats.Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('cumulative').print_runtime(20)
        print "\n--->>>"

    def foo(self, approximation_type):
        def function(X, mi): 
            return np.sin(3 * X[:, 0] + 2 ** (-mi[0])) + 2 ** (-mi[0]) * 2 ** (-mi[1]) + snooze(2 ** mi[0] + 2 ** mi[1])
        cdim_acc = 2
        cdim_par = 1
        MIWPA = MIWeightedPolynomialApproximator(function=function, cdim_acc=cdim_acc, cdim_par=cdim_par)
        if approximation_type == 'continuation':
            decomp=Decomposition(func=MIWPA.expand,init_dims=cdim_acc + cdim_par, is_bundled=lambda dim: dim >= cdim_acc,
                              external=True)
            SA = Approximator(decomp)
            SA.continuation(L_min=1, T_max=10, work_exponents=[np.log(2), np.log(2), np.log(2)],
                            contribution_exponents=[np.log(2), np.log(2), 2], reset=MIWPA.reset)
        elif approximation_type == 'adaptive':
            decomp=Decomposition(func=MIWPA.expand,
                              init_dims=cdim_acc + cdim_par, is_bundled=lambda dim: dim >= cdim_acc,
                              have_work_factor=lambda dim: dim >= cdim_acc,
                              work_factor=MIWPA.estimated_work,
                              external=True)
            SA = Approximator(decomp)
            SA.expand_adaptive(c_steps=75, T_max=4, reset=MIWPA.reset)   
        elif approximation_type == 'nonadaptive':
            decomp=Decomposition(decomposition=MIWPA.expand,
                              init_dims=cdim_acc + cdim_par, is_bundled=lambda dim: dim >= cdim_acc,
                              work_factor=[np.log(2), np.log(2), np.log(2)],
                              contribution_factor=[np.log(2), np.log(2), 2],
                              external=True)
            SA = Approximator(decomp)
            SA.expand_nonadaptive(10, 3)
        if not approximation_type == 'nonadaptive':
            print([SA.get_contribution_exponent(dim) for dim in range(cdim_acc + cdim_par)])
            print([SA.get_work_exponent(dim) for dim in [0, 1]])
            self.assertAlmostEqual(SA.get_contribution_exponent(0), np.log(2), delta=0.1) 
            self.assertAlmostEqual(SA.get_contribution_exponent(1), np.log(2), delta=0.1) 
        A = MIWPA.get_approximation()
        self.assertAlmostEqual(A(np.array([0.5]).reshape(1, 1)), np.sin(1.5), delta=0.1)
        
    def test_continuation(self):
        self.foo('continuation')
        
    def test_adaptive(self):
        self.foo('adaptive')
        
    def test_nonadaptive(self):
        self.foo('nonadaptive')

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
