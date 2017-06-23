'''
Test smolyak.polreg.orthogonal_polynomials
'''
import unittest
from smolyak.applications.polynomials.orthogonal_polynomials import evaluate_orthonormal_polynomials
import numpy as np

class TestOrthogonalPolynomials(unittest.TestCase):

    def test_legendre(self):
        cEvaluations = 100000
        degree_max = 1000
        a = -2
        b = 100
        X = np.linspace(a, b, cEvaluations)
        Y = evaluate_orthonormal_polynomials(X, degree_max, 'u', interval=(a, b))
        self.assertEqual(Y.shape, (cEvaluations, degree_max + 1))
        normsquared_start = Y[:, 0].transpose().dot(Y[:, 0]) / cEvaluations
        normsquared_1 = Y[:, 1].transpose().dot(Y[:, 1]) / cEvaluations
        normsquared_end = Y[:, -1].transpose().dot(Y[:, -1]) / cEvaluations
        scalarproduct_10_20 = Y[:, 10].transpose().dot(Y[:, 20]) / cEvaluations
        self.assertAlmostEqual(normsquared_start, 1, delta=0.05)
        self.assertAlmostEqual(normsquared_1, 1, delta=0.05)
        self.assertAlmostEqual(normsquared_end, 1, delta=0.05)
        self.assertAlmostEqual(scalarproduct_10_20, 0, delta=0.05) 
        
    def test_chebyshev(self):
        cEvaluations = 100000
        degree_max = 1000
        a = -0.12
        b = 12
        X = np.cos(np.linspace(0, np.pi, cEvaluations)) * (b - a) / 2 + (b + a) / 2 
        Y = evaluate_orthonormal_polynomials(X, degree_max, 'c', interval=(a, b))
        self.assertEqual(Y.shape, (cEvaluations, degree_max + 1))
        normsquared_start = Y[:, 0].transpose().dot(Y[:, 0]) / cEvaluations
        normsquared_1 = Y[:, 1].transpose().dot(Y[:, 1]) / cEvaluations
        normsquared_end = Y[:, -1].transpose().dot(Y[:, -1]) / cEvaluations
        scalarproduct_10_20 = Y[:, 10].transpose().dot(Y[:, 20]) / cEvaluations
        self.assertAlmostEqual(normsquared_start, 1, delta=0.05)
        self.assertAlmostEqual(normsquared_1, 1, delta=0.05)
        self.assertAlmostEqual(normsquared_end, 1, delta=0.05)
        self.assertAlmostEqual(scalarproduct_10_20, 0, delta=0.05) 
        
    def test_hermite(self):
        cEvaluations = 10000
        degree_max = 170
        L = 30
        X = np.linspace(-L, L, cEvaluations)
        W = np.exp(-np.power(X, 2) / 2) * 1 / np.sqrt(2 * np.pi)
        Y = evaluate_orthonormal_polynomials(X, degree_max, 'h')
        self.assertEqual(Y.shape, (cEvaluations, degree_max + 1))
        normsquared_start = Y[:, 0].transpose().dot(Y[:, 0] * W) / cEvaluations * 2 * L
        normsquared_1 = Y[:, 1].transpose().dot(Y[:, 1] * W) / cEvaluations * 2 * L
        normsquared_end = Y[:, -1].transpose().dot(Y[:, -1] * W) / cEvaluations * 2 * L
        scalarproduct_10_20 = Y[:, 10].transpose().dot(Y[:, 20] * W) / cEvaluations * 2 * L
        self.assertAlmostEqual(normsquared_start, 1, delta=0.05)
        self.assertAlmostEqual(normsquared_1, 1, delta=0.05)
        self.assertAlmostEqual(normsquared_end, 1, delta=0.05)
        self.assertAlmostEqual(scalarproduct_10_20, 0, delta=0.05)
    
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test1']
    unittest.main()
