from smolyak.applications.polynomials.polynomial_spaces import TensorPolynomialSpace
from smolyak.applications.polynomials.probability_spaces import UnivariateProbabilitySpace
from smolyak.applications.polynomials.polynomial_approximation import PolynomialApproximation
def polynomial_combination_factory(ps):
    a=PolynomialApproximation(ps)
    def func(coefficients,X):
        a.set_coefficients(coefficients)
        return a(X)
    return func
def bool_combination(selector,x_1,x_2):
    if selector>0:
        return x_1
    else:
        return x_2