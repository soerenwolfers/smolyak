from smolyak.applications.particle_systems.value_functions import univariate_integral_approximation
import numpy as np
from smolyak.demonstration.demonstration import demonstrate
def get_numerical_algorithm():
    case = 3
    shift = True
    invcdf = None
    pdf = None
    bound = None
    if case == 1:
        phi = lambda x: np.exp(x)
        invcdf = lambda x: x
    if case == 2:
        phi = lambda x: np.sin(1.5 * np.pi * (x - 0.1))
        invcdf = lambda x: np.power(x, 1. / 2)
    if case == 3:
        phi = lambda x: x ** 2
        invcdf = lambda x:np.power(x, 1. / 2)
    if case == 4:
        phi = lambda x: x ** 2
        pdf = lambda x: 2 * x
        bound = 2
    def value_function_approximation(m, n):
        work_model = float(m) * float(n) ** 2
        value = univariate_integral_approximation(n * 2, m ** (-1) / 2, invcdf, pdf, bound, phi, shift, app_type='time_stepping')
        return (work_model, value)
    return value_function_approximation

if __name__ == '__main__':
    numerical_algorithm = get_numerical_algorithm()
    demonstrate(numerical_algorithm=numerical_algorithm,
     convergence_type='algebraic',
     work_parameters=[1, 2],
     convergence_parameters=[1, 1],
     has_work_model=True,
     runtime_limit=[80, 300, 300])
