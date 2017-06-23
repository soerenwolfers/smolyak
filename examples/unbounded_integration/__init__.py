from smolyak.applications.particle_systems.value_functions import univariate_integral_approximation
import numpy as np
from smolyak.demonstration.demonstration import demonstrate
def get_numerical_algorithm():
    def integral_approximation(k_1,k_2):
        X=np.linspace(-2**k_1, 2**k_1, 2*2**(k_1+k_2)+1, True)
        X=X[1:-1]
        work_model=len(X)
        value=np.sum(np.abs(np.cos(X))/(1+np.power(X,2)))/2**k_2
        return (work_model, value)
    return integral_approximation

if __name__ == '__main__':
    numerical_algorithm = get_numerical_algorithm()
    demonstrate(numerical_algorithm=numerical_algorithm,
         convergence_type='exponential',
         work_parameters=[1, 1],
         convergence_parameters=[1, 1],
         has_work_model=True,
         runtime_limit=[150, 150, 150])
