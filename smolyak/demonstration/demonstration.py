from __future__ import division
from smolyak.sparse.mixed_differences import MixedDifferences
from smolyak.sparse.sparse_approximator import SparseApproximator
import numpy as np
from smolyak.misc.plots import plot_convergence
import matplotlib.pyplot as plt
from smolyak.sparse.approximation_problem import ApproximationProblem
from smolyak.sparse.sparse_index import rectangle, SparseIndex
import timeit

def demonstrate(numerical_algorithm, convergence_type, n=None, work_parameters=None, convergence_parameters=None, has_work_model=False, runtime_limit=60):
    '''
    Compare non-adaptive Smolyak algorithm, adaptive Smolyak algorithm, and direct computations.
    
    :param numerical_algorithm: Algorithm to be accelerated
    :type numerical_algorithm: Function with n positive integer parameters
    :param n: Number of input parameters
    :param convergence_type: 'algebraic' if algorithm converges algebraically;
                             'exponential' if algorithm converges exponentially
    :param work_parameters: List of n positive reals gamma_j such that 
                    :math:`\text{Work(numerical_algorithm}(k_1,\dots,k_n_)\text{)}=\prod(k_j^\gamma_j)` (if convergence_type='algebraic')
                    or
                    :math:`\text{Work(numerical_algorithm}(k_1,\dots,k_n_)\text{)}=\prod(\exp(\gamma_j k_j))` (if convergence_type='exponential')
    :param convergence_parameters: List of n positive reals beta_j such that
                    :math:`\|\text{numerical_algorithm}(k_1,\dots,k_n)-\text{limit}\| < \sum(k_j^{-\beta_j})` (if convergence_type='algebraic')
                    or
                    :math:`\|\text{numerical_algorithm}(k_1,\dots,k_n)-\text{limit}\| < \sum(\exp(-\beta_j k_j))` (if convergence_type='exponential')
    :param has_work_model: Does the numerical_algorithm come with its own cost specification?
        If yes, numerical_algorithm needs to return tuples (cost,value)
    :type has_work_model: Boolean
    :param runtime_limit: Runtime limit for the convergence plots, in seconds
    '''
    if not hasattr(runtime_limit, "__getitem__"):
        runtime_limit = [runtime_limit] * 3
    if not n:
        if convergence_parameters:
            n = len(convergence_parameters)
        elif work_parameters:
            n = len(work_parameters)
        else:
            raise ValueError('Specify either number of input parameters or list of convergence or work parameters')
    if work_parameters:
        if not n == len(work_parameters):
            raise ValueError('Number of work parameters does not match number of input parameters')
    if convergence_parameters:
        if not n == len(convergence_parameters):
            raise ValueError('Number of convergence parameters does not match number of input parameters')
    def numerical_algorithm_wrapper(mi):
        if convergence_type == 'algebraic':
            return numerical_algorithm(*[2 ** mi[i] for i in range(n)])
        elif convergence_type == 'exponential':
            return numerical_algorithm(*mi.full_tuple(dim_max=n))    
        else:
            raise ValueError('Convergence_type must be \'algebraic\' or \'exponential\'')
    MD = MixedDifferences(numerical_algorithm_wrapper, store_output=False)
    problem = ApproximationProblem(decomposition=MD, n=n, has_work_model=has_work_model)    
    SA = SparseApproximator(problem=problem)
    if convergence_parameters and work_parameters:
        scale = min([work_parameters[i] + convergence_parameters[i] for i in range(len(work_parameters))])
        SA.expand_by_mis(rectangle([scale * 13. / (work_parameters[i] + convergence_parameters[i]) for i in range(len(work_parameters))]))
    else:
        SA.expand_by_mis(rectangle(10), n)
    SA.plot_mis(range(n), weighted='contribution/work_model', percentiles=5)
    plt.show()
    
    MD = MixedDifferences(numerical_algorithm_wrapper, store_output=True)
    if has_work_model:
        work_type = 'work_model'
    else:
        work_type = 'runtime' 
    # DIRECT COMPUTATION
    values_direct = []
    runtimes_direct = []
    work_model_direct = []   
    L = 0
    scale = max(convergence_parameters)
    while True:
        parameters = [scale * L / convergence_parameters[i] for i in range(n)]
        tic = timeit.default_timer()
        (work_model, value) = numerical_algorithm_wrapper(SparseIndex(parameters))
        runtimes_direct.append(timeit.default_timer() - tic)
        work_model_direct.append(work_model)
        values_direct.append(value)
        print 'Direct runtime at level {}: {}s'.format(L, runtimes_direct[-1])
        if sum(runtimes_direct) > runtime_limit[0]:
            break
        else:
            L = L + 1
    print('Direct computations convergence order: {}'.format(plot_convergence(work_model_direct, values_direct)))   
    
    # ADAPTIVE SMOLYAK ALGORITHM
    values_adaptive = []
    runtimes_adaptive = []
    work_model_adaptive = []
    problem = ApproximationProblem(decomposition=MD, n=n, has_work_model=has_work_model)    
    L = 0
    while True:  # 20 
        problem.reset()
        SA = SparseApproximator(problem=problem, work_type=work_type)
        SA.expand_adaptive(T_max=2 ** (max(convergence_parameters) * (L - 10)))
        values_adaptive.append(SA.get_approximation())
        runtimes_adaptive.append(SA.get_total_runtime())
        work_model_adaptive.append(SA.get_total_work_model())
        print 'Adaptive runtime at level {}: {}s'.format(L, runtimes_adaptive[-1])
        if sum(runtimes_adaptive) > runtime_limit[1]:
            break
        else:
            L = L + 1
    print('Adaptive Smolyak convergence order: {}'.format(plot_convergence(work_model_adaptive, values_adaptive)))
    
    # NONADAPTIVE SMOLYAK ALGORITHM
    if work_parameters and convergence_parameters:
        values_nonadaptive = []
        runtimes_nonadaptive = []
        work_model_nonadaptive = []
        problem = ApproximationProblem(decomposition=MD, n=n, work_factor=work_parameters,
                                    contribution_factor=convergence_parameters, has_work_model=has_work_model) 
        scale = min([np.log(2) * (work_parameters[i] + convergence_parameters[i]) for i in range(len(work_parameters))])
        L = 0
        while True:  # 28
            problem.reset()
            SA = SparseApproximator(problem=problem, work_type=work_type)
            SA.expand_nonadaptive(L, scale=scale)
            values_nonadaptive.append(SA.get_approximation())
            runtimes_nonadaptive.append(SA.get_total_runtime())
            work_model_nonadaptive.append(SA.get_total_work_model())
            print 'Nonadaptive runtime at level {}: {}s'.format(L, runtimes_nonadaptive[-1])
            if sum(runtimes_nonadaptive) > runtime_limit[2]:
                break
            else:
                L = L + 1
        print('Nonadaptive Smolyak algorithm convergence order: {}'.format(plot_convergence(work_model_nonadaptive, values_nonadaptive)))
            
    plt.legend(['Direct', 'Adaptive Smolyak', 'Nonadaptive Smolyak'])
    plt.show()