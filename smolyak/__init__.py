'''
smolyak includes the following packages:

* applications: Various numerical approximation algorithms that can be 
    accelerated using Smolyak's algorithm, including:
    * Simulation of particle systems
    * Solution of partial differential equations
    * Polynomial approximation
* misc: Miscellaneous tools
* sparse: The core of the toolbox, including:
    * General, adaptive and non-adaptive, implementations of Smolyak's algorithm
        in class sparse_approximator.SparseApproximator
    * Implementation of the combination rule
        in class combination_rule.CombinationRule
    * Problem abstraction module to be used with SparseApproximator 
        in class approximation_problem.ApproximationProblem
'''
from .approximator import Approximator
from .decomposition import Decomposition