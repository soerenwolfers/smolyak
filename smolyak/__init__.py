'''
smolyak includes the following modules and packages:

* smolyak.py: General, adaptive and non-adaptive, implementations of Smolyak's algorithm
        in class SparseApproximator. 
         
* indices.py: Various tools for handling multi-indices, such as:
        * a sparse object oriented representation of multi-indices in class MultiIndex
        * the class MISet for connected multi-index sets
        * a class that computes mixed differences
        * various functions for the generation of multi-index sets in different shapes, such as
            simplex, hyperbolic_cross, pyramids, or determined by user-defined functions (get_admissible_indices)
        * computation of combination rule coefficients

* applications: Various numerical approximation algorithms that can be 
    accelerated using Smolyak's algorithm, including:
    * Simulation of particle systems
    * Solution of partial differential equations
    * Polynomial approximation
    
'''
from smolyak.smolyak import SparseApproximator
from smolyak import indices
