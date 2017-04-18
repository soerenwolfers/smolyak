from operator import mul
import functools
import numpy as np
import math
from numpy import Inf

class ApproximationProblem():
    def __init__(self, decomposition, n, init_dims=None, next_dims=None, is_bundled=None, is_external=False,
                 is_md=None, have_work_factor=None, have_contribution_factor=None,
                  work_factor=None, contribution_factor=None, has_work_model=False):
        r'''        
        :param decomposition: Computes the elements :math:`(\Delta f)(\mathbf{k})`
        :type decomposition: Optional. :math:`\mathbb{N}^{n}\to Y` for some vector space :math:`Y`
        :param n: Number of discretization parameters
        :type n: Integers including numpy.Inf (or 'inf') 
        :param init_dims: In most cases this should be :math:`n`. However, for large or infinite :math:`n`, instead provide list of 
           dimensions that are explored initially. Each time a multi-index with non-zero entry in one of these initial dimensions is selected, 
           new dimensions are added, according to  nextDims
        :type init_dims: Integer or list of integers
        :param next_dims: Assigns to each dimension a list of child dimensions (see above)
        :type next_dims: Optional. :math:`\mathbb{N}\to 2^{\mathbb{N}}`
        :param is_bundled: In some problems, the decomposition terms :math:`(\Delta f)(\mathbf{k})` cannot be computed independent of 
           each other. In this case, :code:`decomposition` is called with subsets of :math:`\mathcal{I}` whose entries agree
           except for the dimensions specified in is_bundled
        :type is_bundled: Optional. :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param have_work_factor: Dimensions for which work_factor is available
        :type have_work_factor: Optional. Boolean or :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param have_contribution_factor: Dimensions for which contribution factor is available
        :type have_contribution_factor: Optional. Boolean or :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param work_factor: If specified, then the work required for the computation of :math:`f(\mathbf{k})` is assumed to behave
        like :math:`\verb|work_factor|(\mathbf{k})\times w(\tilde{\mathbf{k}})` where :math:`\tilde{\mathbf{k}}` contains the
        entries of :math:`\mathbf{k}` that are not in :code:`have_work_factor`. 
           If this is a list of real numbers, the work is assumed to grow exponentially in given parameters with exponents specified in list.
        :type work_factor: Optional. :math:`\mathbb{N}^n\to(0,\infty)` or list of positive numbers
        :param contribution_factor: If specified, then the contribution of :math:`f(\mathbf{k})` is assumed to behave
        like :math:`\verb|work_factor|(\mathbf{k})\times e(\tilde{\mathbf{k}})` where :math:`\tilde{\mathbf{k}}` contains the
        entries of :math:`\mathbf{k}` that are not in :code:`have_work_factor`. 
           If this is a list of real numbers, the contribution is assumed to decay exponentially in given parameters with exponents specified in list.
        :type contribution_factor: Optional. :math:`\mathbb{N}^n\to(0,\infty)` or list of positive numbers
        :param is_md: Specifies whether decomposition is a multi-index decomposition. This information is used when fitting work parameters.
        :type is_md: Optional. Boolean
        :param is_external: Specifies whether approximation is computed externally. If True, decomposition is called multiple times 
        with a SparseIndex as argument (or a list of SparseIndices if :code:`is_bundled` is True as well) and must collect the associated
        decomposition terms itself. If False, decomposition is also called multiple times with a SparseIndex (or a list of SparseIndices)
        and must return the associated decomposition terms, which are then stored within the SparseApproximator instance
        :type is_external: Optional. Boolean
        :param has_work_model: Does the decomposition come with its own cost specification?
            If yes, it needs to return tuples (cost,value)
        :type has_work_model: Optional. Boolean
        '''
        self.decomposition = decomposition
        self.__set_n(n)
        self.has_work_model = has_work_model
        self.is_external = is_external
        if math.isinf(self.n):
            self.__set_init_dims(init_dims)
            self.__set_next_dims(next_dims)  
        elif init_dims or next_dims:
            raise ValueError('Parameters init_dims and next_dims only valid for infinite-dimensional problems')
        else:
            self.__set_init_dims(self.n)
        self.__set_is_bundled(is_bundled)
        self.__set_is_md(is_md)
        self.__process_work_factor(have_work_factor, work_factor, self.is_md)
        self.__process_contribution_factor(have_contribution_factor, contribution_factor, self.is_md)
        
    def __process_work_factor(self, have_work_factor, work_factor, is_md):
        if have_work_factor is True or (have_work_factor is None and work_factor):
            self.have_work_factor = lambda dim: True
        elif hasattr(have_work_factor, '__contains__'):
            self.have_work_factor = lambda dim: dim in have_work_factor
        elif have_work_factor:
            self.have_work_factor = have_work_factor
        else:
            self.have_work_factor = lambda dim: False
        if work_factor:
            if hasattr(work_factor, '__contains__'):
                list_have = []
                i = 0
                dim = 0
                while i < len(work_factor):
                    if self.have_work_factor(dim):
                        list_have.append(dim)
                        i += 1
                    dim += 1
                if self.is_bundled:
                    self.work_factor = lambda mi: functools.reduce(
                            mul, [np.exp(work_factor[i] * mi[0][dim]) 
                            for i, dim in enumerate(list_have)])
                else:
                    self.work_factor = lambda mi: functools.reduce(
                            mul, [np.exp(work_factor[i] * mi[dim]) 
                            for i, dim in enumerate(list_have)])
            else:
                self.work_factor = work_factor
        else: 
            self.work_factor = lambda mi: 1
            
    def __process_contribution_factor(self, have_contribution_factor, contribution_factor, is_md):
        if have_contribution_factor is True or (have_contribution_factor is None and contribution_factor):
            self.have_contribution_factor = lambda dim: True
        elif hasattr(have_contribution_factor, '__contains__'):
            self.have_contribution_factor = lambda dim: dim in have_contribution_factor
        elif have_contribution_factor:
            self.have_contribution_factor = have_contribution_factor
        else:
            self.have_contribution_factor = lambda dim: False
        if contribution_factor:
            if hasattr(contribution_factor, '__contains__'):
                list_have = []
                i = 0
                dim = 0
                while i < len(contribution_factor):
                    if self.have_contribution_factor(dim):
                        list_have.append(dim)
                        i += 1
                    dim += 1
                self.contribution_factor = lambda mi: functools.reduce(
                        mul, [np.exp(-contribution_factor[i] * mi[dim]) 
                        for i, dim in enumerate(list_have)])
            else:
                self.contribution_factor = contribution_factor
        else: 
            self.contribution_factor = lambda mi: 1
    
    def __set_is_md(self, is_md):
        if not is_md:
            self.is_md = lambda dim: False
        elif is_md is True:
            self.is_md = lambda dim: True
        else:
            self.is_md = is_md
            
    def __set_is_bundled(self, bundled):
        if hasattr(bundled, '__contains__'):
            self.is_bundled = lambda dim: dim in bundled
        else: 
            self.is_bundled = bundled
       
    def __set_n(self, n):
        if hasattr(n, 'upper') and n.upper() in ['INF', 'INFINITY', 'INFTY']:
            self.n = Inf   
        else:
            self.n = n
         
    def __set_next_dims(self, next_dims):
        if (not next_dims) and self.n == Inf:
            self.next_dims = lambda dim: dim + 1 if dim + 1 not in self.init_dims else []
        else:
            self.next_dims = next_dims
        
    def __set_init_dims(self, init_dims):
        if isinstance(init_dims, int):
            self.init_dims = range(init_dims)
        elif hasattr(init_dims, '__contains__'):
            self.init_dims = init_dims
        elif init_dims is None:
            self.init_dims = [0]
            
    def reset(self):
        if hasattr(self.decomposition, 'reset'):
            self.decomposition.reset()
