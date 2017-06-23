from operator import mul
import functools
import numpy as np
import math
from numpy import Inf

class Decomposition():
    def __init__(self, func, n, init_dims=None, next_dims=None, is_bundled=None,
                 is_external=False, is_md=None, have_work_factor=None,
                 have_contribution_factor=None, work_factor=None, contribution_factor=None,
                 has_work_model=False, has_contribution_model=False, kronecker_exponents=None,reset=None):
        r'''        
        :param func: Computes decomposition terms.
            In the most basic form, a single multi-index is passed to 
            :code:`func` and a single value, which represents the corresponding 
            decomposition term and supports vector space operations, is expected. 
            
            If is_bundled, then :code:`func` is passed an iterable of multi-indices
            (which agree in the dimensions that are not bundled, see below) must return 
            the sum of the corresponding decomposition elements.
            This may be useful for parallelization, or for problems where joint
            computation of decomposition elements is analytically more efficient.
            
            If :code:`is_external`, no output is required and decomposition terms
            are expected to be stored by :code:`func` itself. Each call 
            will contain the full multi-index set representing the current approximation.
            
            If :code:`has_work_model`, return (value,work) (or only work if is_external),
            where work is a single float representing the work that was required for the 
            computation of value.
            
            If :code:`has_contribution_model`, return (value,contribution) (or only 
            contribution if is_external), where contribution is:
                *a single float if not is_external and not is_bundled or else
                *a dictionary  {(mi,contribution(float) for mi in bundle}
                    where bundle is the multi-index set that has been passed
            
            If both, return (value,work,contribution) (or only (work,contribution))
            if is_external 
        :type func:  Function.
        :param n: Number of discretization parameters
        :type n: Integers including numpy.Inf (or 'inf') 
        :param init_dims: In most cases this should be :math:`n`. However, for large or infinite :math:`n`, instead provide list of 
           dimensions that are explored initially. Each time a multi-index with non-zero group in one of these initial dimensions is selected, 
           new dimensions are added, according to  nextDims
        :type init_dims: Integer or list of integers
        :param next_dims: Assigns to each dimension a list of child dimensions (see above)
        :type next_dims: :math:`\mathbb{N}\to 2^{\mathbb{N}}`
        :param is_bundled: In some problems, the decomposition terms cannot be computed independent of 
           each other. In this case, :code:`func` is called with subsets of :math:`\mathcal{I}` whose entries agree
           except for the dimensions specified in is_bundled
        :type is_bundled: :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param have_work_factor: Dimensions for which work_factor is available
        :type have_work_factor: Boolean or :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param have_contribution_factor: Dimensions for which contribution factor is available
        :type have_contribution_factor: Boolean or :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param work_factor: If specified, then the work required for the computation of :math:`f(\mathbf{k})` is assumed to behave
        like :math:`\verb|work_factor|(\mathbf{k})\times w(\tilde{\mathbf{k}})` where :math:`\tilde{\mathbf{k}}` contains the
        entries of :math:`\mathbf{k}` that are not in :code:`have_work_factor`. 
           If this is a list of real numbers, the work is assumed to grow exponentially in given parameters with exponents specified in list.
        :type work_factor: :math:`\mathbb{N}^n\to(0,\infty)` or list of positive numbers
        :param contribution_factor: If specified, then the contribution of :math:`f(\mathbf{k})` is assumed to behave
        like :math:`\verb|work_factor|(\mathbf{k})\times e(\tilde{\mathbf{k}})` where :math:`\tilde{\mathbf{k}}` contains the
        entries of :math:`\mathbf{k}` that are not in :code:`have_work_factor`. 
           If this is a list of real numbers, the contribution is assumed to decay exponentially in given parameters with exponents specified in list.
        :type contribution_factor: :math:`\mathbb{N}^n\to(0,\infty)` or list of positive numbers
        :param is_md: Specifies whether decomposition is a multi-index decomposition. This information is used when fitting work parameters.
        :type is_md: Boolean
        :param is_external: Specifies whether approximation is computed externally. If True, :code:`func` is called multiple times 
        with a SparseIndex as argument (or a list of SparseIndices if :code:`is_bundled` is True as well) and must collect the associated
        decomposition terms itself. If False, :code:`func` is also called multiple times with a SparseIndex (or a list of SparseIndices)
        and must return the associated decomposition terms, which are then stored within the SparseApproximator instance
        :type is_external: Boolean
        :param has_work_model: Does the decomposition come with its own cost specification?
        :type has_work_model: Boolean
        :param has_contribution_model: Does the decomposition come with its own contribution specification?
        :param kronecker_exponents: For infinite dimensional problems, the 
        contribution of Kronecker multi-index e_j is estimated as exp(contribution_exponent(j))
        :type kronecker_exponents: Function from integers to negative reals
        :param reset: If is_external, this function will reset the externally stored approximation
        :type reset: Function
        '''
        self.func = func
        self._set_n(n)
        self.has_work_model = has_work_model
        self.has_contribution_model = has_contribution_model
        self.is_external = is_external
        if math.isinf(self.n):
            self._set_init_dims(init_dims)
            self._set_next_dims(next_dims)  
        elif init_dims or next_dims:
            raise ValueError('Parameters init_dims and next_dims only valid for infinite-dimensional problems')
        else:
            self._set_init_dims(self.n)
        self._set_is_bundled(is_bundled)
        self._set_is_md(is_md)
        self._process_work_factor(have_work_factor, work_factor, self.is_md)
        self._process_contribution_factor(have_contribution_factor, contribution_factor, self.is_md)
        self.kronecker_exponents = kronecker_exponents
        self.reset=reset
        
    def _process_work_factor(self, have_work_factor, work_factor, is_md):
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
            
    def _process_contribution_factor(self, have_contribution_factor, contribution_factor, is_md):
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
    
    def _set_is_md(self, is_md):
        if not is_md:
            self.is_md = lambda dim: False
        elif is_md is True:
            self.is_md = lambda dim: True
        else:
            self.is_md = is_md
            
    def _set_is_bundled(self, bundled):
        if hasattr(bundled, '__contains__'):
            self.is_bundled = lambda dim: dim in bundled
        else: 
            self.is_bundled = bundled
       
    def _set_n(self, n):
        if hasattr(n, 'upper') and n.upper() in ['INF', 'INFINITY', 'INFTY']:
            self.n = Inf   
        else:
            self.n = n
         
    def _set_next_dims(self, next_dims):
        if (not next_dims) and self.n == Inf:
            self.next_dims = lambda dim: dim + 1 if dim + 1 not in self.init_dims else []
        else:
            self.next_dims = next_dims
        
    def _set_init_dims(self, init_dims):
        if isinstance(init_dims, int):
            self.init_dims = range(init_dims)
        elif hasattr(init_dims, '__contains__'):
            self.init_dims = init_dims
        elif init_dims is None:
            self.init_dims = [0]
