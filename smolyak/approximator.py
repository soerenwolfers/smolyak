'''
Sparse approximation using Smolyak's algorithm
'''
from __future__ import division
import numpy as np
import timeit
from smolyak.indices import  MultiIndexDict, get_admissible_indices, DCSet,\
    kronecker
import copy
import warnings
import math
from smolyak.aux.plots import plot_indices
from smolyak import indices
from smolyak.aux.logs import Log
from smolyak.aux.decorators import log_calls
from smolyak.aux.more_collections import DefaultDict
from smolyak.aux.np_tools import weighted_median
from operator import mul
import functools
from numpy import Inf


class Approximator(object):
    r'''
    Computes sparse approximation based on multi-index decomposition.
    
    Given a decomposition,
    
    .. math::
    
        f_{\infty}=\sum_{\mathbf{k}\in\mathbb{N}^{n}} (\Delta f)(\mathbf{k}),
    
    this class computes approximations of the form
    
    .. math:: 
    
       \mathcal{S}_{\mathcal{I}}f:=\sum_{\mathbf{k}\in\mathcal{I}} (\Delta f)(\mathbf{k}),
    
    where :math:`\mathcal{I}` is an efficiently chosen finite multi-index set. 

    Currently supported choices for the construction of :math:`\mathcal{I}` are 
     :code:`expand_adaptive`, :code:`expand_nonadaptive` and :code:`continuation`
    '''
    def __init__(self, decomposition, work_type='runtime', log=None):
        r'''        
        :param decomposition: Decomposition of an approximation problem
        :type decomposition: Decomposition
        :param work_type: Optimize [work model associated with decomposition] or runtime
        :type work_type: String. 'work_model' or 'runtime' 
        :param log: Log object used for logging
        '''
        self.decomposition = decomposition
        self.md = _MetaData(self.decomposition)
        self.ad = _AdaptiveData(self.decomposition)
        self.app = _Approximation(self.decomposition)
        self.work_type = work_type  # MOVE INTO ALGORITHMS?
        self.dry_run = not self.decomposition.is_external and not self.decomposition.func
        #if self.work_type == 'runtime' and self.dry_run:
        #    raise ValueError('Cannot compute runtime without doing computations.')
        if self.work_type == 'work_model':
            assert(self.decomposition.has_work_model) 
        self.log = log or Log()   
            
    @log_calls   
    def continuation(self , L_max=None, T_max=None, L_min=2, work_exponents=None, contribution_exponents=None, find_work_exponents=False):
        '''
        Compute sparse approximation adaptively by using increasing multi-index 
        sets and determining at each step the next set by fitting contribution
        and work parameters. 
        
        :param L_max: Maximal level
        :type L_max: Integer
        :param T_max: Maximal runtime
        :type T_max: Positive real.
        :param L_min: Initial level
        :type L_min: Integer
        :param work_exponents: Initial guess of work exponents
        :type work_exponents: List of positive reals.
        :param contribution_exponents: Initial guess of contribution exponents
        :type contribution_exponents: List of positive reals.
        :param find_work_exponents: Specifies whether work exponents should be fitted
        :type find_work_exponents: Boolean. 
        '''
        if self.decomposition.is_external and not self.decomposition.reset:
                raise ValueError('If approximation is stored externally, need to specify reset function')
        if not work_exponents:
            if self.decomposition.is_bundled:
                raise ValueError('Need to specify work exponents for is_bundled parameters.')
            else:
                work_exponents = [1] * self.decomposition.n
        if not contribution_exponents:
            contribution_exponents = [1] * self.decomposition.n
        if len(work_exponents) != self.decomposition.n or len(contribution_exponents) != self.decomposition.n:
            raise ValueError('Incorrect number of exponents provided')
        if T_max and not L_max:
            L_max = np.Inf
        elif L_max and not T_max:
            T_max = np.Inf
        elif not L_max and not T_max:
            raise ValueError('Specify L_max or T_max')
        l = L_min
        tic_init = timeit.default_timer()
        C = 1
        while l < L_max and timeit.default_timer() - tic_init < T_max:
            if self.decomposition.is_external:
                self.decomposition.reset()
            self.app = _Approximation(self.decomposition)
            self.ad = _AdaptiveData(self.decomposition)
            tic = timeit.default_timer()
            rho = max([work_exponents[dim] / contribution_exponents[dim] for dim in range(self.decomposition.n)])
            mu = rho / (1 + rho)
            guess = C * np.exp(mu * l)
            def admissible(mi):  # Scale?
                return sum([mi[dim] * (work_exponents[dim] + contribution_exponents[dim]) 
                            for dim in range(self.decomposition.n)]) <= l  # #(all([dim<self.decomposition.n for dim in mi.active_dims()]) and
            mis = get_admissible_indices(admissible, self.decomposition.n)
            self.update_approximation(mis)
            if find_work_exponents:
                work_exponents = [self.get_work_exponent(dim) 
                                    if not (self.decomposition.is_bundled and self.decomposition.is_bundled(dim)) 
                                        and self.ad.work_estimator.ratios[dim] 
                                    else work_exponents[dim] 
                                    for dim in range(self.decomposition.n)]
            contribution_exponents = [self.get_contribution_exponent(dim) 
                                    if self.ad.contribution_estimator.ratios[dim] 
                                    else contribution_exponents[dim]
                                    for dim in range(self.decomposition.n)]
            real = timeit.default_timer() - tic
            C *= real / guess
            l += 1
        return work_exponents, contribution_exponents, mis
    
    @log_calls    
    def expand_nonadaptive(self, L, c_dim=-1, scale=1):
        '''
        Compute sparse approximation non-adaptively.
        
        Use estimates of contributions and work provided by self.decomposition to determine
        multi-indices to be added. 
        
        :param L: Threshold parameter
        :type L: Real
        :param c_dim: Bound on maximal active dimension
            Use if and only self.decomposition is infinite-dimensional
        :type c_dim: Integer
        :param scale: Make larger (>1) or smaller (<1) steps between different values of L
        :type scale: Positive real.
        '''
        if not math.isinf(self.decomposition.n):
            if c_dim != -1:
                raise ValueError('Use c_dim only for infinite-dimensional problems')
            else:
                c_dim = self.decomposition.n
        def admissible(mi):
            return self.ad.evaluator(mi) ** (-1) <= np.exp(scale * (L + 1e-6))
        try:
            mis = get_admissible_indices(admissible, c_dim)
        except KeyError:
            raise KeyError('Did you specify the work for all dimensions?')
        self.update_approximation(mis)
        
    @log_calls
    def expand_adaptive(self, c_steps=np.Inf, reset=False, T_max=np.Inf):
        '''
        Compute sparse approximation adaptively.
        
        To decide on the multi-index to be added at each step, estimates of contributions and work are maintained. 
        These estimates are based on neighbors that are already in the set :math:`\mathcal{I}`,
        unless they are specified in the arguments :code:`contribution_factors` and :code:`work_factors`.
        If user specifies in the arguments :code:`have_work_factor` and :code:`have_contribution_factor` 
        that only estimates for some of the :code:`n` involved parameters are available, 
        then the estimates from :code:`contribution_factor` and :code:`work_factor` for those parameters
        are combined with neighbor estimates for the remaining parameters.
        
        :param c_steps: Maximal number of new multi-indices.
        :type c_steps: Integer.
        :param reset: Specify whether computations should be redone at the end,
         using adaptively constructed multi-index set
        :type reset: Boolean.
        :param T_max: Maximal time (in seconds).
        :type T_max: Float
        '''
        if self.decomposition.is_bundled and not self.decomposition.has_contribution_model:  # WHY?
            raise ValueError('Cannot run adaptively when decomposition.is_bundled but not decomposition.has_contribution_model')
        if c_steps == np.Inf and T_max == np.Inf:
            raise ValueError('Specify either c_steps or T_max')
        if reset:
            ad_original = copy.deepcopy(self.ad)
        tic_init = timeit.default_timer()
        step = 0
        while step < c_steps:
            tic = timeit.default_timer()
            mi_update = max(self.app.mis.candidates, key=lambda mi: self.ad.evaluator(mi))
            if self.decomposition.is_bundled:
                self._expand_by_mi_or_bundle(indices.get_bundle(mi_update, self.app.mis, self.decomposition.is_bundled) + [mi_update])
            else:
                self._expand_by_mi_or_bundle(mi_update)
            if self.md.runtimes[mi_update] < (timeit.default_timer() - tic) / 2.:
                warnings.warn('Large overhead. Reparametrize decomposition?')
            if (timeit.default_timer() - tic_init > T_max or (timeit.default_timer() - tic_init > T_max / 2. and reset)):
                c_steps = step
            step += 1
        if reset:
            tic_init = timeit.default_timer()
            ad_final = copy.deepcopy(self.ad)
            mis = self.get_indices()
            self.decomposition.reset()
            self.ad = ad_original
            self.app = _Approximation(self.decomposition)
            self.md = _MetaData(self.decomposition)
            self.update_approximation(mis)
            self.ad = ad_final
        return timeit.default_timer() - tic_init
    
    @log_calls    
    def update_approximation(self, mis):
        '''
        Compute approximation based on multi-indices in mis.
        '''
        if self.decomposition.is_bundled:
            mis_or_miss = indices.get_bundles(mis, self.decomposition.is_bundled)
        else:
            mis_or_miss = mis
        for mi_or_mis in mis_or_miss:
            self._expand_by_mi_or_bundle(mi_or_mis)
                    
    def get_approximation(self):
        if self.decomposition.is_external:
            raise ValueError('Decomposition is stored externally')
        else:
            return sum([self.app.object_slices[si] for si in self.app.object_slices])   
    
    def get_work_exponent(self, dim):
        if not self.ad.work_estimator.dims_ignore(dim) and not (self.decomposition.is_bundled and self.decomposition.is_bundled(dim)):
            return self.ad.work_estimator.exponents[dim]
        else:
            raise KeyError('No work fit for this dimension')
        
    def get_contribution_exponent(self, dim):
        if not self.ad.contribution_estimator.dims_ignore(dim):
            return -self.ad.contribution_estimator.exponents[dim]
        else:
            raise KeyError('No contribution fit for this dimension') 
        
    def get_total_work_model(self):
        return sum(self.md.work_models._dict.values())
    
    def get_total_runtime(self):
        return sum(self.md.runtimes._dict.values())
    
    def get_indices(self):
        return copy.deepcopy(self.app.mis.mis)
    
    def plot_indices(self, dims=None, weighted=False, percentiles=1):
        '''
        :param dims: Dimensions that should be used for plotting
        :type dims: List of integers, length at most 3
        :param weighted: Determines size of points
        :type weighted: 'contribution' or 'work_model' or 'runtime' or 'contribution/work_model' or 'contribution/runtime'
        :param percentiles: Plot given number of weight-percentile groups in different colors
        :type perentiles: Integer
        '''
        if not dims:
            dims = list(self.app.mis.active_dims)
        if not weighted:
            weight_dict = None
        elif weighted == 'contribution':
            weight_dict = {mi: self.md.contributions[mi] for mi in self.get_indices()}
        elif weighted == 'work_model':
            assert(self.decomposition.has_work_model)
            weight_dict = {mi: self.md.work_models[mi] for mi in self.get_indices()}
        elif weighted == 'runtime':
            weight_dict = {mi: self.md.runtimes[mi] for mi in self.get_indices()}
        elif weighted == 'contribution/work_model':
            assert(self.decomposition.has_work_model)
            weight_dict = {mi:self.md.contributions[mi] / self.md.work_models[mi] for mi in self.get_indices()}
        elif weighted == 'contribution/runtime':
            weight_dict = {mi: self.md.contributions[mi] / self.md.runtimes[mi] for mi in self.get_indices()}
        else: 
            raise ValueError('Cannot use weights {}'.format(weighted))
        plot_indices(mis=self.get_indices(), dims=dims, weight_dict=weight_dict, N_q=percentiles) 
          
    @log_calls
    def _expand_by_mi_or_bundle(self, mi_or_bundle):
        '''
        Expands approximatoin by given multi-index or multi-index-bundle.
        
        :param mi_or_bundle: Single multi-index or single multi-index-bundle
        :return: Time required to compute decomposition term(s) if not self.dry_run
        '''
        if self.decomposition.is_bundled:
            mis_update = mi_or_bundle
            mi_update = mis_update[0]
        else:
            mis_update = [mi_or_bundle]
            mi_update = mi_or_bundle
        for mi in mis_update:
            self.app.mis.add(mi)
        if self.dry_run:
            return  
        external_work_factor = self.decomposition.work_factor(mi_or_bundle)
        tic = timeit.default_timer()
        if self.decomposition.is_external:
            output = self.decomposition.func(self.app.mis.mis)
        else:
            output = self.decomposition.func(mi_or_bundle)
        runtime = timeit.default_timer() - tic
        self.md.runtimes[mi_update] = runtime
        n_arg = sum(map(int, [not self.decomposition.is_external, self.decomposition.has_work_model, self.decomposition.has_contribution_model]))
        if n_arg == 1:
            output = [output]
        if not self.decomposition.is_external:
            self.app.object_slices[mi_update] = output[0]
            output = output[1:]
        if self.decomposition.has_work_model and self.decomposition.has_contribution_model:
            work_model, contribution = output
        if self.decomposition.has_work_model and not self.decomposition.has_contribution_model:
            work_model = output[0]
        if not self.decomposition.has_work_model and self.decomposition.has_contribution_model:
            contribution = output[0]
        if self.decomposition.has_contribution_model:
            if not self.decomposition.is_bundled:
                contribution = {mi_update: contribution}
        elif not self.decomposition.is_external:
            try:
                if self.decomposition.is_bundled:
                    contribution = {mi: self.app.object_slices[mi_update].norm(mi) for mi in mis_update}
                else:
                    contribution = {mi_update: self.app.object_slices[mi_update].norm()}
            except AttributeError:
                try:
                    if self.decomposition.is_bundled:
                        contribution = {mi: np.linalg.norm(self.app.object_slices[mi_update]) for mi in mis_update}
                    else:
                        contribution = {mi_update: np.linalg.norm(self.app.object_slices[mi_update])}
                except AttributeError:
                    pass
        if self.work_type == 'runtime':
            work = runtime
        else:
            work = work_model
        self.ad.work_estimator[mi_update] = work / external_work_factor  # CANNOT KEEP APART DIFFERENT CONTRIBUTIONS TO WORK IF BUNDLED>NEED WORK FACTORS FOR BUNDLED
        if self.decomposition.has_work_model:
            self.md.work_models[mi_update] = work_model
        try:
            for mi in mis_update:
                self.ad.contribution_estimator[mi] = contribution[mi] / self.decomposition.contribution_factor(mi)
                self.md.contributions[mi] = contribution[mi]
        except (KeyError,NameError): 
            pass  # Contribution could not be determined, contribution was never created
        if math.isinf(self.decomposition.n):
            self.ad.find_new_dims(mis_update, self.app.mis)
        return runtime

class Decomposition():
    def __init__(self, func=None, n=None, init_dims=None, next_dims=None, is_bundled=None,
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
        if hasattr(n, 'upper') and n.upper() in ['INF', 'INFINITY', 'INFTY'] or n is None:
            self.n = Inf   
        else:
            self.n = n
         
    def _set_next_dims(self, next_dims):
        if (not next_dims) and self.n == Inf:
            self.next_dims = lambda dim: [dim + 1] if dim + 1 not in self.init_dims else []
        else:
            self.next_dims = next_dims
        
    def _set_init_dims(self, init_dims):
        if isinstance(init_dims, int):
            self.init_dims = range(init_dims)
        elif hasattr(init_dims, '__contains__'):
            self.init_dims = init_dims
        elif init_dims is None:
            self.init_dims = [0]


class _Estimator(object):
    
    def __init__(self, dims_ignore, exponent_max, exponent_min, md_correction=None, init_exponents=None):
        self.quantities = {}
        self.dims_ignore = dims_ignore
        self.ratios = DefaultDict(lambda dim: [])
        init_exponents = init_exponents or (lambda dim:0)
        self.fallback_exponents = DefaultDict(init_exponents)  # USED AS PRIOR IN EXPONENT ESTIMATION AND AS INITIAL GUESS OF EXPONENT WHEN NO DATA AVAILABLE AT ALL
        self.exponents = DefaultDict(lambda dim: self.fallback_exponents[dim])
        self.reliability = DefaultDict(lambda dim: 1)
        self.md_correction = md_correction or (lambda dim: False)
        self.exponent_max = exponent_max
        self.exponent_min = exponent_min
        self.FIT_WINDOW = np.Inf
        self.active_dims=set()
        
    def _base_estimate(self,mi):
        q_neighbors = []
        q_neighbors.append(self.quantities[mi])
        for dim in self.active_dims:
            neighbor1=mi-kronecker(dim)
            if neighbor1 in self.quantities:
                q_neighbors.append(self.quantities[neighbor1]*np.exp(self.exponents[dim]))
            neighbor2=mi+kronecker(dim)
            if neighbor2 in self.quantities:
                q_neighbors.append(self.quantities[neighbor2]*np.exp(-self.exponents[dim]))
        return np.mean(q_neighbors)
        
    def set_fallback_exponent(self, dim, fallback_exponent):
        self.fallback_exponents[dim] = fallback_exponent
        
    def __contains__(self, mi):
        mi = mi.mod(self.dims_ignore)
        return mi in self.quantities
    
    def __setitem__(self, mi, q):
        self.active_dims.update(set(mi.active_dims()))
        mi = mi.mod(self.dims_ignore)
        self.quantities[mi] = q
        for dim in [dim for dim in mi.active_dims()]:
            mi_compare = mi - kronecker(dim)
            if self.quantities[mi_compare] > 0:
                ratio_new = q / self.quantities[mi_compare]
                if self.md_correction(dim) and mi_compare[dim] == 0:
                    ratio_new -= 1
                    if ratio_new < 0:
                        ratio_new = 0
            else:
                ratio_new = np.Inf  
            if len(self.ratios[dim]) < self.FIT_WINDOW:
                self.ratios[dim].append(ratio_new)
            else:
                self.ratios[dim] = self.ratios[dim][1:] + [ratio_new]
        self._update_exponents()
        
    def _update_exponents(self):
        for dim in self.ratios:
            ratios = self.ratios[dim]
            estimate = max(min(np.median(ratios), np.exp(self.exponent_max)), np.exp(self.exponent_min))
            c = len(ratios)
            self.exponents[dim] = (self.fallback_exponents[dim] + c * np.log(estimate)) / (c + 1.)
            self.reliability[dim] = 1. / (1 + 1/math.sqrt(c)+np.median([np.abs(ratio - estimate) for ratio in ratios]) / estimate)
            
    def __call__(self, mi):
        mi = mi.mod(self.dims_ignore)
        if mi in self.quantities:
            return self.quantities[mi]
        else:
            if mi.active_dims():
                q_neighbors = []
                w_neighbors = []
                for dim in mi.active_dims():
                    neighbor = mi - kronecker(dim)
                    q_neighbor = self._base_estimate(neighbor)*np.exp(self.exponents[dim])
                    q_neighbors.append(q_neighbor)
                    w_neighbors.append(self.reliability[dim])
                return sum([q*w for (q,w) in zip(q_neighbors, w_neighbors)])/sum(w_neighbors)
            else:
                return 1

class _AdaptiveData(object):
    
    def __init__(self, decomposition):
        self.WORK_EXPONENT_MAX = 10
        self.WORK_EXPONENT_MIN = 0
        self.CONTRIBUTION_EXPONENT_MAX = 0
        self.CONTRIBUTION_EXPONENT_MIN = -10
        self.decomposition = decomposition
        self.work_estimator = _Estimator(self.decomposition.have_work_factor,
                                      exponent_max=self.WORK_EXPONENT_MAX,
                                      exponent_min=self.WORK_EXPONENT_MIN,
                                      md_correction=self.decomposition.is_md)
        self.contribution_estimator = _Estimator(self.decomposition.have_contribution_factor,
                                              exponent_max=self.CONTRIBUTION_EXPONENT_MAX,
                                              exponent_min=self.CONTRIBUTION_EXPONENT_MIN,
                                              init_exponents=self.decomposition.kronecker_exponents)
              
    def evaluator(self, mi):
        contribution = self.contribution_estimator(mi) * self.decomposition.contribution_factor(mi)
        if self.decomposition.is_bundled:
            work = self.work_estimator(mi) * self.decomposition.work_factor([mi])
        else:
            work = self.work_estimator(mi) * self.decomposition.work_factor(mi)
        return contribution / work
    
    def find_new_dims(self, mis_update, dc_set):
        for mi in mis_update:
            if mi.is_kronecker() and not mi in dc_set:
                dim_trigger = mi.active_dims()[0]
                dims_new = self.decomposition.next_dims(dim_trigger)
                for dim in dims_new:
                    dc_set.add_dimensions([dim])
                    self.work_estimator.set_fallback_exponent(dim, self.work_estimator.exponents[dim_trigger])
                    if not self.decomposition.kronecker_exponents:
                        self.contribution_estimator.set_fallback_exponent(dim, self.contribution_estimator.exponents[dim_trigger])
    
class _Approximation(object):
    def __init__(self, decomposition):
        self.mis = DCSet(dims=decomposition.init_dims)
        self.object_slices = MultiIndexDict(decomposition.is_bundled)
        
class _MetaData(object):
    def __init__(self, decomposition):
        self.contributions = dict()
        self.work_models = MultiIndexDict(decomposition.is_bundled)
        self.runtimes = MultiIndexDict(decomposition.is_bundled)
        