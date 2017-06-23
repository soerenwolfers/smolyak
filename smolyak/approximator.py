'''
Sparse approximation using Smolyak's algorithm
'''
from __future__ import division
import numpy as np
import timeit
from smolyak.indices import  MultiIndexDict, get_admissible_indices, DCSet
import copy
import warnings
import math
from smolyak.misc.plots import plot_indices
from smolyak import indices
from smolyak.misc.logs import Log
from smolyak.misc.decorators import log_instance_method
from smolyak.misc.collections import DefaultDict
from smolyak.misc.np_tools import weighted_median

class Approximator(object):
    r'''
    Computes sparse approximation based on multi-index decomposition.
    
    Given a decomposition
    
    .. math::
    
        f_{\infty}=\sum_{\mathbf{k}\in\mathbb{N}^{n}} (\Delta f)(\mathbf{k}),
    
    approximations of the form
    
    .. math:: 
    
       \mathcal{S}_{\mathcal{I}}f:=\sum_{\mathbf{k}\in\mathcal{I}} (\Delta f)(\mathbf{k})
    
    are computed, where :math:`\mathcal{I}` is an efficiently chosen finite 
    multi-index set. 

    Currently supported choices for the construction of :math:`\mathcal{I}` are 
     :code:`expand_adaptive`, :code:`expand_nonadaptive` and :code:`continuation`
    '''
    def __init__(self, decomposition, work_type='runtime', log=None):
        r'''        
        :param decomposition: Decomposition of an approximation problem
        :type decomposition: Decomposition
        :param work_type: Optimize work model associated with decomposition or runtime
        :type work_type: String. 'work_model' or 'runtime' 
        :param log: Log
        '''
        self.decomposition = decomposition
        self.md = _MetaData(self.decomposition)
        self.ad = _AdaptiveData(self.decomposition)
        self.app = _Approximation(self.decomposition)
        self.work_type = work_type  # MOVE INTO ALGORITHMS?
        self.dry_run = not self.decomposition.is_external and not self.decomposition.func
        if self.work_type == 'runtime' and self.dry_run:
            raise ValueError('Cannot compute runtime without doing computations.')
        if self.work_type == 'work_model':
            assert(self.decomposition.has_work_model) 
        self.log = log or Log()   
            
    @log_instance_method   
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
    
    @log_instance_method    
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
        
    @log_instance_method
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
    
    @log_instance_method    
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
        return sum(self.md.work_models.dict.values())
    
    def get_total_runtime(self):
        return sum(self.md.runtimes.dict.values())
    
    def get_indices(self):
        return copy.deepcopy(self.app.mis.mis)
    
    def plot_indices(self, dims=None, weighted=False, percentiles=1):
        '''
        :param dims: Dimensions that should be used for plotting
        :type dims: List of integers, length at most 3
        :param weighted: Determines size of points
        :type weighted: 'contribution' or 'work_model' or 'runtime'
        :param percentiles: Plot given number of weight-percentile groups in different colors
        :type perentiles: Integer
        '''
        if not dims:
            dims = list(self.app.mis.active_dims)
        if not weighted:
            weight_dict = None
        elif weighted == 'contribution':
            weight_dict = self.md.contributions
        elif weighted == 'work_model':
            assert(self.decomposition.has_work_model)
            weight_dict = self.md.work_models
        elif weighted == 'runtime':
            weight_dict = self.md.runtimes
        elif weighted == 'contribution/work_model':
            assert(self.decomposition.has_work_model)
            weight_dict = {mi:self.md.contributions[mi] / self.md.work_models[mi] for mi in self.md.contributions}
        elif weighted == 'contribution/runtime':
            weight_dict = {mi: self.md.contributions[mi] / self.md.runtimes[mi] for mi in self.md.contributions}
        plot_indices(mis=self.get_indices(), dims=dims, weight_dict=weight_dict, N_q=percentiles) 
          
    @log_instance_method
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
        except NameError: 
            pass  # Contribution could not be determined, contribution was never created
        if math.isinf(self.decomposition.n):
            self.ad.find_new_dims(mis_update, self.app.mis)
        return runtime

class _Estimator(object):
    
    def __init__(self, dims_ignore, exponent_max, exponent_min, is_md, init_exponents=None):
        self.quantities = {}
        self.dims_ignore = dims_ignore
        self.ratios = DefaultDict(lambda dim: [])
        init_exponents = init_exponents or (lambda dim:0)
        self.fallback_exponents = DefaultDict(init_exponents)  # USED AS PRIOR IN EXPONENT ESTIMATION AND AS INITIAL GUESS OF EXPONENT WHEN NO DATA AVAILABLE AT ALL
        self.exponents = DefaultDict(lambda dim: self.fallback_exponents[dim])
        self.reliability = DefaultDict(lambda dim: 1)
        self.is_md = is_md
        self.exponent_max = exponent_max
        self.exponent_min = exponent_min
        self.FIT_WINDOW = np.Inf
        
    def set_fallback_exponent(self, dim, fallback_exponent):
        self.fallback_exponents[dim] = fallback_exponent
        
    def __contains__(self, mi):
        mi = mi.mod(self.dims_ignore)
        return mi in self.quantities
    
    def __setitem__(self, mi, q):
        mi = mi.mod(self.dims_ignore)
        self.quantities[mi] = q
        for dim in [dim for dim in mi.active_dims()]:
            mi_compare = mi.copy()
            mi_compare[dim] = mi_compare[dim] - 1
            if self.quantities[mi_compare] > 0:
                ratio_new = q / self.quantities[mi_compare]
                if self.is_md(dim) and mi_compare[dim] == 0:
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
            self.reliability[dim] = 1. / (1 + np.median([np.abs(ratio - estimate) for ratio in ratios]) / estimate)
            
    def __call__(self, mi):
        mi = mi.mod(self.dims_ignore)
        if mi in self.quantities:
            return self.quantities[mi]
        else:
            if mi.active_dims():
                q_neighbors = []
                w_neighbors = []
                for dim in mi.active_dims():
                    neighbor = mi.copy()
                    neighbor[dim] = neighbor[dim] - 1
                    q_neighbor = self.quantities[neighbor] * np.exp(self.exponents[dim])
                    q_neighbors.append(q_neighbor)
                    w_neighbors.append(self.reliability[dim])
                return weighted_median(q_neighbors, w_neighbors)
            else:
                return 1

class _AdaptiveData(object):
    
    def __init__(self, decomposition):
        self.WORK_EXPONENT_MAX = 100
        self.WORK_EXPONENT_MIN = 0
        self.CONTRIBUTION_EXPONENT_MAX = 0
        self.CONTRIBUTION_EXPONENT_MIN = -100
        self.decomposition = decomposition
        self.work_estimator = _Estimator(self.decomposition.have_work_factor,
                                      exponent_max=self.WORK_EXPONENT_MAX,
                                      exponent_min=self.WORK_EXPONENT_MIN,
                                      is_md=self.decomposition.is_md)
        self.contribution_estimator = _Estimator(self.decomposition.have_contribution_factor,
                                              exponent_max=self.CONTRIBUTION_EXPONENT_MAX,
                                              exponent_min=self.CONTRIBUTION_EXPONENT_MIN,
                                              is_md=self.decomposition.is_md,
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
                if not hasattr(dims_new, '__contains__'):
                    dims_new = [dims_new]
                for dim in dims_new:
                    dc_set.candidates |= {indices.kronecker(dim)}
                    self.work_estimator.set_fallback_exponent(dim, self.work_estimator.exponents[dim_trigger])
                    if not self.decomposition.kronecker_exponents:
                        self.contribution_estimator.set_fallback_exponent(dim, self.contribution_estimator.exponents[dim_trigger])
    
class _Approximation(object):
    def __init__(self, problem):
        self.mis = DCSet()
        self.object_slices = MultiIndexDict(problem.is_bundled)
        
class _MetaData(object):
    def __init__(self, problem):
        self.contributions = dict()
        self.work_models = MultiIndexDict(problem.is_bundled)
        self.runtimes = MultiIndexDict(problem.is_bundled)
        