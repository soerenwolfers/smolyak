'''
Sparse approximation using Smolyak's algorithm
'''
from __future__ import division
import numpy as np
import timeit
from smolyak.sparse.dc_set import DCSet
from smolyak.sparse.sparse_index import SparseIndex, SparseIndexDict, get_admissible_indices
import copy
from smolyak.sparse.estimator import Estimator
import warnings
import math
from smolyak.misc.plots import plot_mis

class AdaptiveData(object):
    def __init__(self, problem):
        self.INIT_WORK_EXPONENT = 0
        self.INIT_CONTRIBUTION_EXPONENT = 0
        self.WORK_EXPONENT_MAX = 100
        self.WORK_EXPONENT_MIN = 0
        self.CONTRIBUTION_EXPONENT_MAX = 0
        self.CONTRIBUTION_EXPONENT_MIN = -100
        self.problem = problem
        self.work_estimator = Estimator(self.problem.have_work_factor,
                                      exponent_max=self.WORK_EXPONENT_MAX,
                                      exponent_min=self.WORK_EXPONENT_MIN,
                                      is_md=self.problem.is_md)
        self.contribution_estimator = Estimator(self.problem.have_contribution_factor,
                                              exponent_max=self.CONTRIBUTION_EXPONENT_MAX,
                                              exponent_min=self.CONTRIBUTION_EXPONENT_MIN,
                                              is_md=self.problem.is_md)
        self.candidates = {SparseIndex()}
        self.contribution_fallback_exponents = None  # ALLOWS FOR USER TO SPECIFY DECAY OF IMPORTANCE OF DIMENSIONS. THIS IS USED FOR INITIAL GUESS OF EXPONENT WHEN NEW DIMENSION IS ADDED TO      
            
    def get_update(self):
        values = dict()
        for mi in self.candidates:
            values[mi] = self.evaluator(mi)
        mi_update = max(self.candidates, key=lambda mi: values[mi])
        self.candidates -= {mi_update}
        return mi_update
              
    def evaluator(self, mi):
        contribution = self.contribution_estimator(mi) * self.problem.contribution_factor(mi)
        if self.problem.is_bundled:
            work = self.work_estimator(mi) * self.problem.work_factor([mi])
        else:
            work = self.work_estimator(mi) * self.problem.work_factor(mi)
        return contribution / work
    
    def find_new_dims(self, mis_update):
        def add_dims(dims, init_work_exponent, init_contribution_exponent):
                for dim in dims:
                    T = SparseIndex()
                    T[dim] = 1
                    self.candidates |= {T}
                    self.work_estimator.set_fallback_exponent(dim, init_work_exponent)
                    if self.contribution_fallback_exponents:  #
                        self.contribution_estimator.set_fallback_exponent(dim, self.contribution_fallback_exponents(dim))
                    else:
                        self.contribution_estimator.set_fallback_exponent(dim, init_contribution_exponent)
        if any(mi == SparseIndex() for mi in mis_update):
            add_dims(self.problem.init_dims, self.INIT_WORK_EXPONENT, self.INIT_CONTRIBUTION_EXPONENT)
        if math.isinf(self.problem.n):
            if self.problem.next_dims:
                for mi in [mi for mi in mis_update if mi.is_unit()]:
                    dim_trigger = mi.active_dims()[0]
                    dims_new = self.problem.next_dims(dim_trigger)
                    if not hasattr(dims_new, '__contains__'):
                        dims_new = [dims_new]
                    init_work_exponent = self.work_estimator.exponents[dim_trigger]
                    init_contribution_exponent = self.contribution_estimator.exponents[dim_trigger]
                    add_dims(dims_new, init_work_exponent, init_contribution_exponent)
    
class Approximation(object):
    def __init__(self, problem):
        self.mis = DCSet()
        self.object_slices = SparseIndexDict(problem.is_bundled)
        
class MetaData(object):
    def __init__(self, problem):
        self.contributions = dict()
        self.work_models = SparseIndexDict(problem.is_bundled)
        self.runtimes = SparseIndexDict(problem.is_bundled)

class Diary(object):
    def __init__(self, verbosity):
        self.entries = []
        self.verbosity = verbosity
    def write(self, text, rank=1):
        self.entries.append(text, rank)
    def read(self, verbosity):
        return [entry[0] for entry in self.entries if entry[1] <= verbosity]

class SparseApproximator(object):
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
    def __init__(self, problem, work_type='runtime', verbosity=0):
        r'''        
        :param problem: Approximation problem
        :type problem: ApproximationProblem
        :param work_type: Keep track of work model associated with problem or simple runtime
        :type work_type: Optional. String. 'work_model' or 'runtime' 
        '''
        self.problem = problem
        self.md = MetaData(self.problem)
        self.ad = AdaptiveData(self.problem)
        self.app = Approximation(self.problem)
        self.work_type = work_type  # MOVE INTO ALGORITHMS?
        if self.work_type == 'work_model':
            assert(self.problem.has_work_model) 
        self.diary = Diary(verbosity)
        
    def continuation(self , L_max=None, T_max=None, L_min=2, work_exponents=None, contribution_exponents=None, find_work_exponents=False):
        '''
        :param L_max: Maximal level
        :type L_max: Integer
        :param T_max: Maximal runtime
        :type T_max: Optional. Positive real.
        :param L_min: Initial level
        :type L_min: Optional. Integer
        :param work_exponents: Initial guess of work exponents
        :type work_exponents: Optional (unless problem.is_bundled). List of positive reals.
        :param contribution_exponents: Initial guess of contribution exponents
        :type contribution_exponents: optional. List of positive reals.
        :param find_work_exponents: Specifies whether work exponents should be fitted
        :type find_work_exponents: Optional. Boolean. 
        '''
        if self.problem.is_external and not self.problem.reset:
                raise ValueError('If approximation is stored externally, need to specify reset function')
        if not work_exponents:
            if self.problem.is_bundled:
                raise ValueError('Need to specify work exponents for is_bundled parameters.')
            else:
                work_exponents = [1] * self.problem.n
        if not contribution_exponents:
            contribution_exponents = [1] * self.problem.n
        if len(work_exponents) != self.problem.n or len(contribution_exponents) != self.problem.n:
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
            if self.problem.is_external:
                self.problem.reset()
            self.app = Approximation(self.problem)
            self.ad = AdaptiveData(self.problem)
            tic = timeit.default_timer()
            rho = max([work_exponents[dim] / contribution_exponents[dim] for dim in range(self.problem.n)])
            mu = rho / (1 + rho)
            guess = C * np.exp(mu * l)
            def admissible(mi):  # Scale?
                return sum([mi[dim] * (work_exponents[dim] + contribution_exponents[dim]) 
                            for dim in range(self.problem.n)]) <= l  # #(all([dim<self.problem.n for dim in mi.active_dims()]) and
            mis = get_admissible_indices(admissible, self.problem.n)
            self.expand_by_mis(mis)
            if find_work_exponents:
                work_exponents = [self.get_work_exponent(dim) 
                                    if not (self.problem.is_bundled and self.problem.is_bundled(dim)) 
                                        and self.ad.work_estimator.ratios[dim] 
                                    else work_exponents[dim] 
                                    for dim in range(self.problem.n)]
            contribution_exponents = [self.get_contribution_exponent(dim) 
                                    if self.ad.contribution_estimator.ratios[dim] 
                                    else contribution_exponents[dim]
                                    for dim in range(self.problem.n)]
            real = timeit.default_timer() - tic
            C *= real / guess
            l += 1
        return work_exponents, contribution_exponents, mis
        
    def expand_nonadaptive(self, L, c_dim=-1, scale=1):
        if not math.isinf(self.problem.n):
            if c_dim != -1:
                raise ValueError('Use c_dim only for infinite-dimensional problems')
            else:
                c_dim = self.problem.n
        def admissible(mi):
            return self.ad.evaluator(mi) ** (-1) <= np.exp(scale * (L + 1e-6))
        try:
            mis = get_admissible_indices(admissible, c_dim)
        except KeyError:
            raise KeyError('Did you specify the work for all parameters?')
        self.expand_by_mis(mis)
        
    def expand_adaptive(self, c_steps=None, reset=False, T_max=None, contribution_exponents=None):
        '''
        Compute sparse approximation adaptively.
        
        To decide on the multi-index to be added at each step, estimates of contributions and work are maintained. 
        These estimates are based on neighbors that are already in the set :math:`\mathcal{I}`,
        unless they are specified in the arguments :code:`contribution_factors` and :code:`work_factors`.
        If user specifies in the arguments :code:`have_work_factor` and :code:`have_contribution_factor` 
        that only estimates for some of the :code:`n` involved parameters are available, 
        then the estimates from :code:`contribution_factor` and :code:`work_factor` for those parameters
        are combined with neighbor estimates for the remaining parameters.
        
        :param c_steps: Maximal size of multi-index set.
        :type c_steps: Optional. Integer.
        :param reset: Specify whether computations should be redone at the end,
         using adaptively constructed multi-index set
        :type reset: Optional. Boolean.
        :param T_max: Maximal time (in seconds).
        :type T_max: Optional.
        :param contribution_exponents: For infinite dimensional problems, the 
        contribution of Kronecker multi-index e_j is estimated as exp(contribution_exponent(j))
        :type contribution_exponents: Function from integers to negative reals
        '''
        if contribution_exponents:
            self.ad.contribution_fallback_exponents = contribution_exponents
        if self.problem.is_bundled and not self.problem.is_external:  # WHY?
            raise ValueError('Cannot run adaptively when problem.is_bundled but not problem.is_external')
        if T_max and not c_steps:
            c_steps = np.Inf
        elif c_steps and not T_max:
            T_max = np.Inf
        elif not c_steps and not T_max:
            raise ValueError('Specify either c_steps or T_max')
        assert(c_steps > 0)
        if reset:
            ad_original = copy.deepcopy(self.ad)
        tic_init = timeit.default_timer()
        step = 0
        while step < c_steps:
            tic = timeit.default_timer()
            mi_update = self.ad.get_update()
            if self.problem.is_bundled:
                decomposition_work = self.__expand_by_mi_or_bundle([mi_update])
            else:
                decomposition_work = self.__expand_by_mi_or_bundle(mi_update)
            if decomposition_work < (timeit.default_timer() - tic) / 2.:
                warnings.warn('Organizational work large compared to computational work. Reparametrize problem?')
            if (timeit.default_timer() - tic_init > T_max or (timeit.default_timer() - tic_init > T_max / 2. and reset)):
                c_steps = step
            step += 1
        if reset:
            tic_init = timeit.default_timer()
            ad_final = copy.deepcopy(self.ad)
            mis = self.get_mis()
            self.problem.reset()
            self.ad = ad_original
            self.app = Approximation(self.problem)
            self.md = MetaData(self.problem)
            self.expand_by_mis(mis)
            self.ad = ad_final
        return timeit.default_timer() - tic_init
        
    def expand_by_mis(self, mis):
        if self.problem.is_bundled:
            mis_or_miss = self.__bundles_from_mis(mis)
        else:
            mis_or_miss = mis
        for mi_or_mis in mis_or_miss:
            self.__expand_by_mi_or_bundle(mi_or_mis)
                    
    def get_approximation(self):
        if self.problem.is_external:
            raise ValueError('Decomposition is stored externally')
        else:
            return sum([self.app.object_slices[si] for si in self.app.object_slices])   
    
    def get_work_exponent(self, dim):
        if not self.ad.work_estimator.dims_ignore(dim) and not (self.problem.is_bundled and self.problem.is_bundled(dim)):
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
    
    def get_mis(self):
        return copy.deepcopy(self.app.mis.mis)
    
    def plot_mis(self, dims=None, weighted=False, percentiles=1):
        '''
        :param dims: Dimensions that should be used for plotting
        :type dims: List of integers, length at most 3
        :param weighted: Determines size of points
        :type weighted: Optional. 'contribution' or 'work_model' or 'runtime'
        :param percentiles: Plot given number of weight-percentile groups in different colors
        :type perentiles: Optional. Integer
        '''
        if not dims:
            dims = list(self.app.mis.active_dims)
        if not weighted:
            weight_dict = None
        elif weighted == 'contribution':
            weight_dict = self.md.contributions
        elif weighted == 'work_model':
            assert(self.problem.has_work_model)
            weight_dict = self.md.work_models
        elif weighted == 'runtime':
            weight_dict = self.md.runtimes
        elif weighted == 'contribution/work_model':
            assert(self.problem.has_work_model)
            weight_dict = {mi:self.md.contributions[mi] / self.md.work_models[mi] for mi in self.md.contributions}
        elif weighted == 'contribution/runtime':
            weight_dict = {mi: self.md.contributions[mi] / self.md.runtimes[mi] for mi in self.md.contributions}
        plot_mis(mis=self.get_mis(), dims=dims, weight_dict=weight_dict, N_q=percentiles) 
            
    def __expand_by_mi_or_bundle(self, mi_or_bundle):
        '''
        Expands decomposition by given multi-index or multi-index-bundle.
        
        :param mi_or_bundle: Single multi-index or single multi-index-bundle
        :return: Time required to compute decomposition term(s)
        '''
        if self.problem.is_bundled:
            mis_update = mi_or_bundle
            mi_update = mis_update[0]
        else:
            mis_update = [mi_or_bundle]
            mi_update = mi_or_bundle
        self.ad.candidates |= set(self.app.mis.add_mi(mi_or_bundle))
        have_contributions = False
        if self.problem.is_external or self.problem.decomposition:
            external_work_factor = self.problem.work_factor(mi_or_bundle)
            if self.problem.is_external:
                tic = timeit.default_timer()
                output = self.problem.decomposition(mi_or_bundle)
                decomposition_work = timeit.default_timer() - tic
                if self.problem.has_work_model:
                    work_model = output[0]
                    contributions = output[1]
                else:
                    contributions = output
                contributions = {mis_update[i]: contribution for i, contribution in enumerate(contributions)}
                have_contributions = True
            elif self.problem.decomposition:
                tic = timeit.default_timer()
                output = self.problem.decomposition(mi_or_bundle)
                decomposition_work = timeit.default_timer() - tic
                if self.problem.has_work_model:
                    work_model = output[0]
                    self.app.object_slices[mi_update] = output[1]
                else:
                    self.app.object_slices[mi_update] = output
                try:
                    if self.problem.is_bundled:
                        contributions = {mi: self.app.object_slices[mi_update].norm(mi) for mi in mis_update}
                    else:
                        contributions = {mi_update: self.app.object_slices[mi_update].norm()}
                    have_contributions = True
                except AttributeError:
                    try:
                        if self.problem.is_bundled:
                            contributions = {mi: np.linalg.norm(self.app.object_slices[mi_update]) for mi in mis_update}
                        else:
                            contributions = {mi_update: np.linalg.norm(self.app.object_slices[mi_update])}
                        have_contributions = True
                    except AttributeError:
                        warnings.warn("Couldn't compute contribution estimate")
        if self.work_type == 'runtime':
            work = decomposition_work
        else:
            work = work_model
        self.ad.work_estimator[mi_update] = work / external_work_factor  # CANNOT KEEP APART DIFFERENT CONTRIBUTIONS TO WORK IF BUNDLED>NEED WORK FACTORS FOR BUNDLED
        self.md.runtimes[mi_update] = decomposition_work
        if self.problem.has_work_model:
            self.md.work_models[mi_update] = work_model
        if have_contributions:
            for mi in mis_update:
                self.ad.contribution_estimator[mi] = contributions[mi] / self.problem.contribution_factor(mi)
                self.md.contributions[mi] = contributions[mi]
        self.ad.find_new_dims(mis_update)
        return decomposition_work
    
    def __bundles_from_mis(self, mis):
            if not hasattr(mis, '__contains__'):
                mis = [mis]
            miss = []
            for mi in mis:
                bundle = [mi2 for mi2 in mis if mi.equal_mod(mi2, self.problem.is_bundled)]
                if not bundle in miss:
                    miss += [bundle]
            return miss
        
