'''
Sparse approximation using Smolyak's algorithm
'''
from __future__ import division
import numpy as np
import timeit
from smolyak.indices import  MultiIndexDict, get_admissible_indices, DCSet, \
    kronecker
import copy
import warnings
import math
from smolyak import indices
from swutil.logs import Log
from swutil.decorators import log_calls
from swutil.collections import DefaultDict
from numpy import Inf
from swutil.validation import NotPassed, Positive, Integer, Float, validate_args, \
    Nonnegative, Instance, DefaultGenerator, Function, Iterable, Bool, Dict, \
    List, Equals, InInterval, Arg, Passed, In
import itertools
from swutil import plots
# TODO: Put online

class _Factor():
    @validate_args('multipliers>(~func,~dims) multipliers==n_acc',warnings=False)
    def __init__(self,
                 func:Function,
                 multipliers:Dict(value_spec=Positive & Float),
                 n:Positive & Integer,
                 dims:Function(value_spec=Bool)=lambda dim: True,
                 bundled:Bool=False,
                 #have_all:Bool=False,
                 #have_none:Bool=False,
                 ):
        self.bundled = bundled
        if Passed(multipliers):
            self.multipliers = multipliers
            func = lambda mi: np.prod([
                    self.multipliers[dim] ** mi[dim]
                    for dim in self.multipliers
            ])
            if self.bundled:
                self.func= lambda mis: sum(func(mi) for mi in mis)
            else:
                self.func = func
            self.have_none = False
            if math.isinf(n):
                self.have_all = False
            else:
                self.have_all = all(dim in self.multipliers for dim in range(n))
            self.dims = self.multipliers.__contains__
        else:
            self.multipliers = {}
            #self.have_all = have_all
            #self.have_non = have_none
            self.func = func
            self.dims = dims 
    def __call__(self, *args):
        return self.func(*args)
    
class WorkFunction(_Factor):
    pass

class ContributionFunction(_Factor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bundled:
            raise ValueError('Contribution function cannot be bundled')

class Decomposition():
    @validate_args('~work_multipliers|~work_function',
                   '~contribution_multipliers|~contribution_function',
                   'bundled_dims>bundled',
                   Arg('init_dims|next_dims', Passed) > Arg('n_acc', Equals(math.inf)),
                   warnings=False)
    def __init__(self,
                 func:Function,
                 n:Positive & Integer,
                 work_multipliers:Dict(value_spec=InInterval(l=1, lo=False), lenience=2),
                 work_function:Instance(WorkFunction),
                 contribution_multipliers:Dict(value_spec=InInterval(l=0, lo=True, r=1, ro=True), lenience=2),
                 contribution_function:Instance(ContributionFunction),
                 returns_work:Bool=False,
                 returns_contributions:Bool=False,
                 init_dims:List(Nonnegative & Integer, lenience=2)=NotPassed,
                 next_dims:Function=NotPassed,
                 bundled:Bool=False,
                 bundled_dims:Function(value_spec=Bool) | List(value_spec=Integer)=NotPassed,
                 #is_md:Bool=False,
                 kronecker_exponents:Function(value_spec=Nonnegative & Float)=NotPassed,
                 stores_approximation:Bool=False,
                 reset:Function=NotPassed):
        r'''        
        :param func: Computes decomposition terms.
            In the most basic form, a single multi-index is passed to 
            :code:`func` and a single value, which represents the corresponding 
            decomposition term and supports vector space operations, is expected. 
            
            If :code:`returns_work == True`, :code:`func` must return (value,work) (or only work if stores_approximation),
            where work is a single float representing some abstract type of work
            that was required for the computation of value. Otherwise, algorithms
            are guided by runtime.
            
            If :code:`returns_contributions == True`, return (value,contribution) (or only 
            contribution if stores_approximation), where contribution is:
                *a single float if not stores_approximation and not bundled_dims or else
                *a dictionary  {(mi,contribution(float) for mi in bundle}
                    where bundle is the multi-index set that has been passed
            Otherwise, the return values will be normed by means of np.linalg.norm to
            assess their contribution, or they may implement norm() methods. 
            
            If both :code:`returns_contributions == True` and :code:`returns_work`
            then :code:`func` must return (value,work,contribution) (or only (work,contribution) in case of external storage)
            
            If :code:`bundled == True`, then :code:`func` is passed an iterable of multi-indices
            (which agree in the dimensions for which :code:`bundled_dims` returns True) must return 
            the sum of the corresponding decomposition elements.
            This may be useful for parallelization, or for problems where joint
            computation of decomposition elements is analytically more efficient.
            
            If :code:`stores_approximation == True`, no n_results is required and decomposition terms
            are expected to be stored by :code:`func` itself. Each call 
            will contain the full multi-index set representing the current approximation.
            
        :type func:  Function.
        :param n_acc: Number of discretization parameters
        :type n_acc: Integer, including math.inf (or 'inf') 
        :param work_multipliers: Specify factor by which work increases if index is increased in a given dimension
        :type work_multipliers: Dict
        :param work_function: If work is more complex, use work_function instead of work_multipliers to compute
            expected work associated with a given multi-index
        :type work_function: Function MultiIndex->Positive reals
        :param contribution_multipliers: Specify factor by which contribution decreases if index is increased in a given dimension
        :type work_multipliers: Dict
        :param work_function: If contribution is more complex, use contribution_function instead of contribution_multipliers to compute
            expected contribution of a given multi-index
        :type work_function: Function MultiIndex->Positive reals
        :param returns_work: Does the decomposition come with its own cost specification?
        :type returns_work: Boolean
        :param returns_contributions: Does the decomposition come with its own contribution specification? 
        :type returns_contributions: Boolean  
        :param init_dims: Initial dimensions used to create multi-index set. Defaults to :code:`range(n_acc)`. However,
            for large or infinite `n_acc`, it may make sense (or be necessary) to restrict this initially. 
            Each time a multi-index with non-zero entry in one of these initial dimensions, say j,  is selected, 
            new dimensions are added, according to output of :code:`next_dims(j)`
        :type init_dims: Integer or list of integers
        :param next_dims: Assigns to each dimension a list of child dimensions (see above)
        :type next_dims: :math:`\mathbb{N}\to 2^{\mathbb{N}}`
        :param: bundled: To be used when the decomposition terms cannot be computed independent of 
           each other. In this case, :code:`func` is called with subsets of :math:`\mathcal{I}` whose entries agree
           except for the dimensions specified in bundled_dims
        :type bundled: Boolean
        :param bundled_dims: Specifies dimensions used for bundling
        :type bundled_dims: :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param kronecker_exponents: For infinite dimensional problems, the 
            contribution of Kronecker multi-index e_j is estimated as exp(kronecker_exponents(j))
        :type kronecker_exponents: Function from integers to negative reals  
        :param stores_approximation: Specifies whether approximation is computed externally. If True, :code:`func` is called multiple times 
            with a SparseIndex as argument (or a list of SparseIndices if :code:`bundled_dims` is True as well) and must collect the associated
            decomposition terms itself. If False, :code:`func` is also called multiple times with a SparseIndex (or a list of SparseIndices)
            and must return the associated decomposition terms, which are then stored within the SparseApproximator instance
        :type stores_approximation: Boolean
        :param reset: If stores_approximation, this function will reset the externally stored approximation
        :type reset: Function
        '''
        self.func = func
        self.n_acc = n
        self.returns_work = returns_work
        self.returns_contributions = returns_contributions
        self.stores_approximation = stores_approximation
        self.kronecker_exponents = kronecker_exponents
        self.reset = reset
        #self.is_md = is_md
        if math.isinf(self.n_acc):
            if not init_dims:
                init_dims = [0]
            self.init_dims = init_dims
            self._set_next_dims(next_dims)  
        else:
            self.init_dims = range(self.n_acc)
        self._set_is_bundled(bundled, bundled_dims)
        self._set_work(work_multipliers, work_function)
        self._set_contribution(contribution_multipliers, contribution_function)
        
    def _set_work(self, work_multipliers, work_function):
        if work_multipliers:
            self.work_function = WorkFunction(multipliers=work_multipliers, n=self.n_acc,bundled=self.bundled)
        elif work_function:
            if work_function.bundled==self.bundled:
                self.work_function = work_function
            elif work_function.bundled and not self.bundled:
                raise ValueError("Work function cannot act on bundles when computations don't")
            elif self.bundled and not work_function.bundled:
                self.work_function = WorkFunction(func = lambda mis: sum(work_function(mi) for mi in mis),dims = work_function.dims,bundled=True)
        else:
            self.work_function = WorkFunction(func=lambda _: 1, dims=lambda dim: False, bundled=self.bundled)
        if self.bundled!=self.work_function.bundled:
            raise ValueError('If computations are bundled, work function must act on bundles as well (and conversely)')
        
    def _set_contribution(self, contribution_multipliers, contribution_function):
        if contribution_multipliers:
            self.contribution_function = ContributionFunction(multipliers=contribution_multipliers, n=self.n_acc,bundled = False)
        elif contribution_function:
            self.contribution_function = contribution_function
        else:
            self.contribution_function = ContributionFunction(func=lambda _: 1, dims=lambda dim:False, bundled=False)

    def _set_is_bundled(self, bundled, bundled_dims):
        self.bundled = bundled
        if hasattr(bundled_dims, '__contains__'):
            self.bundled_dims = lambda dim: dim in bundled_dims
        else: 
            self.bundled_dims = bundled_dims
         
    def _set_next_dims(self, next_dims):
        if (not next_dims) and self.n_acc == Inf:
            self.next_dims = lambda dim: [dim + 1] if dim + 1 not in self.init_dims else []
        else:
            self.next_dims = next_dims

class SparseApproximation():
    r'''
    Sparse approximation based on multi-index decomposition.

    Given a decomposition of a vector :math:`f_{\infty}` as    
    .. math::
    
        f_{\infty}=\sum_{\mathbf{k}\in\mathbb{N}^{n_acc}} (\Delta f)(\mathbf{k}),
        
    this class computes and stores approximations of the form
    .. math:: 
    
       \mathcal{S}_{\mathcal{I}}f:=\sum_{\mathbf{k}\in\mathcal{I}} (\Delta f)(\mathbf{k}),
    
    for finite index sets :math:`\mathcal{I}\subset\mathbb{R}^n_acc`. 

    Currently supported ways to specify :math:`\mathcal{I}` are:
     :code:`update_approximation`, which requires passing the multi-index set,
     :code:`expand_adaptive`, which constructs the set adaptively, one multi-index at a time,
     :code:`expand_apriori`, which constructs the set based on a-priori knowledge about work and contribution of the decomposition terms, and
     :code:`expand_continuation`, which constructs the set by a combination of first learning the behavior of work and contribution, and then using this knowledge to create optimal sets.
    '''
    @validate_args(warnings=False)
    def __init__(self, decomposition:Instance(Decomposition), log:Instance(Log)=DefaultGenerator(lambda: Log(print_filter=False))):
        r'''        
        :param decomposition: Decomposition of an approximation problem
        :type decomposition: Decomposition
        :param log: Log object used for logging
        '''
        self.decomposition = decomposition
        self.log = log
        self.reset()
        
    @validate_args('L|T', warnings=False)
    def expand_continuation(self , L:Positive&Integer=np.Inf, T:Positive&Float=np.Inf, L_init:Nonnegative&Integer=2,reset:Bool=True):
        '''
        Compute sparse approximation adaptively by using increasing multi-index 
        sets and determining at each step the next set by fitting contribution
        and work parameters. 
        
        :param L: Maximal level
        :type L: Integer
        :param T: Maximal runtime
        :type T: Positive real.
        :param L_init: Initial level
        :type L_init: Integer
        '''
        tic_init = timeit.default_timer()
        if self.decomposition.stores_approximation and not self.decomposition.reset:
                raise ValueError('If approximation is stored externally, decomposition needs to specify reset function')
        work_exponents, contribution_exponents = np.ones((2, self.decomposition.n_acc))  
        C = 1
        rho = lambda: max(work_exponents / contribution_exponents)
        mu = lambda: rho() / (1 + rho())
        estimated_time = lambda l : C * np.exp(mu() * l)  # only correct with the scaling below
        work_estimator = self.storage.work_model_estimator if self.decomposition.returns_work else self.storage.runtime_estimator
        done_something = False
        for l in (range(L_init,L) if not math.isinf(L) else itertools.count(L_init)):
            for i in range(self.decomposition.n_acc):
                if i not in self.decomposition.work_function.multipliers and work_estimator.ratios[i]:
                    work_exponents[i] = self._get_work_exponent(i)
                if i not in self.decomposition.work_function.multipliers and self.storage.contribution_estimator.ratios[i]:
                    contribution_exponents[i] = self._get_contribution_exponent(i)
            if reset: 
                self.reset()
            tic_level = timeit.default_timer()
            scale = min(work_exponents[dim] + contribution_exponents[dim] for dim in range(self.decomposition.n_acc))
            mis = indices.simplex(L=l, weights=(work_exponents + contribution_exponents) / scale, n=self.decomposition.n_acc)
            if not set(mis)<set(self.storage.mis.mis):#in particular, always when reset==True
                self.expand_by_indices(mis)
                done_something = True
            observed_time = timeit.default_timer() - tic_level
            C *= observed_time / estimated_time(l)
            if (timeit.default_timer() - tic_init) + estimated_time(l + 1) > T:
                break
        if not done_something:
            warnings.warn("Call of expand_apriori didn't end up expanding multi-index set")    
        #return work_exponents, contribution_exponents
      
    @validate_args('L^T','reset>T', warnings=False)
    def expand_apriori(self, L:Nonnegative&Integer=NotPassed, scale:Positive&Float=1,T:Positive&Float=NotPassed,reset:Bool = NotPassed):
        '''
        Compute sparse approximation non-adaptively.
        
        Use estimates of contributions and work provided by self.decomposition to determine
        multi-indices to be added. 
        
        :param L: Threshold parameter
        :type L: Real
        :param scale: Make larger (>1) or smaller (<1) steps between different values of L
        :type scale: Positive real.
        '''
        #if not self.decomposition.work_function.have_all or not self.decomposition.contribution_function.have_all:
        #    raise ValueError('Cannot run nonadaptively unless work is known for all dimensions.')
        tic_init = timeit.default_timer()
        if reset and self.decomposition.stores_approximation and not self.decomposition.reset:
            raise ValueError('If approximation is stored externally, decomposition needs to specify reset function')
        def admissible(mi,l):
                return self.storage.profit_estimate(mi) ** (-1) <= np.exp(scale * (l + 1e-6))
        done_something = False
        for l in ([L] if Passed(L) else itertools.count()):
            if reset:
                self.reset()
            tic_level = timeit.default_timer()
            try:
                mis = get_admissible_indices(lambda mi: admissible(mi,l), self.decomposition.n_acc)
            except KeyError:
                raise KeyError('Did you specify the work for all dimensions?')
            if not mis < self.storage.mis.mis:
                self.expand_by_indices(mis)
                done_something = True
            if Passed(T) and (timeit.default_timer() - tic_init) + (timeit.default_timer() - tic_level) > T:#Crude estimate: Next level as long as current one
                break
        if not done_something:
            warnings.warn("Call of expand_apriori didn't end up expanding multi-index set")
        
    @validate_args('N|T', warnings=False)
    def expand_adaptive(self, N:Positive & Integer = NotPassed, T:Positive & Float = NotPassed, reset:Bool=False):
        '''
        Compute sparse approximation adaptively.
        
        To decide on the multi-index to be added at each step, estimates of contributions and work are maintained. 
        These estimates are based on neighbors that are already in the set :math:`\mathcal{I}`,
        unless they are specified in :code:`contribution_factors` and :code:`work_factors`.
        If user defines :code:`have_work_factor` and :code:`have_contribution_factor` 
        that only estimates for some of the :code:`n_acc` involved parameters are available, 
        then the estimates from :code:`contribution_factor` and :code:`work_factor` for those parameters
        are combined with neighbor estimates for the remaining parameters.
        
        :param N: Maximal number of new multi-indices.
        :type N: Integer.
        :param T: Maximal time (in seconds).
        :type T: Float
        :param reset: Specify whether computations should be redone at the end,
            using adaptively constructed multi-index set
        :type reset: Boolean.
        '''
        if reset and self.decomposition.stores_approximation and not self.decomposition.reset:
            raise ValueError('Cannot reset: Decomposition stores approximation but does not specify reset function')
        if self.decomposition.bundled and not self.decomposition.returns_contributions:  # WHY?
            raise ValueError('Cannot run adaptively when decomposition.bundled but not decomposition.returns_contributions')
        tic_init = timeit.default_timer()
        for _ in (range(N) if Passed(N) else itertools.count()):
            tic = timeit.default_timer()
            mi_update = max(self.storage.mis.candidates, key=lambda mi: self.storage.profit_estimate(mi))
            self._expand(mi_update)
            if self.storage.runtimes[mi_update] < (timeit.default_timer() - tic) / 2.:
                warnings.warn('Large overhead. Reparametrize decomposition?')
            if Passed(T) and (timeit.default_timer() - tic_init > T or (timeit.default_timer() - tic_init > T / 2. and reset)):
                break
        if reset:
            mis = self.get_indices()
            self.reset()
            self.expand_by_indices(mis)
    
    @log_calls    
    def reset(self):
        if Passed(self.decomposition.reset):
            self.decomposition.reset()
        self.storage = _Storage(self.decomposition)
        
    @log_calls    
    def expand_by_indices(self, mis):
        '''
        Expand approximation by set of multi-indices.
        
        :param mis: Multi-indices to add to approximation
        :type mis: Iterable of multi-indices
        '''
        if self.decomposition.bundled:
            miss = indices.get_bundles(mis, self.decomposition.bundled_dims)
            not_bundled_dims = lambda dim: not self.decomposition.bundled_dims(dim)
            key = lambda mis: mis[0].restrict(not_bundled_dims)
            miss = sorted(miss, key=key)
            for mis in miss:
                self._expand(mis_update=mis)
        else:
            for mi in mis:
                self._expand(mi_update=mi)
                    
    def get_approximation(self):
        if self.decomposition.stores_approximation:
            raise ValueError('Decomposition is stored externally')
        else:
            return sum([self.storage.object_slices[mi] for mi in self.storage.object_slices])   
    
    def _get_work_exponent(self, dim):
        if not self.decomposition.returns_work:
            raise ValueError('Decomposition does not provide abstract work model')
        if not self.storage.work_model_estimator.dims_ignore(dim):
                return self.storage.work_model_estimator.exponents[dim]
        else:
            raise KeyError('No work fit for this dimension')
    
    def _get_runtime_exponent(self, dim):
        if not self.storage.runtime_estimator.dims_ignore(dim):
            return self.storage.runtime_estimator.exponents[dim]
        else:
            raise KeyError('No runtime fit for this dimension')
       
    def _get_contribution_exponent(self, dim):
        if not self.storage.contribution_estimator.dims_ignore(dim):
            return -self.storage.contribution_estimator.exponents[dim]
        else:
            raise KeyError('No contribution fit for this dimension') 
    
    def get_runtime_multiplier(self, dim):
        return np.exp(self._get_runtime_exponent(dim))
    
    def get_work_multiplier(self, dim):
        return np.exp(self._get_work_exponent(dim))
    
    def get_contribution_multiplier(self, dim):
        return np.exp(-self._get_contribution_exponent(dim))
    
    def get_total_work_model(self):
        return sum(self.storage.work_models._dict.values())
    
    def get_total_runtime(self):
        return sum(self.storage.runtimes._dict.values())
    
    def get_indices(self):
        return copy.deepcopy(self.storage.mis.mis)
    
    @validate_args(warnings=False)
    def plot_indices(self, dims:Iterable, weights:In('contribution','work_model','runtime','contribution/work_model','contribution/runtime',False)=False, percentiles:Positive & Integer=4):
        '''
        :param dims: Dimensions that should be used for plotting
        :type dims: List of integers, length at most 3
        :param weights: Determines size of points
        :type weights: 'contribution' or 'work_model' or 'runtime' or 'contribution/work_model' or 'contribution/runtime'
        :param percentiles: Plot given number of weight-percentile groups in different colors
        :type perentiles: Integer
        '''
        if NotPassed(dims):
            dims = list(self.storage.mis.active_dims)
        if not weights:
            percentiles = 1
            weight_dict = None
        elif weights == 'contribution':
            weight_dict = {mi: self.storage.contributions[mi] for mi in self.get_indices()}
        elif weights == 'work_model':
            if not self.decomposition.returns_work:
                raise ValueError('Decomposition does not provide abstract work model')
            weight_dict = {mi: self.storage.work_models[mi] for mi in self.get_indices()}
        elif weights == 'runtime':
            weight_dict = {mi: self.storage.runtimes[mi] for mi in self.get_indices()}
        elif weights == 'contribution/work_model':
            assert(self.decomposition.returns_work)
            weight_dict = {mi:self.storage.contributions[mi] / self.storage.work_models[mi] for mi in self.get_indices()}
        elif weights == 'contribution/runtime':
            weight_dict = {mi: self.storage.contributions[mi] / self.storage.runtimes[mi] for mi in self.get_indices()}
        else: 
            raise ValueError('Cannot use weights {}'.format(weights))
        plots.plot_indices(mis=self.get_indices(), dims=dims, weight_dict=weight_dict, N_q=percentiles) 
          
          
    @log_calls
    def _expand(self, mi_update=NotPassed,mis_update=NotPassed):
        '''
        Expands approximation by given multi-index or multi-index-bundle.
        
        :param mi: Single multi-index to add
        :param mis: Bundle of multi-indices to add
        '''
        work_model = None
        contribution = None
        object_slice = None
        if self.decomposition.bundled:
            if NotPassed(mi_update):
                mi_update = mis_update[0]
            else:
                mis_update = indices.get_bundle(mi_update, self.storage.mis, self.decomposition.bundled_dims) + [mi_update]
        else:
            if Passed(mis_update):
                raise ValueError('Only specify mi_update')
            mis_update = [mi_update]
        if not self.decomposition.stores_approximation and not self.decomposition.func:  # Dry run
            return  
        if self.decomposition.bundled:
            external_work_factor = self.decomposition.work_function(mis_update) # Want to be able to provide work function with whole bundle without checking what is actually new and then asking each of those separately, thats why it is required that work_function covers all bundled dimensions
        else:
            external_work_factor = self.decomposition.work_function(mi_update)
        self.storage.expand(mis_update)
        
        if self.decomposition.stores_approximation:
            argument = self.storage.mis.mis
        else:
            argument = mis_update if self.decomposition.bundled else mi_update
        tic = timeit.default_timer()
        output = self.decomposition.func(argument) # Always provide full set, leave it to external to reuse computations or not 
        runtime = timeit.default_timer() - tic
        n_arg = sum(map(int, [not self.decomposition.stores_approximation, self.decomposition.returns_work, self.decomposition.returns_contributions]))  # Possible outputs: Decomposition term, work, contribution
        if n_arg == 1 and not isinstance(output, tuple):  # Allow user to not return tuples if not necessary
            output = [output]
        if not self.decomposition.stores_approximation:  # Decomposition term
            object_slice, output = output[0], output[1:]
            output = output[1:]
        if self.decomposition.returns_work and self.decomposition.returns_contributions: # Remaining 2 outputs
            work_model, contribution = output
        elif self.decomposition.returns_work and not self.decomposition.returns_contributions: # Remaining 1 output
            work_model = output[0]
        elif not self.decomposition.returns_work and self.decomposition.returns_contributions:  # Remaining 1 output
            contribution = output[0]
        if self.decomposition.returns_contributions:  # User decides what contribution means
            if not self.decomposition.bundled_dims:  # Allow user to not return dictionary if not_bundled (which means user is only passed single multi-index)
                contribution = {mi_update: contribution}
        elif not self.decomposition.stores_approximation:  # If approximation is stored here instead of by user, try to figure out contribution
            try:
                if self.decomposition.bundled_dims:  # If approximation is grouped into bundles, norm function must be able to divide contribution into single multi-indices
                    contribution = {mi: object_slice.norm(mi) for mi in mis_update}
                else:
                    contribution = {mi_update: object_slice.norm()}
            except AttributeError:  # User didn't implement .norm()
                try:
                    if self.decomposition.bundled_dims:  # TODO: Does it make sense to assign each term in bundle the same contribution? 
                        contribution = {mi: np.linalg.norm(object_slice) for mi in mis_update}
                    else:  # Only one new term in approximation
                        contribution = {mi_update: np.linalg.norm(object_slice)}
                except AttributeError:
                    pass
        if self.decomposition.returns_contributions:
            if contribution is None:
                raise ValueError("Decomposition didn't return contributions") 
            if set(contribution.keys())!=set(argument):
                raise ValueError('Contributions did not match multi-index set')
        if self.decomposition.returns_work and work_model == None:
            raise ValueError("Decomposition didn't return work")
        self.storage.update_estimates(mis_update, mi_update, object_slice, contribution, work_model, runtime, external_work_factor)

class _Estimator():
    
    def __init__(self, dims_ignore, exponent_max, exponent_min, md_correction=None, init_exponents=None):
        self.quantities = {}
        self.dims_ignore = dims_ignore
        self.ratios = DefaultDict(lambda dim: [])
        init_exponents = init_exponents or (lambda dim:0)
        self.fallback_exponents = DefaultDict(init_exponents)  # USED AS PRIOR IN EXPONENT ESTIMATION AND AS INITIAL GUESS OF EXPONENT WHEN NO DATA AVAILABLE AT ALL
        self.exponents = DefaultDict(lambda dim: self.fallback_exponents[dim])
        self.reliability = DefaultDict(lambda dim: 1)
        #self.md_correction = md_correction or (lambda dim: False)
        self.exponent_max = exponent_max
        self.exponent_min = exponent_min
        self.FIT_WINDOW = np.Inf
        self.active_dims = set()   
        
    def set_fallback_exponent(self, dim, fallback_exponent):
        self.fallback_exponents[dim] = fallback_exponent
        
    def __contains__(self, mi):
        mi = mi.mod(self.dims_ignore)
        return mi in self.quantities
    
    def __setitem__(self, mi, q):
        self.active_dims.update(set(mi.active_dims()))
        mi = mi.mod(self.dims_ignore)
        self.quantities[mi] = q  # This should not depend on the ignored dimension. It does in practice, but the override makes sense as it makes sure that up-to-date information is used
        for dim in [dim for dim in mi.active_dims()]:
            mi_compare = mi - kronecker(dim)
            if self.quantities[mi_compare] > 0:
                ratio_new = q / self.quantities[mi_compare]
                #if self.md_correction(dim) and mi_compare[dim] == 0:
                #    ratio_new -= 1
                #    if ratio_new < 0:
                #        ratio_new = 0
            else:
                ratio_new = np.Inf  
            if len(self.ratios[dim]) < self.FIT_WINDOW:
                self.ratios[dim].append(ratio_new)
            else:
                self.ratios[dim] = self.ratios[dim][1:] + [ratio_new]
        self._update_exponents()
        
    def _update_exponents(self):
        for dim in self.ratios:
            ratios = np.array(self.ratios[dim])
            estimate = max(min(np.median(ratios), np.exp(self.exponent_max)), np.exp(self.exponent_min))
            c = len(ratios)
            self.exponents[dim] = (self.fallback_exponents[dim] + c * np.log(estimate)) / (c + 1.)
            self.reliability[dim] = 1. / (1 + 1 / math.sqrt(c) + 10 * np.median(np.abs(ratios - estimate) / ratios))
    
    def _base_estimate(self, mi):
        q_neighbors = []
        q_neighbors.append(self.quantities[mi])
        for dim in self.active_dims:
            neighbor1 = mi - kronecker(dim)
            if neighbor1 in self.quantities:
                q_neighbors.append(self.quantities[neighbor1] * np.exp(self.exponents[dim]))
            neighbor2 = mi + kronecker(dim)
            if neighbor2 in self.quantities:
                q_neighbors.append(self.quantities[neighbor2] * np.exp(-self.exponents[dim]))
        return np.mean(q_neighbors)
      
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
                    try:
                        q_neighbor = self._base_estimate(neighbor) * np.exp(self.exponents[dim])#Could replace self._base_estimate[neighbor] by self.quantities[neighbor], but would be more prone to getting fooled by initial unimportant indices in some dimensions
                    except:
                        raise KeyError('Could not access required contribution or work estimate. Were they specified?')
                    q_neighbors.append(q_neighbor)
                    w_neighbors.append(self.reliability[dim])
                if sum(w_neighbors) > 0:
                    return sum([q * w for (q, w) in zip(q_neighbors, w_neighbors)]) / sum(w_neighbors)
                else:
                    return np.nan
            else:
                return 1

class _Storage():
    def __init__(self, decomposition):
        self.WORK_EXPONENT_MAX = 10
        self.WORK_EXPONENT_MIN = 0
        self.CONTRIBUTION_EXPONENT_MAX = 0
        self.CONTRIBUTION_EXPONENT_MIN = -10
        self.decomposition = decomposition 
        #IF decomposition is bundled, runtime_estimator only makes sense if runtime_function has all bundled dimensions . runtime_function may or not be bundled; if it is not it will be called for each index in bundled and the observed runtime will then be divided by the sum 
        # WORK_MODEL FUNCTION DOES NOT HAVE TO BE BUNDLED IF DECOMPOSITION IS. IF IT IS BUNDLED, IT DOES HAVE TO INCLUDE BUNDLED DIMS IN ITS DIMS
        # CONTRIBUTION FUNCTION CANNOT BE BUNDLED
        self.runtime_estimator = _Estimator(self.decomposition.work_function.dims,
                                      exponent_max=self.WORK_EXPONENT_MAX,
                                      exponent_min=self.WORK_EXPONENT_MIN)#md_correction=self.decomposition.is_md)
        self.runtimes = MultiIndexDict(lambda dim: self.decomposition.bundled and self.decomposition.bundled_dims(dim))
        self.contribution_estimator = _Estimator(self.decomposition.contribution_function.dims,
                                              exponent_max=self.CONTRIBUTION_EXPONENT_MAX,
                                              exponent_min=self.CONTRIBUTION_EXPONENT_MIN,
                                              init_exponents=self.decomposition.kronecker_exponents)
        if self.decomposition.returns_work:  # CHECK HOW WORK FUNCTION AND CONTIRBUTION FUNCTION ARE CALLED, POSSIBLE CALL THEM FOR ALL MI IN MIS UPDATE
            self.work_model_estimator = _Estimator(self.decomposition.work_function.dims,
                                      exponent_max=self.WORK_EXPONENT_MAX,
                                      exponent_min=self.WORK_EXPONENT_MIN)#md_correction=self.decomposition.is_md)
            self.work_models = MultiIndexDict(decomposition.bundled_dims)
        self.mis = DCSet(dims=decomposition.init_dims)
        self.object_slices = MultiIndexDict(decomposition.bundled_dims)
        self.contributions = dict()
    
    def expand(self,mis_update):
        if math.isinf(self.decomposition.n_acc):
            self._find_new_dims(mis_update)
        self.mis.add_many(mis_update)        
        
    def update_estimates(self, mis_update, mi_update, object_slice, contribution, work_model, runtime, external_work_factor):
        self.object_slices[mi_update] = object_slice
        self.runtimes[mi_update] = runtime
        if self.decomposition.returns_work:  # external_work_factor means work_model 
            self.work_model_estimator[mi_update] = work_model / external_work_factor  # Here lies reason why work factor is needed if bundled: Cannot keep different contribution to work apart else
            self.work_models[mi_update] = work_model    
        self.runtime_estimator[mi_update] = runtime / external_work_factor
        self.runtimes[mi_update] = runtime
        try:
            for mi in mis_update:
                self.contribution_estimator[mi] = contribution[mi] / self.decomposition.contribution_function(mi)
                self.contributions[mi] = contribution[mi]
        except (KeyError, NameError): 
            pass  # Contribution could not be determined, contribution was never created 
    
    def profit_estimate(self, mi):
        contribution = self.contribution_estimator(mi) * self.decomposition.contribution_function(mi)
        if self.decomposition.bundled:
            mi_to_bundle = lambda mi: indices.get_bundle(mi, self.mis, self.decomposition.bundled_dims) + [mi]
        else:
            mi_to_bundle = lambda mi: mi
        if self.decomposition.returns_work:
            work = self.work_model_estimator(mi) * self.decomposition.work_function(mi_to_bundle(mi))
            return contribution / work
        else:
            runtime = self.runtime_estimator(mi) * self.decomposition.work_function(mi_to_bundle(mi))
            return contribution / runtime
        
    def _find_new_dims(self, mis_update): 
        for mi in mis_update:
            if mi.is_kronecker() and not mi in self.mis:
                dim_trigger = mi.active_dims()[0]
                dims_new = self.decomposition.next_dims(dim_trigger)
                for dim in dims_new:
                    #if dim in self.mis.active_dims:#
                    #    break##mi had been added before! (->parts of mis_update were already in self.mis) 
                    self.mis.add_dimensions([dim])
                    self.runtime_estimator.set_fallback_exponent(dim, self.runtime_estimator.exponents[dim_trigger])
                    if self.decomposition.returns_work:
                        self.work_model_estimator.set_fallback_exponent(dim, self.work_model_estimator.exponents[dim_trigger])
                    if not self.decomposition.kronecker_exponents:
                        self.contribution_estimator.set_fallback_exponent(dim, self.contribution_estimator.exponents[dim_trigger])
        
