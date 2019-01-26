'''
Sparse approximation using Smolyak's algorithm

Usage:
    1) Setup a function instance that computes elements in a multi-index
    decomposition (and possibly auxiliary information about work and 
    contribution of the computed terms).
    Here, it can be helpful to use MixedDifferences from the module indices
    which turns regular algorithms into the form required for sparse
    approximations. As a special case, function approximation using 
    least squares polynomial approximations is already implemented in
    PolynomialApproximator.
    
    n = 2
    f = lambda x: np.sin(np.prod(x,1))
    pa = PolynomialApproximator(f,domain = [[0,1]]**n)

    2) Pass that function to a SparseApproximator instance, along
    with information about work and runtime estimates, etc.
    In the case of PolynomialApproximator instances, additional information
    is optional.

    sa = SparseApproximator(pa)

    3) Use :code:`update_approximation` to perform the computations

    sa.update_approximation(T=10)
    pa.get_approximation().plot()

'''
import copy
import warnings
import math
from timeit import default_timer as timer
import itertools
import random
import collections

import numpy as np
from numpy import Inf
from swutil.logs import Log
from swutil.decorators import log_calls
from swutil.collections import DefaultDict
from swutil.validation import NotPassed, Positive, Integer, Float, validate_args, \
    Nonnegative, Instance, DefaultGenerator, Function, Iterable, Bool, Dict, \
    List, Equals, InInterval, Arg, Passed, In
from swutil import plots

from smolyak.indices import  MultiIndexDict, get_admissible_indices, MISet, \
    kronecker, MultiIndex, get_bundles, get_bundle
from smolyak import indices
from smolyak.applications.polynomials import PolynomialApproximator

class _Factor:
    @validate_args('multipliers>(~func,~dims) multipliers==n',warnings=False)
    def __init__(self,
                 func:Function,
                 multipliers:Dict(value_spec=Positive & Float),
                 n:Positive & Integer,
                 dims:Function(value_spec=Bool)=lambda dim: True,
                 bundled:Bool=False,
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

class _Decomposition:
    @validate_args(
        '~work_multipliers|~work_function',
        '~contribution_multipliers|~contribution_function',
        'bundled_dims>bundled',
        Arg('init_dims|next_dims', Passed) > Arg('n', Equals(math.inf)),
        warnings=False,
    )
    def __init__(
        self,
        Delta,
        n: Positive & Integer,
        work_multipliers: Dict(value_spec = InInterval(l = 1, lo = False), lenience = 2),
        work_function: Instance(WorkFunction),
        contribution_multipliers: Dict(value_spec = InInterval(l = 0, lo = True, r = 1, ro = True), lenience = 2),
        contribution_function: Instance(ContributionFunction),
        returns_work: Bool = False,
        returns_contributions: Bool = False,
        init_dims: List(Nonnegative & Integer, lenience = 2) = NotPassed,
        next_dims: Function = NotPassed,
        bundled: Bool = False,
        bundled_dims: Function(value_spec = Bool) | List(value_spec = Integer) = NotPassed,
        kronecker_exponents: Function(value_spec = Nonnegative & Float) = NotPassed,
        stores_approximation: Bool = False,
        structure=None,
        callback=None,
    ):
        r'''        
        :param Delta: Computes decomposition terms.
            In the most basic form, a single multi-index is passed to 
            :code:`Delta` and a single value, which represents the corresponding 
            decomposition term and supports vector space operations, is expected. 
            
            If :code:`returns_work == True`, :code:`Delta` must return (value,work) (or only work if stores_approximation),
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
            then :code:`Delta` must return (value,work,contribution) (or only (work,contribution) in case of external storage)
            
            If :code:`bundled == True`, then :code:`Delta` is passed an iterable of multi-indices
            (which agree in the dimensions for which :code:`bundled_dims` returns True) must return 
            the sum of the corresponding decomposition elements.
            This may be useful for parallelization, or for problems where joint
            computation of decomposition elements is analytically more efficient.
            
            If :code:`stores_approximation == True`, no n_results is required and decomposition terms
            are expected to be stored by :code:`Delta` itself. Each call 
            will contain the full multi-index set representing the current approximation.
            
        :type Delta:  Function.
        :param n: Number of discretization parameters
        :type n: Integer, including math.inf (or 'inf') 
        :param work_multipliers: Specify factor by which work increases if index is increased in a given dimension
        :type work_multipliers: Dict
        :param work_function: If work is more complex, use work_function instead of work_multipliers to compute
            expected work associated with a given multi-index
        :type work_function: Function MultiIndex->Positive reals
        :param contribution_multipliers: Specify factor by which contribution decreases if index is increased in a given dimension
        :type contribution_multipliers: Dict
        :param contribution_function: If contribution is more complex, use contribution_function instead of contribution_multipliers to compute
            expected contribution of a given multi-index
        :type contribution_function: Function MultiIndex->Positive reals
        :param returns_work: Are the deltas returned together with their own work specification?
        :type returns_work: Boolean
        :param returns_contributions: Are the deltas returned together with their own contribution specification?
        :type returns_contributions: Boolean  
        :param init_dims: Initial dimensions used to create multi-index set. Defaults to :code:`range(n)`. However,
            for large or infinite `n`, it may make sense (or be necessary) to restrict this initially. 
            Each time a multi-index with non-zero entry in one of these initial dimensions, say j,  is selected, 
            new dimensions are added, according to output of :code:`next_dims(j)`
        :type init_dims: Integer or list of integers
        :param next_dims: Assigns to each dimension a list of child dimensions (see above)
        :type next_dims: :math:`\mathbb{N}\to 2^{\mathbb{N}}`
        :param: bundled: To be used when the decomposition terms cannot be computed independent of 
           each other. In this case, :code:`Delta` is called with subsets of :math:`\mathcal{I}` whose entries agree
           except for the dimensions specified in bundled_dims
        :type bundled: Boolean
        :param bundled_dims: Specifies dimensions used for bundling
        :type bundled_dims: :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param kronecker_exponents: For infinite dimensional problems, the 
            contribution of Kronecker multi-index e_j is estimated as exp(kronecker_exponents(j))
        :type kronecker_exponents: Function from integers to negative reals  
        :param stores_approximation: Specifies whether approximation is computed externally. If True, :code:`Delta` is called multiple times 
            with a SparseIndex as argument (or a list of SparseIndices if :code:`bundled_dims` is True as well) and must collect the associated
            decomposition terms itself. If False, :code:`Delta` is also called multiple times with a SparseIndex (or a list of SparseIndices)
            and must return the associated decomposition terms, which are then stored within the SparseApproximator instance
        :type stores_approximation: Boolean
        :param structure: Used to enforce symmetries in approximating set.  
        :type structure: Function: multiindices->setsofmultiindices
        :param callback: If this returns a Trueish value, approximation is stopped (use for accuracy based approximations)
        :type callback: Function(): Bool
        '''
        if isinstance(Delta,PolynomialApproximator):
            if Passed(n) or returns_work:
                raise ValueError('Do not specify `n` or `returns_work` to SparseApproximator for polynomial approximation')  
            self.Delta = Delta.update_approximation
            self.n = Delta.n_acc+Delta.n
            _work_function = WorkFunction(func = Delta.estimated_work, dims = lambda n: (True if Passed(work_function) else n>=Delta.n_acc),bundled=True)
            self.returns_work = True 
            self.returns_contributions = True
            self.stores_approximation = True
            self.kronecker_exponents = kronecker_exponents 
            self._set_is_bundled(True,Delta.bundled_dims)
            self._set_work(work_multipliers,_work_function)
            self._set_contribution(contribution_multipliers,contribution_function)
            if isinstance(structure,str):
                if structure.lower() == 'td':
                    structure = lambda mi: [mi.restrict(lambda n:n<Delta.n_acc) + mi2.shifted(Delta.n_acc) 
                        for mi2 in indices.simplex(n = Delta.n,L=mi.mod(lambda n:n<Delta.n_acc).sum())]
                elif structure.lower() == 'pd':
                    structure = lambda mi: [mi.restrict(lambda n:n<Delta.n_acc) + mi2.shifted(Delta.n_acc) 
                        for mi2 in indices.rectangle(n = Delta.n,L=max(mi.mod(lambda n:n<Delta.n_acc)))]
                elif structure.lower() == 'sym':
                    from sympy.utilities.iterables import multiset_permutations
                    def structure(mi):
                        mim = mi.mod(lambda n:n<Delta.n_acc)
                        if mim==MultiIndex():
                            return []
                        else:
                            ret =  [mi.restrict(lambda n:n<Delta.n_acc) + mi2.shifted(Delta.n_acc)
                                for mi2 in [MultiIndex(perm) for perm in multiset_permutations(mi.mod(lambda n:n<Delta.n_acc).full_tuple())]]
                            return ret
        else:
            self.Delta = Delta
            self.n = n
            self.returns_work = returns_work
            self.returns_contributions = returns_contributions
            self.stores_approximation = stores_approximation
            self.kronecker_exponents = kronecker_exponents
            self._set_is_bundled(bundled, bundled_dims)
            self._set_work(work_multipliers, work_function)
            self._set_contribution(contribution_multipliers, contribution_function)
        self.callback = callback or (lambda: False)
        self.structure = structure or (lambda mi: set())
        if math.isinf(self.n):
            if not init_dims:
                init_dims = [0]
            self.init_dims = init_dims
            self._set_next_dims(next_dims)  
        else:
            self.init_dims = list(range(self.n))
        
    def _set_work(self, work_multipliers, work_function):
        if work_multipliers:
            self.work_function = WorkFunction(multipliers=work_multipliers, n=self.n,bundled=self.bundled)
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
            self.contribution_function = ContributionFunction(multipliers=contribution_multipliers, n=self.n,bundled = False)
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
        if (not next_dims) and self.n == Inf:
            self.next_dims = lambda dim: [dim + 1] if dim + 1 not in self.init_dims else []
        else:
            self.next_dims = next_dims

class SparseApproximator:
    r'''
    Sparse approximation based on multi-index decomposition.

    Given a decomposition of :math:`f_{\infty}` as    
    .. math::

        f_{\infty}=\sum_{\mathbf{k}\in\mathbb{N}^{n}} (\Delta f)(\mathbf{k}),

    this class computes and stores approximations of the form
    .. math:: 

       \mathcal{S}_{\mathcal{I}}f:=\sum_{\mathbf{k}\in\mathcal{I}} (\Delta f)(\mathbf{k}),

    for finite index sets :math:`\mathcal{I}\subset\mathbb{R}^n`. 

    To compute the approximation, use `update_approximation`. This method provides
    multiple ways to choose the index set :math:`\mathcal{I}`. 
    '''
    @validate_args(
        '~work_multipliers|~work_function',
        '~contribution_multipliers|~contribution_function',
        'bundled_dims>bundled',
        Arg('init_dims|next_dims', Passed) > Arg('n', Equals(math.inf)),
        warnings=False,
    )
    def __init__(
        self,
        Delta,
        n: Positive & Integer,
        work_multipliers: Dict(value_spec = InInterval(l = 1, lo = False), lenience = 2),
        work_function: Instance(WorkFunction),
        contribution_multipliers: Dict(value_spec = InInterval(l = 0, lo = True, r = 1, ro = True), lenience = 2),
        contribution_function: Instance(ContributionFunction),
        returns_work: Bool = False,
        returns_contributions: Bool = False,
        init_dims: List(Nonnegative & Integer, lenience = 2) = NotPassed,
        next_dims: Function = NotPassed,
        bundled: Bool = False,
        bundled_dims: Function(value_spec = Bool) | List(value_spec = Integer) = NotPassed,
        kronecker_exponents: Function(value_spec = Nonnegative & Float) = NotPassed,
        stores_approximation: Bool = False,
        structure=None,
        callback=None,
    ):
        r'''        
        :param Delta: Computes decomposition terms.
            In the most basic form, a single multi-index is passed to 
            :code:`Delta` and a single value, which represents the corresponding 
            decomposition term and supports vector space operations, is expected. 
            
            If :code:`returns_work == True`, :code:`Delta` must return (value,work) (or only work if stores_approximation),
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
            then :code:`Delta` must return (value,work,contribution) (or only (work,contribution) in case of external storage)
            
            If :code:`bundled == True`, then :code:`Delta` is passed an iterable of multi-indices
            (which agree in the dimensions for which :code:`bundled_dims` returns True) must return 
            the sum of the corresponding decomposition elements.
            This may be useful for parallelization, or for problems where joint
            computation of decomposition elements is analytically more efficient.
            
            If :code:`stores_approximation == True`, no n_results is required and decomposition terms
            are expected to be stored by :code:`Delta` itself. Each call 
            will contain the full multi-index set representing the current approximation.
            
        :type Delta:  Function.
        :param n: Number of discretization parameters
        :type n: Integer, including math.inf (or 'inf') 
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
        :param init_dims: Initial dimensions used to create multi-index set. Defaults to :code:`range(n)`. However,
            for large or infinite `n`, it may make sense (or be necessary) to restrict this initially. 
            Each time a multi-index with non-zero entry in one of these initial dimensions, say j,  is selected, 
            new dimensions are added, according to output of :code:`next_dims(j)`
        :type init_dims: Integer or list of integers
        :param next_dims: Assigns to each dimension a list of child dimensions (see above)
        :type next_dims: :math:`\mathbb{N}\to 2^{\mathbb{N}}`
        :param: bundled: To be used when the decomposition terms cannot be computed independent of 
           each other. In this case, :code:`Delta` is called with subsets of :math:`\mathcal{I}` whose entries agree
           except for the dimensions specified in bundled_dims
        :type bundled: Boolean
        :param bundled_dims: Specifies dimensions used for bundling
        :type bundled_dims: :math:`\mathbb{N}\to\{\text{True},\text{False}\}` or list of dimensions
        :param kronecker_exponents: For infinite dimensional problems, the 
            contribution of Kronecker multi-index e_j is estimated as exp(kronecker_exponents(j))
        :type kronecker_exponents: Function from integers to negative reals  
        :param stores_approximation: Specifies whether approximation is computed externally. If True, :code:`Delta` is called multiple times 
            with a SparseIndex as argument (or a list of SparseIndices if :code:`bundled_dims` is True as well) and must collect the associated
            decomposition terms itself. If False, :code:`Delta` is also called multiple times with a SparseIndex (or a list of SparseIndices)
            and must return the associated decomposition terms, which are then stored within the SparseApproximator instance
        :type stores_approximation: Boolean
        :param structure: Used to enforce symmetries in approximating set.  
        :type structure: Function: multiindices->setsofmultiindices
        :param callback: If this returns a Trueish value, approximation is stopped (use for accuracy based approximations)
        :type callback: Function(): Bool
        '''
        self.decomposition = _Decomposition(
            Delta,
            n,
            work_multipliers,
            work_function,
            contribution_multipliers,
            contribution_function,
            returns_work,
            returns_contributions,
            init_dims,
            next_dims,
            bundled,
            bundled_dims,
            kronecker_exponents,
            stores_approximation,
            structure,
            callback,
        )
        self.log = Log(print_filter=False)
        self.data = _Data(self.decomposition)
        
    def update_approximation(self, mode=None,indices=None, T=None, L=None):
        '''
        Update the multi-index set and compute the resulting updated approximation.

        There are four ways to determine the updated multi-index set:
         :code:`indices`, which requires passing the new multi-indices with the argument `indices`
         :code:`adaptive`, which increases the multi-index set adaptively, one multi-index at a time
         :code:`apriori`, which constructs the set based on a-priori knowledge about work and contribution of the decomposition terms
         :code:`continuation`, which constructs the set by a combination of first learning the behavior of work and contribution, and then using this knowledge to create optimal sets.
        
        ADAPTIVE:
            To decide on the multi-index to be added at each step, estimates of contributions and work are maintained. 

            These estimates are based on neighbors that are already in the set :math:`\mathcal{I}`,
            unless they are specified in :code:`contribution_factors` and :code:`work_factors`.
            If user defines :code:`have_work_factor` and :code:`have_contribution_factor` 
            that only estimates for some of the :code:`n` involved parameters are available, 
            then the estimates from :code:`contribution_factor` and :code:`work_factor` for those parameters
            are combined with neighbor estimates for the remaining parameters.
        
            Must pass exactly one of the following additional arguments:
            :param N: Maximal number of new multi-indices.
            :type N: Integer
            :param T: Maximal time (in seconds).
            :type T: Positive

        APRIORI:
            Use apriori estimates of contributions and work provided to determine.
        
            Must pass exactly one of the following additional arguments:
            :param L: Threshold 
            :type L: Positive
            :param T: Maximal time (in seconds).
            :type T: Positive

        CONTINUATION:
            In an iterative manner, determine optimal multi-index-simplices by fitting contribution
            and work parameters to simplex of previous iteration.

            :param T: Maximal time (in seconds).
            :type T: Positive

        INDICES:
            Use user-specified multi-index set 
            
            :param indices: Multi-index set
            :type indices: iterable of MultiIndex instances

        :param mode: Update mode
        :type mode: One of 'indices', 'adaptive', 'apriori', 'continuation'
        '''
        tic_init = timer()
        Passed = lambda x: x is not None
        if not Passed(mode):
            if Passed(indices): 
                mode = 'indices'
            if Passed(T):
                mode = 'adaptive'
            if Passed(L):
                mode = 'apriori'
        if mode == 'indices':
            if Passed(L) or Passed(T):
                raise ValueError('Cannot pass L or T in indices mode')
            it = [0]
        elif mode == 'apriori':
            if Passed(T) == Passed(L):
                raise ValueError('Must pass either L or T in apriori mode')
            if Passed(L):
                it = [L]
            else:
                it = itertools.count()
        elif mode == 'adaptive':
            if Passed(T) == Passed(L):
                raise ValueError('Must pass either L or T in adaptive mode')
            if self.decomposition.bundled and not self.decomposition.returns_contributions:
                raise ValueError('Cannot run adaptively when decomposition.bundled but not decomposition.returns_contributions')
            if Passed(L):
                it = range(L)
            else:
                it = itertools.count()
        elif mode== 'continuation':
            if Passed('L'):
                raise ValueError('Cannot pass L in continuation mode')
            if np.isinf(self.decomposition.n):
                raise ValueError('Cannot use continuation for infinite-dimensional problems')
            it = itertools.count()
            n = self.decomposition.n
            work_exponents = lambda: [
                self._get_work_exponent(i) 
                if (
                        self.data.work_model_estimator if 
                        self.decomposition.returns_work 
                        else self.data.runtime_estimator
                ).ratios[i] 
                else 1 
                for i in range(n)
            ]
            contribution_exponents = lambda: [
                self._get_contribution_exponent(i) 
                if 
                self.data.contribution_estimator.ratios[i]
                else 1 
                for i in range(n)
            ]
        else: 
            raise ValueError('No mode selected')
        max_time=0
        for l in it:
            tic_iter = timer()
            if Passed(T) and l>0 and max_time + (timer() - tic_init) > T: 
                break
            if self.decomposition.callback():
                break
            if mode == 'apriori':
                mis_update = self.data.apriori_indices(l)
            elif mode == 'adaptive':
                mis_update = [self.data.next_best_mi()]
            elif mode == 'continuation':
                scale = min(work_exponents() + contribution_exponents() for dim in range(n))
                mis_update = indices.simplex(L=l, weights=(work_exponents() + contribution_exponents()) / scale, n=n)
            elif mode == 'indices':
                mis_update = indices
            self._extend(mis_update) # check whether this rbreaks when mis_udpate is a subset of current mis
            max_time = max(timer() - tic_iter,max_time)
        
    @log_calls    
    def _extend(self, mis):
        '''
        :param mis: Multi-indices to add to approximation
        :type mis: Iterable of multi-indices
        '''
        if self.decomposition.bundled:
            miss = get_bundles(mis, self.decomposition.bundled_dims)
            not_bundled_dims = lambda dim: not self.decomposition.bundled_dims(dim)
            key = lambda mis: mis[0].restrict(not_bundled_dims)
            it = sorted(miss, key=key)
        else:
            it = mis
        for temp in it:
            work_model = None
            contribution = None
            object_slice = None
            if self.decomposition.bundled:
                mi_update = temp[0]
                mis_update = temp
            else:
                mi_update = temp
                mis_update = [temp]
            if not self.decomposition.stores_approximation and not self.decomposition.Delta:  # Dry run
                return  
            if self.decomposition.bundled:
                external_work_factor = self.decomposition.work_function(mis_update) # Want to be able to provide work function with whole bundle without checking what is actually new and then asking each of those separately, thats why it is required that work_function covers all bundled dimensions
            else:
                external_work_factor = self.decomposition.work_function(mi_update)
            self.data.extend(mis_update)
            if self.decomposition.stores_approximation:
                argument = self.data.mis.mis # provide full set, leave it to external to reuse computations or not 
            else:
                argument = mis_update if self.decomposition.bundled else mi_update
            tic = timer()
            output = self.decomposition.Delta(copy.deepcopy(argument))
            runtime = timer() - tic
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
                if not self.decomposition.bundled_dims and not Dict.valid(contribution):  # Allow user to not return dictionary if not_bundled (which means user is only passed single multi-index)
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
            self.data.update_estimates(mis_update, mi_update, object_slice, contribution, work_model, runtime, external_work_factor)
             
    def get_approximation(self):
        if self.decomposition.stores_approximation:
            raise ValueError('Decomposition is stored externally')
        else:
            return sum([self.data.object_slices[mi] for mi in self.data.object_slices])   
    
    def get_contribution_multiplier(self, dim):
        return np.exp(-self._get_contribution_exponent(dim))

    def get_runtime_multiplier(self, dim):
        if self.decomposition.returns_work:
            raise ValueError('Since decomposition provides abstract work model, no runtime estimates are kept. Try `get_work_multiplier` instead.')
        return np.exp(self._get_runtime_exponent(dim))

    def get_work_multiplier(self, dim):
        return np.exp(self._get_work_exponent(dim))
    
    def get_total_work_model(self):
        return sum(self.data.work_models._dict.values())
    
    def get_total_runtime(self):
        return sum(self.data.runtimes._dict.values())
    
    def get_indices(self):
        return copy.deepcopy(self.data.mis.mis)

    def _get_work_exponent(self, dim):
        estimator = self.data.work_model_estimator if self.decomposition.returns_work else self.data.runtime_estimator
        if not estimator.dims_ignore(dim):
                return estimator.exponents[dim]
        else:
            raise KeyError('No work fit for this dimension')
    
    def _get_runtime_exponent(self, dim):
        if not self.data.runtime_estimator.dims_ignore(dim):
            return self.data.runtime_estimator.exponents[dim]
        else:
            raise KeyError('No runtime fit for this dimension')
       
    def _get_contribution_exponent(self, dim):
        if not self.data.contribution_estimator.dims_ignore(dim):
            return -self.data.contribution_estimator.exponents[dim]
        else:
            raise KeyError('No contribution fit for this dimension') 
    
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
            dims = list(self.data.mis.active_dims)
        if not weights:
            percentiles = 1
            weight_dict = None
        elif weights == 'contribution':
            weight_dict = {mi: self.data.contributions[mi] for mi in self.get_indices()}
        elif weights == 'work_model':
            if not self.decomposition.returns_work:
                raise ValueError('Decomposition does not provide abstract work model')
            weight_dict = {mi: self.data.work_models[mi] for mi in self.get_indices()}
        elif weights == 'runtime':
            weight_dict = {mi: self.data.runtimes[mi] for mi in self.get_indices()}
        elif weights == 'contribution/work_model':
            assert(self.decomposition.returns_work)
            weight_dict = {mi:self.data.contributions[mi] / self.data.work_models[mi] for mi in self.get_indices()}
        elif weights == 'contribution/runtime':
            weight_dict = {mi: self.data.contributions[mi] / self.data.runtimes[mi] for mi in self.get_indices()}
        else: 
            raise ValueError('Cannot use weights {}'.format(weights))
        plots.plot_indices(mis=self.get_indices(), dims=dims, weights=weight_dict, groups=percentiles) 

class _Estimator:
    def __init__(self, dims_ignore, exponent_max, exponent_min, init_exponents=None,name=''):
        self.quantities = {}
        self.dims_ignore = dims_ignore
        self.FIT_WINDOW = int(1e6)
        self.ratios = DefaultDict(lambda dim: collections.deque([],self.FIT_WINDOW))
        init_exponents = init_exponents or (lambda dim:0)
        self.fallback_exponents = DefaultDict(init_exponents)  # USED AS PRIOR IN EXPONENT ESTIMATION AND AS INITIAL GUESS OF EXPONENT WHEN NO DATA AVAILABLE AT ALL
        self.exponents = DefaultDict(lambda dim: self.fallback_exponents[dim])
        self.exponent_max = exponent_max
        self.exponent_min = exponent_min
        self.active_dims = set()   
        self.name=name
        
    def set_fallback_exponent(self, dim, fallback_exponent):
        self.fallback_exponents[dim] = fallback_exponent
        
    def __contains__(self, mi):
        mi = mi.mod(self.dims_ignore)
        return mi in self.quantities
    
    def __setitem__(self, mi, q):
        self.active_dims.update(set(mi.active_dims()))
        mi = mi.mod(self.dims_ignore)
        q = float(q)
        have = mi in self.quantities
        self.quantities[mi] = q  #two reasons to overwrite: 1) in least squares polynomials for the contribution estimate, the estimate of every single coefficient gets better and better over time 2) in general, for dimensions that are modulod out because their work contribution factor is known, this entry is theoretically the same but practically different. for example, look at MLMC, assume the work factor of the first parameter is theoretically 2. then this stores effectively the cost per sample divided by 2**l. however, this cost is not actually indpendent of the level  
        if not have:
            get_ratio = lambda a,b: a/b if b>0 else (np.Inf if a>0 else 1)
            for dim in self.active_dims:
                neighbor = mi + (-1)*kronecker(dim)
                if neighbor in self.quantities:
                    self.ratios[dim].append(get_ratio(q,self.quantities[neighbor])) 
            self._update_exponents()
        
    def _update_exponents(self):
        for dim in self.ratios:
            ratios = np.array(self.ratios[dim])
            estimate = max(min(np.median(ratios), np.exp(self.exponent_max)), np.exp(self.exponent_min))
            c = len(ratios)
            self.exponents[dim] = (self.fallback_exponents[dim] + c * np.log(estimate)) / (c + 1.)
    
    def __call__(self, mi):
        mi = mi.mod(self.dims_ignore)
        if mi in self.quantities:
            return self.quantities[mi]
        else:
            if mi.is_kronecker():
                dim = mi.active_dims()[0]
                return self.quantities[MultiIndex()]*np.exp(self.exponents[dim])
            else:
                estimate = 0
                for dim,sign in itertools.product(self.active_dims,(-1,1)):
                    neighbor = mi + sign*kronecker(dim)
                    if neighbor in self.quantities:
                        estimate = max(estimate,self.quantities[neighbor])
                return estimate

class _Data:
    def __init__(self, decomposition):
        self.WORK_EXPONENT_MAX = 10
        self.WORK_EXPONENT_MIN = 0
        self.CONTRIBUTION_EXPONENT_MAX = 0
        self.CONTRIBUTION_EXPONENT_MIN = -10
        self.decomposition = decomposition 
        # If decomposition is bundled, runtime_estimator only makes sense if runtime_function has all bundled dimensions. runtime_function may or not be bundled; if it is not it will be called for each index in bundled and the observed runtime will then be divided by the sum 
        # work_model function does not have to be bundled if decomposition is. if it is bundled, it does have to include bundled dims in its dims
        # contribution function cannot be bundled
        self.runtime_estimator = _Estimator(self.decomposition.work_function.dims,
                                      exponent_max=self.WORK_EXPONENT_MAX,
                                      exponent_min=self.WORK_EXPONENT_MIN,
                                      name='runtime')
        self.runtimes = MultiIndexDict(lambda dim: self.decomposition.bundled and self.decomposition.bundled_dims(dim))
        self.contribution_estimator = _Estimator(self.decomposition.contribution_function.dims,
                                              exponent_max=self.CONTRIBUTION_EXPONENT_MAX,
                                              exponent_min=self.CONTRIBUTION_EXPONENT_MIN,
                                              init_exponents=self.decomposition.kronecker_exponents,
                                              name='contribution')
        if self.decomposition.returns_work:
            self.work_model_estimator = _Estimator(self.decomposition.work_function.dims,
                                      exponent_max=self.WORK_EXPONENT_MAX,
                                      exponent_min=self.WORK_EXPONENT_MIN,
                                      name='work_model')
            self.work_models = MultiIndexDict(decomposition.bundled_dims)
        self.mis = MISet(dims=decomposition.init_dims)
        self.mis.structure_constraints = set([MultiIndex()])
        self.mis.structure_constraints |= set(MultiIndex(((i,1),),sparse=True) for i in self.decomposition.init_dims)
        self.object_slices = MultiIndexDict(decomposition.bundled_dims)
        self.contributions = dict()

    def next_best_mi(self):
        mis = self.mis
        for mi in mis.structure_constraints.copy():
            if mi in mis:
                mis.structure_constraints.discard(mi)
            elif mis.is_admissible(mi):
                mi_update = mi
                break
        else:
            estimates = {mi:self.profit_estimate(mi) for mi in mis.candidates} 
            # plots.plot_indices(mis.candidates)
            # plots.plot_indices(mis.mis,colors=[[0,0,0]])
            # import matplotlib.pyplot as plt
            # plt.show()
            mi_update = max(mis.candidates, key=lambda mi: self.profit_estimate(mi))
            if self.decomposition.structure:
                mis.structure_constraints |= set(self.decomposition.structure(mi_update))
        return mi_update

    def apriori_indices(self,L):
        def admissible(mi):
            return self.profit_estimate(mi) ** (-1) <= np.exp(L + 1e-12)
        try:
            mis = get_admissible_indices(lambda mi: admissible(mi), self.decomposition.n)
        except KeyError:
            raise KeyError('Did you specify the work for all dimensions?')
    
    def extend(self,mis_update):
        if math.isinf(self.decomposition.n):
            self._find_new_dims(mis_update)
        self.mis.update(mis_update)        
        
    def update_estimates(self, mis_update, mi_update, object_slice, contribution, work_model, runtime, external_work_factor):
        self.object_slices[mi_update] = object_slice
        self.runtimes[mi_update] = runtime
        if self.decomposition.returns_work:  # external_work_factor refers to work_model 
            if external_work_factor>0:
                self.work_model_estimator[mi_update] = work_model / external_work_factor  # Here lies reason why work factor must be bundled if decomposition is bundled: `work_model` does not distinguish different contributions to work (its for the entire mi_update-bundle), so external_work_factor must also make its predictions for entires bundles
            self.work_models[mi_update] = work_model    
        else: # external_work_factor refers to runtime
            self.runtime_estimator[mi_update] = runtime / external_work_factor
        try:
            for mi in contribution:
                self.contribution_estimator[mi] = contribution[mi] / self.decomposition.contribution_function(mi) # Here lies reason why the reasoning above does not apply to contributions: they can always be split up among all the multi-indices in a bundle (this is implicit in the assumption that the used provided contributions are separate for each mi in mi_update) 
                self.contributions[mi] = contribution[mi]
        except (KeyError, NameError): 
            pass  # Contribution could not be determined, contribution was never created 
    
    def profit_estimate(self, mi):
        contribution = self.contribution_estimator(mi) * self.decomposition.contribution_function(mi)
        if self.decomposition.bundled:
            mi_to_bundle = lambda mi: get_bundle(mi, self.mis, self.decomposition.bundled_dims) + [mi]
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
                    self.mis.add_dimensions([dim])
                    if self.decomposition.returns_work:
                        self.work_model_estimator.set_fallback_exponent(dim, self.work_model_estimator.exponents[dim_trigger])
                    else:
                        self.runtime_estimator.set_fallback_exponent(dim, self.runtime_estimator.exponents[dim_trigger])
                    if not self.decomposition.kronecker_exponents:
                        self.contribution_estimator.set_fallback_exponent(dim, self.contribution_estimator.exponents[dim_trigger])

