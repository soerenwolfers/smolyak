import warnings
import math
import copy
import timeit
import functools

import numpy as np
import scipy
from scipy.linalg import solve
from scipy.linalg.misc import LinAlgError
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport @UnresolvedImport

from swutil.np_tools import grid_evaluation
from swutil.collections import RFunction, VFunction
from swutil.collections import DefaultDict
from swutil.decorators import print_runtime
from swutil.np_tools import grid_evaluation
from swutil.time import Timer
import swutil.validation

from smolyak.applications.polynomials import samples
from smolyak.indices import MixedDifferences
from smolyak import indices
from smolyak.indices import MultiIndex, cartesian_product
from smolyak.applications.polynomials.probability_distributions import ProbabilityDistribution,\
    ProductProbabilityDistribution, AbstractProbabilityDistribution 
from smolyak.applications.polynomials.orthogonal_polynomials import evaluate_orthonormal_polynomials

class PolynomialSpace:
    '''
    Vector space of polynomials for least-squares polynomial approximation on open 
    subsets of Euclidean space.
    
    Combines domain (in form of probability space) with description
    of finite-dimensional subspace of polynomials

    :param n: Dimension of domain
    :param k: Polynomial degree
    '''
    def __init__(self, probability_distribution = None,basis=None,warnings = False, n = None, k = None):
        if isinstance(probability_distribution,ProbabilityDistribution):
            self.probability_distribution = ProductProbabilityDistribution([probability_distribution])
        else:
            if probability_distribution is None:
                probability_distribution = 't'
            self.probability_distribution = ProbabilityDistribution(probability_distribution)**n
            if basis is None and k is None:
                raise ValueError('Must provide basis or polynomial degree')
            basis = basis or indices.simplex(L=k,n=n)
        self.set_basis(basis)
        self.warnings = warnings
    
    def get_dimension(self): 
        '''
        Return dimension of polynomial subspace (not of domain, use get_c_var() for that)
        '''
        return len(self.basis)

    @property
    def dimension(self):
        return self.get_dimension()
    
    def optimal_weights(self, X):
        '''
        Return inverse of optimal sample density (=normalized Christoffel function)
        '''
        if self.get_dimension()>0:
            if X.shape[0] == 0:
                return np.zeros((0, 1))
            else:
                B = self.evaluate_basis(X)
                W = self.get_dimension() / np.sum(np.power(B, 2), axis=1)
                return W   
        else:
            raise ValueError('Polynomial subspace is zero-dimensional')
    
    def get_approximation(self, X, Y, W=None):
        coeff,*_ = self.weighted_least_squares(X=X,Y=Y,W=W)
        return PolynomialApproximation(self,coeff)
        
    def weighted_least_squares(self, X, Y, W=None, basis_extension=None):
        '''
        Compute least-squares approximation. 
        
        :param X: sample locations
        :type X: `N x self.get_c_var()` numpy array
        :param W: weights
        :param Y: sample values
        :return: coefficients
        '''
        W = W if W is not None else np.ones([len(X)])
        B = self.evaluate_basis(X)
        if basis_extension:
            b_extra = basis_extension(X)
            B = np.concatenate([B,b_extra],axis=1)
        W = W.reshape(-1)
        Y = Y.reshape(-1)
        cond = None
        if self.warnings:
            G = B.transpose().dot(B * W)
            cond = np.linalg.cond(G)
            if cond > 3:
                warnings.warn('Ill conditioned Gramian matrix encountered') 
        if B.shape[1]>0:
            tol=1e-9
            rls = np.sqrt(W)*Y
            M = np.sqrt(W)[:,None]*B
            coefficients,*info = scipy.sparse.linalg.lsmr(M,rls,atol=tol,btol=tol)
            if self.warnings and not np.isfinite(coefficients).all():
                warnings.warn('Numerical instability encountered')
        if basis_extension:
            return {pol: coefficients[i] for i, pol in enumerate(self.basis)},cond, coefficients[len(self.basis):]
        else:
            return {pol: coefficients[i] for i, pol in enumerate(self.basis)},cond
         
    def get_active_dims(self):
        if self.basis:
            return set.union(*[set(pol.active_dims()) for pol in self.basis]) 
        else:
            return set()
    
    def get_c_var(self):
        '''
        Return dimension of domain (not of polynomial subspace, use get_dimension() for that)
        '''
        return self.probability_distribution.get_c_var()

    def evaluate_basis(self, X,derivative=None):
        '''
        Evaluates basis polynomials at given sample locations.
        
        :param X: Sample locations
        :type X: Numpy array of shape `N x self.get_c_var()` 
        :return: Basis polynomials evaluated at X
        :rtype: `N x self.get_dimension()` np.array
        '''
        values = np.ones((X.shape[0], self.get_dimension()))
        basis_values = {}
        if self.basis:
            active_vars = sorted(list(set.union(*[set(pol.active_dims()) for pol in self.basis])))
        else:
            active_vars = list()
        if not derivative:
            derivative = np.zeros(self.get_c_var())
        elif len(derivative)<self.get_c_var():
            derivative = derivative + [0]*(self.get_c_var()-len(derivative))
        if any(order>0 and dim not in active_vars for dim,order in enumerate(derivative)):
            values[:] = 0
        elif active_vars:
            for dim in active_vars:
                basis_values[dim] = evaluate_orthonormal_polynomials(
                    X[:,dim],
                    max(pol[dim] for pol in self.basis), 
                    measure=self.probability_distribution.ups[dim].measure, 
                    interval=self.probability_distribution.ups[dim].interval,
                    derivative = derivative[dim]
                )
            projected_basis = [pol.retract(lambda i: active_vars[i]).full_tuple(c_dim = len(active_vars)) for pol in self.basis]
            for i,pol in enumerate(self.basis):
                if any(order>0 and dim not in pol.active_dims() for dim,order in enumerate(derivative)):#Check necessity 
                    values[:,i] = 0                                                                                      
                else:
                    values[:,i] = np.prod([basis_values[active_vars[d]][:,projected_basis[i][d]] for d in range(len(active_vars))],axis=0)
        return values
    
    def set_basis(self,polynomials):
        self.basis=[]
        if polynomials:
            self.extend_basis(polynomials)
        
    def extend_basis(self, polynomials):
        '''
        Extend polynomial subspace.
        
        If necessary, extend dimension of domain as well. 
        
        :param polynomials: Polynomials to be added
        '''
        self.basis += polynomials
        if self.basis:
            max_var = max([pol.max_dim()for pol in self.basis])
            if max_var > self.get_c_var():
                for __ in range(max_var - self.get_c_var()):
                    self.probability_distribution.ups.append(copy.deepcopy(self.probability_distribution.ups[-1]))
        
    def plot_optimal_distribution(self, N=200, L=1):
        '''
        Plot optimal sampling distribution
        '''
        if self.get_c_var() == 1:
            X = self.probability_distribution.get_range(N, L)
            Z = self.probability_distribution.lebesgue_density(X) / self.optimal_weights(X)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(X, Z)
        elif self.get_c_var() == 2:
            X, Y = self.probability_distribution.get_range(N, L)
            Z = grid_evaluation(X, Y,self.probability_distribution.lebesgue_density) / grid_evaluation(X, Y,self.optimal_weights)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5)  # @UndefinedVariable
        fig.suptitle('Lebesgue density of optimal distribution')
        plt.show()
        return np.sum(Z) / Z.size

class PolynomialApproximator:
    r'''
    Maintains polynomial approximation of a given function 
    :math:`f\colon [a,b]^d\to\mathbb{R}`
    that can be sampled at different levels of accuracy.
    '''
    def __init__(self, function=None,domain=None,C=1,sampler='optimal',reparametrization=False,n = None, n_acc=0,warnings = False):
        r'''
        :param function: Function :math:`[a,b]^d\to\mathbb{R}` that is being
        approximated. Needs to support :code:`__call__(X,mi)` where :code:`X` 
        is an np.array of size N x :math:`d` describing the sample locations and :code:`mi` is a
        multi-index describing the required accuracy of the samples.
        :param cdim_acc: Number of discretization parameters of `self.function`
        (=length of `mi` above)
        :param domain: Probability distribution
        :type domain: List of intervals or ProbabilityDistribution
        :param C: see SinglelevelPolynomialApproximator
        :param sampler: see SinglelevelPolynomialApproximator
        :param reparametrization: Determines whether polynomial subspaces are indexed with 
        an exponential reparametrization
        :type reparametrization: Function MultiIndex to List-of-MultiIndex, or
            simply True, which corresponds to exponential reparametrization
        '''
        #if C<1:
        #    raise ValueError('Choose C>=1')
        self.C=C
        if isinstance(domain,AbstractProbabilityDistribution):
            self.probability_distribution = domain
        elif domain:
            if swutil.validation.Iterable.valid(domain[0]):
                self.probability_distribution = ProbabilityDistribution(interval = domain[0])
                for dom in domain[1:]:
                    self.probability_distribution = self.probability_distribution * ProbabilityDistribution(interval = dom)
            else:
                self.probability_distribution = ProbabilityDistribution(interval = domain)
        else:
            self.probability_distribution = ProbabilityDistribution()**n
        if swutil.validation.String.valid(sampler) and n_acc:
            self.sampler = lambda mi: sampler
        elif swutil.validation.Instance(np.ndarray).valid(sampler) and n_acc:
            if self.n_acc>1:
                raise ValueArray("Don't currently support predefined samples with multi-index regression")
            self.sampler = lambda mi: sampler[...,mi]
        else:
            self.sampler = functools.partial(sampler,mi) if n_acc else sampler
        self.n_acc = n_acc
        self.warnings=warnings
        if n is None:
            self.n = self.probability_distribution.get_c_var()
        else:
            self.n = n
        self.bundled_dims = lambda dim: dim >= self.n_acc
        self.function = function
        if self.n_acc>0:
            def T(mi_acc): 
                if np.isinf(self.n_acc): # prevent user from accidentally modifying mi_acc
                    mi = copy.deepcopy(mi_acc)
                else:
                    mi = mi_acc.full_tuple(c_dim=self.n_acc)
                return VFunction(lambda X: self.function(mi,X))
            mdT=MixedDifferences(T)
        get_spa = lambda mi_acc: SinglelevelPolynomialApproximator(
            function = mdT(mi_acc) if self.n_acc else self.function,
            probability_distribution = copy.deepcopy(self.probability_distribution),
            C = self.C,
            sampler = self.sampler(mi) if self.n_acc else self.sampler,
            warnings = self.warnings,
        )    
        self.spas = DefaultDict(default=get_spa)
        self.reparametrization=reparametrization
        self.work_per_sample = None
     
    def plot_samples(self,mi_acc=None,ax=None,kwargs=None):
        if mi_acc is None and self.n_acc>0:
            raise ValueError('Must specify which delta to plot')
        if not isinstance(mi_acc,MultiIndex):
            mi_acc = MultiIndex(mi_acc)
        if mi_acc not in self.spas:
            raise ValueError('Only have samples with accuracy parameters {}'.format(list(self.spas.keys())))
        return self.spas[mi_acc].plot_samples(ax=ax,kwargs=kwargs)

    def get_samples(self,mi_acc=None):
        if mi_acc is not None and not isinstance(mi_acc,MultiIndex):
            mi_acc = MultiIndex(mi_acc)
            if mi_acc not in self.spas:
                raise ValueError('Only have samples with accuracy parameters {}'.format(list(self.spas.keys())))
        if mi_acc is None:
            if self.n_acc>0:
                return {mi_acc:(spa.X,spa.Y) for spa in self.spas}
            else:
                return self.spas[MultiIndex()].X,self.spas[MultiIndex()].Y
                 
    #def get_active_dims(self):
    #    return set.union(*[spa.get_active_dims() for spa in self.spas.values()]) 
    
    def get_approximation(self,mi_acc=None):
        '''
        Returns polynomial approximation
        
        :return: Polynomial approximation of :math:`f\colon [a,b]^d\to\mathbb{R}`
        :rtype: Function
        '''
        if mi_acc is not None:
            if not isinstance(mi_acc,MultiIndex):
                mi_acc = MultiIndex(mi_acc)
            if mi_acc not in self.spas:
                raise ValueError('Only have approximation of Deltas with accuracy parameters {}'.format(list(self.spas.keys())))
            return self.spas[mi_acc].get_approximation()
        else:
            return sum([self.spas[mi_acc].get_approximation() for mi_acc in self.spas])
           
    def update_approximation(self, mis):
        r'''
        Expands polynomial approximation.
        
        Converts list of multi-indices into part that describes polynomial basis
        and part that describes accuracy of samples that are being used, then 
        extend polynomial approximation of self.functon based on this information.
        
        
        :param mis: Multi-indices 
        :return: work and contribution associated to mis
        :rtype: (work,contribution) where work is real number and contribution is dictionary mis>reals
        '''
        bundles=indices.get_bundles(mis, self.bundled_dims)
        work = 0
        contributions=dict()
        for bundle in bundles:
            mis_pols,mi_acc= self._handle_mis(bundle)
            (new_work,_) = self.spas[mi_acc].update_approximation(self._pols_from_mis(mis_pols))
            work+=new_work
            #if work>0:
            pa = self.spas[mi_acc].get_approximation()
            contributions.update({mi_acc+mi.shifted(self.n_acc): pa.norm(self._pols_from_mi(mi)) for mi in mis_pols})
        return work, contributions
    
    def reset(self):
        '''
        Delete all stored samples
        '''
        self.__init__(function=self.function, n_acc=self.n_acc, n = self.n , 
            warnings = self.warnings,domain=self.probability_distribution,C=self.C,sampler=self.sampler,reparametrization = self.reparametrization)
    
    def estimated_work(self, mis,multibundles=False):          
        '''
        Return number of samples that would be generated if instance were 
        extended with given multi-index set.
        
        :param mis: (List of) MultiIndex
        :param multibundles: Allow for multi-index sets that contain multiple different accuracies
        :return: Number of new sampls
        '''
        if multibundles:
            bundles = indices.get_bundles(mis,self.bundled_dims)
        else:
            bundles = [mis]
        work = 0
        for bundle in bundles:
            mis_pols, mi_acc = self._handle_mis(bundle)
            work += (self.spas[mi_acc].estimated_work(self._pols_from_mis(mis_pols))
                     *(self.work_per_sample(mi_acc) if self.work_per_sample else 1))
        return work

    def _handle_mis(self, mis):
        mis_pols = [mi.shifted(-self.n_acc) for mi in mis]
        mi_acc = mis[0].mod(self.bundled_dims)
        return mis_pols, mi_acc
     
    def _pols_from_mi(self, mi):
        '''
        Convert multi-index to corresponding polynomials
        
        :param mi: Multi-index
        :return: List of polynomials corresponding to mi
        '''
        if self.reparametrization is True:
            if mi == MultiIndex():
                    return [mi]
            else:
                univariate_entries = []
                for dimension in mi.active_dims():
                    init_range = 2 ** (mi[dimension])-1
                    end_range = 2 ** (mi[dimension]+1)-1
                    univariate_entries.append(range(init_range, end_range))
                return cartesian_product(univariate_entries, mi.active_dims())
        elif self.reparametrization is False:
            return [mi]
        else:
            return self.reparametrization(mi)
            
    def _pols_from_mis(self,mis):
        '''
        Convert multi-indices to corresponding polynomials
        
        :param mis: Multi-indices
        :return: List of polynomials corresponding to mis
        '''
        if self.reparametrization:
            pols=[]
            for mi in mis:
                pols+= self._pols_from_mi(mi)
            return pols
        else:
            return mis

    def _get_info(self):
        return [spa._info for spa in self.spas.values()]

    def accuracy(self):
        try:
            active_spas = [spa for spa in self.spas.values() if spa.polynomial_space.basis]
            recent_spa = active_spas[-1]
            remainder = sum([spa.remainder for spa in active_spas]) + (recent_spa.get_approximation().norm() + recent_spa.remainder if self.n_acc>0 else 0)
            have = sum([spa.get_approximation().norm() for spa in active_spas])
            return remainder/(have+remainder)
        except IndexError:
            return np.inf
        
class SinglelevelPolynomialApproximator:
    '''
    Maintains polynomial approximation of given function on :math:`[a,b]^d`.

    Do not use. Even for single-level approximation it is more convenient to use
    with the general PolynomialApproximator
    '''
    def __init__(self, function, probability_distribution, C=2,sampler='optimal',warnings=False):
        ''' 
        :param function: Function that is approximated. Needs to support __call__
        :param probability_distribution: Probability distribution used for approximation
        :type probability_distribution: ProbabilityDistribution
        :param C: Multiplier for number of samples used for reconstruction
        :type C: Positive real
        '''
        if swutil.validation.String.valid(sampler) and sampler != 'optimal' and sampler != 'arcsine': 
            raise ValueError('Sampling strategy not supported')
        self.keep = True
        self.sampler=sampler
        self.C = C
        self.function = function
        polynomial_space = PolynomialSpace(probability_distribution,warnings= warnings)
        self.polynomial_space = polynomial_space
        self.X = np.zeros((0, self.polynomial_space.get_c_var()))
        self.Y = np.zeros((0, 1))
        self.W = np.zeros((0, 1))
        self._approximation = PolynomialApproximation(self.polynomial_space)
        self._recompute_approximation = True
        self.remainder = np.inf;
       
    def c_samples_per_polynomial(self,dim):
        return max(3,math.ceil(self.C*np.log2(dim+1)))

    def c_samples_from_c_pols(self,dim):
        return dim*self.c_samples_per_polynomial(dim)

    def estimated_work(self,pols):
        '''
        Return number of new samples required to determine polynomial 
        coefficients onto given polynomial subspace
        
        :param pols 
        :return: Number of required samples
        '''
        # if all(pol in self.polynomial_space.basis for pol in pols):
        #     return 0
        c_pols = len(pols)
        c_samples = self.c_samples_from_c_pols(c_pols)
        if self.keep:
            return int(max(c_samples - self.X.shape[0],0))
        else:
            return int(c_samples) 
        
    def update_approximation(self, pols):
        '''
        Get near-optimal projection onto subspace described by mis
        
        :param mis: List of multi-indices describing polynomial span to be added
        :return: Time to compute new samples
        '''
        self._recompute_approximation=True
        old_approximation = copy.deepcopy(self._approximation)
        old_basis = self.polynomial_space.basis
        c_samples = self.estimated_work(pols) # this must come before next call
        self.polynomial_space.set_basis(pols)
        too_few_samples = False
        if self.X.shape[1] < self.polynomial_space.get_c_var():
            c_samples += self.X.shape[0]  # contrary to previous belief we cannot keep samples, thus need to add the number of old ones
            self.X = np.zeros((0, self.polynomial_space.get_c_var()))
            self.Y = np.zeros((0, 1))
            self.W = np.zeros((0, 1))
        if swutil.validation.String.valid(self.sampler):
            if self.sampler == 'arcsine':
                (Xnew,Wnew) = samples.importance_samples(self.polynomial_space.probability_distribution, c_samples, 'arcsine')
            elif self.sampler == 'optimal': 
                (Xnew,Wnew) = samples.samples_per_polynomial(self.polynomial_space,old_basis, pols, self.c_samples_per_polynomial)
            tic = timeit.default_timer()
            Ynew = self.function(Xnew).reshape(-1,1)
            work_model = timeit.default_timer() - tic
        elif swutil.validation.Instance(np.ndarray).valid(self.sampler):
            too_few_samples = len(self.sampler)<c_samples
            if too_few_samples:
                warnings.warn("Need {} more samples".format(c_samples-len(self.sampler)))
                c_samples = len(self.sampler)
                self.polynomial_space.set_basis(old_basis)
                if c_samples:
                    N = len(pols)
                    while c_samples < self.estimated_work(pols[:N]) and N>0:
                        N -= 1 
                    self.polynomial_space.set_basis(pols[:N])
            (samplesnew,self.sampler) = self.sampler[:c_samples],self.sampler[c_samples:]
            (Xnew,Ynew) = samplesnew[:,:-1],samplesnew[:,[-1]]
            Wnew = np.ones((samplesnew.shape[0],1))
            for dim in range(Xnew.shape[1]):
                interval = self.polynomial_space.probability_distribution.ups[dim].interval
                Wnew *= np.pi * np.sqrt((Xnew[:,[dim]] - interval[0]) * (interval[1] - Xnew[:,[dim]]))
            work_model = 0
        else:
            tic = timeit.default_timer()
            (Xnew,Ynew) = self.sampler(c_samples)
            work_model = timeit.default_timer() - tic
            Wnew = np.ones((c_samples, 1))
            for dim in range(Xnew.shape[1]):
                interval = self.polynomial_space.probability_distribution.ups[dim].interval
                Wnew *= np.pi * np.sqrt((Xnew[:,[dim]] - interval[0]) * (interval[1] - Xnew[:,[dim]]))
        if Ynew.shape[0] != Xnew.shape[0]:
            raise ValueError('Function must return as many outputs values as it is given inputs')
        self.X = np.concatenate((self.X,Xnew),axis=0)
        self.W = np.concatenate([self.W, Wnew.reshape(-1,1)])  
        self.Y = np.concatenate([self.Y, Ynew])
        if too_few_samples:
            app = self.get_approximation()
            if not app.coefficients:
                app.coefficients = RFunction()
            for pol in pols:
                if pol not in app.coefficients:
                    app.coefficients[pol] = 0
        if Xnew.shape[0]>0:
            self.remainder = np.sqrt(np.sum(Wnew/np.sum(Wnew)*(old_approximation(Xnew).reshape([-1,1])-Ynew)**2)) 
        return work_model,self.get_contributions()
        
    def get_contributions(self):
        pa=self.get_approximation()
        return {pol:pa.norm([pol]) for pol in self.polynomial_space.basis}
    
    def get_approximation(self):
        if self._recompute_approximation:
            old_err_settings = np.seterr(all='raise')
            while self._recompute_approximation:
                try:
                    (coeffs,self._info) = self.polynomial_space.weighted_least_squares(X = self.X, Y = self.Y, W=self.W)
                    self._approximation =  PolynomialApproximation(self.polynomial_space,coeffs)
                    self._recompute_approximation = False
                except LinAlgError:
                    warnings.warn('Singular matrix, doubling number of samples')
                    c_samples = len(self.X)
                    (Xnew,Wnew) = samples.importance_samples(self.polynomial_space.probability_distribution, c_samples, 'arcsine')
                    self.X = np.concatenate((self.X, Xnew), axis=0)  
                    self.W = np.concatenate([self.W, Wnew])  
                    Ynew=self.function(Xnew).reshape(-1,1)
                    if not Ynew.shape[0]==Xnew.shape[0]:
                        raise ValueError('Function must return as many outputs values as it is given inputs')
                    self.Y = np.concatenate([self.Y, Ynew])
            np.seterr(**old_err_settings)
        return self._approximation
        
    def plot_samples(self,ax=None,kwargs=None):
        '''
        Scatter plot of stored samples.
        '''
        kwargs = kwargs or {}
        if self.polynomial_space.get_c_var() == 1:
            order = np.argsort(self.X, axis=0).reshape(-1)
            if ax is None:
                fig = plt.figure()
                ax = fig.gca()
            ax.scatter(self.X[order, :], self.Y[order, :],**kwargs)
        elif self.polynomial_space.get_c_var() == 2:
            if ax is None:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.Y,**kwargs)
        return ax

class PolynomialApproximation:
    def __init__(self,ps,coefficients=None,basis = None):
        '''
        :param ps: Polynomial subspace in which approximation lives or ProbabilityDistribution that defines the domain of the approximations
        :type ps: PolynomialSpace or ProbabilityDistribution or 
        :param coefficients: Coefficients with respect to basis determined by
            :code:`basis `
        :type coefficients: Dictionary ps.basis->reals
        '''
        if isinstance(ps,PolynomialSpace):
            self.polynomial_space = ps
            if basis is not None:
                self.polynomial_space.set_basis(basis)
        else:
            self.polynomial_space = PolynomialSpace(ps)
            if basis is None:
                if hasattr(coefficients,'keys'):
                    self.polynomial_space.set_basis(coefficients.keys())
                else:
                    raise ValueError('Must specfiy either polynomial space or probability space and basis or probability space and coefficients dictionary whose keys determine basis')
        self.set_coefficients(coefficients)
        
    def set_coefficients(self,coefficients):
        if hasattr(coefficients,'keys'):
            if set(coefficients.keys()) != set(self.polynomial_space.basis):
                raise ValueError('Coefficients do not match polynomial basis')
            self.coefficients=RFunction(coefficients)
        elif coefficients is not None:
            t=dict()
            if len(self.polynomial_space.basis)!=len(coefficients):
                raise ValueError('Number of coefficients does not match polynomial subspace dimension')
            for i,p in enumerate(self.polynomial_space.basis):
                t[p]=coefficients[i]
            self.coefficients=RFunction(t)
        else:
            self.coefficients = None
        
    def _sum_sq_coeff(self, pols=None):
        '''
        Return sum of squared coefficients of given polynomials
        
        :param mi: Multi-index
        :return: Sum of squares of coefficients corresponding to mi
        '''
        if pols is None:
            pols=self.polynomial_space.basis
        return sum([self.coefficients[pol] ** 2 for pol in pols])  
        
    def __call__(self, X,derivative=None):
        '''
        Return approximation at specified locations.
        
        :param X: Locations of evaluations
        :return: Values of approximation at specified locations
        '''
        if self.coefficients:
            T = self.polynomial_space.evaluate_basis(X,derivative=derivative)
            return T.dot(np.array([self.coefficients[pol] for pol in self.polynomial_space.basis]).reshape(-1))
        else:
            return np.zeros(X.shape[0])
    
    def norm(self, pols=None):
        '''
        Return norm of projection onto polynomial subspace.
        
        :param mi: Multi-index
        :return: Norm of projection onto polynomial space described by mi
        '''
        if pols is None:
            pols=self.polynomial_space.basis
        return np.sqrt(self._sum_sq_coeff(pols))
    
    def __radd__(self, other):
        if other == 0:
            return copy.deepcopy(self)
        else:
            return self.__add__(other)
    
    def __add__(self, other):
        '''
        Add two polynomial approximations. 
        
        :rtype: Function
        '''
        new_coefficients=self.coefficients+other.coefficients
        new_ps=copy.deepcopy(self.polynomial_space)
        new_ps.basis=list(new_coefficients.keys())
        return PolynomialApproximation(ps=new_ps,coefficients=new_coefficients)
        
    def __rmul__(self,other):
        return self.__mul__(other) 
    
    def __mul__(self,other):
        new = copy.deepcopy(self)
        new.coefficients = other*self.coefficients
        return new
    
    def plot(self,ax = None):
        '''
        Plot polynomial approximation.
        '''
        if self.polynomial_space.get_c_var() == 1:
            X = self.polynomial_space.probability_distribution.get_range()
            Z = self(X)
            ax = ax or plg.figure().gca()
            ax.plot(X, Z)
        elif self.polynomial_space.get_c_var() == 2:
            X, Y = self.polynomial_space.probability_distribution.get_range()
            Z = grid_evaluation(X, Y, self)
            ax = ax or plt.figure().gca(projection='3d')
            ax.plot_surface(X, Y, Z,alpha=0.5)
        else:
            raise ValueError('Cannot plot {}-dimensional function'.format(self.polynomial_space.get_c_var()))
        return ax
        
