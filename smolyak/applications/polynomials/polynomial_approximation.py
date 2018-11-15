import warnings
import math
import copy
import timeit

import numpy as np
from scipy.linalg import solve
from scipy.linalg.misc import LinAlgError
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport @UnresolvedImport

from swutil.np_tools import grid_evaluation
from swutil.collections import RFunction
from swutil.v_function import VFunction
from swutil.collections import DefaultDict
from swutil.np_tools import grid_evaluation

from smolyak.applications.polynomials import samples
from smolyak.indices import MixedDifferences
from smolyak import indices
from smolyak.indices import MultiIndex, cartesian_product
from smolyak.applications.polynomials.probability_spaces import ProbabilityDistribution,\
    ProductProbabilityDistribution, ProbabilitySpace
from smolyak.applications.polynomials.orthogonal_polynomials import evaluate_orthonormal_polynomials
class PolynomialSpace:
    '''
    Vector space of polynomials for least-squares polynomial approximation on open 
    subsets of Euclidean space.
    
    Combines domain (in form of probability space) with description
    of finite-dimensional subspace of polynomials
    '''
    
    def __init__(self, probability_space,warnings = False):
        if isinstance(probability_space,ProbabilityDistribution):
            probability_space = ProductProbabilityDistribution(probability_space)
        self.probability_space=probability_space
        self.basis = []
        self.warnings = warnings
    
    def get_dimension(self): 
        '''
        Return dimension of polynomial subspace (not of domain, use get_c_var() for that)
        '''
        return len(self.basis)
    
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
    
    def weighted_least_squares(self, X, W, Y):
        '''
        Compute least-squares approximation. 
        
        :param X: sample locations
        :type X: `N x self.get_c_var()` numpy array
        :param W: weights
        :param Y: sample values
        :return: coefficients
        '''
        B = self.evaluate_basis(X)
        W = W.reshape((W.size, 1))
        R = B.transpose().dot(Y * W)
        G = B.transpose().dot(B * W)
        if self.warnings and np.linalg.cond(G) > 3:
            warnings.warn('Ill conditioned Gramian matrix encountered') 
        if G.shape[0]>0:
            coefficients = solve(G, R, sym_pos=True)
        if self.warnings and not np.isfinite(coefficients).all():
            warnings.warn('Numerical instability encountered')
        return {pol: coefficients[i] for i, pol in enumerate(self.basis)}
         
    def get_active_dims(self):
        if self.basis:
            return set.union(*[set(pol.active_dims()) for pol in self.basis]) 
        else:
            return set()
    
    def get_c_var(self):
        '''
        Return dimension of domain (not polynomial subspace, use get_dimension() for that)
        '''
        return self.probability_space.get_c_var()

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
        if any(order>0 and dim not in active_vars for dim,order in enumerate(derivative)):
            values[:] = 0
        elif active_vars:
            for dim in active_vars:
                basis_values[dim] = evaluate_orthonormal_polynomials(
                    X[:,dim],
                    max(pol[dim] for pol in self.basis), 
                    measure=self.probability_space.ups[dim].measure, 
                    interval=self.probability_space.ups[dim].interval,
                    derivative = derivative[dim]
                )
            values = np.empty_like(values)
            projected_basis = [pol.retract(lambda i: active_vars[i]).full_tuple(c_dim = len(active_vars)) for pol in self.basis]
            prelim_values = {tuple():np.ones((X.shape[0]))}
            for d in range(len(active_vars)-1):
                prelim_values_new = {}
                for i,pol in enumerate(projected_basis):
                    initial = pol[:d+1]
                    if initial not in prelim_values_new:
                        if pol[d] == 0:
                            prelim_values_new[initial] = prelim_values[initial[:-1]]
                        else:
                            prelim_values_new[initial] = prelim_values[initial[:-1]]*basis_values[active_vars[d]][:,pol[d]]
                prelim_values = prelim_values_new
            for i,pol in enumerate(self.basis):
                if any(order>0 and dim not in pol.active_dims() for dim,order in enumerate(derivative)):#Check if necessary
                    values[:,i] = 0                                                                                      
                pre_initial = projected_basis[i][:-1]
                if pol[active_vars[-1]] == 0:
                    values[:,i] = prelim_values[pre_initial]
                else:
                    values[:,i] = prelim_values[pre_initial]*basis_values[active_vars[-1]][:,pol[active_vars[-1]]]
        return values
    
    def set_basis(self,polynomials):
        self.basis=[]
        self.expand_basis(polynomials)
        
    def expand_basis(self, polynomials):
        '''
        Expand polynomial subspace.
        
        If necessary, expand dimension of domain, as well. 
        
        :param polynomials: Polynomials to be added
        '''
        self.basis += polynomials
        max_var = max([pol.max_dim()for pol in self.basis])
        if max_var > self.get_c_var():
            for __ in range(max_var - self.get_c_var()):
                self.probability_space.ups.append(copy.deepcopy(self.probability_space.ups[-1]))
        
    def plot_optimal_distribution(self, N=200, L=1):
        '''
        Plot optimal sampling distribution
        '''
        if self.get_c_var() == 1:
            X = self.probability_space.get_range(N, L)
            Z = self.probability_space.lebesgue_density(X) / self.optimal_weights(X)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(X, Z)
        elif self.get_c_var() == 2:
            X, Y = self.probability_space.get_range(N, L)
            Z = grid_evaluation(X, Y,self.probability_space.lebesgue_density) / grid_evaluation(X, Y,self.optimal_weights)
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
    def __init__(self, function,domain=None,C=1,sampler='optimal',reparametrization=False,n = None, n_acc=0,warnings = False):
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
        if isinstance(domain,ProbabilitySpace):
            self.probability_distribution = domain
        elif domain:
            self.probability_distribution = ProbabilityDistribution(interval = domain[0])
            for dom in domain[1:]:
                self.probability_distribution = self.probability_distribution * ProbabilityDistribution(interval = dom)
        else:
            self.probability_distribution = ProbabilityDistribution()**n
        self.sampler=sampler
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
                #mi_acc = mi.mod(self.bundled_dims)
                return VFunction(lambda X: self.function(mi_acc,X))
            mdT=MixedDifferences(T)
        get_wpa = lambda mi_acc: SinglelevelPolynomialApproximator(
            function = mdT(mi_acc) if self.n_acc else self.function,
            probability_distribution = copy.deepcopy(self.probability_distribution),
            C = self.C,
            sampler = self.sampler,
            warnings = self.warnings
        )    
        self.WPAs = DefaultDict(default=get_wpa)
        self.reparametrization=reparametrization
     
    def plot_samples(self,mi_acc=None):
        if mi_acc is None and self.n_acc>0:
            raise ValueError('Must specify which delta to plot')
        if not isinstance(mi_acc,MultiIndex):
            mi_acc = MultiIndex(mi_acc)
        if mi_acc not in self.WPAs:
            raise ValueError('Only have samples with accuracy parameters {}'.format(list(self.WPAs.keys())))
        self.WPAs[mi_acc].plot_samples()
                 
    #def get_active_dims(self):
    #    return set.union(*[WPA.get_active_dims() for WPA in self.WPAs.values()]) 
    
    def get_approximation(self,mi_acc=None):
        '''
        Returns polynomial approximation
        
        :return: Polynomial approximation of :math:`f\colon [a,b]^d\to\mathbb{R}`
        :rtype: Function
        '''
        if mi_acc is not None:
            if not isinstance(mi_acc,MultiIndex):
                mi_acc = MultiIndex(mi_acc)
            if mi_acc not in self.WPAs:
                raise ValueError('Only have approximation of Deltas with accuracy parameters {}'.format(list(self.WPAs.keys())))
            return self.WPAs[mi_acc].get_approximation()
        else:
            return sum([self.WPAs[mi_acc].get_approximation() for mi_acc in self.WPAs])
           
    def update_approximation(self, mis):
        r'''
        Expands polynomial approximation.
        
        Converts list of multi-indices into part that describes polynomial basis
        and part that describes accuracy of samples that are being used, then 
        expands polynomial approximation of self.functon based on this information.
        
        
        :param mis: Multi-indices 
        :return: work and contribution associated to mis
        :rtype: (work,contribution) where work is real number and contribution is dictionary mis>reals
        '''
        bundles=indices.get_bundles(mis, self.bundled_dims)
        work = 0
        contributions=dict()
        for bundle in bundles:
            mis_pols,mi_acc= self._handle_mis(bundle)
            (new_work,_) = self.WPAs[mi_acc].update_approximation(self._pols_from_mis(mis_pols))
            work+=new_work
            #if work>0:
            pa = self.WPAs[mi_acc].get_approximation()
            contributions.update({mi_acc+mi.shifted(self.n_acc): pa.norm(self._pols_from_mi(mi)) for mi in mis_pols})
        return work, contributions
    
    def reset(self):
        '''
        Delete all stored samples
        '''
        self.__init__(function=self.function, n_acc=self.n_acc, n = self.n , 
            warnings = self.warnings,domain=self.probability_distribution,C=self.C,sampler=self.sampler,reparametrization = self.reparametrization)
    
    def estimated_work(self, mis):          
        '''
        Return number of samples that would be generated if instance were 
        expanded with given multi-index set.
        
        :param mis: (List of) multi-index
        :return: Number of new sampls
        '''
        bundles = indices.get_bundles(mis,self.bundled_dims)
        work = 0
        for bundle in bundles:
            mis_pols, mi_acc = self._handle_mis(bundle)
            work += self.WPAs[mi_acc].estimated_work(self._pols_from_mis(mis_pols))
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
        if sampler != 'optimal' and sampler != 'arcsine' and sampler != 'MCMC':
            raise ValueError('Sampling strategy not supported')
        else:
            if sampler == 'optimal':
                self.keep = False
            elif sampler == 'arcsine':
                self.keep = True
            elif sampler == 'MCMC':
                self.keep = False
        self.sampler=sampler
        self.C = C
        self.function = function
        polynomial_space = PolynomialSpace(probability_distribution,warnings= warnings)
        self.polynomial_space = polynomial_space
        self.X = np.zeros((0, self.polynomial_space.get_c_var()))
        self.Y = np.zeros((0, 1))
        self.W = np.zeros((0, 1))
        self._recompute_approximation = True
        
    def c_samples_from_c_pols(self,dim):
        return int((4 if dim == 1 else 0) + math.ceil(self.C * dim * np.log2(dim + 1)))

    def estimated_work(self,pols):
        '''
        Return number of new samples required to determine polynomial 
        coefficients onto given polynomial subspace
        
        :param pols 
        :return: Number of required samples
        '''
        if all(pol in self.polynomial_space.basis for pol in pols):
            return 0
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
        c_samples = self.estimated_work(pols)
        #if  c_samples > 0:#Can lead to errors when c_samples_from_c_pols is not strictly monotonic
        self._recompute_approximation=True
        old_basis = self.polynomial_space.basis
        self.polynomial_space.set_basis(pols)
        if self.sampler == 'arcsine':
            if self.X.shape[1] < self.polynomial_space.get_c_var():
                c_samples += self.X.shape[0]  # contrary to previous belief we cannot keep samples, thus need to add the number of old ones
                self.X = np.zeros((0, self.polynomial_space.get_c_var()))
                self.Y = np.zeros((0, 1))
                self.W = np.zeros((0, 1))
            (Xnew,Wnew) = samples.importance_samples(self.polynomial_space.probability_distribution, c_samples, 'arcsine')
            self.X = np.concatenate((self.X, Xnew), axis=0)  
            self.W = np.concatenate([self.W, Wnew])  
            tic = timeit.default_timer()
            Ynew=self.function(Xnew).reshape(-1,1)
            if not Ynew.shape[0]==Xnew.shape[0]:
                raise ValueError('Function must return as many outputs values as it is given inputs')
            self.Y = np.concatenate([self.Y, Ynew])
            work_model = timeit.default_timer() - tic
        else: 
            #(self.X, self.W) = samples.optimal_samples(self.polynomial_space, c_samples)
            #tic = timeit.default_timer()
            #self.Y = self.function(self.X).reshape(-1, 1)
            #if not self.Y.shape[0]==self.X.shape[0]:
            #    raise ValueError('Function must return as many outputs values as it is given inputs')
            #work_model = timeit.default_timer() - tic
            tic=timeit.default_timer()
            (Xnew,Wnew) = samples.samples_per_polynomial(self.polynomial_space,old_basis, pols,self.C)
            self.X = np.concatenate((self.X,Xnew),axis=0)
            self.W = np.concatenate([self.W,Wnew.reshape(-1,1)])
            tic = timeit.default_timer()
            Ynew = self.function(Xnew).reshape(-1,1)
            if not Ynew.shape[0] == Xnew.shape[0]:
                raise ValueError('Function must return as many outputs values as it is given inputs')
            self.Y = np.concatenate([self.Y, Ynew])
            work_model = timeit.default_timer() - tic
        return work_model,self.get_contributions()
        
    def get_contributions(self):
        pa=self.get_approximation()
        return {pol:pa.norm([pol]) for pol in self.polynomial_space.basis}
    
    def get_approximation(self):
        if self._recompute_approximation:
            old_err_settings = np.seterr(all='raise')
            while self._recompute_approximation:
                try:
                    self._approximation =  PolynomialApproximation(self.polynomial_space,self.polynomial_space.weighted_least_squares(self.X, self.W, self.Y))
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
        
    def plot_samples(self):
        '''
        Scatter plot of stored samples.
        '''
        fig = plt.figure()
        if self.polynomial_space.get_c_var() == 1:
            order = np.argsort(self.X, axis=0).reshape(-1)
            ax = fig.gca()
            ax.scatter(self.X[order, :], self.Y[order, :])
        elif self.polynomial_space.get_c_var() == 2:
            ax = fig.gca(projection='3d')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.Y)
        fig.suptitle('Samples')
        plt.show()

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
        T = self.polynomial_space.evaluate_basis(X,derivative=derivative)
        return T.dot(np.array([self.coefficients[pol] for pol in self.polynomial_space.basis]).reshape(-1))
    
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
    
    def plot(self):
        '''
        Plot polynomial approximation.
        '''
        fig = plt.figure()
        if self.polynomial_space.get_c_var() == 1:
            X = self.polynomial_space.probability_space.get_range()
            Z = self(X)
            ax = fig.gca()
            ax.plot(X, Z)
        elif self.polynomial_space.get_c_var() == 2:
            X, Y = self.polynomial_space.probability_space.get_range()
            Z = grid_evaluation(X, Y, self)
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z)
        else:
            raise ValueError('Cannot plot {}-dimensional function'.format(self.polynomial_space.get_c_var()))
        fig.suptitle('Polynomial approximation')
        
