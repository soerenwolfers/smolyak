'''
Multi-index weighted polynomial approximation
'''
import numpy as np
from smolyak.applications.polynomials.polynomial_approximation import PolynomialApproximation
import math
from smolyak.applications.polynomials import samples
import timeit
import matplotlib.pyplot as plt
from smolyak.applications.polynomials.polynomial_spaces import PolynomialSpace
from smolyak.indices import MixedDifferences
from swutil.v_function import VFunction
from swutil.collections import DefaultDict
import copy
from smolyak import indices
from smolyak.indices import MultiIndex, cartesian_product


class WeightedPolynomialApproximator(object):
    r'''
    Maintains polynomial approximation of a given function 
    :math:`f\colon [a,b]^d\to\mathbb{R}`
    that can be sampled at different levels of accuracy.
    '''
    def __init__(self, function, probability_space,C=2,sampler='optimal',reparametrization=False,n_acc=0):
        r'''
        :param function: Function :math:`[a,b]^d\to\mathbb{R}` that is being
        approximated. Needs to support :code:`__call__(X,mi)` where :code:`X` 
        is an np.array of size N x :math:`d` describing the sample locations and :code:`mi` is a
        multi-index describing the required accuracy of the samples.
        :param cdim_acc: Number of discretization parameters of `self.function`
        (=length of `mi` above)
        :param probability_space: Probability space
        :param C: see _WeightedPolynomialApproximator
        :param sampler: see _WeightedPolynomialApproximator
        :param reparametrization: Determines whether polynomial subspaces are indexed with 
        an exponential reparametrization
        :type reparametrization: Function MultiIndex to List-of-MultiIndex, or
            simply True, which corresponds to exponential reparametrization
        '''
        self.C=C
        self.probability_space = probability_space
        self.sampler=sampler
        self.n_acc = n_acc
        self.WPAs = DefaultDict(default=self.__default_WPA)
        self.bundled_dims = lambda dim: dim >= self.n_acc
        self.function = function
        if self.n_acc>0:
            def T(mi_acc): 
                #mi_acc = mi.mod(self.bundled_dims)
                return VFunction(lambda X: self.function(X, mi_acc))
            mdT=MixedDifferences(T)
            self.function_delta = lambda mi_acc: mdT(mi_acc)
        else:#there will only ever be one function_delta request, namely the function itself.
            self.function_delta = lambda mi_acc: self.function    
        self.reparametrization=reparametrization
     
    def plot_samples(self,mi_acc=None):
        if mi_acc == None and self.n_acc>0:
            raise ValueError('Must specify which delta to plot')
        self.WPAs[mi_acc].plot_xy()
                 
    def __default_WPA(self, mi_acc):
        return _WeightedPolynomialApproximator(function=self.function_delta(mi_acc),
                                              probability_space=copy.deepcopy(self.probability_space),C=self.C,sampler=self.sampler)
    
    #def get_active_dims(self):
    #    return set.union(*[WPA.get_active_dims() for WPA in self.WPAs.values()]) 
    
    def get_approximation(self):
        '''
        Returns polynomial approximation
        
        :return: Polynomial approximation of :math:`f\colon [a,b]^d\to\mathbb{R}`
        :rtype: Function
        '''
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
            mis_pols,mi_acc= self.__handle_mis(bundle)
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
        self.__init__(function=self.function, n_acc=self.n_acc, probability_space=self.probability_space,C=self.C,sampler=self.sampler)
    
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
            mis_pols, mi_acc = self.__handle_mis(bundle)
            work += self.WPAs[mi_acc].estimated_work(self._pols_from_mis(mis_pols))
        return work

    def __handle_mis(self, mis):
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
                    init_range = 2 ** (mi[dimension] - 1)
                    end_range = 2 ** (mi[dimension])
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
        
class _WeightedPolynomialApproximator(object):
    '''
    Maintains polynomial approximation of given function on :math:`[a,b]^d`.
    '''
    def __init__(self, function, probability_space, C=2,sampler='optimal'):
        ''' 
        :param function: Function that is approximated. Needs to support __call__
        :param probability_space: Probability space used for approximation
        :type probability_space: ProbabilitySpace
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
        polynomial_space = PolynomialSpace(probability_space)
        self.probability_space = polynomial_space
        self.X = np.zeros((0, self.probability_space.get_c_var()))
        self.Y = np.zeros((0, 1))
        self.W = np.zeros((0, 1))
        self._recompute_approximation = True
        
    def c_samples_from_c_pols(self,dim):
        return (4 if dim == 1 else 0) + math.ceil(self.C * dim * np.log2(dim + 1))

    def estimated_work(self,pols):
        '''
        Return number of new samples required to determine polynomial 
        coefficients onto given polynomial subspace
        
        :param pols 
        :return: Number of required samples
        '''

        c_pols = len(pols)
        c_samples = self.c_samples_from_c_pols(c_pols)
        if self.keep:
            return int(c_samples - self.X.shape[0])
        else:
            return int(c_samples) 
        
    def update_approximation(self, pols):
        '''
        Get near-optimal projection onto subspace described by mis
        
        :param mis: List of multi-indices describing polynomial span to be added
        :return: Time to compute new samples
        '''
        c_samples = self.estimated_work(pols)
        if c_samples > 0:
            self._recompute_approximation=True
            self.probability_space.set_basis(pols)
            if self.sampler == 'arcsine':
                if self.X.shape[1] < self.probability_space.get_c_var():
                    c_samples += self.X.shape[0]  # contrary to previous belief we cannot keep samples, thus need to add the number of old ones
                    self.X = np.zeros((0, self.probability_space.get_c_var()))
                    self.Y = np.zeros((0, 1))
                    self.W = np.zeros((0, 1))
                (Xnew,Wnew) = samples.importance_samples(self.probability_space.probability_space, c_samples, 'arcsine')
                self.X = np.concatenate((self.X, Xnew), axis=0)  
                self.W = np.concatenate([self.W, Wnew])  
                tic = timeit.default_timer()
                Y_new=self.function(Xnew).reshape(-1,1)
                if not Y_new.shape[0]==Xnew.shape[0]:
                    raise ValueError('Function must return as many outputs values as it is given inputs')
                self.Y = np.concatenate([self.Y, Y_new])
                work_model = timeit.default_timer() - tic
            else: 
                (self.X, self.W) = samples.optimal_samples(self.probability_space, c_samples)
                tic = timeit.default_timer()
                self.Y = self.function(self.X).reshape(-1, 1)
                if not self.Y.shape[0]==self.X.shape[0]:
                    raise ValueError('Function must return as many outputs values as it is given inputs')
                work_model = timeit.default_timer() - tic
            return work_model,self.get_contributions()
        else:
            return 0,self.get_contributions()
        
    def get_contributions(self):
        pa=self.get_approximation()
        return {pol:pa.norm([pol]) for pol in self.probability_space.basis}
    
    def get_approximation(self):
        if self._recompute_approximation:
            self._approximation =  PolynomialApproximation(self.probability_space,self.probability_space.weighted_least_squares(self.X, self.W, self.Y))
            self._recompute_approximation = False
        return self._approximation
        
    def plot_xy(self):
        '''
        Scatter plot of stored samples.
        '''
        fig = plt.figure()
        if self.probability_space.get_c_var() == 1:
            order = np.argsort(self.X, axis=0).reshape(-1)
            ax = fig.gca()
            ax.scatter(self.X[order, :], self.Y[order, :])
        elif self.probability_space.get_c_var() == 2:
            ax = fig.gca(projection='3d')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.Y)
        fig.suptitle('Samples')
        plt.show()