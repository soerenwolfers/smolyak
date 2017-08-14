'''
Multi-index weighted polynomial approximation
'''
from smolyak.applications.polynomials.weighted_polynomial_approximator import WeightedPolynomialApproximator
from smolyak.indices import MixedDifferences
from smolyak.aux.v_function import VFunction
from smolyak.aux.more_collections import DefaultDict
import copy
from smolyak import indices
from smolyak.indices import MultiIndex, cartesian_product

class MIWeightedPolynomialApproximator(object):
    r'''
    Maintains polynomial approximation of a given function 
    :math:`f\colon [a,b]^d\to\mathbb{R}`
    that can be sampled at different levels of accuracy.
    '''
    def __init__(self, function, c_dim_acc, ps,C=2,sampler='optimal',reparametrization=False):
        r'''
        :param function: Function :math:`[a,b]^d\to\mathbb{R}` that is being
        approximated. Needs to support :code:`__call__(X,mi)` where :code:`X` 
        is an np.array of size N x :math:`d` describing the sample locations and :code:`mi` is a
        multi-index describing the required accuracy of the samples.
        :param cdim_acc: Number of discretization parameters of `self.function`
        (=length of `mi` above)
        :param ps: Polynomial space
        :param C: see WeightedPolynomialApproximator
        :param sampler: see WeightedPolynomialApproximator
        :param reparametrization: Determines whether polynomial subspaces are indexed with 
        an exponential reparametrization
        :type reparametrization: Function MultiIndex to List-of-MultiIndex, or
            simply True, which corresponds to exponential reparametrization
        '''
        self.C=C
        self.sampler=sampler
        self.c_dim_acc = c_dim_acc
        if len(ps.basis) > 0:
            raise ValueError('Polynomial subspace must be empty on initialization')
        else:
            self.ps = ps
        self.WPAs = DefaultDict(default=self.__default_WPA)
        self.is_bundled = lambda dim: dim >= self.c_dim_acc
        def T(mi): 
            mi_acc = mi.mod(self.is_bundled)
            return VFunction(lambda X: function(X, mi_acc))
        self.function = function
        self.mixed_differences = MixedDifferences(T)
        self.reparametrization=reparametrization
                 
    def __default_WPA(self, mi_acc):
        return WeightedPolynomialApproximator(function=self.mixed_differences(mi_acc),
                                              ps=copy.deepcopy(self.ps),C=self.C,sampler=self.sampler)
    
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
        :rtype: (work,contribution) where work is real number and contribution is list of same length as mis  
        '''
        bundles=indices.get_bundles(mis, self.is_bundled)
        work = 0
        contributions=dict()
        for bundle in bundles:
            mis_pols,mi_acc= self.__handle_mis(bundle)
            work += self.WPAs[mi_acc].update_approximation(self._pols_from_mis(mis_pols))
            if work>0:
                pa = self.WPAs[mi_acc].get_approximation()
                contributions.update({mi_acc+mi.shifted(self.c_dim_acc): pa.norm(self._pols_from_mi(mi)) for mi in mis_pols})
        return work, contributions
    
    def reset(self):
        '''
        Delete all stored samples
        '''
        self.__init__(function=self.function, c_dim_acc=self.c_dim_acc, ps=self.ps,C=self.C,sampler=self.sampler)
    
    def estimated_work(self, mis):          
        '''
        Return number of samples that would be generated if instance were 
        expanded with given multi-index set.
        
        :param mis: (List of) multi-index
        :return: Number of new sampls
        '''
        bundles = indices.get_bundles(mis,self.is_bundled)
        work = 0
        for bundle in bundles:
            mis_pols, mi_acc = self.__handle_mis(bundle)
            work += self.WPAs[mi_acc].estimated_work(self._pols_from_mis(mis_pols))
        return work

    def __handle_mis(self, mis):
        mis_pols = [mi.shifted(-self.c_dim_acc) for mi in mis]
        mi_acc = mis[0].mod(self.is_bundled)
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