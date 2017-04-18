'''
Multi-index weighted polynomial approximation
'''
from smolyak.applications.polreg.weighted_polynomial_approximator import WeightedPolynomialApproximator
from smolyak.sparse.mixed_differences import MixedDifferences
from smolyak.misc.v_function import VFunction
#import timeit
from smolyak.misc.default_dict import DefaultDict
import copy

class MIWeightedPolynomialApproximator(object):
    r'''
    Maintains polynomial approximation of a given function 
    :math:`f\colon [a,b]^d\to\mathbb{R}`
    that can be sampled at different levels of accuracy.
    '''
    def __init__(self, function, c_dim_acc, ps):
        r'''
        :param function: Function :math:`[a,b]^d\to\mathbb{R}` that is being
        approximated. Needs to support :code:`__call__(X,mi))` where :code:`X` 
        is an np.array of size N x :math:`d` describing the sample locations and :code:`mi` is a
        multi-index describing the required accuracy of the samples.
        :param cdim_acc: Number of discretization parameters of `self.function`
        (=length of `mi` above)
        :param ps: Polynomial space
        '''
        self.c_dim_acc = c_dim_acc
        if len(ps.pols) > 0:
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
                 
    def __default_WPA(self, mi_acc):
        return WeightedPolynomialApproximator(function=self.mixed_differences(mi_acc),
                                              ps=copy.deepcopy(self.ps))
    
    def get_active_dims(self):
        return set.union(*[WPA.get_active_dims() for WPA in self.WPAs.values()]) 
    
    def get_approximation(self):
        '''
        Returns polynomial approximation
        
        :return: Polynomial approximation of :math:`f\colon [a,b]^d\to\mathbb{R}`
        :rtype: Function
        '''
        return sum([self.WPAs[mi_acc] for mi_acc in self.WPAs])
           
    def expand(self, mis):
        r'''
        Expands polynomial approximation.
        
        Converts list of multi-indices into part that describes polynomial basis
        and part that describes accuracy of samples that are being used, then 
        expands polynomial approximation of self.functon based on this information.
        
        
        :param mis: (List of) multi-index. 
        :return: work and contribution associated to mis
        :rtype: (work,contribution) where work is real number and contribution is list of same length as mis  
        '''
        #tic = timeit.default_timer()
        mis_pol, mi_acc = self.__handle_mis(mis)
        work = self.WPAs[mi_acc].expand(mis_pol)
        contributions = [self.WPAs[mi_acc].norm(mi_pol) for mi_pol in mis_pol]
        #work = timeit.default_timer() - tic
        return work, contributions
    
    def reset(self):
        '''
        Delete all stored samples
        '''
        self.__init__(function=self.function, c_dim_acc=self.c_dim_acc, ps=self.ps)
    
    def required_samples(self, mis):          
        '''
        Return number of samples that would be generated if instance were 
        expanded with given multi-index set.
        
        :param mis: (List of) multi-index
        :return: Number of new sampls
        '''
        mis_pol, mi_acc = self.__handle_mis(mis)
        work = self.WPAs[mi_acc].c_samples(mis_pol)
        return work

    def __handle_mis(self, mis):
        if not hasattr(mis, '__contains__'):  # is single mi?
            mis = [mis]
        mis_pol = [mi.leftshift(self.c_dim_acc) for mi in mis]
        mi_acc = mis[0].mod(self.is_bundled)
        return mis_pol, mi_acc
            
