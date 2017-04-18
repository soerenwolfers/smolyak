'''
Weighted Polynomial approximation
'''
import numpy as np
import functools
import timeit
from operator import mul
import math
from smolyak.sparse.sparse_index import cartesian_product, SparseIndex
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport
from smolyak.misc.z_func import z_func
WARNING = False

class WeightedPolynomialApproximator(object):
    '''
    Maintains polynomial approximation of given function on :math:`[a,b]^d`.
    '''
    def __init__(self, function, ps, C=2):
        ''' 
        :param function: Function that is approximated. Needs to support __call__
        :param ps: Polynomial subspace used for approximation
        :type ps: PolynomialSubspace instance
        :param C: Multiplier for number of samples used for reconstruction
        :type C: Optional. Positive real
        '''
        self.C = C
        self.function = function
        self.mis = []
        self.coefficients = {}
        self.ps = ps
        self.X = np.zeros((0, self.ps.get_c_var()))
        self.Y = np.zeros((0, 1))
        self.W = np.zeros((0, 1))
        self.c_samples_from_c_pols = lambda dim: (4 if dim == 1 else 0) + math.ceil(self.C * dim * np.log2(dim + 1))
        
    def __sum_sq_coeff_from_mi(self, mi):
        '''
        Return sum of squared coefficients
        
        :param mi: Multi-index
        :return: Sum of squares of coefficients corresponding to mi
        '''
        return sum([self.coefficients[pol] ** 2 for pol in self.__pols_from_mi(mi)])  
    
    def __pols_from_mi(self, mi):
        '''
        Convert multi-index to corresponding polynomials
        
        :param mi: Multi-index
        :return: List of polynomials corresponding to mi
        '''
        if mi == SparseIndex():
            return [SparseIndex()]
        else:
            univariate_entries = []
            for dimension in mi.active_dims():
                init_range = 2 ** (mi[dimension] - 1)
                end_range = 2 ** (mi[dimension])
                univariate_entries.append(range(init_range, end_range))
            return cartesian_product(univariate_entries, mi.active_dims())
        
    def __c_pols_from_mis(self, mis): 
        '''
        Return number of polynomials corresponding to multi-index set
        
        :param mis: List of multi-indices (no repetitions allowed)
        :return: Number of polynomials associated with mis
        '''
        return np.sum([functools.reduce(mul, [2 ** (mi[dim] - 1) if mi[dim] > 0 else 1 for dim in mi.active_dims()]) 
                       if mi.active_dims() else 1 for mi in mis])
        
    def c_samples(self, mis):
        '''
        Return number of new samples required to determine polynomial 
        coefficients if new multi-indices are added to approximation 
        
        :param mis: Multi-indices to be added
        :return: Number of required samples
        '''
        c_pols = self.__c_pols_from_mis(set(mis) | set(self.mis))
        c_samples = self.c_samples_from_c_pols(c_pols)
        if self.ps.sampler == 'arcsine':
            return int(c_samples - self.X.shape[0])
        else:
            return int(c_samples)
        
    def expand(self, mis):
        '''
        Expand basis and sample accordingly to obtain new approximation
        
        :param mis: List of multi-indices describing polynomial span to be added
        '''
        c_samples = self.c_samples(mis)
        if c_samples > 0:
            for mi in mis:
                if not mi in self.mis:
                    self.mis.append(mi)
                    pols = self.__pols_from_mi(mi)
                    self.ps.expand_pols(pols)
            if self.ps.sampler == 'arcsine':
                if self.X.shape[1] < self.ps.c_var:
                    c_samples += self.X.shape[0]  # unlike prior believe we cannot keep samples, thus need to add the number of old ones
                    self.X = np.zeros((0, self.ps.c_var))
                    self.Y = np.zeros((0, 1))
                    self.W = np.zeros((0, 1))
                (Xnew, Wnew) = self.ps.generate_samples(c_samples)
                self.X = np.concatenate((self.X, Xnew), axis=0)  
                self.W = np.concatenate([self.W, Wnew])  
                tic = timeit.default_timer()
                self.Y = np.concatenate([self.Y, self.function(Xnew).reshape(-1, 1)])
                work_model = timeit.default_timer() - tic
            else: 
                (self.X, self.W) = self.ps.generate_samples(c_samples)
                tic = timeit.default_timer()
                self.Y = self.function(self.X).reshape(-1, 1)
                work_model = timeit.default_timer() - tic
            self.coefficients = self.ps.weighted_least_squares(self.X, self.W, self.Y)
            return work_model
      
    def __call__(self, X):
        '''
        Return approximation at specified locations.
        
        :param X: Locations of evaluations
        :return: Values of approximation at specified locations
        '''
        return self.ps.evaluate_basis(X).dot(np.array([self.coefficients[pol] for pol in self.ps.pols]).reshape(-1, 1))
        
    def norm(self, mi):
        '''
        Return norm of projection onto polynomial subspace.
        
        :param mi: Multi-index
        :return: Norm of projection onto polynomial space described by mi
        '''
        return np.sqrt(self.__sum_sq_coeff_from_mi(mi))
    
    def __radd__(self, other):
        if other == 0:
            def evaluator(X):
                return self(X)
            return evaluator
        else:
            return self.__add__(other)
    
    def __add__(self, other):
        '''
        Add two polynomial approximations. 
        
        :rtype: Function
        '''
        def evaluator(X):
            return self(X) + other(X)
        return evaluator
    
    def plot(self, L=10):
        '''
        Plot polynomial approximation.
        '''
        fig = plt.figure()
        if self.ps.c_var == 1:
            X = self.ps.get_range()
            Z = self(X)
            ax = fig.gca()
            ax.plot(X, Z)
        elif self.ps.c_var == 2:
            X, Y = self.ps.get_range()
            Z = z_func(self, X, Y)
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z)
        fig.suptitle('Polynomial approximation')
        plt.show()
    
    def plot_xy(self):
        '''
        Scatter plot of stored samples.
        '''
        fig = plt.figure()
        if self.ps.c_var == 1:
            order = np.argsort(self.X, axis=0).reshape(-1)
            ax = fig.gca()
            ax.scatter(self.X[order, :], self.Y[order, :])
        elif self.ps.c_var == 2:
            ax = fig.gca(projection='3d')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.Y)
        fig.suptitle('Samples')
        plt.show()
        
    def get_active_dims(self):
        if self.mis:
            return set.union(*[set(mi.active_dims()) for mi in self.mis]) 
        else:
            return set()
