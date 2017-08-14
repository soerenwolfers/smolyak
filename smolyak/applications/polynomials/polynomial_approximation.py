import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport
from smolyak.aux.np_tools import grid_evaluation
import copy
from smolyak.aux.more_collections import RFunction

class PolynomialApproximation(object):
    def __init__(self,ps=None,coefficients=None):
        '''
        
        :param ps: Polynomial subspace in which approximation lives
        :type ps: PolynomialSubspace
        :param coefficients: Coefficients with respect to basis determined by
            :code:`ps`
        :type coefficients: Dictionary ps.basis->reals
        '''
        if set(coefficients.keys()) != set(ps.basis):
            raise ValueError('Coefficients do not match polynomial basis')
        self.coefficients=RFunction(coefficients)
        self.ps = ps
        
    def _sum_sq_coeff(self, pols=None):
        '''
        Return sum of squared coefficients of given polynomials
        
        :param mi: Multi-index
        :return: Sum of squares of coefficients corresponding to mi
        '''
        if not pols:
            pols=self.ps.basis
        return sum([self.coefficients[pol] ** 2 for pol in pols])  
        
    def __call__(self, X):
        '''
        Return approximation at specified locations.
        
        :param X: Locations of evaluations
        :return: Values of approximation at specified locations
        '''
        return self.ps.evaluate_basis(X).dot(np.array([self.coefficients[pol] for pol in self.ps.basis]).reshape(-1, 1))
    
    def norm(self, pols=None):
        '''
        Return norm of projection onto polynomial subspace.
        
        :param mi: Multi-index
        :return: Norm of projection onto polynomial space described by mi
        '''
        if not pols:
            pols=self.ps.basis
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
        new_ps=copy.deepcopy(self.ps)
        new_ps.basis=new_coefficients.keys()
        return PolynomialApproximation(ps=new_ps,coefficients=new_coefficients)
        
    def __rmul__(self,other):
        return self.__mul__(other) 
    
    def __mul__(self,other):
        new = copy.deepcopy(self)
        new.coefficients = other*self.coefficients
        return new
    
    def plot(self, L=10):
        '''
        Plot polynomial approximation.
        '''
        fig = plt.figure()
        if self.ps.get_c_var() == 1:
            X = self.ps.probability_space.get_range()
            Z = self(X)
            ax = fig.gca()
            ax.plot(X, Z)
        elif self.ps.get_c_var() == 2:
            X, Y = self.ps.probability_space.get_range()
            Z = grid_evaluation(X, Y, self)
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z)
        fig.suptitle('Polynomial approximation')
        plt.show()
        