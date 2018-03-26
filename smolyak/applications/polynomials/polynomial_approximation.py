import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport @UnresolvedImport
from swutil.np_tools import grid_evaluation
import copy
from swutil.collections import RFunction
from smolyak.applications.polynomials.probability_spaces import ProbabilitySpace
from smolyak.applications.polynomials.polynomial_spaces import PolynomialSpace

class PolynomialApproximation(object):
    def __init__(self,ps,coefficients=None,basis = None):
        '''
        
        :param polynomial_space: Polynomial subspace in which approximation lives
        :type polynomial_space: PolynomialSubspace
        :param coefficients: Coefficients with respect to basis determined by
            :code:`polynomial_space`
        :type coefficients: Dictionary polynomial_space.basis->reals
        '''
        if isinstance(ps,ProbabilitySpace):
            self.polynomial_space = PolynomialSpace(ps)
            if basis is None:
                if hasattr(coefficients,'keys'):
                    self.polynomial_space.set_basis(coefficients.keys())
                else:
                    raise ValueError('Must specfiy either polynomial space or probability space and basis or probability space and coefficients dictionary whose keys determine basis')
            else:
                self.polynomial_space.set_basis(basis)
        else:
            self.polynomial_space = ps
            if basis is not None:
                self.polynomial_space.set_basis(basis)
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
        print(T.shape)
        return T.dot(np.array([self.coefficients[pol] for pol in self.polynomial_space.basis]).reshape(-1, 1))
    
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
        fig.suptitle('Polynomial approximation')
        