import copy
import warnings
import numpy as np

from scipy.linalg import solve
from matplotlib import cm
import matplotlib.pyplot as plt

from swutil.np_tools import grid_evaluation
from smolyak.applications.polynomials.probability_spaces import UnivariateProbabilitySpace,\
    TensorProbabilitySpace
from smolyak.applications.polynomials.orthogonal_polynomials import evaluate_orthonormal_polynomials

class PolynomialSpace():
    '''
    Vector space of polynomials for least-squares polynomial approximation on open 
    subsets of Euclidean space.
    
    Combines domain (in form of probability space) with description
    of finite-dimensional subspace of polynomials
    '''
    
    def __init__(self, probability_space,warnings = False):
        if isinstance(probability_space,UnivariateProbabilitySpace):
            probability_space = TensorProbabilitySpace(probability_space)
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
        if self.warnings and np.linalg.cond(G) > 100:
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
            active_vars = set.union(*[set(pol.active_dims()) for pol in self.basis]) 
        else:
            active_vars = set()
        if not derivative:
            derivative = np.zeros(self.get_c_var())
        if any(order>0 and dim not in active_vars for dim,order in enumerate(derivative)):
            values[:] = 0
        else:
            for dim in active_vars:
                basis_values[dim] = evaluate_orthonormal_polynomials(
                    X[:,dim],
                    max(pol[dim] for pol in self.basis), 
                    measure=self.probability_space.ups[dim].measure, 
                    interval=self.probability_space.ups[dim].interval,
                    derivative = derivative[dim]
                )
            for i, pol in enumerate(self.basis):
                if any(order>0 and dim not in pol.active_dims() for dim,order in enumerate(derivative)):
                    values[:,i] = 0
                else:
                    for dim in pol.active_dims():
                        values[:, i] *= basis_values[dim][:, pol[dim]]   
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
        max_var = max([max(pol.active_dims()) + 1 if pol.active_dims() else 0 for pol in self.basis])#else 0 or else 1?
        if max_var > self.get_c_var():
            for __ in range(max_var - self.get_c_var()):
                self.probability_space.ups.append(copy.deepcopy(self.probability_space.ups[-1]))
        
    def plot_optimal_distribution(self, N=200, L=1):
        '''
        Plot optimal sampling distribution
        '''
        if self.get_c_var() == 1:
            X = self.get_range(N, L)
            Z = self.probability_space.lebesgue_density(X) / self.optimal_weights(X)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(X, Z)
        elif self.get_c_var() == 2:
            X, Y = self.get_range(N, L)
            Z = grid_evaluation(X, Y,self.probability_space.lebesgue_density) / grid_evaluation(X, Y,self.optimal_weights)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5)  # @UndefinedVariable
        fig.suptitle('Lebesgue density of optimal distribution')
        plt.show()
        return np.sum(Z) / Z.size
