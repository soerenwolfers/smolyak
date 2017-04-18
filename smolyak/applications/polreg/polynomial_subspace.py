from abc import ABCMeta, abstractmethod
import numpy as np
from smolyak.applications.polreg.orthogonal_polynomials import evaluate_orthonormal_polynomials
import random
import warnings
from scipy.linalg import solve
from numpy import meshgrid
from smolyak.misc.z_func import z_func
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
WARNING = False

class PolynomialSubspace:
    '''
    Maintains description of polynomial subspace for least-squares polynomial approximation
    
    '''
    __metaclass__ = ABCMeta
    @abstractmethod
    def generate_samples(self, N): 
        '''
        Generate samples from optimal density (=inverse normalized Christoffel function)
        '''
        pass
    
    @abstractmethod
    def evaluate_basis(self, X): 
        '''
        Evaluate basis polynomials
        
        :param X: Sample locations
        '''
        pass
    
    @abstractmethod
    def get_dimension(self): 
        '''
        Return dimension of polynomial subspace (not of domain)
        '''
        pass
    
    @abstractmethod
    def get_pols(self): 
        '''
        Return basis of polynomial subspace
        '''
        pass
    
    def optimal_weights(self, X):
        '''
        Return inverse of optimal sample density (=normalized Christoffel function)
        '''
        if X.shape[0] == 0:
            return np.zeros((0, 1))
        else:
            B = self.evaluate_basis(X)
            W = self.get_dimension() / np.sum(np.power(B, 2), axis=1)
            return W   
        
    @abstractmethod
    def lebesgue_density(self, X): 
        '''
        Return lebesgue density of measure at given locations
        '''
        pass
    
    @abstractmethod
    def get_c_var(self): 
        '''
        Return dimension of domain = number of variables
        '''
        pass
    
    def weighted_least_squares(self, X, W, Y):
        '''
        Compute least-squares approximation. 
        
        :param X: sample locations
        :param W: weights
        :param Y: sample values
        :return: coefficients
        '''
        B = self.evaluate_basis(X)
        W = W.reshape((W.size, 1))
        R = B.transpose().dot(Y * W)
        # from scipy.sparse.linalg import LinearOperator, cg
        # def mv(v):
        #    return B.transpose().dot(W*B.dot(v.reshape([-1,1])))
        # GO=LinearOperator((B.shape[1],B.shape[1]),matvec=mv)
        # import timeit
        # tic=timeit.default_timer()
        # coefficients1,info = cg(GO,R,tol=1e-12)
        # print('CG:',timeit.default_timer()-tic)
        G = B.transpose().dot(B * W)
        if WARNING and np.linalg.cond(G) > 100:
            warnings.warn('Ill conditioned Gramian matrix encountered') 
        coefficients = solve(G, R, sym_pos=True)
        if WARNING and not np.isfinite(coefficients).all():
            warnings.warn('Numerical instability encountered')
        return {pol: coefficients[i] for i, pol in enumerate(self.get_pols())}
    
class UnivariatePolynomialSubspace(PolynomialSubspace):
    '''
    Maintains description of univariate polynomial subspace.
    '''
    def __init__(self, measure='u', interval=(0, 1), sampler='optimal'):
        if measure != 'u' and measure != 'c' and measure != 'h':
            raise ValueError('Measure not supported')
        else:
            self.measure = measure
        if sampler != 'optimal' and sampler != 'arcsine' and sampler != 'MCMC':
            raise ValueError('Sampling strategy not supported')
        else:
            if sampler == 'optimal':
                self.keep = False
            elif sampler == 'arcsine':
                if self.measure in ['h']:
                    raise ValueError('Cannot use arcsine samples for Hermite polynomials')
                self.keep = True
            elif sampler == 'MCMC':
                self.keep = False
        self.sampler = sampler
        self.pols = []
        self.interval = (float(interval[0]), float(interval[1]))
    
    def get_dimension(self):
        return len(self.pols)
    
    def get_c_var(self): 
        return 1 
    
    def get_pols(self): 
        return self.pols
    
    def lebesgue_density(self, X):
        if self.measure == 'u':
            return np.ones((X.shape[0], 1)) / (self.interval[1] - self.interval[0])
        elif self.measure == 'c':
            return 1 / (np.pi * np.sqrt((X - self.interval[0]) * (self.interval[1] - X)))
        elif self.measure == 'h':
            return np.exp(-(X ** 2.) / 2.) / np.sqrt(2 * np.pi)
  
    def sample_from_polynomial(self, pol):
        def dens_goal(X):
            T = np.power(evaluate_orthonormal_polynomials(X,
                                                          pol,
                                                          measure=self.measure,
                                                          interval=self.interval)[0, -1], 2)
            return T * self.lebesgue_density(X)
        if self.measure == 'u':
            acceptance_ratio = 1. / (4 * np.exp(1))
        elif self.measure == 'c':
            acceptance_ratio = 1. / (2 * np.exp(1) * (2 + np.sqrt(1. / 2)))
        elif self.measure == 'h':
            acceptance_ratio = 1. / 8 * (pol + 1) ** (-1. / 3)
        accept = False
        while not accept:
            if self.measure in ['u', 'c']:
                X_temp = (np.cos(np.pi * np.random.rand(1, 1)) + 1) / 2
                X = self.interval[0] + X_temp * (self.interval[1] - self.interval[0])
                dens_prop_X = 1 / (np.pi * np.sqrt((X - self.interval[0]) * (self.interval[1] - X)))
            elif self.measure in ['h']:
                an = 8 * (pol + 1) ** (1. / 2)
                X = np.random.uniform(low=-an, high=an, size=(1, 1)) 
                dens_prop_X = 1 / (2 * an)
            dens_goal_X = dens_goal(X)
            alpha = acceptance_ratio * dens_goal_X / dens_prop_X
            U = np.random.rand(1, 1)
            accept = (U < alpha)
            if accept:
                return X
            
    def evaluate_basis(self, X):
        return evaluate_orthonormal_polynomials(X, max(self.pols), measure=self.measure, interval=self.interval)
        
    def generate_samples(self, N):
        if self.sampler == 'optimal':
            X = np.zeros((N, 1))
            for i in range(N):
                pol = random.randrange(0, len(self.pols))
                degree = self.pols[pol]
                X[i] = self.sample_from_polynomial(degree)
            W = self.optimal_weights(X)
        elif self.sampler == 'arcsine':
            X_temp = (np.cos(np.pi * np.random.rand(int(N), 1)) + 1) / 2   
            X = self.interval[0] + X_temp * (self.interval[1] - self.interval[0])
            W = np.pi * np.sqrt((X - self.interval[0]) * (self.interval[1] - X))
        return (X, W)
        
    def c_samples(self, pols):
        PolynomialSubspace.c_samples(self, pols)
        
    def expand_pols(self, pols):
        '''
        Expand polynomial subspace.
        
        If necessary, expand dimension of domain, as well. 
        
        :param pols: Polynomials to be added
        '''
        self.pols += pols
    
class MultivariatePolynomialSubspace(PolynomialSubspace):
    '''
    Maintains description of multivariate polynomial subspace.
    '''
    def __init__(self, ups_list=None, ups=None, c_var=None, sampler='optimal'):
        if (ups_list and (ups or c_var)) or not (ups_list or(ups and c_var)):
            raise ValueError('Specify either list of univariate polynomial subspaces' 
            'or single univariate polynomial subspace and dimension')
        if ups_list:
            self.ups_list = ups_list
            self.c_var = len(ups_list)
        else:
            self.ups_list = []
            for __ in range(c_var):
                self.ups_list += [copy.deepcopy(ups)]
            self.c_var = c_var
        self.pols = []
        self.sampler = sampler
        
    def get_dimension(self):
        return len(self.pols)
    
    def get_c_var(self): 
        return self.c_var
    
    def get_pols(self): 
        return self.pols
    
    def evaluate_basis(self, X):
        '''
        Evaluates basis polynomials at given sample locations.
        
        :param X: Sample locations
        :return: Basis polynomials evaluated at X
        :rtype: `X.shape[0] x len(self.pols)` np.array
        '''
        values = np.ones((X.shape[0], len(self.pols)))
        basis_values = {}
        if self.pols:
            active_vars = set.union(*[set(pol.active_dims()) for pol in self.pols]) 
        else:
            active_vars = set()
        for dim in active_vars:
            self.ups_list[dim].pols = range(max([pol[dim] for pol in self.pols]) + 1)
            basis_values[dim] = self.ups_list[dim].evaluate_basis(X[:, dim])
        for i, pol in enumerate(self.pols):
            for dim in pol.active_dims():
                values[:, i] *= basis_values[dim][:, pol[dim]]   
        return values
        
    def generate_samples(self, N):
        if self.sampler == 'optimal':
            X = np.zeros((N, self.c_var))
            for i in range(N):
                pol = random.randrange(0, len(self.pols))
                for dim in range(self.c_var):
                    degree = self.pols[pol][dim]
                    X[i, dim] = self.ups_list[dim].sample_from_polynomial(degree)
            W = self.optimal_weights(X)
        elif self.sampler == 'arcsine':
            X = np.zeros((N, self.c_var))
            W = np.ones((N, 1))
            for dim in range(self.c_var):
                self.ups_list[dim].sampler = 'arcsine'
                (X_temp, W_temp) = self.ups_list[dim].generate_samples(N)
                X[:, [dim]] = X_temp
                W *= W_temp
        return (X, W)
    
    def lebesgue_density(self, X):
        Y = np.ones((X.shape[0], 1))
        for dim in range(self.c_var):
            Y *= self.ups_list[dim].lebesgue_density(X[:, dim])
        return Y
        
    def expand_pols(self, pols):
        '''
        Expand polynomial subspace.
        
        If necessary, expand dimension of domain, as well. 
        
        :param pols: Polynomials to be added
        '''
        self.pols += pols
        max_var = max([max(pol.active_dims()) + 1 if pol.active_dims() else 1 for pol in self.pols])
        if max_var > self.c_var:
            for __ in range(max_var - self.c_var):
                self.ups_list.append(copy.deepcopy(self.ups_list[-1]))
            self.c_var = max_var
      
    def get_range(self, N=200, L=1):
        '''
        Return mesh of points within domain
        '''
        if self.c_var == 1:
            if self.ups_list[0].measure in ['u', 'c']:
                interval = self.ups_list[0].interval
                L = interval[1] - interval[0]
                X = np.linspace(interval[0] + L / N, interval[1] - L / N, N)
            else:
                X = np.linspace(-L, L, N)
            return X.reshape((-1, 1))
        elif self.c_var == 2:
            T = np.zeros((N, 2))
            for i in [0, 1]:
                if self.ups_list[0] in ['u', 'c']:
                    interval = self.ups_list[i].interval
                    L = interval[1] - interval[0]
                    T[:, i] = np.linspace(interval[0] + L / N, interval[1] - L / N, N) 
                else:
                    T[:, i] = np.linspace(-L, L, N)
            X, Y = meshgrid(T[:, 0], T[:, 1])
            return (X, Y)
        
    def plot_optimal_distribution(self, N=200, L=1):
        '''
        Plot optimal sampling distribution
        '''
        if self.c_var == 1:
            X = self.get_range(N, L)
            Z = self.lebesgue_density(X) * self.optimal_weights(X)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(X, Z)
        elif self.c_var == 2:
            X, Y = self.get_range(N, L)
            Z = z_func(self.lebesgue_density, X, Y) / z_func(self.optimal_weights, X, Y)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5)
        fig.suptitle('Lebesgue density of optimal distribution')
        plt.show()
        return np.sum(Z) / Z.size
