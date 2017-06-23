import numpy as np
from smolyak.applications.polynomials.polynomial_approximation import PolynomialApproximation
import math
from smolyak.applications.polynomials import samples
import timeit
import matplotlib.pyplot as plt

class WeightedPolynomialApproximator(object):
    '''
    Maintains polynomial approximation of given function on :math:`[a,b]^d`.
    '''
    def __init__(self, function, ps, C=2,sampler='optimal'):
        ''' 
        :param function: Function that is approximated. Needs to support __call__
        :param ps: Polynomial subspace used for approximation
        :type ps: PolynomialSubspace instance
        :param C: Multiplier for number of samples used for reconstruction
        :type C: Positive real
        '''
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
        self.sampler=sampler
        self.C = C
        self.function = function
        self.ps = ps
        self.X = np.zeros((0, self.ps.get_c_var()))
        self.Y = np.zeros((0, 1))
        self.W = np.zeros((0, 1))
        
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
        '''
        c_samples = self.estimated_work(pols)
        if c_samples > 0:
            self.ps.set_basis(pols)
            if self.sampler == 'arcsine':
                if self.X.shape[1] < self.ps.get_c_var():
                    c_samples += self.X.shape[0]  # contrary to previous belief we cannot keep samples, thus need to add the number of old ones
                    self.X = np.zeros((0, self.ps.get_c_var()))
                    self.Y = np.zeros((0, 1))
                    self.W = np.zeros((0, 1))
                (Xnew,Wnew) = samples.importance_samples(self.ps, c_samples, 'arcsine')
                self.X = np.concatenate((self.X, Xnew), axis=0)  
                self.W = np.concatenate([self.W, Wnew])  
                tic = timeit.default_timer()
                self.Y = np.concatenate([self.Y, self.function(Xnew).reshape(-1, 1)])
                work_model = timeit.default_timer() - tic
            else: 
                (self.X, self.W) = samples.optimal_samples(self.ps, c_samples)
                tic = timeit.default_timer()
                self.Y = self.function(self.X).reshape(-1, 1)
                work_model = timeit.default_timer() - tic
            return work_model
        else:
            return 0
        
    def get_approximation(self):
        return PolynomialApproximation(self.ps,self.ps.weighted_least_squares(self.X, self.W, self.Y))
    
    def plot_xy(self):
        '''
        Scatter plot of stored samples.
        '''
        fig = plt.figure()
        if self.ps.get_c_var() == 1:
            order = np.argsort(self.X, axis=0).reshape(-1)
            ax = fig.gca()
            ax.scatter(self.X[order, :], self.Y[order, :])
        elif self.ps.get_c_var() == 2:
            ax = fig.gca(projection='3d')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.Y)
        fig.suptitle('Samples')
        plt.show()
        