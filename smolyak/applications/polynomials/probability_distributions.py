from abc import abstractmethod, ABC
import numpy as np
from numpy import meshgrid
from swutil.validation import Integer, Positive, validate_args

class AbstractProbabilityDistribution(ABC):
    '''
    Probability spaces over subsets of Euclidean space
    '''
    
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

    @abstractmethod
    def get_range(self, N):
        '''
        Return mesh of points within domain
        '''
        pass

class ProbabilityDistribution(AbstractProbabilityDistribution):
    def __init__(self, measure='u', interval=None):
        if measure  not in ['u', 'c', 'h','t']:
            raise ValueError('Measure not supported')
        else:
            self.measure = measure
        if measure == 'h':
            interval = interval or 1
            self.interval = (None,float(interval)) # replace None by center in the future
        else:
            interval = interval or (-1,1)
            self.interval = (float(interval[0]), float(interval[1]))
        
    def lebesgue_density(self, X):
        if self.measure == 'u':
            return np.ones((X.shape[0], 1)) / (self.interval[1] - self.interval[0])
        elif self.measure == 'c':
            return 1 / (np.pi * np.sqrt((X - self.interval[0]) * (self.interval[1] - X)))
        elif self.measure == 'h':
            return np.exp(-(X ** 2.) / (2.*self.interval[1])) / np.sqrt(2 *self.interval[1]*np.pi)
        
    def get_c_var(self):
        return 1
    
    def get_range(self, N=200,L=None):
        if self.measure in ['u', 'c']:
            interval = self.interval
            L = interval[1] - interval[0]
            X = np.linspace(interval[0] + L / N, interval[1] - L / N, N)
        else:
            L = L or self.interval[1] 
            X = np.linspace(-L, L, N)
        return X.reshape((-1, 1))
    
    def __mul__(self,other):
        if isinstance(other,ProductProbabilityDistribution):
            return ProductProbabilityDistribution([self]+other.ups)
        elif isinstance(other,ProbabilityDistribution):
            return ProductProbabilityDistribution([self,other])
        else:
            raise TypeError()
        
    @validate_args(warnings=False)
    def __pow__(self,other:Positive&Integer):
        return ProductProbabilityDistribution([self]*other)
        

class ProductProbabilityDistribution(AbstractProbabilityDistribution):
    def __init__(self,univariate_probability_distributions):
        self.ups=univariate_probability_distributions
    
    def __mul__(self,other):
        if isinstance(other,ProductProbabilityDistribution):
            return ProductProbabilityDistribution(self.ups+other.ups)
        elif isinstance(other,ProbabilityDistribution):
            return ProductProbabilityDistribution(self.ups+[other])
        else:
            raise TypeError()
    
    @validate_args(warnings=False)
    def __pow__(self,other:Positive&Integer):
        return ProductProbabilityDistribution(self.ups*other)

    def lebesgue_density(self,X):
        D=np.ones((X.shape[0],1))
        for dim in range(self.get_c_var()):
            D*=self.ups[dim].lebesgue_density(X[:,[dim]])
        return D
        
    def get_c_var(self):
        return len(self.ups)
    
    def get_range(self, N=30,L=1):
        if self.get_c_var() == 1:
            if self.ups[0].measure in ['u', 'c']:
                interval = self.ups[0].interval
                L = interval[1] - interval[0]
                X = np.linspace(interval[0] + L / N, interval[1] - L / N, N)
            else:
                X = np.linspace(-L, L, N)
            return X.reshape((-1, 1))
        elif self.get_c_var() == 2:
            T = np.zeros((N, 2))
            for i in [0, 1]:
                if self.ups[i].measure in ['u', 'c']:
                    interval = self.ups[i].interval
                    L = interval[1] - interval[0]
                    T[:, i] = np.linspace(interval[0] + L / N, interval[1] - L / N, N) 
                else:
                    T[:, i] = np.linspace(-L, L, N)
            X, Y = meshgrid(T[:, 0], T[:, 1])
            return (X, Y)
        
        
