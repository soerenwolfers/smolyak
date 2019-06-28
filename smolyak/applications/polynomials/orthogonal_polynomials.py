'''
Orthogonal polynomials
'''
import numpy as np

def evaluate_orthonormal_polynomials(X, max_degree, measure, interval=(0, 1),derivative = 0):
    r'''
    Evaluate orthonormal polynomials in :math:`X`.
    The endpoints of `interval` can be specified when `measure` is uniform or Chebyshev.
    
    :param X: Locations of desired evaluations
    :type X:  One dimensional np.array
    :param max_degree: Maximal degree of polynomial basis.
    :param measure: Orthogonality measure. `u` for uniform, `c` for Chebyshev, `h` for Gauss/Hermite
    :param interval: Domain
    :type interval: tuple or list
    :rtype: numpy.array of size :code:`X.shape[0] x (max_degree+1)`
    '''
    max_degree = int(max_degree)
    if derivative and measure not in ['t','u','h']:
        raise ValueError('Derivative not implemented for Chebyshev polynomials')
    if derivative>1 and measure not in ['h','t']:
        raise ValueError('Second derivative only implemented for Taylor and Hermite polynomials')
    if measure in ['u', 'c']:
        Xtilde = (X - (interval[1] + interval[0]) / 2.) / ((interval[1] - interval[0]) / 2.) 
        if measure == 'u':
            y = legendre_polynomials(Xtilde, max_degree + 1)
            if derivative ==0:
                return y
            if derivative > 0:
                for _ in range(derivative):
                    y1 = np.zeros(y.shape)
                    y1[:,1] = np.sqrt(3)*np.ones(y.shape[0])
                    for j in range(2,max_degree+1):
                        y1[:,j] = np.sqrt(2*j+1)*((2*j-1)*y[:,j-1]/np.sqrt(2 * j - 1) + y1[:,j-2]/np.sqrt(2*j-3))
                    y=y1
                return (2/(interval[1]-interval[0]))**derivative*y
        elif measure == 'c':
            return chebyshev_polynomials(Xtilde, max_degree + 1)
    elif measure == 'h':
        Xtilde = X / interval[1]
        y = hermite_polynomials(Xtilde, max_degree + 1)
        factorial = np.ones((1,max_degree+1))
        for i in range(1,max_degree+1):
            factorial[0,i:]*=i
        orthonormalizer = 1 / np.sqrt(factorial)
        if derivative>0:
            y /= orthonormalizer
            for _ in range(derivative):
                y = np.concatenate([np.zeros([X.shape[0],1]),np.arange(1,y.shape[1])**1*y[:,:-1]],axis=1)
            y = 1/interval[1]**derivative*y
            y *= orthonormalizer
        return y
    elif measure == 't':
        y = taylor_polynomials(X,max_degree+1)
        if derivative>0:
            for _ in range(derivative):
                y = np.concatenate([np.zeros([X.shape[0],1]),np.arange(1,y.shape[1])*y[:,:-1]],axis=1)
        return y
    
def taylor_polynomials(X,N):
    return X.reshape(-1,1)**np.arange(N)

def chebyshev_polynomials(X, N):
    r'''
    Evaluate the orthonormal Chebyshev polynomials on
    :math:`([-1,1],dx/2)` in :math:`X\subset [-1,1]`
    
    :param X: Locations of desired evaluations
    :type X:  One dimensional np.array
    :param N: Number of polynomials
    :rtype: numpy.array of shape :code:`X.shape[0]xN`
    '''
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = np.concatenate((np.array([1]).reshape(1, 1), np.sqrt(2) * np.ones((1, deg))), axis=1)
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))      
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = 2 * X * out[:, n] - out[:, n - 1]
    return out * orthonormalizer

def legendre_polynomials(X, N):
    r'''
    Evaluate the orthonormal Legendre polynomials on 
    :math:`([-1,1],dx/2)` in :math:`X\subset [-1,1]`
    
    :param X: Locations of desired evaluations
    :type X:  One dimensional np.array
    :param N: Number of polynomials
    :rtype: numpy.array of shape :code:`X.shape[0]xN`
    '''
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = np.sqrt(2 * (np.arange(deg + 1)) + 1)
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))      
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = 1. / (n + 1) * ((2 * n + 1) * X * out[:, n] - n * out[:, n - 1])
    return out * orthonormalizer

def hermite_polynomials(X, N):
    r'''
    Evaluate the orthonormal Hermite polynomials on 
    :math:`(\mathbb{R},\frac{1}{\sqrt{2\pi}}\exp(-x^2/2)dx)` in :math:`X\subset\mathbb{R}`
    
    
    :param X: Locations of desired evaluations
    :type X:  One dimensional np.array
    :param N: Number of polynomials
    :rtype: numpy.array of shape :code:`X.shape[0] x N`
    '''
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    factorial = np.ones((1,N))
    for i in range(1,N):
        factorial[0,i:]*=i
    orthonormalizer = 1 / np.sqrt(factorial)
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))      
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = X * out[:, n] - n * out[:, n - 1]
    return out * orthonormalizer
