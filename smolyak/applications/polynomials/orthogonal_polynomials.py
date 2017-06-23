'''
Orthogonal polynomials
'''
import numpy as np
import math

def evaluate_orthonormal_polynomials(X, max_degree, measure, interval=(0, 1)):
    r'''
    Compute values of orthonormal polynomials on `interval` in :math:`X`.
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
    
    if measure in ['u', 'c']:
        Xtilde = (X - (interval[1] + interval[0]) / 2.) / ((interval[1] - interval[0]) / 2.) 
        if measure == 'u':
            temp = legendre_polynomials(Xtilde, max_degree + 1)
        elif measure == 'c':
            temp = chebyshev_polynomials(Xtilde, max_degree + 1)
    elif measure == 'h':
        temp = hermite_polynomials(X, max_degree + 1)
    return temp
    
def chebyshev_polynomials(X, N):
    r'''
    Compute values of the orthonormal Chebyshev polynomials on
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
    Compute values of the orthonormal Legendre polynomials on 
    :math:`([-1,1],dx/2)` in :math:`X\subset [-1,1]`
    
    :param X: Locations of desired evaluations
    :type X:  One dimensional np.array
    :param N: Number of polynomials
    :rtype: numpy.array of shape :code:`X.shape[0]xN`
    '''
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = np.reshape(np.sqrt(2 * (np.array(range(deg + 1))) + 1), (1, N))
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
    Compute values of the orthonormal Hermite polynomials on 
    :math:`(\mathbb{R},\frac{1}{\sqrt{2\pi}}\exp(-x^2/2)dx)` in :math:`X\subset\mathbb{R}`
    
    
    :param X: Locations of desired evaluations
    :type X:  One dimensional np.array
    :param N: Number of polynomials
    :rtype: numpy.array of shape :code:`X.shape[0]xN`
    '''
    out = np.zeros((X.shape[0], N))
    deg = N - 1
    orthonormalizer = 1 / np.reshape([math.sqrt(math.factorial(n)) for n in range(N)], (1, N))
    if deg < 1:
        out = np.ones((X.shape[0], 1))
    else:
        out[:, 0] = np.ones((X.shape[0],))      
        out[:, 1] = X
        for n in range(1, deg):
            out[:, n + 1] = X * out[:, n] - n * out[:, n - 1]
    return out * orthonormalizer
