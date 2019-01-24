import numpy as np
from smolyak.applications.polynomials.orthogonal_polynomials import evaluate_orthonormal_polynomials
import random
import math

def arcsine_samples(probability_space,N): 
    def univariate_arcsine_samples(N,interval): 
        X_temp = (np.cos(np.pi * np.random.rand(int(N), 1)) + 1) / 2   
        X = interval[0] + X_temp * (interval[1] - interval[0])
        D = 1./(np.pi * np.sqrt((X - interval[0]) * (interval[1] - X)))
        return (X,D)
    X = np.zeros((N, probability_space.get_c_var()))
    D = np.ones((N, 1))
    for dim in range(X.shape[1]):
        (X_temp, D_temp) = univariate_arcsine_samples(N,probability_space.ups[dim].interval)
        X[:, [dim]] = X_temp
        D *= D_temp
    return (X,D)
    
def importance_samples(probability_space,N,importance):  
    if importance == 'arcsine':
        (X,D)=arcsine_samples(probability_space, N)
        W=probability_space.lebesgue_density(X)/D
        return (X, W)
    else:
        raise ValueError('Sampling measure not implemented yet')

def optimal_samples(tensor_polynomial_subspace, N):
    X = np.zeros((N, tensor_polynomial_subspace.get_c_var()))
    for i in range(N):
        j = random.randrange(0, tensor_polynomial_subspace.get_dimension())
        for dim in range(tensor_polynomial_subspace.get_c_var()):
            degree = tensor_polynomial_subspace.basis[j][dim]
            X[i, dim] = sample_from_polynomial(tensor_polynomial_subspace.probability_distribution.ups[dim],degree)
    W = tensor_polynomial_subspace.optimal_weights(X)
    return (X, W)

def samples_per_polynomial(tensor_polynomial_subspace,old_basis,pols,c_samples):
    l_old = len(old_basis)
    N_new = math.ceil(c_samples(len(pols)))
    N_add = math.ceil(N_new) - math.ceil(c_samples(l_old))
    news = np.zeros(len(pols))
    for j,pol in enumerate(pols):
        if not pol in old_basis:
            news[j] = True
    N = int(N_new * np.sum(news)+N_add*(len(pols)-np.sum(news)))
    X = np.zeros((N, tensor_polynomial_subspace.get_c_var()))
    i=0
    for new,pol in zip(news,pols):
        for _ in range(N_new if new else N_add):
            for dim in range(tensor_polynomial_subspace.get_c_var()):
                degree = pol[dim]
                X[i, dim] = sample_from_polynomial(tensor_polynomial_subspace.probability_distribution.ups[dim],degree)
            i+=1
    W = tensor_polynomial_subspace.optimal_weights(X)
    return (X, W)
            
def sample_from_polynomial(probability_space, pol):
    def dens_goal(X):
        T = np.power(
                evaluate_orthonormal_polynomials(
                    X,
                    pol,
                    measure=probability_space.measure,
                    interval=probability_space.interval
                )[0, -1], 
                2
        )
        return T * probability_space.lebesgue_density(X)
    if probability_space.measure == 'u':
        acceptance_ratio = 1. / (4 * np.exp(1))
    elif probability_space.measure == 'c':
        acceptance_ratio = 1. / (2 * np.exp(1) * (2 + np.sqrt(1. / 2)))
    elif probability_space.measure == 'h':
        acceptance_ratio = 1. / 8 * (pol + 1) ** (-1. / 3)
    accept = False
    while not accept:
        if probability_space.measure in ['u', 'c']:
            X_temp = (np.cos(np.pi * np.random.rand(1, 1)) + 1) / 2
            X = probability_space.interval[0] + X_temp * (probability_space.interval[1] - probability_space.interval[0])
            dens_prop_X = 1 / (np.pi * np.sqrt((X - probability_space.interval[0]) * (probability_space.interval[1] - X)))
        elif probability_space.measure in ['h']:
            an = 8 * (pol + 1) ** (1. / 2)
            X = np.random.uniform(low=-an, high=an, size=(1, 1)) 
            dens_prop_X = 1 / (2 * an)
        dens_goal_X = dens_goal(X)
        alpha = acceptance_ratio * dens_goal_X / dens_prop_X
        U = np.random.rand(1, 1)
        accept = (U < alpha)
        if accept:
            return X
