'''
Methods to generate Dirac approximations to probability distributions
'''
import numpy as np

def inverse_transform_sampling(invcdf, N):
    X = np.linspace(0, 1, N, endpoint=True)
    return invcdf(X)

def random_sampling(pdf, bound, N):
    np.random.seed(1)
    X = np.zeros((N, 1))
    for i in range(int(N)):
        accept = False
        while not accept:
            U = np.random.rand(1, 1)
            X[i] = np.random.rand(1, 1)
            accept = U < pdf(X[i])
    return X
