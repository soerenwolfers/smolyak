import numpy as np
def is_1d(array):
    return np.squeeze(array).ndim==1

def grid_evaluation(X, Y, f):
    '''
    Evaluate function on given grid and return values in grid format
    
    Assume X and Y are 2-dimensional arrays containing x and y coordinates, 
    respectively, of a two-dimensional grid, and f is a function that takes
    1-d arrays with two entries. This function evaluates f on the grid points
    described by X and Y and returns another 2-dimensional array of the shape 
    of X and Y that contains the values of f.
    :param X: 2-dimensional array of x-coordinates
    :param Y: 2-dimensional array of y-coordinates
    :param f: function to be evaluated on grid
    :return: 2-dimensional array of values of f
    '''
    XX = np.reshape(np.concatenate([X[..., None], Y[..., None]], axis=2), (X.size, 2), order='C')
    return np.reshape(f(XX), X.shape, order='C')     

def weighted_median(values, weights):
    '''
    Returns element such that sum of weights below and above are (roughly) equal
    
    :param values: Values whose median is sought
    :type values: List of reals
    :param weights: Weights of each value
    :type weights: List of positive reals
    :return: value of weighted median
    :rtype: Real
    '''
    if len(values) == 1:
        return values[0]
    if len(values) == 0:
        raise ValueError('Cannot take median of empty list')
    values = [float(value) for value in values]
    indices_sorted = np.argsort(values)
    values = [values[ind] for ind in indices_sorted]
    weights = [weights[ind] for ind in indices_sorted]
    total_weight = sum(weights)
    below_weight = 0
    i = -1
    while below_weight < total_weight / 2:
        i += 1
        below_weight += weights[i]
    return values[i]
