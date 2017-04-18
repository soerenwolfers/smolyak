import numpy as np
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
