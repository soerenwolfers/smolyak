import numpy as np

def z_func(A, x_value_grid, y_value_grid):
    '''
    Applies function that takes inputs of the form (x,y) to two-dimensional grid 
    and returns output on same two-dimensional grid, as needed by various plotting
    functions.
    
    :param A: Function to be evaluated on grid
    :param x_value_grid: Two-dimensional array of x_value_grid values (as returned by Matlab's meshgrid) 
    :type x_value_grid: numpy.array with shape (m,n)
    :param y_value_grid: Two-dimensional array of y_value_grid values (as returned by Matlab's meshgrid)
    :type y_value_grid: numpy.array with shape (m,n)
    :return: :math:`(A(x_i,y_j))_{i\in\{1,\dots,m\},j\in\{1,\dots,n\}}`
    :rtype: numpy.array with shape(m,n)  
    '''
    X = np.reshape(np.concatenate([x_value_grid[..., None], y_value_grid[..., None]], axis=2), (x_value_grid.size, 2), order='C')
    return np.reshape(A(X), x_value_grid.shape, order='C')  
