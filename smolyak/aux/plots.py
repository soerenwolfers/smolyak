'''
Various plotting functions
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib
import warnings
from numpy import inf
def plot_indices(mis, dims, weight_dict=None, N_q=1):
    '''
    Plot multi-index set
    
    :param mis: Multi-index set
    :type mis: Iterable of SparseIndices
    :param dims: Which dimensions to use for plotting
    :type dims: List of integers.
    :param weight_dict: Weights associated with each multi-index
    :type weight_dict: Dictionary
    :param N_q: Number of percentile-groups plotted in different colors
    :type N_q: Integer>=1
    '''
    if len(dims) > 3:
        raise ValueError('Cannot plot more than three dimensions.')
    if len(dims) < 1:
        warnings.warn('Sure you don\'t want to plot anything?')
        return
    if weight_dict:
        values = weight_dict.values()
        weight_function = lambda mi: weight_dict[mi]
    else:
        if N_q > 1:
            raise ValueError('Cannot create percentile-groups without weight dictionary')
        weight_function = lambda mi: 1
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, N_q))  # @UndefinedVariable
    fig = plt.figure()
    if len(dims) == 3:
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()
    ax.set_aspect('equal')
    size_function = lambda mi: sum([weight_function(mi2) for mi2 in mis if mi.equal_mod(mi2, lambda dim: dim not in dims)])
    sizes = {mi: np.power(size_function(mi), 0.1) for mi in mis}
    for q in range(N_q):
        if N_q > 1:
            plot_indices = [mi for mi in mis if weight_function(mi) >= np.percentile(values, 100 / N_q * q) and weight_function(mi) <= np.percentile(values, 100 / N_q * (q + 1))]
        else:
            plot_indices = mis
        X = np.array([mi[dims[0]] for mi in plot_indices])
        if len(dims) > 1:
            Y = np.array([mi[dims[1]] for mi in plot_indices])
        else:
            Y = np.array([0 for mi in plot_indices])
        if len(dims) > 2:
            from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport, @UnresolvedImport
            Z = np.array([mi[dims[2]] for mi in plot_indices])
        else:
            Z = np.array([0 for mi in plot_indices])   
        sizes_plot = np.array([sizes[mi] for mi in plot_indices])
        if weight_dict:
            if len(dims) == 3:
                ax.scatter(X, Y, Z, s=100 * sizes_plot / max(sizes.values()), color=colors[q], alpha=1)
            else:
                ax.scatter(X, Y, s=100 * sizes_plot / max(sizes.values()), color=colors[q], alpha=1)
        else:
            if len(dims) == 3:
                ax.scatter(X, Y, Z)
            else:
                ax.scatter(X, Y)
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        if len(dims) == 3:
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')
        else:
            for xb, yb in zip(Xb, Yb):
                ax.plot([xb], [yb], 'w')
        ax.set_xlabel('Dim. ' + str(dims[0]))
        if len(dims) > 1:
            ax.set_ylabel('Dim. ' + str(dims[1]))
        if len(dims) > 2:
            ax.set_zlabel('Dim. ' + str(dims[2]))
        plt.grid()

def plot3D(X, Y, Z):
    '''
    Surface plot.
    
    Generate X and Y using X,Y = np.mgrid[0:1:50j, 0:1:50j] for example.
    
    :param X: 2D-Array of x-coordinates
    :param Y: 2D-Array of y-coordinates
    :param Z: 2D-Array of z-coordinates
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    light = LightSource(90, 90)
    illuminated_surface = light.shade(Z, cmap=cm.coolwarm)
    Xmin = np.amin(X)
    Xmax = np.amax(X)
    Ymin = np.amin(Y)
    Ymax = np.amax(Y)
    Zmin = np.amin(Z)
    Zmax = np.amax(Z)
    ax.contourf(X, Y, Z, zdir='x', offset=Xmin - 0.1 * (Xmax - Xmin), cmap=cm.coolwarm, alpha=1)
    ax.contourf(X, Y, Z, zdir='y', offset=Ymax + 0.1 * (Ymax - Ymin), cmap=cm.coolwarm, alpha=1)
    ax.contourf(X, Y, Z, zdir='z', offset=Zmin - 0.1 * (Zmax - Zmin), cmap=cm.coolwarm, alpha=1)
    ax.plot_surface(X, Y, Z, cstride=5, rstride=5, facecolors=illuminated_surface, alpha=0.5)
    plt.show()
    
def plot_convergence(times, values, expect_limit=None, expect_residuals=None,
                     expect_times=None, expect_order=None, ignore=1, p=2,
                     ignore_start=0):
    '''
    Show loglog convergence plot.
    
    Specify :code:`expect_limit` if exact limit is known. Otherwise limit is 
    taken to be last entry of :code:`values`.
    
    Distance to limit is computed as RMSE (or analogous p-norm if p is specified)
    
    Specify either :code:`expect_order`(pass number or 'fit') or 
    :code:`expect_residuals` and :code:`expect_times` to add a second plot with
    the expected convergence.
    
    :param times: Runtimes
    :type times: List of positive numbers
    :param values: Outputs
    :type values: List of arrays
    :param expect_limit: Exact solution
    :type expect_limit: Array
    :param expect_residuals: Expected residuals
    :type expect_residuals: List of positive numbers
    :param expect_times: Expected runtimes
    :type expect_times: List of positive numbers
    :param expect_order: Expected convergence order
    :type expect_order: Real or 'fit'
    :param ignore: If expect_limit is not provided, how many entries (counting
       from the end) should be ignored for the computation of residuals. 
    :type ignore: Integer.
    :param ignore_start: How many entries counting from start should be ignored.
    :type ignore_start: Integer.
    :return: fitted convergence order
    '''
    c_ticks = 30;
    for value in values:
        if hasattr(value, 'shape') and len(value.shape) == 1:
            value = value.reshape(1, -1)  
    assert(len(times) == len(values))
    sorting = np.argsort(times)
    times = [times[i] for i in sorting]
    values = [values[i] for i in sorting]
    if expect_limit is None:
        plotlength = len(times) - ignore
        if plotlength < 2:
            raise KeyError('Too few values')
        limit = values[-1]
        times = times[0:-ignore]
    else:
        plotlength = len(times);
        limit = expect_limit
        if hasattr(limit, 'shape') and len(limit.shape) == 1:
            limit = limit.reshape(1, -1)
    residuals = np.zeros([plotlength, 1])
    for L in range(plotlength):
        if hasattr(values[L], 'shape') and len(values[L].shape) > 0:
            N = values[L].shape[0]
        else:
            N = 1
        if p < np.Inf:
            residuals[L] = np.power(np.sum(np.power(np.abs(values[L] - limit), p) / N), 1. / p)  #
        else:
            residuals[L] = np.amax(np.abs(values[L] - limit))
    plt.loglog(times, residuals)
    if expect_times is not None and expect_residuals is not None:
        plt.loglog(expect_times, expect_residuals) 
    logx = np.log(times[ignore_start:]).reshape(-1)
    logy = np.log(residuals[ignore_start:]).reshape(-1)
    logy[logy == -inf] = -17
    coeffs = np.polyfit(logx, logy, deg=1)
    fitted_order = coeffs[0]
    if expect_order is not None:
        X = np.linspace(min(times), max(times), c_ticks)
        if expect_order == 'fit':
            plt.loglog(X, np.exp(coeffs[1]) * X ** (fitted_order))
        else:
            plt.loglog(X, X ** (expect_order) / X[-1] ** (expect_order) * residuals[-1])
    return fitted_order
