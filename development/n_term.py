'''
Find best n term approximation, given coefficients with respect to orthonormal basis
'''
import pickle
import numpy as np
from smolyak.misc import plots
import matplotlib.pyplot as plt
if __name__ == '__main__':
    with open('coefficients/coeffs4') as f:
        obj = pickle.load(f)
    abs_coeff = np.abs(obj.values())
    abs_sorted = np.sort(abs_coeff, axis=0)
    dict_sorted = sorted(obj.items(), key=lambda x:abs(x[1]), reverse=True)
    remainders = []
    for j in range(1, abs_sorted.shape[0]):
        remainders.append(np.linalg.norm(abs_sorted[0:-j]))
    order = plots.plot_convergence([n * np.log(n + 1) for n in range(1, len(remainders) + 1)], remainders, expect_limit=0, expect_order='fit', ignore=400)
    print(order)
    plt.show()
    1
