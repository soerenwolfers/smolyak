'''
Nonadaptive and adaptive multi-level weighted polynomial least squares approximation for a random PDE that is
parametrized using a Karhunen Loeve type expansion.
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import timeit
from smolyak.misc import plots
from smolyak.applications.polreg.mi_weighted_polynomial_approximator import MIWeightedPolynomialApproximator
from smolyak.sparse.sparse_approximator import SparseApproximator
from smolyak.applications.pde.kl import kl
from matplotlib2tikz import save as tikz_save
from smolyak.applications.polreg.polynomial_subspace import MultivariatePolynomialSubspace,\
    UnivariatePolynomialSubspace
    
def kl_nonadaptive():
    xi=2.
    exponent=4.
    def PDE(X, mi):
        return kl(X, 32 * np.sqrt(2) ** mi[0], order=1)
    plt.figure()
    runtimes = []
    L_max=12
    L_min=2
    scale=np.log(4)
    As=[]
    for L in range(L_min,L_max):
        ps=MultivariatePolynomialSubspace(ups_list=[UnivariatePolynomialSubspace()])
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=1,ps=ps)
        def work(mis):
            mi=mis[0]   
            return 2.**mi[0]*np.prod([2.**v for __,v in mi.leftshift()])
        def contribution(mi):
            return 2.**(-mi[0])*np.prod([np.exp(-np.log(xi*(dim+1)**exponent)*(2.**(v-1))) for dim,v in mi.leftshift()])
        SA = SparseApproximator(decomposition=mipa.expand,
                            is_bundled=lambda dim: dim>=1, 
                            work_factor=work,
                            contribution_factor=contribution,
                            external=True)
        tic = timeit.default_timer()
        SA.expand_nonadaptive(L=L,scale=scale)
        runtimes.append(timeit.default_timer() - tic)
        A = mipa.get_approximation()
        As.append(A)
        print('Runtime', runtimes[-1])
    dims=max(mipa.get_active_dims())+1
    X = np.random.rand(10000, dims)
    Zl = [A(X) for A in As]
    order = plots.plot_convergence(runtimes, [np.array(Z).reshape(-1, 1) for Z in Zl])
    print('Nonadaptive: Fitted convergence order: {}'.format(order))
    print('Number of active dimensions: ', dims)
    tikz_save('kl2_nonadaptiveNEW.tex');
    #plt.show()
    
def kl_adaptive():
    xi=2.
    exponent=4.
    def PDE(X, mi):
        return kl(X, 32 * np.sqrt(2) ** mi[0], order=1)
    CSTEPS = 7
    runtimes = []
    As=[]
    for step in range(CSTEPS):
        ps=MultivariatePolynomialSubspace(ups_list=[UnivariatePolynomialSubspace()],sampler='arcsine')
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=1,ps=ps)
        SA = SparseApproximator(decomposition=mipa.expand, 
                            init_dims=1,
                            next_dims=True, 
                            is_bundled=lambda dim: dim>=1,
                            work_factor=lambda mis: mipa.required_samples(mis)*2**mis[0][0],
                            external=True,is_md=True)
        tic = timeit.default_timer()
        SA.expand_adaptive(T_max=4.**step,contribution_exponents=lambda dim: -np.log(xi*(dim+1)**exponent))
        runtimes.append(timeit.default_timer() - tic)
        A = mipa.get_approximation()
        As.append(A)
        print('Runtime', runtimes[-1])
    dims=max(mipa.get_active_dims())+1
    X = np.random.rand(10000, dims)
    Zl = [A(X) for A in As]
    order = plots.plot_convergence(runtimes, [np.array(Z).reshape(-1, 1) for Z in Zl],
                                    expect_order=-1)
    print('Adaptive: Fitted convergence order: {}'.format(order))
    print('Number of active dimensions: ', dims)
    tikz_save('kl2_adaptiveNEW.tex');
    #plt.show()

if __name__ == '__main__':
    import cProfile
    cProfile.run('kl_adaptive()', 'restats')
    import pstats
    p = pstats.Stats('restats') 
    p.sort_stats('cumulative').print_stats(20)