'''
Multi-level polynomial least squares approximation for a parametric Poisson problem.
'''
from __future__ import division
import numpy as np
import timeit
from smolyak.applications.polreg.mi_weighted_polynomial_approximator import MIWeightedPolynomialApproximator
from smolyak.sparse.sparse_approximator import SparseApproximator
from smolyak.applications.pde.poisson import poisson_kink
from smolyak.misc import plots
from matplotlib2tikz import save as tikz_save
import matplotlib.pyplot as plt
from smolyak.sparse.sparse_index import \
    pyramid, rectangle
from smolyak.applications.polreg.polynomial_subspace import MultivariatePolynomialSubspace,\
    UnivariatePolynomialSubspace

def adaptive_multilevel(pardim):
    def PDE(X, mi):
        return poisson_kink(X, 64 * np.sqrt(2) ** mi[0], order=1)
    CSTEPS = 13 #FACTOR 2
    runtimes = []
    As=[]
    for step in range(CSTEPS):
        ps=MultivariatePolynomialSubspace(ups=UnivariatePolynomialSubspace(),c_var=pardim)
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=1,ps=ps)
        SA = SparseApproximator(decomposition=mipa.expand, 
                            init_dims=pardim+1,
                            is_bundled=lambda dim: dim>=1,
                            work_factor=lambda mis: mipa.required_samples(mis)*2**mis[0][0],
                            external=True,is_md=True)
        tic = timeit.default_timer()
        SA.expand_adaptive(T_max=2**step)
        runtimes.append(timeit.default_timer() - tic)
        A = mipa.get_approximation()
        As.append(A)
        print('Runtime', runtimes[-1])
    X = np.random.rand(10000, pardim)
    Zl = [A(X) for A in As]
    order2 = plots.plot_convergence(runtimes, [np.array(Z).reshape(-1, 1) for Z in Zl])
    print('Adaptive ml_d{}_p2 order: {}'.format(pardim,order2))
    tikz_save('./results/kink_ad_ml_d{}_p2.tex'.format(pardim));
    plt.close()
    
def adaptive_singlelevel(pardim):
    if pardim==1:
        CSTEPS = 7 #2*2^(1/pardim/2)
        kappa=1.115
    else:
        CSTEPS = 8 #2*2^(1/pardim/2)
        kappa=0.75
    runtimes = []
    As=[]
    for step in range(CSTEPS):
        def PDE(X, mi):
            return poisson_kink(X, 64*2**(kappa/2.*step), order=1)
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=0,c_dim_par=pardim,measure='u',interval=[-1,1])
        SA = SparseApproximator(decomposition=mipa.expand, 
                            init_dims=pardim,
                            is_bundled=lambda dim: dim>=0,
                            work_factor=lambda mis: mipa.required_samples(mis),
                            external=True,is_md=True)
        tic = timeit.default_timer()
        SA.expand_adaptive(T_max=2**step*2**(kappa/1.*step))
        runtimes.append(timeit.default_timer() - tic)
        A = mipa.get_approximation()
        As.append(A)
        print('Runtime', runtimes[-1])
    X = np.random.rand(10000, pardim)
    Zl = [A(X) for A in As]
    order2 = plots.plot_convergence(runtimes, [np.array(Z).reshape(-1, 1) for Z in Zl])
    print('adaptive sl_d{}_p2 order: {}'.format(pardim,order2))
    tikz_save('./results/kink_ad_sl_d{}_p2.tex'.format(pardim));
    plt.close()
    
def nonadaptive_multilevel(pardim):
    def PDE(X, mi):
        return poisson_kink(X, 64 * np.sqrt(2) ** mi[0], order=1)
    if pardim==1:
        CSTEPS = 10 #2*pardim*log
        kappa=1.115
    else:
        CSTEPS=6
        kappa=1.5
    runtimes = []
    As=[]
    Gamma_PDE=1
    Beta_PDE=1
    Gamma_LS=pardim
    Beta_LS=kappa
    for step in range(CSTEPS):
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=1,c_dim_par=pardim,measure='u',interval=[-1,1])
        SA = SparseApproximator(decomposition=mipa.expand, 
                            init_dims=pardim+1,
                            is_bundled=lambda dim: dim>=1,
                            external=True,is_md=True)
        mis=pyramid(step, Gamma_PDE, Beta_PDE, Gamma_LS, Beta_LS, pardim+1)
        tic = timeit.default_timer()
        SA.expand_by_mis(mis)
        runtimes.append(timeit.default_timer() - tic)
        A = mipa.get_approximation()
        As.append(A)
        print('Runtime', runtimes[-1])
    X = np.random.rand(10000, pardim)
    Zl = [A(X) for A in As]
    order2 = plots.plot_convergence(runtimes, [np.array(Z).reshape(-1, 1) for Z in Zl])
    print('Non-adaptive ml_d{}_p2 order: {}'.format(pardim,order2))
    tikz_save('./results/kink_non_ml_d{}_p2.tex'.format(pardim));
    plt.close()
    
def nonadaptive_singlelevel(pardim):
    if pardim==1:
        CSTEPS = 7#FACTOR 2**kappa*2*pardim*log
        kappa=1.115
    else:
        CSTEPS=5
        kappa=1.5
    
    runtimes = []
    As=[]
    for step in range(CSTEPS):
        def PDE(X, mi):
            return poisson_kink(X, 64*2**(kappa/2.*step), order=1)
        ps=MultivariatePolynomialSubspace(ups=UnivariatePolynomialSubspace(interval=[-1,1]),c_var=pardim)
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=0,ps=ps)
        SA = SparseApproximator(decomposition=mipa.expand, 
                            init_dims=pardim,
                            is_bundled=lambda dim: dim>=0,
                            external=True,is_md=True)
        mis=rectangle(step,pardim)
        tic = timeit.default_timer()
        SA.expand_by_mis(mis)
        runtimes.append(timeit.default_timer() - tic)
        A = mipa.get_approximation()
        As.append(A)
        print('Runtime', runtimes[-1])
    X = np.random.rand(10000, pardim)
    Zl = [A(X) for A in As]
    order2 = plots.plot_convergence(runtimes, [np.array(Z).reshape(-1, 1) for Z in Zl])
    print('Non-adaptive sl_d{}_p2 order: {}'.format(pardim,order2))
    tikz_save('./results/kink_non_sl_d{}_p2.tex'.format(pardim));
    plt.close()

def find_kappa(pardim):
    if pardim==1:
        CSTEPS = 11#FACTOR 2*pardim*log
    else:
        CSTEPS=4
    runtimes = []
    As=[]
    for step in range(CSTEPS):
        def PDE(X, mi):
            return poisson_kink(X, 64, order=1)
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=0,c_dim_par=pardim,measure='u',interval=[-1,1])
        SA = SparseApproximator(decomposition=mipa.expand, 
                            init_dims=pardim,
                            is_bundled=lambda dim: dim>=0,
                            external=True,is_md=True)
        mis=rectangle(step,pardim)
        tic = timeit.default_timer()
        SA.expand_by_mis(mis)
        runtimes.append(timeit.default_timer() - tic)
        A = mipa.get_approximation()
        As.append(A)
        print('Runtime', runtimes[-1])
    X = np.random.rand(10000, pardim)
    Zl = [A(X) for A in As]
    order2 = plots.plot_convergence(runtimes, [np.array(Z).reshape(-1, 1) for Z in Zl])
    print('Non-adaptive kappa_d{}_p2 order: {}'.format(pardim,order2))
    tikz_save('./results/kink_kappa_d{}_p2.tex'.format(pardim));
    plt.close()

def find_kappa_ad(pardim):
    CSTEPS = 9 #2*2^(1/pardim/2)
    runtimes = []
    As=[]
    for step in range(CSTEPS):
        def PDE(X, mi):
            return poisson_kink(X, 64, order=1)
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=0,c_dim_par=pardim,measure='u',interval=[-1,1])
        SA = SparseApproximator(decomposition=mipa.expand, 
                            init_dims=pardim,
                            is_bundled=lambda dim: dim>=0,
                            work_factor=lambda mis: mipa.required_samples(mis),
                            external=True,is_md=True)
        tic = timeit.default_timer()
        SA.expand_adaptive(T_max=2**step, reset=mipa.reset)
        runtimes.append(timeit.default_timer() - tic)
        A = mipa.get_approximation()
        As.append(A)
        print('Runtime', runtimes[-1])
    X = np.random.rand(10000, pardim)
    Zl = [A(X) for A in As]
    order2 = plots.plot_convergence(runtimes, [np.array(Z).reshape(-1, 1) for Z in Zl])
    print('adaptive kappa_d{}_p2 order: {}'.format(pardim,order2))
    tikz_save('./results/kink_ad_kappa_d{}_p2.tex'.format(pardim));
    plt.close()  
    
if __name__ == '__main__':
    #find_kappa(1)
    #find_kappa(2)
    #find_kappa_ad(2)
    for d in [1]:
        nonadaptive_singlelevel(d)#redo with 8 - > 7 hours
        #nonadaptive_multilevel(d)
        #adaptive_singlelevel(d)
        #adaptive_multilevel(d)