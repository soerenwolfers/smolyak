import numpy as np
from smolyak.applications.pde.poisson import poisson_kink
from smolyak.approximator import Approximator, Decomposition
from swutil import plots
from matplotlib2tikz import save as tikz_save
from smolyak import scilog
import pickle
from smolyak.applications.polynomials.mi_weighted_polynomial_approximator import MIWeightedPolynomialApproximator
from smolyak.applications.polynomials.probability_spaces import TensorProbabilitySpace,\
    UnivariateProbabilitySpace
from smolyak.applications.polynomials.polynomial_spaces import TensorPolynomialSpace
from swutil.logs import Log
from matplotlib.pyplot import savefig
import os
        
class ResponseSurfaceApproximation(object):
    def __init__(self,opts):
        self.opts=opts
        
    def run_test(self,test):   
        def PDE(X, mi):
            return poisson_kink(X,self.opts['n_0']*2** mi[0], order=1)
        prob_space=TensorProbabilitySpace(
            [UnivariateProbabilitySpace(interval=(-1,1))]*self.opts['c_dim_par']
        )
        ps=TensorPolynomialSpace(prob_space)
        mipa = MIWeightedPolynomialApproximator(PDE, c_dim_acc=1,ps=ps,sampler=self.opts['sampler'],reparametrization=True)
        decomposition = Decomposition(func=mipa.update_approximation, n=self.opts['c_dim_par']+1, 
                                       is_bundled=lambda dim: dim>=1, 
                                       is_external=True,
                                       work_factor=lambda mis: mipa.estimated_work(mis)*2**mis[0][0],
                                       has_work_model=True,
                                       has_contribution_model=True)
        SA = Approximator(decomposition=decomposition,work_type='work_model',log=Log())
        SA.expand_adaptive(T=2**test['step'])
        return mipa.get_approximation()
    
    def analyze(self,results,info):
        X = np.random.rand(10000, self.opts['c_dim_par'])
        with open('evaluations.pkl','wb') as fp:
            pickle.dump(X,fp)
        order = plots.plot_convergence(info['runtime'], [np.array(A(X)).reshape(-1, 1) for A in results])
        print('Convergence order ({}): {}'.format(info['name'],order))
        tikz_save('convergence.tex')
        savefig('convergence.pdf', bbox_inches='tight')

if __name__ == '__main__':
    opts={
        'n_0':8,
        'c_dim_par':2,
        'sampler':'optimal'
    }
    rsa=ResponseSurfaceApproximation(opts)
    tests=[
        {'step':7} for l in range(9)
    ]
    path=scilog.conduct(name='rsa_kink_d{}'.format(opts['c_dim_par']), tests=tests, func=rsa.run_test, overwrite=True,supp_data=opts)
    info,results=scilog.load(path=path)
    os.chdir(path)
    rsa.analyze(results,info)
    #demonstrate(numerical_algorithm=sparse_response_surface_approximation_factory),
    #convergence_type='algebraic',
    #work_parameters=[1, 2],
    #convergence_parameters=[1, 1],
    #has_work_model=True,
    #runtime_limit=[80, 300, 300])
