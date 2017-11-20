import numpy as np
from swutil import plots
from matplotlib2tikz import save as tikz_save
import pickle
from smolyak.applications.polynomials.probability_spaces import TensorProbabilitySpace,\
    UnivariateProbabilitySpace
from smolyak.applications.polynomials.polynomial_spaces import TensorPolynomialSpace
from swutil.logs import Log
from matplotlib.pyplot import savefig
from smolyak.applications.polynomials.weighted_polynomial_approximator import WeightedPolynomialApproximator
from smolyak.smolyak import Decomposition, Approximator
from matplotlib import pyplot
from smolyak import indices
from swutil.decorators import print_runtime
from swutil.plots import plot_convergence, plot_nterm_convergence
import matplotlib
def main(d,time):   
    def kink(X):
        return np.clip(np.sum(X,1),0,np.Inf)
    prob_space=TensorProbabilitySpace(
        [UnivariateProbabilitySpace(interval=(-1,1))]*d
    )
    ps=TensorPolynomialSpace(prob_space)
    wpa = WeightedPolynomialApproximator(kink, ps,sampler='arcsine')
    decomposition = Decomposition(func=wpa.update_approximation, n=d, 
                                   is_bundled=True ,
                                   is_external=True,
                                   work_factor=lambda mis: wpa.estimated_work(mis),
                                   has_work_model=True,
                                   has_contribution_model=True)
    SA = Approximator(decomposition=decomposition,work_type='work_model')
    print_runtime(SA.update_approximation)(indices.rectangle(150, d))
    #SA.expand_adaptive(T_max=20)
    #wpa.get_approximation().plot()
    SA.plot_indices(weighted='contribution',percentiles=20)
    matplotlib.pyplot.figure()
    rate=plot_nterm_convergence(list(SA.md.contributions.values()))
    print(rate)
    pyplot.show()

if __name__ == '__main__':
    main(d=2,time=60)
