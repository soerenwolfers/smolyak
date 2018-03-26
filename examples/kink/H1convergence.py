from smolyak.applications.polynomials.mi_weighted_polynomial_approximator import WeightedPolynomialApproximator
from smolyak import indices
from swutil.plots import plot_convergence
import numpy as np
from matplotlib import pyplot as plt
from smolyak.applications.polynomials.probability_spaces import UnivariateProbabilitySpace
k = 3
N = 40
def f(x):
    return np.maximum(0, np.linalg.norm(x, axis=1) ** k)
def f1(x): #Derivative w.r.t. x_1
    return k * (np.linalg.norm(x, axis=1) ** (k - 2)) * x[:, 0]
prob_space = UnivariateProbabilitySpace('u', (-1, 1))**2
miwpa = WeightedPolynomialApproximator(function=f, probability_space=prob_space, reparametrization=True)
x=np.random.uniform(-1,1,size=(N,2))
zs = []
z1s = []
M = 6
for j in range(M):
    miwpa.update_approximation(indices.rectangle(L=j, n=2))
    pa = miwpa.get_approximation()
    zs.append(pa(x))
    z1s.append(pa(x, [1, 0]))#Evaluate derivative of approximation
plot_convergence(times=4 ** np.arange(M), values=zs, reference=f(x), plot_rate='fit')#L^2 convergence
plot_convergence(times=4 ** np.arange(M), values=z1s, reference=f1(x), plot_rate='fit')#H^1 convergence
plt.show()

