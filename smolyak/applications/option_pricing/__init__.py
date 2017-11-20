import tensorflow as tf
from smolyak.applications.polynomials.polynomial_spaces import TensorPolynomialSpace
from smolyak.applications.polynomials.probability_spaces import UnivariateProbabilitySpace,\
    TensorProbabilitySpace
from smolyak import indices
import numpy as np
from swutil.np_tools import orthonormal_complement_basis
from smolyak.applications.polynomials.polynomial_approximation import PolynomialApproximation
from smolyak.applications.option_pricing.tflow.ops import polynomial_combination_factory, bool_combination
from tensorflow.python.ops.script_ops import py_func
#############
d=1
basis_exercise_boundary=indices.rectangle(L=10,c_dim=d-1)
basis_free_boundary_solution=indices.rectangle(L=10,c_dim=d-1)
option_weights=np.array([1]).reshape([1,d])

prob_space_eb=TensorProbabilitySpace([UnivariateProbabilitySpace()]*(d-1))
ps_eb=TensorPolynomialSpace(prob_space_eb)
ps_eb.set_basis(basis_exercise_boundary)
prob_space_fbs=TensorProbabilitySpace([UnivariateProbabilitySpace()]*d)
ps_fbs=TensorPolynomialSpace(prob_space_fbs)
ps_fbs.set_basis(basis_free_boundary_solution)
#############
i_stock_prices = tf.placeholder(dtype=tf.float64,shape=[d,1])
n_option_weights = tf.constant(option_weights,dtype=tf.float64)
orthogonal_projection=orthonormal_complement_basis(option_weights)
n_orthogonal_projection = tf.constant(orthogonal_projection,dtype=tf.float64)
n_orthogonal_complement = tf.matmul(n_orthogonal_projection,i_stock_prices)
n_bag_price = tf.matmul(n_option_weights,i_stock_prices)
v_exercise_boundary = tf.Variable(np.zeros([len(basis_exercise_boundary)]))
n_exercise_boundary = py_func(
    polynomial_combination_factory(ps_eb),
    [v_exercise_boundary,n_orthogonal_complement],
    tf.float64,
    stateful=True,
    name=None
)
v_free_boundary_solution = tf.Variable(np.zeros([len(basis_free_boundary_solution)]))
n_free_boundary_solution = py_func(
    polynomial_combination_factory(ps_fbs),
    [v_free_boundary_solution,i_stock_prices],
    tf.float64,
    stateful=True,
    name=None
)
n_model_output = py_func(
    bool_combination,
    [n_exercise_boundary,n_bag_price,n_free_boundary_solution],
    tf.float64
)
d_values = tf.placeholder(shape=[1],dtype=tf.float64)
squared_deltas=tf.square(n_model_output-d_values)
loss = tf.reduce_sum(squared_deltas)
