from dolfin import *  # @UnusedWildImport
import logging
import numpy as np
logging.getLogger('FFC').setLevel(logging.warnings)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('UFL').setLevel(logging.warnings)
set_log_level(WARNING)

def kl(y, N, order=1):
    deg = order
    mesh = UnitSquareMesh(int(N), int(N))
    V = FunctionSpace(mesh, 'CG', deg)
    bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
    output = np.zeros((y.shape[0], 1))
    ut = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    L = f * v * dx
    d = y.shape[1]
    expression_string = diffusion_coefficient(d)
    for j in range(y.shape[0]):
        # Compute solution
        arg_dict = { 'y%d' % (i + 1): v for i, v in enumerate(y[j, :])}
        arg_dict['degree'] = 2 * deg
        D = Expression(expression_string, **arg_dict)
        a = dot(D * grad(ut), grad(v)) * dx
        u = Function(V)
        problem = LinearVariationalProblem(a, L, u, bc)
        solver = LinearVariationalSolver(problem)
        solver.parameters['linear_solver'] = 'cg'
        solver.parameters['preconditioner'] = 'amg'
        cg_prm = solver.parameters['krylov_solver']
        cg_prm['absolute_tolerance'] = 1E-14
        cg_prm['relative_tolerance'] = 1E-14
        cg_prm['maximum_iterations'] = 10000
        solver.solve()
        integrand = u * dx
        a = assemble(integrand)
        # print(a)
        output[j] = a
    return output

def diffusion_coefficient(d):
    expression = 'exp( 0 '
    for dd in range(d):
        dd = dd + 1
        line = int(np.ceil(-1. / 2 + np.sqrt(1. / 4 + 2 * dd) - 1))
        remains = int(dd - (line ** 2 + line) / 2.)
        if remains > 0:
            s1 = line + 2 - remains
            s2 = remains
        expression += '+'
        expression += 'pow({},-4)*y{}*'.format(dd, dd)
        if s1 % 2 == 0:
            expression += 'sin({}*pi*x[0])'.format(int(s1 / 2.))
        else:
            expression += 'cos({}*pi*x[0])'.format(int((s1 + 1) / 2.))
        if s2 % 2 == 0:
            expression += '*sin({}*pi*x[1])'.format(int(s2 / 2.))
        else:
            expression += '*cos({}*pi*x[1])'.format(int((s2 + 1) / 2.))
    expression += ')'
    return expression
