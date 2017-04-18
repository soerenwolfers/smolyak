from dolfin import *  # @UnusedWildImport

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

import logging
import numpy as np
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_level(WARNING)

def poisson_smooth(y, N, order=1):
    deg = order
    mesh = UnitSquareMesh(int(N), int(N))
    # File("mesh.pvd") << mesh
    V = FunctionSpace(mesh, 'CG', deg)
    # Define boundary condition
    u_D = Constant(0)
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_D, boundary)
    
    output = np.zeros((y.shape[0], 1))
    ut = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    L = f * v * dx
    for i in range(y.shape[0]):
        # Compute solution
        D = Constant(0.1 + np.sum(np.abs(y[i, :] - 0.5) ** 2))
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
        a = (assemble(integrand))
        # print(a)    
        output[i] = a
    return output

def poisson_kink(y, N, order=1):
    deg = order
    mesh = UnitSquareMesh(int(N), int(N))
    # File("mesh.pvd") << mesh
    V = FunctionSpace(mesh, 'P', deg)
    # Define boundary condition
    u_D = Constant(0)
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_D, boundary)
    
    output = np.zeros((y.shape[0], 1))
    ut = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    L = f * v * dx
    for i in range(y.shape[0]):
        # Compute solution
        D = Expression('y+pow(4*(x[0]-0.5)*(x[0]-0.5)+4*(x[1]-0.5)*(x[1]-0.5),0.5)', y=1 + np.sqrt(np.sum(np.abs(y[i, :]) ** 2)), degree=2)
        # D = Constant(1 + np.sqrt(np.sum(np.abs(y[i, :]) ** 2)))
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
        a = (assemble(integrand))
        # print(a)    
        output[i] = a
    return output
