from dolfin import *  # @UnusedWildImport

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

import logging
import numpy as np
logging.getLogger('FFC').setLevel(5)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('UFL').setLevel(5)
set_log_level(50)

def poisson_smooth(y, N, order=1):
    deg = order
    mesh = UnitSquareMesh(int(N), int(N))
    # File("mesh.pvd") << mesh
    V = FunctionSpace(mesh, 'CG', deg)
    V = FunctionSpace(mesh,'Lagrange',deg)
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
        D = Constant(0.1 + np.sum(np.abs(y[i, :] - 0.5) ** 2))
        a = dot(D * grad(ut), grad(v)) * dx
        u = Function(V)
        problem = LinearVariationalProblem(a, L, u, bc)
        solver = LinearVariationalSolver(problem)
        # solver.parameters['linear_solver'] = 'cg'
        # solver.parameters['preconditioner'] = 'amg'
        # cg_prm = solver.parameters['krylov_solver']
        # cg_prm['absolute_tolerance'] = 1E-14
        # cg_prm['relative_tolerance'] = 1E-14
        # cg_prm['maximum_iterations'] = 10000
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
    class kink_coefficients(Expression):
        def eval(self, value, x):
            value[0]=1+np.power(np.linalg.norm(self.y),3)+np.linalg.norm(x)
    D=kink_coefficients(degree=2*deg)    
    for i in range(y.shape[0]):
        # Compute solution
        #D = Expression(expression_string, y=np.power(np.sum(np.abs(y[i, :]) ** 2),3./2.), degree=2*deg)
        # D = Constant(1 + np.sqrt(np.sum(np.abs(y[i, :]) ** 2))) 
        D.y=y[i,:]
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
