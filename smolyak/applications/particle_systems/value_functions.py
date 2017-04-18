import numpy as np
from smolyak.applications.particle_systems.particle_approximations import inverse_transform_sampling, \
    random_sampling

def univariate_integral(particles, phi, h, shift):
    N = len(particles)
    if N == 0:
        raise ValueError('Need at least one particle')
    if shift:
        phi_particles = phi(np.array(particles) + h)
    else:
        phi_particles = phi(np.array(particles))
    return 1. / N * np.sum(phi_particles) + h

def time_stepping(particles, phi, h, shift):
    N_steps = int(1. / h)
    h = 1. / N_steps
    N_particles = len(particles)
    for __ in range(N_steps):
        new_particles = particles.copy()
        for (i, particle) in enumerate(particles):
            for (j, particle_2) in enumerate(particles):
                if i != j:
                    new_particles[i] = new_particles[i] + 1. / N_particles * h * (particle_2 - particle) * np.power(np.absolute(particle_2 - particle), 3)
        particles = new_particles
    phi_particles = phi(new_particles)
    return 1. / N_particles * np.sum(phi_particles)
                    
    
    

def univariate_integral_approximation(N_particles, h, invcdf=None, pdf=None, bound=None, phi=None, shift=None, app_type='time_stepping'):
    if invcdf:
        particles = inverse_transform_sampling(invcdf, N_particles)
        
    else:
        particles = random_sampling(pdf, bound, N_particles)
    if app_type == 'artificial':
        return univariate_integral(particles, phi, h, shift)
    elif app_type == 'time_stepping':
        return time_stepping(particles, phi, h, shift)
