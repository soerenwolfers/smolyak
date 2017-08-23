from __future__ import division #WHY DOES THIS NOT WORK????????????????????????
import numpy as np
import scipy.stats
import itertools
def forward_stepping(particles,weights,h,potential):
    N_steps = int(1. / h)
    d=particles.shape[0]
    h = 1. / N_steps
    N_particles = particles.shape[1]
    particle_history=np.empty((d,N_particles,N_steps+1))
    particle_history[:,:,0]=particles
    for step in range(N_steps):
        for (i, p1) in enumerate(particle_history[:,:,step].T):
            drift=np.zeros((d,))
            for (j, p2) in enumerate(particle_history[:,:,step].T):
                if i != j:
                    drift+= float(weights[j])/ N_particles * h * potential(p1,p2)
            particle_history[:,i,step+1] =p1 + drift
    return particle_history
        
def backward_stepping(particles,weights,h,d_potential):
    N_steps = int(1. / h)
    d=particles.shape[0]
    h = 1. / N_steps
    N_particles = particles.shape[1]
    particle_history=np.empty((d,N_particles,N_steps+1))
    particle_history[:,:,0]=particles
    for step in range(N_steps):
        for (i, p1) in enumerate(particle_history[:,:,step].T):
            drift=np.zeros((d,))
            for (j, p2) in enumerate(particle_history[:,:,step].T):
                if i != j:
                    drift+= -(p1 *float(weights[j]))/N_particles*h*d_potential(p1,p2,1)
                    drift+=  -(p2 *float(weights[i]))/N_particles*h*d_potential(p2,p1,2)       
            particle_history[:,i,step+1] = p1 +drift
    return particle_history
        
def update_control(control,particle_history,costate_history,alpha,potential,rho):
    N_steps=particle_history.shape[2]-1
    N_particles=particle_history.shape[1]
    d=particle_history.shape[0]
    new_control=control.copy()
    new_control[0]=control[0]+rho*(-alpha/control[0]**2-1/N_particles*np.sum(costate_history[:,:,-1]*particle_history[:,:,0]))
    new_control[1:]=control[1:]+15*rho*(-1/N_particles*np.sum(costate_history[:,:,-1],axis=1))
    #control[1]=max([control[1],0])
    return new_control

def iteration(c_particles,c_steps,c_iter,alpha,d,rho,potential,d_potential,control,random=False):
    print(c_particles,c_steps)
    h=1./c_steps
    
    new_control=control
    if random:
        np.random.seed(10)
        normal_discretization = np.random.multivariate_normal(np.zeros((d,)),np.eye(d),size=(c_particles,)).T
    else:
        normal_discretization = np.array(list(itertools.product(*([scipy.stats.norm.ppf(np.linspace(0,1,c_particles+2)[1:-1])]*d)))).T
    N_particles=normal_discretization.shape[1]
    weights=np.ones((N_particles,))
    for __ in range(c_iter):
        control=new_control
        init_particles=np.tile(control[1:,None],(1,N_particles))+control[0]*normal_discretization
        particle_history=forward_stepping(init_particles,weights,h,potential)
        costate_history=backward_stepping(-particle_history[:,:,-1],weights,h,d_potential)
        new_control=update_control(control,particle_history,costate_history,alpha,potential,rho)
        #print(new_control)
    return control
        
if __name__=='__main__':
    np.seterr(all='raise')
    N=4
    c_steps=20
    c_iter=100
    alpha=1
    rho=-0.05
    power_reject=1
    power_attract=1
    d=2
    def potential(p1,p2,power):
        return (p2 - p1) * np.linalg.norm(p2 - p1)**(power-1)
    def d_potential(p1,p2,power,k):
        return (-1)**k*(np.linalg.norm(p2-p1)**(power-1)+(power-1)*np.power(p2-p1,2)*np.linalg.norm(p2-p1)**(power-3))
        #return (-1)**k*power*np.linalg.norm(p2-p1)**(power-1)
    def total_potential(p1,p2):
        return -potential(p1,p2,power_reject)#+potential(p1,p2,power_attract)
    def total_d_potential(p1,p2,k):
        return -d_potential(p1,p2,power_reject,k)#+d_potential(p1,p2,power_attract,k)
    control=np.array([0.1]+d*[1])
    print(iteration(N,c_steps,c_iter,alpha,1,rho,total_potential,total_d_potential,control))
    