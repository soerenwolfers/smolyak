from __future__ import division
from scipy.sparse.linalg.interface import LinearOperator
import numpy as np
import scipy
from smolyak.aux.decorators import print_runtime
from smolyak.aux.plots import plot_convergence
from time import sleep

def fn(X,s,r,b,sigma,L):
    n=int(round(np.power(len(X),1/3)))
    X=X.reshape([n,n,n]);
    Y=np.zeros([n,n,n]);
    dx=2*L/(n-1);
    for i in range(n):
        for j in range(n):
            for k in range(n):
                con=-convection(lambda i,j,k: padded(X,i,j,k),i,j,k,s,r,b,dx,n,L)
                Y[i,j,k]=con
                Y[i,j,k]=Y[i,j,k]+diffusion(lambda i,j,k: padded(X,i,j,k),i,j,k,s,r,b,sigma,dx)
                #Y(i,j,k)=diffusion(lambda i,j,k: padded(X,i,j,k),i,j,k,s,r,b,sigma,dx)
    Y=Y.reshape([Y.size,1]);
    return Y

def ind2co(i,j,k,n,L):
    co=np.array([[-L+2*L*i/(n-1)],#TODO: IS THIS CORRECT?
                 [-L+2*L*j/(n-1)],
                  [-L+2*L*k/(n-1)]]);
    return co

def drift(co,ind,s,r,b,L):
    dr=np.array([-s*co[0]+s*co[1],
        r*co[0]-co[1]-co[0]*co[2],
        co[0]*co[1]-b*co[2]])
    dr=dr[ind];
    #if co(1)<-L || co(1)>L || co(2)<-L ||co(2)>L||co(3)<-L||co(3)>L
    #    dr=0;
    return dr

def convection(X,i,j,k,s,r,b,dx,n,L):
    con=(drift(ind2co(i+1,j,k,n,L),0,s,r,b,L)*X(i+1,j,k)-drift(ind2co(i-1,j,k,n,L),0,s,r,b,L)*X(i-1,j,k))/(2*dx);
    con=con+(drift(ind2co(i,j+1,k,n,L),1,s,r,b,L)*X(i,j+1,k)-drift(ind2co(i,j-1,k,n,L),1,s,r,b,L)*X(i,j-1,k))/(2*dx);
    con=con+(drift(ind2co(i,j,k+1,n,L),2,s,r,b,L)*X(i,j,k+1)-drift(ind2co(i,j,k-1,n,L),2,s,r,b,L)*X(i,j,k-1))/(2*dx);
    return con 

def diffusion(X,i,j,k,s,r,b,sigma,dx):
    dif=(X(i+1,j,k)-2*X(i,j,k)+X(i-1,j,k))/(dx*dx);
    dif=dif+(X(i,j+1,k)-2*X(i,j,k)+X(i,j-1,k))/(dx*dx);
    dif=dif+(X(i,j,k+1)-2*X(i,j,k)+X(i,j,k-1))/(dx*dx);
    dif=1/2*sigma*dif;
    return dif
def padded(X,i,j,k):
    n=len(X);
    return X[max(0,min(n-1,i)),max(0,min(n-1,j)),max(0,min(n-1,k))];

def main(n,**kwargs):
    L=75;
    s=10;
    r=28;
    b=8/3;
    sigma=5;
    T=1;
    dt=1/50;
    c_iter=round(T/dt);
    dt=T/c_iter;
    class LO(LinearOperator):
        def __init__(self):
            self.shape=[n*n*n,n*n*n]
        def _matvec(self,vec):
            return fn(vec,s,r,b,sigma,L)      
       
    #A= scipy.sparse.linalg.eigs(A=LO(),k=5,which='SM',return_eigenvectors=False,tol=0.1)
    sleep(10)
    print(n)
    print(kwargs)
    return n#np.random.rand(n*n*n,n*n*n)

def analyze(results,info):
    print(results)
    #plot_convergence(results, info['runtime'])
    
if __name__=='__main__':
    print(main(90))
    
