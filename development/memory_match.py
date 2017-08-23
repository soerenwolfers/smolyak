'''
Created on Aug 16, 2017

@author: wolfersf
'''
import re



if __name__=='__main__':
    print max_mem('''Filename: /home/wolfersf/Dropbox/Professional/Projects/smolyak/development/lorenz_lambda.py

Line #    Mem usage    Increment   Line Contents
================================================
    54  73.7188 MiB   0.0000 MiB   def main(n):
    55  73.7188 MiB   0.0000 MiB       L=75;
    56  73.7188 MiB   0.0000 MiB       s=10;
    57  73.7188 MiB   0.0000 MiB       r=28;
    58  73.7188 MiB   0.0000 MiB       b=8/3;
    59  73.7188 MiB   0.0000 MiB       sigma=5;
    60  73.7188 MiB   0.0000 MiB       T=1;
    61  73.7188 MiB   0.0000 MiB       dt=1/50;
    62  73.7188 MiB   0.0000 MiB       c_iter=round(T/dt);
    63  73.7188 MiB   0.0000 MiB       dt=T/c_iter;
    64  73.7188 MiB   0.0000 MiB       class LO(LinearOperator):
    65  73.7188 MiB   0.0000 MiB           def __init__(self):
    66                                         self.shape=[n*n*n,n*n*n]
    67  73.7188 MiB   0.0000 MiB           def _matvec(self,vec):
    68                                         return fn(vec,s,r,b,sigma,L)      
    69                                    
    70                                 #A= scipy.sparse.linalg.eigs(A=LO(),k=5,which='SM',return_eigenvectors=False,tol=0.1)
    71 1531.5117 MiB 1457.7930 MiB       return np.random.rand(n*n*n,n*n*n)''')