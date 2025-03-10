#===========================================
# Statistical Computing - Project 1
# Implementation of Numerical Methods
#===========================================

"""
This module implements various numerical methods for solving linear systems.
Data is generated for testing purposes; these algorithms work with any valid input data.
"""

#===========================================
# Problem 1 - Part 1: LS-SVM Implementation
# Conjugate Gradient Method
#===========================================

import numpy as np
from scipy.linalg import norm

def ConjugateGrad(A,B,x0,i,thresh):
    """
    Conjugate Gradient Method for solving Ax = B
    Args:
        A: System matrix
        B: Target vector
        x0: Initial guess
        i: Maximum iterations
        thresh: Convergence threshold
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    r0 = B
    p_1 = r0
    x_i = x0
    k=1
    
    while(k<i):
        lambda_i = np.dot(r0,r0) / np.dot(np.dot(p_1,A),p_1)	
        x_i = x0 + np.dot(lambda_i,p_1)							
        r_i = r0 - np.dot(np.dot(lambda_i,A),p_1)
            
        beta_i = np.dot(r_i,r_i) / np.dot(r0,r0)				
        p_i = r_i + np.dot(beta_i,p_1)							

        if(norm(r_i) <= thresh):
            break
        
        x0 = x_i
        p_1 = p_i
        r0 = r_i
        
        k+=1
        
    return x_i

A = np.array([[1.,-1.,0.],\
              [-1.,2.,1.],\
              [0.,1.,5.]])
B = np.array([3.,-3.,4.])				
x = ConjugateGrad(A,B,np.array([0.,0.,0.]),100,1e-9)


print("ConjugateGrad = {}.".format(x.round(2)))

#===========================================
# Problem 1 - Part 3: Linear System Solver
# Incremental Method Implementation
#===========================================

import numpy as np

# Input matrix
R = np.array([[6, 5, 0],
              [5, 1, 4],
              [0, 4, 3]])
b = np.array([3.,-3.,4.])


def incremental(A, b):
    """
    Incremental method for solving triangular system
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    n = np.size(b)
    x = np.zeros_like(b)

    x[-1] = 1. / A[-1, -1] * b[-1]
    # Fix: Use range instead of xrange for Python 3 compatibility
    for i in range(n-2, -1, -1):
        x[i] = 1. / A[i, i] * (b[i] - np.sum(A[i, i+1:] * x[i+1:]))

    return x
x = incremental(R,b)
    
print ("incremental = {}".format(x.round(2)))
