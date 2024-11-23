import numpy as np
from ODESolving import *
from numpy.linalg import norm

# X = [x1;x2;x3]
# F = [F1;F2]
# F1 = sin(x1) + x3**3 + 2
# F2 = e**x1 + x2

def F1(x1,x3):
    return np.sin(x1) + x3**3 + 2

def F2(x1,x2):
    return np.exp(x1) + x2

def F(X):
    return [F1(X[0],X[2]),F2(X[0],X[1])]

def DF(X):
    return [[np.cos(X[0]), 0, 3*(X[2]**2)],
            [np.exp(X[0]), 1, 0]]

tol = 1e-12

X0 = [-2,0,-1]
Xi = X0
Fmag = 1
Xvec = []
Fvec = []


max_iters = 50
i=0

while Fmag > tol and i < max_iters:
    Fi = np.array(F(Xi))
    Fmag = norm(Fi)
    print("Fmag",Fmag)
    Fvec.append(Fmag)
    DFi = np.array(DF(Xi))
    Xi = Xi - min_norm(DFi)@Fi
    Xvec.append(Xi)
    i += 1

print("iter",i)
print("Xvec[-1]",Xvec[-1])
print("Fvec",Fmag)