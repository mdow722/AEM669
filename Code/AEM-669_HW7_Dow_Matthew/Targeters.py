import numpy as np
from ThreeBodyModel import *
from ODESolving import *
from numpy.linalg import inv,norm

# def clamp(num, min_value, max_value):
#    return max(min(num, max_value), min_value)

# def is_square(mat):
#     if isinstance(mat,np.ndarray) == False:
#         mat = np.array(mat)
#     return len(mat.shape) == 2 and mat.shape[0] == mat.shape(1)

# def SingleShooter_FixedTime(X0_g,T_g,F,DF,F_args=None,DF_args=None):
