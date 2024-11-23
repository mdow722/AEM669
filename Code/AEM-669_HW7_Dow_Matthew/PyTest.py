import numpy as np

Xvec = np.array(range(36))
print("Xvec",Xvec)

mat6 = Xvec.reshape(6,6)
print("mat6",mat6)

newvec = mat6.flatten()
print("newvec",newvec)