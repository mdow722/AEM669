import numpy as np
from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *

def normalize(vector):
    magnitude = norm(vector)
    return vector / magnitude

Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
Vstar = Lstar/Tstar

xLib,_,_ = Get3BodyColinearLibrationPoints(mu,Lstar)

libration_point = 1
xL1 = xLib[libration_point-1]
print("xL1",xL1)
A_L1 = GetAMatrix_CR3BP([xL1,0,0],mu,dim=2)
eigvals,eigvecs = GetEigenvaluesAndVectors(A_L1)
eigvecs = eigvecs.transpose()
print("eigvals",eigvals)
print("eigvecs",eigvecs)
print("eigvecs[0]",eigvecs[0])
print("mutlts",eigvecs@eigvals)

num_eigvals = len(eigvals)
eigenvecs_for_plot = []
origin = [0,0]
fig, ax1 = plt.subplots(figsize=(5,5), dpi=96)
fig, ax2 = plt.subplots(figsize=(5,5), dpi=96)
fig, ax3 = plt.subplots(figsize=(5,5), dpi=96)
stable_eigvec = None
unstable_eigvec = None
for i in range(num_eigvals):
    # print("eigenvalue: ",eigvals[i])
    realeig = np.real(eigvals[i])
    if np.abs(realeig) < 1e-9 :
        print("central case")
        print("eigenvalue: ",eigvals[i])
        print("eigenvector: ",eigvecs[i,:2])
    elif realeig < 0:
        print("stable")
        print("eigenvalue: ",eigvals[i])
        print("eigenvector: ",eigvecs[i,:2]) 
        eigenvecs_for_plot.append(np.real(eigvecs[i,:2]))
        ax1.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Stable Eigenspace",color="r",linestyle=":")
        ax3.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="L1 Stable Eigenspace",color="r",linestyle=":")
        stable_eigvec = normalize(eigvecs[i])
    elif realeig > 0:
        print("unstable")
        print("eigenvalue: ",eigvals[i])
        print("eigenvector: ",eigvecs[i,:2])
        eigenvecs_for_plot.append(np.real(eigvecs[i,:2]))
        ax1.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Unstable Eigenspace",color="b",linestyle=":")
        ax3.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label=" L1Unstable Eigenspace",color="b",linestyle=":")
        unstable_eigvec = normalize(eigvecs[i])
    else:
        raise ValueError(eigvals)
    
print("eigenvecs_for_plot",eigenvecs_for_plot)
X_L1 = np.array([xL1,0,0])
JC_L1 = GetJacobiConstant(X_L1,[0,0,0],mu)
ax1.set(title=f"Trajectories near L{libration_point}",
        xlabel="X", ylabel="Y")
ax3.set(title=f"Manifolds at L1 and L2",
        xlabel="X", ylabel="Y")
# dXvec = [30,50,200,1000,2000,10000,20000]
dXvec = [50]
ax1.plot(xLib[0], 0, 'mo', label=f"L1")
ax3.plot(xLib[0], 0, 'mo', label=f"L1")
ax3.plot(xLib[1], 0, 'mo', label=f"L2")
for i in range(len(dXvec)):
    print("offset: ",dXvec[i])
    for stb in ["stable","unstable"]:
        eigvec = unstable_eigvec
        propdir = +1
        if stb == "stable":
            eigvec = stable_eigvec
            propdir = -1
        for dir in ["pos","neg"]:
            dirfactor = -1
            if dir == "pos":
                dirfactor = 1
            print("eigvec",eigvec)
            scaled_eigvec = dirfactor*dXvec[i]/Lstar*eigvec
            dX0 = np.array([*scaled_eigvec[:2],0])
            # dX0 = np.array([dXvec[i]/Lstar,0,0])
            X0 = X_L1 + dX0
            print("X0",X0)
            V0 = np.array([*scaled_eigvec[2:],0])
            print("V0",V0)
            XV0 = [*X0,*V0]
            JC_IC = GetJacobiConstant(XV0[:3],XV0[3:],mu)
            tof = 2.5
            sol,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,propdir*tof,5000), args=(mu,))
            ax1.plot(sol[:, 0], sol[:, 1],label=f"{stb} - {dir}")
            ax3.plot(sol[:, 0], sol[:, 1],label=f"L1 {stb} - {dir}")

# ax1.plot(-mu, 0, 'bo', label="Earth")
# ax1.plot(1 - mu, 0, 'go', label="Moon")

ax1.legend()

libration_point = 2
xL1 = xLib[libration_point-1]
print("xL1",xL1)
A_L1 = GetAMatrix_CR3BP([xL1,0,0],mu,dim=2)
eigvals,eigvecs = GetEigenvaluesAndVectors(A_L1)
eigvecs = eigvecs.transpose()
print("eigvals",eigvals)
print("eigvecs",eigvecs)
print("eigvecs[0]",eigvecs[0])
print("mutlts",eigvecs@eigvals)

num_eigvals = len(eigvals)
eigenvecs_for_plot = []
origin = [0,0]
stable_eigvec = None
unstable_eigvec = None
for i in range(num_eigvals):
    # print("eigenvalue: ",eigvals[i])
    realeig = np.real(eigvals[i])
    if np.abs(realeig) < 1e-9 :
        print("central case")
        print("eigenvalue: ",eigvals[i])
        print("eigenvector: ",eigvecs[i,:2])
    elif realeig < 0:
        print("stable")
        print("eigenvalue: ",eigvals[i])
        print("eigenvector: ",eigvecs[i,:2]) 
        eigenvecs_for_plot.append(np.real(eigvecs[i,:2]))
        ax2.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Stable Eigenspace",color="r",linestyle=":")
        ax3.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="L2 Stable Eigenspace",color="r",linestyle=":")
        stable_eigvec = normalize(eigvecs[i])
    elif realeig > 0:
        print("unstable")
        print("eigenvalue: ",eigvals[i])
        print("eigenvector: ",eigvecs[i,:2])
        eigenvecs_for_plot.append(np.real(eigvecs[i,:2]))
        ax2.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Unstable Eigenspace",color="b",linestyle=":")
        ax3.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="L2 Unstable Eigenspace",color="b",linestyle=":")
        unstable_eigvec = normalize(eigvecs[i])
    else:
        raise ValueError(eigvals)
    
print("eigenvecs_for_plot",eigenvecs_for_plot)
X_L1 = np.array([xL1,0,0])
JC_L1 = GetJacobiConstant(X_L1,[0,0,0],mu)
ax2.set(title=f"Trajectories near L{libration_point}",
        xlabel="X", ylabel="Y")
# dXvec = [30,50,200,1000,2000,10000,20000]
dXvec = [50]
ax2.plot(xLib[1], 0, 'mo', label=f"L2")
for i in range(len(dXvec)):
    print("offset: ",dXvec[i])
    for stb in ["stable","unstable"]:
        eigvec = unstable_eigvec
        propdir = +1
        if stb == "stable":
            eigvec = stable_eigvec
            propdir = -1
        for dir in ["pos","neg"]:
            dirfactor = -1
            if dir == "pos":
                dirfactor = 1
            print("eigvec",eigvec)
            scaled_eigvec = dirfactor*dXvec[i]/Lstar*eigvec
            dX0 = np.array([*scaled_eigvec[:2],0])
            # dX0 = np.array([dXvec[i]/Lstar,0,0])
            X0 = X_L1 + dX0
            print("X0",X0)
            V0 = np.array([*scaled_eigvec[2:],0])
            print("V0",V0)
            XV0 = [*X0,*V0]
            JC_IC = GetJacobiConstant(XV0[:3],XV0[3:],mu)
            tof = 3
            sol,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,propdir*tof,5000), args=(mu,))
            ax2.plot(sol[:, 0], sol[:, 1],label=f"{stb} - {dir}")
            ax3.plot(sol[:, 0], sol[:, 1],label=f"L2 {stb} - {dir}")

# ax2.plot(-mu, 0, 'bo', label="Earth")
ax3.plot(1 - mu, 0, 'go', label="Moon")

ax2.legend()
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
ax3.set_aspect("equal")

plt.show()