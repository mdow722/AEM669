import numpy as np
from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *

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
fig, ax = plt.subplots(figsize=(5,5), dpi=96)
fig, ax1 = plt.subplots(figsize=(5,5), dpi=96)
fig, ax2 = plt.subplots(figsize=(5,5), dpi=96)
ax.set(title=f"Stable and Unstable Eigenvectors\nand Eigenspaces at Earth-Moon L{libration_point}",
       xlabel="X", ylabel="Y")
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
        ax.quiver(*origin,*eigvecs[i,:2],color=['r'],label="Stable Eigenvector")
        ax.axline((0,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Stable Eigenspace",color="r",linestyle=":")
        ax1.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Stable Eigenspace",color="r",linestyle=":")
        ax2.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Stable Eigenspace",color="r",linestyle=":")
    elif realeig > 0:
        print("unstable")
        print("eigenvalue: ",eigvals[i])
        print("eigenvector: ",eigvecs[i,:2])
        eigenvecs_for_plot.append(np.real(eigvecs[i,:2]))
        ax.quiver(*origin,*eigvecs[i,:2],color=['b'],label="Unstable Eigenvector")
        ax.axline((0,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Unstable Eigenspace",color="b",linestyle=":")
        ax1.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Unstable Eigenspace",color="b",linestyle=":")
        ax2.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Unstable Eigenspace",color="b",linestyle=":")
    else:
        raise ValueError(eigvals)
    
print(eigenvecs_for_plot)
    
# Part B
X_L1 = np.array([xL1,0,0])
JC_L1 = GetJacobiConstant(X_L1,[0,0,0],mu)
ax1.set(title=f"Trajectories near L{libration_point}",
        xlabel="X", ylabel="Y")
dXvec = [30,50,200]
ax1.plot(X_L1[0], 0, 'bo', label=f"L{libration_point}")
Xgrid,Ygrid,Zgrid = GetZVCGrid_xy(mu,JC_L1,resolution=1e-5,xlimits=[0.8365,0.842],ylimits=[-0.003,0.003])
ax1.contour(Xgrid,Ygrid,Zgrid,levels=[0])
ax1.set(xlim=(0.8365,0.842), ylim=(-0.002,0.002))
for i in range(len(dXvec)):
    print("offset: ",dXvec[i])
    dX0 = np.array([dXvec[i]/Lstar,0,0])
    X0 = X_L1 + dX0
    print("X0",X0)
    V0 = GetVelFromJacobiAndPos(X0,JC_L1,mu)
    print("V0",V0)
    XV0 = [*X0,0,V0,0]
    tof = 3
    solp,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,tof,500), args=(mu,))
    soln,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,-tof,500), args=(mu,))
    ax1.plot(solp[:, 0], solp[:, 1],label=f"{dXvec[i]}km - Forward")
    ax1.plot(soln[:, 0], soln[:, 1],label=f"{dXvec[i]}km - Backward")

dXvec = [100,100,-100]
dYvec = [100,-100,-100]

ax2.set(title=f"Trajectories near L{libration_point}",
        xlabel="X", ylabel="Y")
ax2.plot(X_L1[0], 0, 'bo', label=f"L{libration_point}")
ax2.set(xlim=(0.825,0.85), ylim=(-0.002,0.002))
Xgrid,Ygrid,Zgrid = GetZVCGrid_xy(mu,JC_L1,resolution=1e-5,xlimits=[0.825,0.85],ylimits=[-0.003,0.003])
ax2.contour(Xgrid,Ygrid,Zgrid,levels=[0])
for i in range(len(dXvec)):
    print("offset: ",dXvec[i])
    dX0 = np.array([dXvec[i]/Lstar,dYvec[i]/Lstar,0])
    X0 = X_L1 + dX0
    print("X0",X0)
    V0 = GetVelFromJacobiAndPos(X0,JC_L1,mu)
    print("V0",V0)
    XV0 = [*X0,0,V0,0]
    tof = 3
    solp,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,tof,500), args=(mu,))
    soln,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,-tof,500), args=(mu,))
    ax2.plot(solp[:, 0], solp[:, 1],label=f"({dXvec[i]}km,{dYvec[i]}km) - Forward")
    ax2.plot(soln[:, 0], soln[:, 1],label=f"({dXvec[i]}km,{dYvec[i]}km) - Backward")



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
fig, ax3 = plt.subplots(figsize=(5,5), dpi=96)
fig, ax4 = plt.subplots(figsize=(5,5), dpi=96)
fig, ax5 = plt.subplots(figsize=(5,5), dpi=96)
ax3.set(title=f"Stable and Unstable Eigenvectors\nand Eigenspaces at Earth-Moon L{libration_point}",
       xlabel="X", ylabel="Y")
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
        ax3.quiver(*origin,*eigvecs[i,:2],color=['r'],label="Stable Eigenvector")
        ax3.axline((0,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Stable Eigenspace",color="r",linestyle=":")
        ax4.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Stable Eigenspace",color="r",linestyle=":")
        ax5.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Stable Eigenspace",color="r",linestyle=":")
    elif realeig > 0:
        print("unstable")
        print("eigenvalue: ",eigvals[i])
        print("eigenvector: ",eigvecs[i,:2])
        eigenvecs_for_plot.append(np.real(eigvecs[i,:2]))
        ax3.quiver(*origin,*eigvecs[i,:2],color=['b'],label="Unstable Eigenvector")
        ax3.axline((0,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Unstable Eigenspace",color="b",linestyle=":")
        ax4.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Unstable Eigenspace",color="b",linestyle=":")
        ax5.axline((xL1,0),slope=eigvecs[i][1]/eigvecs[i][0],label="Unstable Eigenspace",color="b",linestyle=":")
    else:
        raise ValueError(eigvals)
    
print(eigenvecs_for_plot)
    
# Part C
X_L1 = np.array([xL1,0,0])
JC_L1 = GetJacobiConstant(X_L1,[0,0,0],mu)
ax4.set(title=f"Trajectories near L{libration_point}",
        xlabel="X", ylabel="Y")
dXvec = [30,50,200]
ax4.plot(X_L1[0], 0, 'bo', label=f"L{libration_point}")
Xgrid,Ygrid,Zgrid = GetZVCGrid_xy(mu,JC_L1,resolution=1e-5,xlimits=[1.1550,1.158],ylimits=[-0.003,0.003])
ax4.contour(Xgrid,Ygrid,Zgrid,levels=[0])
for i in range(len(dXvec)):
    print("offset: ",dXvec[i])
    dX0 = np.array([dXvec[i]/Lstar,0,0])
    X0 = X_L1 + dX0
    print("X0",X0)
    V0 = GetVelFromJacobiAndPos(X0,JC_L1,mu)
    print("V0",V0)
    XV0 = [*X0,0,V0,0]
    tof = 3
    solp,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,tof,500), args=(mu,))
    soln,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,-tof,500), args=(mu,))
    ax4.plot(solp[:, 0], solp[:, 1],label=f"{dXvec[i]}km - Forward")
    ax4.plot(soln[:, 0], soln[:, 1],label=f"{dXvec[i]}km - Backward")
ax4.set(xlim=(1.1550,1.158), ylim=(-0.002,0.002))

dXvec = [100,100,-100]
dYvec = [100,-100,-100]

ax5.set(title=f"Trajectories near L{libration_point}",
        xlabel="X", ylabel="Y")
ax5.plot(X_L1[0], 0, 'bo', label=f"L{libration_point}")
ax5.set(xlim=(1.1550,1.158), ylim=(-0.002,0.002))
Xgrid,Ygrid,Zgrid = GetZVCGrid_xy(mu,JC_L1,resolution=1e-5,xlimits=[1.1550,1.158],ylimits=[-0.003,0.003])
ax5.contour(Xgrid,Ygrid,Zgrid,levels=[0])
for i in range(len(dXvec)):
    print("offset: ",dXvec[i])
    dX0 = np.array([dXvec[i]/Lstar,dYvec[i]/Lstar,0])
    X0 = X_L1 + dX0
    print("X0",X0)
    V0 = GetVelFromJacobiAndPos(X0,JC_L1,mu)
    print("V0",V0)
    XV0 = [*X0,0,V0,0]
    tof = 3
    solp,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,tof,500), args=(mu,))
    soln,_,_ = rungekutta4(CR3BP_nondim, XV0, np.linspace(0,-tof,500), args=(mu,))
    ax5.plot(solp[:, 0], solp[:, 1],label=f"({dXvec[i]}km,{dYvec[i]}km) - Forward")
    ax5.plot(soln[:, 0], soln[:, 1],label=f"({dXvec[i]}km,{dYvec[i]}km) - Backward")

ax.legend()
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
plt.show()