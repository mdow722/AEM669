import numpy as np
from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *

Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
Vstar = Lstar/Tstar

def F1(x,x_target):
    return x_target - x

def F2(y,y_target):
    return y_target - y

def F3(z,z_target):
    return z_target - z

def F(V0,R0,R_target,tof,mu):
    sol,_,_ = rungekutta4(CR3BP_nondim, [*R0,*V0], np.linspace(0,tof,100), args=(mu,))
    RF = sol[-1,:3]
    evec = R_target - RF
    print("evec_dim",evec*Lstar)
    print("emag_dim",norm(evec*Lstar))
    return np.array([F1(RF[0],R_target[0]),F2(RF[1],R_target[1]),F3(RF[2],R_target[2])]), sol

def DF(V0,R0,tof,mu):
    states, STMs = STM_coupled_integration_vec([*R0,*V0],np.linspace(0,tof,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    STM = STMs[-1]
    xdotf = CR3BP_nondim(0,states[-1],mu).reshape(6,1)
    J = np.hstack((STM,xdotf))
    return -np.identity(3)@J[:3,3:7]

tol = 1e-12

R0 = [300000/Lstar,0,0]
V0 = np.array([0,0.5/Vstar,0.5/Vstar])
Fmag = 1
Xvec = []
Fvec = []
trajvec = []

RT = [500000/Lstar,-90000/Lstar,200000/Lstar]
TOF0 = (10*86400)/Tstar
TOF = (10*86400)/Tstar

Xi = [*V0,TOF]
V0d = np.array([*V0])
max_iters = 100
i=0
target_achieved = False

while Fmag > tol and i < max_iters:
    print("current iteration",i)
    Fi,traj = F([*Xi[:3]],R0,RT,TOF,mu)
    Fmag = norm(Fi)
    Fvec.append(Fmag)
    trajvec.append(traj)
    DFi = np.array(DF([*Xi[:3]],R0,TOF,mu))
    # Xi = Xi - inv(DFi)@Fi
    Xi = Xi - min_norm(DFi)@Fi
    Xvec.append(Xi)
    print("Delta V vec:",(V0d-V0)*Vstar)
    print("Delta V mag:",norm((V0d-V0)*Vstar))
    print("TOF:",TOF*Tstar/86400)
    V0d[0:3] = Xi[0:3]
    TOF = Xi[3]
    i += 1
    if Fmag < tol:
        target_achieved = True

print("Target Achieved: ",target_achieved)
print("iteration amount",i)
# print("Xvec",Xvec)
print("V0d_dim",np.array(V0d)*Vstar)
print("Delta V vec:",(V0d-V0)*Vstar)
print("Delta V mag:",norm((V0d-V0)*Vstar))
print("Achieved Time (d):",TOF*Tstar/86400)

sol,_,_ = rungekutta4(CR3BP_nondim, [*R0,*V0], np.linspace(0,TOF,100), args=(mu,))
evec = RT - sol[-1,:3]
print("evec",evec)
print("evec_dim",evec*Lstar)

ax = Plot_CR3BP_3D(trajvec,mu,TOF*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6),names=range(i))
ax.plot(*RT, 'mo', label="Target Position")
ax.plot(trajvec[0][-1][0], trajvec[0][-1][1], trajvec[0][-1][2], 'ro', label="Initial Guess' Final Position")
if target_achieved == False:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], trajvec[-1][-1][2], 'yo', label="Last Iteration's Final Position")








plt.legend()
plt.show()