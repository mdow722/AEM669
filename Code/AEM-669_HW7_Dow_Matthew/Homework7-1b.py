import numpy as np
from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Utils import printProgressBar, getEigTypes

Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
print("Lstar",Lstar)
Vstar = Lstar/Tstar

def F1(y,y_target):
    return y_target - y

def F2(vx,vx_target):
    return vx_target - vx

def F(V0,R0,Xtarget,tof,mu):
    sol,_,_ = rungekutta4(CR3BP_nondim, [*R0,0,*V0,0], np.linspace(0,tof,500), args=(mu,))
    RF = sol[-1,[1,3]]
    return np.array([F1(RF[0],Xtarget[0]),F2(RF[1],Xtarget[1])]), sol

def DF(V0,R0,tof,mu):
    states, STMs = STM_coupled_integration_vec([*R0,0,*V0,0],np.linspace(0,tof,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    STM = STMs[-1]
    xdotf = CR3BP_nondim(0,states[-1],mu).reshape(6,1)
    J = np.hstack((STM,xdotf))
    Jrelevant = np.vstack((J[1,[4,6]],J[3,[4,6]]))
    return -np.identity(2)@Jrelevant

xLib,_,_ = Get3BodyColinearLibrationPoints(mu,Lstar)
dx0 = 0.01
dy0 = 0
dR0 = np.array([dx0,dy0])
RL1 = np.array([xLib[0],0])
R0 = RL1 + dR0

vx0 = 0
vy0 = -0.0837226945866448
V0 = np.array([vx0,vy0])

# Get time to cross x axis:
t_0 = 0
t_f = 10.5
num_points = 100
t_points = np.linspace(t_0, t_f, int(t_f*num_points))
crossing_x_axis_event = [False, True, False, False, False, False]
yvec, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,*V0,0], t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)
TOF0 = tvec[-1]
print("TOF0",TOF0)

# ax = Plot_CR3BP([yvec],mu,TOF0*Tstar/86400,show_initial_pos=True,show_final_pos=True,
#            show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))


tol = 1/Lstar

TOF = tvec[-1]
Xi = [V0[1],TOF]
Fmag = 1
Xvec = []
V0d = np.array([*V0])
Fvec = []
trajvec = []

Xtarget = [0,0] # y and vx

max_iters = 100
i=0
target_achieved = False

while Fmag > tol and i < max_iters:
    Fi,traj = F([V0[0],Xi[0]],R0,Xtarget,TOF,mu)
    Fmag = norm(Fi)
    Fvec.append(Fmag)
    trajvec.append(traj)
    DFi = np.array(DF([V0[0],Xi[0]],R0,TOF,mu))
    Xi = Xi - inv(DFi)@Fi
    Xvec.append(Xi)
    V0d[1] = Xi[0]
    TOF = Xi[1]
    i += 1
    if Fmag < tol and TOF > 0.1:
        target_achieved = True

print("Target Achieved: ",target_achieved)
print("iteration amount",i)
# print("Xvec",Xvec)
print("V0d_dim",np.array(V0d)*Vstar)
print("Delta V vec:",(V0d-V0)*Vstar)
print("Delta V mag:",norm((V0d-V0)*Vstar))
print("Achieved Time (d):",TOF*Tstar/86400)
BaseTraj, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,V0[0],Xi[0],0], np.linspace(0,2*TOF,100), args=(mu,))
_, STMs = STM_coupled_integration_vec([*R0,0,V0[0],Xi[0],0],np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))

ax = Plot_CR3BP([BaseTraj],mu,f"L1 Lyapunov Orbit Plotted for {np.round(2*TOF0*Tstar/86400,3)} Days",show_initial_pos=False,show_final_pos=False,
                show_bodies=True,show_ZVC=False,xlimits=(0.6,1.1),ylimits=(-0.2,.2))
ax.plot(BaseTraj[:,0], BaseTraj[:,1], 'k', label="Initial Orbit")
ax.plot(RL1[0], RL1[1], 'co', label="L1")
ax.legend()

eigvals,eigvecs = np.linalg.eig(STMs[-1][[0,1,3,4],:][:,[0,1,3,4]])
print("eigvals",eigvals)
print("eigvecs",eigvecs)

eigtypes = getEigTypes(eigvals)
print("eigtypes",eigtypes)

orbit_fixed_points,_,_ = rungekutta4(CR3BP_nondim, [*R0,0,V0[0],Xi[0],0], np.linspace(0,2*TOF,20), args=(mu,))
eigval_arr = []
eigvec_arr = []
i = 0

for orbit_point in orbit_fixed_points:
    # print("orbit_point",orbit_point)
    print("point num",i)
    _, STMs = STM_coupled_integration_vec(orbit_point,np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    eigvals,eigvecs = np.linalg.eig(STMs[-1][[0,1,3,4],:][:,[0,1,3,4]])
    eigvecs = eigvecs.transpose()
    print("eigvals",eigvals)
    ax.quiver(*orbit_point[:2],*eigvecs[0,:2],color=['r'],width=0.001)
    ax.quiver(*orbit_point[:2],*eigvecs[1,:2],color=['b'],width=0.001)
    i += 1

plt.show()