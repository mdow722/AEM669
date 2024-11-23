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

def F(V0,R0,R_target,tof,mu):
    sol,_,_ = rungekutta4(CR3BP_nondim, [*R0,0,*V0,0], np.linspace(0,tof,100), args=(mu,))
    RF = sol[-1,:2]
    return np.array([F1(RF[0],R_target[0]),F2(RF[1],R_target[1])]), sol

def DF(V0,R0,tof,mu):
    _, STMs = STM_coupled_integration_vec([*R0,0,*V0,0],np.linspace(0,tof,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    STM = STMs[-1]
    return -np.identity(2)@STM[:2,3:5]

tol = 1e-12

R0 = [0.488,0.200]
V0 = [-0.88,0.200]
Xi = [*V0]
Fmag = 1
Xvec = []
V0d = [*V0]
Fvec = []
trajvec = []

# Get time to cross x axis:
t_0 = 0
t_f = 10.5
num_points = 100
t_points = np.linspace(t_0, t_f, int(t_f*num_points))
crossing_x_axis_event = [False, True, False, False, False, False]
_, tvec, _ = rungekutta4(CR3BP_nondim, [*R0,0,*V0,0], t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)
TOF = tvec[-1]

# RT = [-0.3,0.05]
# RT = [-0.1,0.0]
RT = [-0.4,-0.1]

max_iters = 100
i=0
target_achieved = False

while Fmag > tol and i < max_iters:
    Fi,traj = F(Xi,R0,RT,TOF,mu)
    Fmag = norm(Fi)
    Fvec.append(Fmag)
    trajvec.append(traj)
    DFi = np.array(DF(Xi,R0,TOF,mu))
    Xi = Xi - inv(DFi)@Fi
    Xvec.append(Xi)
    V0d = Xi
    i += 1
    if Fmag < tol:
        target_achieved = True

print("iteration amount",i)
print("Xvec",Xvec)
print("V0d_dim",V0d*Vstar)
print("Delta V vec:",(V0d-V0)*Vstar)
print("Delta V mag:",norm((V0d-V0)*Vstar))
print("Fvec",Fmag)

ax = Plot_CR3BP(trajvec,mu,tvec[-1]*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6),names=range(i))
ax.plot(*RT, 'go', label="Target Position")
ax.plot(trajvec[0][-1][0], trajvec[0][-1][1], 'ro', label="Initial Guess' Final Position")
if target_achieved == False:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'bo', label="Last Iteration's Final Position")





plt.legend()
plt.show()