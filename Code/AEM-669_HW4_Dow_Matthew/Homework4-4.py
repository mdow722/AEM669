import numpy as np
from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *

Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
Vstar = Lstar/Tstar

def F1(y,y_target):
    return y_target - y

def F2(vx,vx_target):
    return vx_target - vx

def F(V0,R0,Xtarget,tof,mu):
    sol,_,_ = rungekutta4(CR3BP_nondim, [*R0,0,*V0,0], np.linspace(0,tof,100), args=(mu,))
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

ax = Plot_CR3BP([yvec],mu,TOF0*Tstar/86400,show_initial_pos=True,show_final_pos=True,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))


tol = 1e-12

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
    if Fmag < tol:
        target_achieved = True

print("Target Achieved: ",target_achieved)
print("iteration amount",i)
# print("Xvec",Xvec)
print("V0d_dim",np.array(V0d)*Vstar)
print("Delta V vec:",(V0d-V0)*Vstar)
print("Delta V mag:",norm((V0d-V0)*Vstar))
print("Achieved Time (d):",TOF*Tstar/86400)

ax = Plot_CR3BP(trajvec,mu,TOF*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2),names=range(i))
# ax.plot(*RT, 'go', label="Target Position")
ax.plot(trajvec[0][-1][0], trajvec[0][-1][1], 'ro', label="Initial Guess' Final Position")
if target_achieved == False:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'bo', label="Last Iteration's Final Position")
else:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'go', label="Last Iteration's Final Position")

BaseTraj, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,V0[0],Xi[0],0], np.linspace(0,2*TOF,100), args=(mu,))
ax = Plot_CR3BP([BaseTraj],mu,tvec[-1]*Tstar/86400,show_initial_pos=True,show_final_pos=True,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))

error = BaseTraj[-1]-BaseTraj[0]
print("error",error)

_, STMs = STM_coupled_integration_vec([*R0,0,V0[0],Xi[0],0],np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
final_STM = STMs[-1]
print("final_STM",final_STM)
print("final_STM[0,1,3,4][0,1,3,4]",final_STM[[0,1,3,4],:][:,[0,1,3,4]])
eigs = np.linalg.eigvals(final_STM[[0,1,3,4],:][:,[0,1,3,4]])
print("eigs",eigs)

# Continuation 1:
dx0 = 0.009
dy0 = 0
dR0 = np.array([dx0,dy0])
RL1 = np.array([xLib[0],0])
R0 = RL1 + dR0

vx0 = 0
vy0 = Xi[0]
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

ax = Plot_CR3BP([yvec],mu,TOF0*Tstar/86400,show_initial_pos=True,show_final_pos=True,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))


tol = 1e-12

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
    if Fmag < tol:
        target_achieved = True

print("Target Achieved: ",target_achieved)
print("iteration amount",i)
# print("Xvec",Xvec)
print("V0d_dim",np.array(V0d)*Vstar)
print("Delta V vec:",(V0d-V0)*Vstar)
print("Delta V mag:",norm((V0d-V0)*Vstar))
print("Achieved Time (d):",TOF*Tstar/86400)

ax = Plot_CR3BP(trajvec,mu,TOF*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2),names=range(i))
# ax.plot(*RT, 'go', label="Target Position")
ax.plot(trajvec[0][-1][0], trajvec[0][-1][1], 'ro', label="Initial Guess' Final Position")
if target_achieved == False:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'bo', label="Last Iteration's Final Position")
else:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'go', label="Last Iteration's Final Position")

ContTraj1, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,V0[0],Xi[0],0], np.linspace(0,2*TOF,100), args=(mu,))
ax = Plot_CR3BP([ContTraj1],mu,tvec[-1]*Tstar/86400,show_initial_pos=True,show_final_pos=True,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))

error = ContTraj1[-1]-ContTraj1[0]
print("error",error)

_, STMs = STM_coupled_integration_vec([*R0,0,V0[0],Xi[0],0],np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
final_STM = STMs[-1]
print("final_STM",final_STM)
print("final_STM[0,1,3,4][0,1,3,4]",final_STM[[0,1,3,4],:][:,[0,1,3,4]])
eigs = np.linalg.eigvals(final_STM[[0,1,3,4],:][:,[0,1,3,4]])
print("eigs",eigs)

# Continuation 2:
dx0 = 0.008
dy0 = 0
dR0 = np.array([dx0,dy0])
RL1 = np.array([xLib[0],0])
R0 = RL1 + dR0

vx0 = 0
vy0 = Xi[0]
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

ax = Plot_CR3BP([yvec],mu,TOF0*Tstar/86400,show_initial_pos=True,show_final_pos=True,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))


tol = 1e-12

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
    if Fmag < tol:
        target_achieved = True

print("Target Achieved: ",target_achieved)
print("iteration amount",i)
# print("Xvec",Xvec)
print("V0d_dim",np.array(V0d)*Vstar)
print("Delta V vec:",(V0d-V0)*Vstar)
print("Delta V mag:",norm((V0d-V0)*Vstar))
print("Achieved Time (d):",TOF*Tstar/86400)

ax = Plot_CR3BP(trajvec,mu,TOF*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2),names=range(i))
# ax.plot(*RT, 'go', label="Target Position")
ax.plot(trajvec[0][-1][0], trajvec[0][-1][1], 'ro', label="Initial Guess' Final Position")
if target_achieved == False:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'bo', label="Last Iteration's Final Position")
else:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'go', label="Last Iteration's Final Position")

ContTraj2, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,V0[0],Xi[0],0], np.linspace(0,2*TOF,100), args=(mu,))
ax = Plot_CR3BP([ContTraj2],mu,tvec[-1]*Tstar/86400,show_initial_pos=True,show_final_pos=True,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))

error = ContTraj2[-1]-ContTraj2[0]
print("error",error)

_, STMs = STM_coupled_integration_vec([*R0,0,V0[0],Xi[0],0],np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
final_STM = STMs[-1]
print("final_STM",final_STM)
print("final_STM[0,1,3,4][0,1,3,4]",final_STM[[0,1,3,4],:][:,[0,1,3,4]])
eigs = np.linalg.eigvals(final_STM[[0,1,3,4],:][:,[0,1,3,4]])
print("eigs",eigs)

# Continuation 3:
dx0 = 0.007
dy0 = 0
dR0 = np.array([dx0,dy0])
RL1 = np.array([xLib[0],0])
R0 = RL1 + dR0

vx0 = 0
vy0 = Xi[0]
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

ax = Plot_CR3BP([yvec],mu,TOF0*Tstar/86400,show_initial_pos=True,show_final_pos=True,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))


tol = 1e-12

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
    if Fmag < tol:
        target_achieved = True

print("Target Achieved: ",target_achieved)
print("iteration amount",i)
# print("Xvec",Xvec)
print("V0d_dim",np.array(V0d)*Vstar)
print("Delta V vec:",(V0d-V0)*Vstar)
print("Delta V mag:",norm((V0d-V0)*Vstar))
print("Achieved Time (d):",TOF*Tstar/86400)

ax = Plot_CR3BP(trajvec,mu,TOF*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2),names=range(i))
# ax.plot(*RT, 'go', label="Target Position")
ax.plot(trajvec[0][-1][0], trajvec[0][-1][1], 'ro', label="Initial Guess' Final Position")
if target_achieved == False:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'bo', label="Last Iteration's Final Position")
else:
    ax.plot(trajvec[-1][-1][0], trajvec[-1][-1][1], 'go', label="Last Iteration's Final Position")

ContTraj3, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,V0[0],Xi[0],0], np.linspace(0,2*TOF,100), args=(mu,))
ax = Plot_CR3BP([ContTraj3],mu,tvec[-1]*Tstar/86400,show_initial_pos=True,show_final_pos=True,
           show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))

error = ContTraj3[-1]-ContTraj3[0]
print("error",error)

_, STMs = STM_coupled_integration_vec([*R0,0,V0[0],Xi[0],0],np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
final_STM = STMs[-1]
print("final_STM",final_STM)
print("final_STM[0,1,3,4][0,1,3,4]",final_STM[[0,1,3,4],:][:,[0,1,3,4]])
eigs = np.linalg.eigvals(final_STM[[0,1,3,4],:][:,[0,1,3,4]])
print("eigs",eigs)

ax = Plot_CR3BP([BaseTraj,ContTraj1,ContTraj2,ContTraj3],mu,tvec[-1]*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=False,xlimits=(0.7,1),ylimits=(-0.2,.2),names=["x=0.01","x=0.009","x=0.008","x=0.007"])

plt.legend()
plt.show()