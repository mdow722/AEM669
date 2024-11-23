import numpy as np
from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *

# Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
Lstar,Mstar,Tstar,mu = Manual3BodyCharacteristics(398600.4480734463,4902.799140594719,384431.4584485)
Vstar = Lstar/Tstar
print("mu",mu)
print("Lstar",Lstar)
print("Tstar",Tstar)
print("Vstar",Vstar)

# Halo 1
X0_nd1 = [0.82575887090385,0,0.08]
V0_nd1 = [0,0.19369724986446,0]
XV0_nd1 = [*X0_nd1,*V0_nd1]

orbit_period_nd1 = 2.77648121127569
print("halo 1 period",orbit_period_nd1*Tstar/86400)
revs = 1

trajectory1,_,_ = rungekutta4(CR3BP_nondim, XV0_nd1, np.linspace(0,orbit_period_nd1*revs,500*revs), args=(mu,))
final_state_nd1 = trajectory1[-1]
print("final_state_nd1",final_state_nd1)
print("final_state1",[*final_state_nd1[:3]*Lstar,*final_state_nd1[3:]*Vstar])

ax = plt.figure().add_subplot(projection='3d')
ax.plot(1 - mu, 0, 0, 'go', label="$m_2$")
ax.plot(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2],label="Halo 1")

# Halo 2
X0_nd2 = [0.82356490862838,0,0.04]
V0_nd2 = [0,0.14924319723734,0]
XV0_nd2 = [*X0_nd2,*V0_nd2]

orbit_period_nd2 = 2.75330620148158
print("halo 2 period",orbit_period_nd2*Tstar/86400)
revs = 1

trajectory2,_,_ = rungekutta4(CR3BP_nondim, XV0_nd2, np.linspace(0,orbit_period_nd2*revs,500*revs), args=(mu,))
final_state_nd2 = trajectory2[-1]
print("final_state_nd2",final_state_nd2)
print("final_state",[*final_state_nd2[:3]*Lstar,*final_state_nd2[3:]*Vstar])

ax.plot(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2],label="Halo 2")

# Continuation Halo
# Initial Guess Creation
XV0_nd3 = [np.mean(x) for x in zip(XV0_nd1,XV0_nd2)]
print("XV0_nd3",XV0_nd3)
orbit_period_guess = np.mean([orbit_period_nd1,orbit_period_nd2])
print("orbit_period_guess",orbit_period_guess)
revs = 1

crossing_x_axis_event = [False, True, False, False, False, False]
trajectory3, tvec, events = rungekutta4(CR3BP_nondim, XV0_nd3, np.linspace(0,orbit_period_guess*revs,500*revs), args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)
final_state_nd3 = trajectory3[-1]
cross_time_ig = tvec[-1]
print("final_state_nd3",final_state_nd3)
print("final_state3",[*final_state_nd3[:3]*Lstar,*final_state_nd3[3:]*Vstar])

ax.plot(trajectory3[:, 0], trajectory3[:, 1], trajectory3[:, 2],label="Halo 3 Initial Guess")
print("*********************************************************************")

def F1(y,y_target):
    return y_target - y

def F2(vx,vx_target):
    return vx_target - vx

def F3(vz,vz_target):
    return vz_target - vz

def F(V0,R0,Xtarget,tof,mu):
    print("V0",V0)
    print("R0",R0)
    print("Xtarget",Xtarget)
    print("tof",tof)
    print("mu",mu)
    sol,_,_ = rungekutta4(CR3BP_nondim, [*R0,*V0], np.linspace(0,tof,100), args=(mu,))
    RF = sol[-1,[1,3,5]]

    print("RF",RF)
    return np.array([F1(RF[0],Xtarget[0]),F2(RF[1],Xtarget[1]),F3(RF[2],Xtarget[2])]), sol

def DF(V0,R0,tof,mu):
    states, STMs = STM_coupled_integration_vec([*R0,*V0],np.linspace(0,tof,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    STM = STMs[-1]
    xdotf = CR3BP_nondim(0,states[-1],mu).reshape(6,1)
    J = np.hstack((STM,xdotf))
    Jrelevant = np.vstack((J[1,[0,4,6]],J[3,[0,4,6]],J[5,[0,4,6]]))
    return -np.identity(3)@Jrelevant

tol = 1e-12

TOF = tvec[-1]
Xi = [XV0_nd3[0],XV0_nd3[4],TOF]
Fmag = 1
Xvec = []
V0d = np.array([*XV0_nd3[3:]])
Fvec = []
trajvec = []

Xtarget = [0,0,0] # y and vx and vz

max_iters = 100
i=0
target_achieved = False

while Fmag > tol and i < max_iters:
    Fi,traj = F([XV0_nd3[3],Xi[1],XV0_nd3[5]],[Xi[0],XV0_nd3[1],XV0_nd3[2]],Xtarget,TOF,mu)
    Fmag = norm(Fi)
    Fvec.append(Fmag)
    trajvec.append(traj)
    DFi = np.array(DF([XV0_nd3[3],Xi[1],XV0_nd3[5]],[Xi[0],XV0_nd3[1],XV0_nd3[2]],TOF,mu))
    Xi = Xi - inv(DFi)@Fi
    Xvec.append(Xi)
    V0d[1] = Xi[1]
    TOF = Xi[2]
    i += 1
    if Fmag < tol:
        target_achieved = True

print("Target Achieved: ",target_achieved)
print("iteration amount",i)
# print("Xvec",Xvec)
print("V0d_dim",np.array(V0d)*Vstar)
print("Delta V vec:",(V0d-XV0_nd3[3:])*Vstar)
print("Delta V mag:",norm((V0d-XV0_nd3[3:])*Vstar))
print("Achieved Time (d):",TOF*Tstar/86400)

final_traj = trajvec[-1]
# ax.plot(final_traj[:, 0], final_traj[:, 1], final_traj[:, 2],label="Halo 3 Final")

achieved_initial_state = final_traj[0]
print("achieved_initial_state",achieved_initial_state)
revs = 1
achieved_halo_traj,_,_ = rungekutta4(CR3BP_nondim, achieved_initial_state, np.linspace(0,2*TOF*revs,500*revs), args=(mu,))
ax.plot(achieved_halo_traj[:, 0], achieved_halo_traj[:, 1], achieved_halo_traj[:, 2],label="Halo 3 Achieved")





ax.set_aspect("equal")
ax.legend()
ax.set(title="Earth Moon Halo Orbits",
       xlabel="X",ylabel="Y",zlabel="Z")
plt.show()