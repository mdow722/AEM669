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

# # Halo 1
# X0_nd1 = [0.82575887090385,0,0.08]
# V0_nd1 = [0,0.19369724986446,0]
# XV0_nd1 = [*X0_nd1,*V0_nd1]

# orbit_period_nd1 = 2.77648121127569
# print("halo 1 period",orbit_period_nd1*Tstar/86400)
# revs = 1

# trajectory1,_,_ = rungekutta4(CR3BP_nondim, XV0_nd1, np.linspace(0,orbit_period_nd1*revs,500*revs), args=(mu,))
# final_state_nd1 = trajectory1[-1]
# print("final_state_nd1",final_state_nd1)
# print("final_state1",[*final_state_nd1[:3]*Lstar,*final_state_nd1[3:]*Vstar])

ax = plt.figure().add_subplot(projection='3d')
ax.plot(1 - mu, 0, 0, 'go', label="$m_2$")
xLib,_,_ = Get3BodyColinearLibrationPoints(mu,Lstar)
RL1 = np.array([xLib[0],0])
ax.plot(RL1[0], RL1[1], 'co', label="L1")
# ax.plot(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2],label="Halo 1")

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
# given_slopes = [(x[1]-x[0])/(XV0_nd2[0]-XV0_nd1[0]) for x in zip(XV0_nd1,XV0_nd2)]
# XV0_nd3 = [np.mean(x) for x in zip(XV0_nd1,XV0_nd2)]
# print("XV0_nd3",XV0_nd3)
# orbit_period_guess = np.mean([orbit_period_nd1,orbit_period_nd2])
# print("orbit_period_guess",orbit_period_guess)
# revs = 1


def F1(y,y_target):
    return y_target - y

def F2(vx,vx_target):
    return vx_target - vx

def F3(vz,vz_target):
    return vz_target - vz

def F(V0,R0,Xtarget,tof,mu):
    # print("V0",V0)
    # print("R0",R0)
    # print("Xtarget",Xtarget)
    # print("tof",tof)
    # print("mu",mu)
    sol,_,_ = rungekutta4(CR3BP_nondim, [*R0,*V0], np.linspace(0,tof,100), args=(mu,))
    RF = sol[-1,[1,3,5]]

    # print("RF",RF)
    return np.array([F1(RF[0],Xtarget[0]),F2(RF[1],Xtarget[1]),F3(RF[2],Xtarget[2])]), sol

def DF(V0,R0,tof,mu):
    states, STMs = STM_coupled_integration_vec([*R0,*V0],np.linspace(0,tof,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    STM = STMs[-1]
    xdotf = CR3BP_nondim(0,states[-1],mu).reshape(6,1)
    J = np.hstack((STM,xdotf))
    Jrelevant = np.vstack((J[1,[0,4,6]],J[3,[0,4,6]],J[5,[0,4,6]]))
    return -np.identity(3)@Jrelevant

XV0_nd3 = [*XV0_nd2]
orbit_period_guess = orbit_period_nd2

max_cont_iters = 100
cont_iter = 1

while cont_iter <= max_cont_iters and XV0_nd3[2] > 1e-4:
    XV0_nd3[2] -= 0.01

    crossing_x_axis_event = [False, True, False, False, False, False]
    trajectory3, tvec, events = rungekutta4(CR3BP_nondim, XV0_nd3, np.linspace(0,orbit_period_guess*revs,500*revs), args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)
    final_state_nd3 = trajectory3[-1]
    cross_time_ig = tvec[-1]
    # print("final_state_nd3",final_state_nd3)
    # print("final_state3",[*final_state_nd3[:3]*Lstar,*final_state_nd3[3:]*Vstar])

    # ax.plot(trajectory3[:, 0], trajectory3[:, 1], trajectory3[:, 2],label="Halo 3 Initial Guess")
    print("*********************************************************************")

    tol = 1e-9

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

    # print("Target Achieved: ",target_achieved)
    # print("iteration amount",i)
    # # print("Xvec",Xvec)
    # print("V0d_dim",np.array(V0d)*Vstar)
    # print("Delta V vec:",(V0d-XV0_nd3[3:])*Vstar)
    # print("Delta V mag:",norm((V0d-XV0_nd3[3:])*Vstar))
    # print("Achieved Time (d):",TOF*Tstar/86400)

    final_traj = trajvec[-1]
    # ax.plot(final_traj[:, 0], final_traj[:, 1], final_traj[:, 2],label="Halo 3 Final")

    achieved_initial_state = final_traj[0]
    print("achieved_initial_state",achieved_initial_state)
    revs = 1
    achieved_halo_traj,_,_ = rungekutta4(CR3BP_nondim, achieved_initial_state, np.linspace(0,2*TOF*revs,500*revs), args=(mu,))
    _, STMs = STM_coupled_integration_vec(achieved_initial_state,np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    halo_eigvals,_ = np.linalg.eig(STMs[-1])
    print("halo_eigvals",halo_eigvals)
    ax.plot(achieved_halo_traj[:, 0], achieved_halo_traj[:, 1], achieved_halo_traj[:, 2],label=f"Cont Halo {cont_iter}")
    cont_iter += 1

achieved_halo_halfper = TOF
achieved_halo_traj_half,_,_ = rungekutta4(CR3BP_nondim, achieved_initial_state, np.linspace(0,TOF*revs,500*revs), args=(mu,))
halo_half_state = achieved_halo_traj_half[-1]
ax.plot(achieved_halo_traj[:, 0], achieved_halo_traj[:, 1], achieved_halo_traj[:, 2], '--k',label=f"Halo Intersection with XY Plane")
ax.set(title=f"Halo Family Continued To XY Plane",
       xlabel="X", ylabel="Y", zlabel="Z")

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
# print("TOF0",TOF0)

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

# print("Target Achieved: ",target_achieved)
# print("iteration amount",i)
# # print("Xvec",Xvec)
# print("V0d_dim",np.array(V0d)*Vstar)
# print("Delta V vec:",(V0d-V0)*Vstar)
# print("Delta V mag:",norm((V0d-V0)*Vstar))
# print("Achieved Time (d):",TOF*Tstar/86400)
BaseTraj, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,V0[0],Xi[0],0], np.linspace(0,2*TOF,100), args=(mu,))
ax.plot(BaseTraj[:, 0], BaseTraj[:, 1], BaseTraj[:, 2], ':r',label=f"Original Lyapunov")

initial_state_original_lyapunov = BaseTraj[0]
print("initial_state_original_lyapunov",initial_state_original_lyapunov)
print("halo_half_state",halo_half_state)

continuation_iter = 0
max_continuation_iters = 100
XV0 = [*initial_state_original_lyapunov]
x_delta = halo_half_state[0] - initial_state_original_lyapunov[0]
dx_iter = x_delta / 20

while continuation_iter < max_continuation_iters and x_delta > 1e-5:
    print("*********************************************************************")
    XV0[0] += dx_iter
    # print("continuation iteration:",continuation_iter)
    # print("dx0_offset:",dx0_offset)
    # Set dx0 based on dir_factor and previous dx0
    # dx0 = dx0m1 + (dir_factor * dx0_offset)
    # # print("dx0",dx0)
    # # Run targeter using solved velocity value from previous case
    # im0 = 0
    # dR0 = np.array([dx0,dy0])
    # RL1 = np.array([xLib[0],0])
    # R0 = RL1 + dR0

    # vx0 = 0
    # vy0 = Xi[0]
    # V0 = np.array([vx0,vy0])

    # Get time to cross x axis:
    t_0 = 0
    t_f = 10.5
    num_points = 100
    t_points = np.linspace(t_0, t_f, int(t_f*num_points))
    crossing_x_axis_event = [False, True, False, False, False, False]
    yvec, tvec, events = rungekutta4(CR3BP_nondim, [*XV0], t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)


    tol = 1/Lstar

    TOF = tvec[-1]
    # TOF = Xi[1]
    Xi = [XV0[4],TOF]
    Fmag = 1
    Xvec = []
    Fvec = []
    trajvec = []

    Xtarget = [0,0] # y and vx

    max_targeter_iters = 500
    i=0
    target_achieved = False

    while Fmag > tol and i < max_targeter_iters:
        Fi,traj = F([XV0[3],Xi[0]],XV0[:2],Xtarget,TOF,mu)
        Fmag = norm(Fi)
        Fvec.append(Fmag)
        trajvec.append(traj)
        DFi = np.array(DF([XV0[3],Xi[0]],XV0[:2],TOF,mu))
        Xi = Xi - inv(DFi)@Fi
        Xvec.append(Xi)
        TOF = Xi[1]
        i += 1
        if Fmag <= tol and TOF > 0.1:
            target_achieved = True

    # print("target achieved:",target_achieved)

    # if successful:
        # save results
        # set 
    # if target_achieved == True:
        # overall_trajvec.append(trajvec[-1])
        # overall_X0vec.append(Xvec[-1])
        # overall_jcvec.append(GetJacobiConstant(trajvec[-1][0,:3],trajvec[-1][0,3:],mu))
        # overall_pervec.append(TOF*2*Tstar/86400)
        # overall_dxvec.append(dx0)

    vy0 = Xi[0]
    XV0[4] = vy0
    x_delta = halo_half_state[0] - XV0[0]
    _, STMs = STM_coupled_integration_vec(trajvec[-1][0],np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    lyapunov_eigvals = np.linalg.eigvals(STMs[-1])
    print("achieved_initial_state",trajvec[-1][0])
    print("lyapunov_eigvals",lyapunov_eigvals)
        # overall_eigsvec.append(eigvals)
        # neweigtypes = getEigTypes(eigvals)
        # if neweigtypes != eigtypes:
        #     overall_trajvec_eigchange.append(trajvec[-1])
        #     print("dx, new eigval structure: ", dx0, neweigtypes,"\n")

        # eigtypes = neweigtypes
    # dx0m1 = dx0
    # else:
        # if not successful
        # dx0_offset *= 0.5


    # printProgressBar(continuation_iter, max_continuation_iters, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    continuation_iter += 1

ax.plot(trajvec[-1][:, 0], trajvec[-1][:, 1], trajvec[-1][:, 2],label="Final Lyapunov @ Halo Bifurcation Point")



ax.legend()
ax.set_aspect('equal', 'box')
plt.show()