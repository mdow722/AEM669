import numpy as np
from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
BaseTraj, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,V0[0],Xi[0],0], np.linspace(0,TOF,100), args=(mu,))
print("baseline opp state:", BaseTraj[-1])
BaseOppTraj, tvec1, events1 = rungekutta4(CR3BP_nondim, BaseTraj[-1], np.linspace(0,TOF,100), args=(mu,))
# ax = Plot_CR3BP([BaseTraj,BaseOppTraj],mu,TOF0*Tstar/86400,show_initial_pos=True,show_final_pos=True,
#            show_bodies=True,show_ZVC=True,xlimits=(0.7,1),ylimits=(-0.2,.2))

error = BaseTraj[-1]-BaseTraj[0]
print("error",error)

_, STMs = STM_coupled_integration_vec([*R0,0,V0[0],Xi[0],0],np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
final_STM = STMs[-1]
print("final_STM",final_STM)
print("final_STM[0,1,3,4][0,1,3,4]",final_STM[[0,1,3,4],:][:,[0,1,3,4]])
eigs = np.linalg.eigvals(final_STM[[0,1,3,4],:][:,[0,1,3,4]])
print("eigs",eigs)
max_continuation_iters = 200
dx0_offset_tol = 1/Lstar
overall_X0vec = []
overall_trajvec = []
overall_pervec = []
overall_jcvec = []
overall_dxvec = []
overall_eigsvec = []
# original_dx0 = BaseTraj[-1,0]-xLib[0]
original_dx0 = dx0
print("original_dx0",original_dx0)
# original_vy0 = BaseTraj[-1,4]
original_vy0 = vy0
print("original_vy0",original_vy0)

for dir_factor in [+1]:
    # while loop to watch offset tolerance and loop counter
    continuation_iter = 0
    dx0m1 = original_dx0
    vy0 = original_vy0
    dx0_offset = 0.001
    while continuation_iter < max_continuation_iters and dx0_offset >= dx0_offset_tol:
        print("continuation iteration:",continuation_iter)
        print("dx0_offset:",dx0_offset)
        # Set dx0 based on dir_factor and previous dx0
        dx0 = dx0m1 + (dir_factor * dx0_offset)
        print("dx0",dx0)
        # Run targeter using solved velocity value from previous case
        im0 = 0
        dR0 = np.array([dx0,dy0])
        RL1 = np.array([xLib[0],0])
        R0 = RL1 + dR0

        vx0 = 0
        # vy0 = Xi[0]
        V0 = np.array([vx0,vy0])

        # Get time to cross x axis:
        t_0 = 0
        t_f = 10.5
        num_points = 100
        t_points = np.linspace(t_0, t_f, int(t_f*num_points))
        crossing_x_axis_event = [False, True, False, False, False, False]
        yvec, tvec, events = rungekutta4(CR3BP_nondim, [*R0,0,*V0,0], t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)


        tol = 1/Lstar

        TOF = tvec[-1]
        # TOF = Xi[1]
        Xi = [V0[1],TOF]
        Fmag = 1
        Xvec = []
        Fvec = []
        trajvec = []

        Xtarget = [0,0] # y and vx

        max_targeter_iters = 500
        i=0
        target_achieved = False

        while Fmag > tol and i < max_targeter_iters:
            Fi,traj = F([V0[0],Xi[0]],R0,Xtarget,TOF,mu)
            Fmag = norm(Fi)
            Fvec.append(Fmag)
            trajvec.append(traj)
            DFi = np.array(DF([V0[0],Xi[0]],R0,TOF,mu))
            Xi = Xi - inv(DFi)@Fi
            Xvec.append(Xi)
            TOF = Xi[1]
            i += 1
            if Fmag <= tol and TOF > 0.1:
                target_achieved = True

        print("target achieved:",target_achieved)

        # if successful:
            # save results
            # set 
        if target_achieved == True and continuation_iter % 1 == 0:
            overall_trajvec.append(trajvec[-1])
            overall_X0vec.append(Xvec[-1])
            overall_jcvec.append(GetJacobiConstant(trajvec[-1][0,:3],trajvec[-1][0,3:],mu))
            overall_pervec.append(TOF*2*Tstar/86400)
            overall_dxvec.append(dx0)

            _, STMs = STM_coupled_integration_vec(trajvec[-1][0],np.linspace(0,2*TOF,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
            overall_eigsvec.append(np.linalg.eigvals(STMs[-1][[0,1,3,4],:][:,[0,1,3,4]]))

            dx0m1 = dx0
            vy0 = Xi[0]
        elif target_achieved == True:
            dx0m1 = dx0
            vy0 = Xi[0]
        else:
            dx0_offset *= 0.5
        # if not successful
        
        continuation_iter += 1


ax = Plot_CR3BP(overall_trajvec,mu,TOF0*Tstar/86400,show_initial_pos=False,show_final_pos=False,
                show_bodies=True,show_ZVC=False,xlimits=(0.6,1.1),ylimits=(-0.6,.2))
ax.plot(BaseTraj[:,0], BaseTraj[:,1], 'k', label="Baseline")
ax.plot(RL1[0], RL1[1], 'co', label="L1")

ax = Plot_CR3BP(overall_trajvec,mu,TOF0*Tstar/86400,show_initial_pos=False,show_final_pos=False,
                show_bodies=True,show_ZVC=False)
ax.plot(BaseTraj[:,0], BaseTraj[:,1], 'k', label="Baseline")
ax.plot(RL1[0], RL1[1], 'co', label="L1")

fig,ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Initial dx from L1')
ax1.set_ylabel('Jacobi Constant', color=color)
ax1.plot(overall_dxvec, overall_jcvec, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Period (d)', color=color)  # we already handled the x-label with ax1
ax2.plot(overall_dxvec, overall_pervec, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fix,ax3 = plt.subplots()
fix,ax4 = plt.subplots()

# mag0 = [abs(ele[0]) for ele in overall_eigsvec]
# mag1 = [abs(ele[1]) for ele in overall_eigsvec]
# mag2 = [abs(ele[2]) for ele in overall_eigsvec]
# mag3 = [abs(ele[3]) for ele in overall_eigsvec]

# extract real part 
real0 = [ele[0].real/abs(ele[0]) for ele in overall_eigsvec] 
real1 = [ele[1].real/abs(ele[1]) for ele in overall_eigsvec] 
real2 = [ele[2].real/abs(ele[2]) for ele in overall_eigsvec] 
real3 = [ele[3].real/abs(ele[3]) for ele in overall_eigsvec] 
# extract imaginary part 
im0 = [ele[0].imag/abs(ele[0]) for ele in overall_eigsvec] 
im1 = [ele[1].imag/abs(ele[1]) for ele in overall_eigsvec] 
im2 = [ele[2].imag/abs(ele[2]) for ele in overall_eigsvec] 
im3 = [ele[3].imag/abs(ele[3]) for ele in overall_eigsvec]


rvec = np.ones(len(overall_eigsvec))
th0 = [np.arctan2(ele[0].imag,ele[0].real) for ele in overall_eigsvec] 
th1 = [np.arctan2(ele[1].imag,ele[1].real) for ele in overall_eigsvec] 
th2 = [np.arctan2(ele[2].imag,ele[2].real) for ele in overall_eigsvec] 
th3 = [np.arctan2(ele[3].imag,ele[3].real) for ele in overall_eigsvec]

r0 = [abs(ele[0]) for ele in overall_eigsvec]
r1 = [abs(ele[1]) for ele in overall_eigsvec]
r2 = [abs(ele[2]) for ele in overall_eigsvec]
r3 = [abs(ele[3]) for ele in overall_eigsvec]



# plot the im comps 
ax3.plot(overall_dxvec, im0, label="eig 1") 
ax3.plot(overall_dxvec, im1, label="eig 2") 
ax3.plot(overall_dxvec, im2, label="eig 3") 
ax3.plot(overall_dxvec, im3, label="eig 4") 
ax3.set_ylabel('Imaginary') 
ax3.set_xlabel('dx')
ax3.set_title('Imaginary Components')

# plot the real comps 
ax4.plot(overall_dxvec, real0, label="eig 1") 
ax4.plot(overall_dxvec, real1, label="eig 2") 
ax4.plot(overall_dxvec, real2, label="eig 3") 
ax4.plot(overall_dxvec, real3, label="eig 4") 
ax4.set_ylabel('Real') 
ax4.set_xlabel('dx')
ax4.set_title('Real Components')

pfig = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}]*2]*2)
pfig.add_trace(go.Scatterpolar(
        r = r0,
        theta = th0,
        thetaunit = "radians",
        mode = 'lines+markers'
    ), 1, 1)
pfig.add_trace(go.Scatterpolar(
        r = r1,
        theta = th1,
        thetaunit = "radians",
        mode = 'lines+markers'
    ), 1, 2)
pfig.add_trace(go.Scatterpolar(
        r = r2,
        theta = th2,
        thetaunit = "radians",
        mode = 'lines+markers'
    ), 2, 1)
pfig.add_trace(go.Scatterpolar(
        r = r2,
        theta = th3,
        thetaunit = "radians",
        mode = 'lines+markers'
    ), 2, 2)

pfig.show()

fix,ax5 = plt.subplots()
ax5.plot(overall_dxvec, th0, label="eig 1") 
ax5.plot(overall_dxvec, th1, label="eig 2") 
ax5.plot(overall_dxvec, th2, label="eig 3") 
ax5.plot(overall_dxvec, th3, label="eig 4") 
ax5.set_ylabel('Imaginary Plane Angle (rad)') 
ax5.set_xlabel('dx')
ax5.set_title('Imaginary Plane Angle')

ax.legend()
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
plt.show()