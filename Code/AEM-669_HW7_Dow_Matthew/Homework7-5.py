from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *
from Utils import printProgressBar

mu = 0.5
JC = 4.5

# y0 = 0

# xrange = np.linspace(-0.635,-0.3085,10)
# # vxrange = np.linspace(-5,5,20)
# vxrange = [0]

# fig,ax = plt.subplots(figsize=(5,5), dpi=96)

# # xvec = []
# # vxvec = []
# max_iters = len(xrange)*len(vxrange)
# current_iter = 0

# printProgressBar(current_iter, max_iters, prefix = 'Progress:', suffix = 'Complete', length = 50)
# for x0 in xrange:
#     for vx0 in vxrange:
#         xvec = []
#         vxvec = []
#         current_iter += 1
#         printProgressBar(current_iter, max_iters, prefix = 'Progress:', suffix = 'Complete', length = 50)
#         try:
#             vy0 = GetVelFromJacobiAndPos([x0,0,0],JC,mu)
#         except:
#             print("skip")
#             continue
#         XV0 = [x0,y0,0,vx0,vy0,0]
#         if current_iter <= 2:
#             print("orbit ICs: ",XV0)
#         xvec.append(x0)
#         vxvec.append(vx0)

#         t_0 = 0
#         t_f = 300
#         num_points = 500
#         t_points = np.linspace(t_0, t_f, int(t_f*num_points))
#         crossing_x_axis_event = [False, True, False, False, False, False]
#         yvec, tvec, events = rungekutta4(CR3BP_nondim, XV0, t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=False)
#         event_index = 0
#         for event in events:
#             if event[4] > 0:
#                 xvec.append(event[0])
#                 vxvec.append(event[3])
#                 if current_iter == 2 and event_index < 15:
#                     ax.annotate(event_index+1,(event[0]+0.005,event[3]+0.005))
#                     event_index += 1
#         ax.scatter(xvec,vxvec,s=6,c='b')
#         ax.plot(xvec,vxvec,'--b')
#         ax.scatter([x0],[vx0],s=8,c='r')
#         printProgressBar(current_iter, max_iters, prefix = 'Progress:', suffix = 'Complete', length = 50)

# # ax.scatter(xvec,vxvec,s=2,c='b')
# Xgrid,VXgrid,Zgrid = GetZVCGrid_xvx(mu,JC,resolution=0.001,xlimits=[-1,0],vxlimits=[-5,5])
# ax.contour(Xgrid,VXgrid,Zgrid,levels=[0])


# ax.set_xlim([-1,0])
# ax.set_ylim([-5, 5])
# ax.set(title=f"Poincare Map for mu={mu}, JC={JC}",
#        xlabel="x", ylabel="vx")

# **********************************************************************************

# qp_ics = np.array([-0.5987222222222223, 0, 0, 0, 2.626413538956279, 0])
# R0 = qp_ics[0:2]
# V0 = qp_ics[3:5]
# # Get time to cross x axis:
# t_0 = 0
# t_f = 500
# num_points = 1000
# t_points = np.linspace(t_0, t_f, int(t_f*num_points))
# crossing_x_axis_event = [False, True, False, False, False, False]
# yvec, tvec, events = rungekutta4(CR3BP_nondim, qp_ics, t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)
# TOF0 = tvec[-1]

# ig_traj, _, _ = rungekutta4(CR3BP_nondim, qp_ics, np.linspace(0, TOF0*30, 1000), args=(mu,))
# fig2,ax2 = plt.subplots(figsize=(5,5), dpi=96)
# ax2.plot(ig_traj[:, 0], ig_traj[:, 1], '-b')
# ax2.set_aspect('equal', 'box')
# ax2.set(title=f"Quasi-Periodic Orbit",
#        xlabel="X", ylabel="Y")



# **********************************************************************************
per_orb_ics = np.array([-0.635, 0, 0, 0, 2.047361637847567, 0])
R0 = per_orb_ics[0:2]
V0 = per_orb_ics[3:5]
# Get time to cross x axis:
t_0 = 0
t_f = 500
num_points = 1000
t_points = np.linspace(t_0, t_f, int(t_f*num_points))
crossing_x_axis_event = [False, True, False, False, False, False]
yvec, tvec, events = rungekutta4(CR3BP_nondim, per_orb_ics, t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)
TOF0 = tvec[-1]


tol = 1e-9

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

trajlist = []
fig1,ax1 = plt.subplots(figsize=(5,5), dpi=96)

while Fmag > tol and i < max_iters:
    Fi,traj = F([V0[0],Xi[0]],R0,Xtarget,TOF,mu)
    # ax1.plot(traj[:, 0], traj[:, 1], ':r',label=f"Original Lyapunov")
    # plt.show()
    trajlist.append(traj)
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

ax1.plot(BaseTraj[:, 0], BaseTraj[:, 1], '-r',label=f"Original Lyapunov")
ax1.set_aspect('equal', 'box')
ax1.set(title="Periodic Orbit",
       xlabel="X", ylabel="Y")


plt.show()