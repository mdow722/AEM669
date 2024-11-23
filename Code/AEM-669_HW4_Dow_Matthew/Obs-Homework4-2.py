from ODESolving import *
from ThreeBodyModel import *
from scipy.integrate import solve_ivp
from PlanetaryDataFuncs import *
from Visualization import *
import matplotlib.pyplot as plt
from numpy.linalg import norm
from Targeters import *

Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
Vstar = Lstar/Tstar

x_0 = 0.488
y_0 = 0.200
z_0 = 0
vx_0 = -0.880
vy_0 = 0.200
vz_0 = 0
X0 = np.array((x_0,y_0,z_0,vx_0,vy_0,vz_0))

# Get time to cross x axis:
t_0 = 0
t_f = 10.5
num_points = 50
t_points = np.linspace(t_0, t_f, int(t_f*num_points))
crossing_x_axis_event = [False, True, False, False, False, False]
_, tvec, _ = rungekutta4(CR3BP_nondim, X0, t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)
final_time = tvec[-1]
print("TOF dim: ",final_time*Tstar)

# # Case a1
# xf_target = -0.3
# yf_target = +0.05
# target_cond = np.array([xf_target,yf_target,None,None,None,None])

# targeter_tolerance = 1e-3
# targeter_max_iters = 500
# fix_time_value = True
# # front_constraints = [(0,0),(0,0),(0,0),None,None,(0,0)]
# front_constraints = [FIXED,FIXED,FIXED,FREE,FREE,FIXED]
# use_coupled_integration = True
# target_achieved, final_state, final_tof, sols, solution_initial_state = TargetFinalStateFixedTime(X0,mu,target_cond,final_time,timesteps=num_points,tol=targeter_tolerance,max_iters=targeter_max_iters,fix_time=fix_time_value,initial_constraints=front_constraints,use_coupled_integration=use_coupled_integration)
# print("target_achieved",target_achieved)
# print("iteration amount",len(sols))
# print("final_state",final_state)
# print("final_tof dim",final_tof*Tstar)
# print("solution_initial_state",solution_initial_state)

# solution_delta = solution_initial_state - X0
# solution_delta_dim = [*solution_delta[:3]*Lstar,*solution_delta[3:]*Vstar]
# print("solution_delta_dim",solution_delta_dim)

# DV = solution_delta_dim[3:]
# print("DV",DV)
# DV_mag = norm(DV)
# print("DV_mag",DV_mag)

# ax = Plot_CR3BP(sols,mu,tvec[-1]*Tstar,show_initial_pos=False,show_final_pos=False,
#            show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6))
# ax.plot(xf_target, yf_target, 'go', label="Target Position")
# ax.plot(sols[0][-1][0], sols[0][-1][1], 'ro', label="Initial Guess' Final Position")
# if target_achieved == False:
#     ax.plot(sols[-1][-1][0], sols[-1][-1][1], 'bo', label="Last Iteration's Final Position")

# plt.legend()
# plt.show()

# Case a2
xf_target = -0.1
yf_target = 0.0
target_cond = np.array([xf_target,yf_target,None,None,None,None])

targeter_tolerance = 1e-3
targeter_max_iters = 3000
fix_time_value = True
front_constraints = [FIXED,FIXED,FIXED,FREE,FREE,FIXED]
use_coupled_integration = True
target_achieved, final_state, final_tof, sols, solution_initial_state = TargetFinalStateFixedTime(X0,mu,target_cond,final_time,timesteps=num_points,tol=targeter_tolerance,max_iters=targeter_max_iters,fix_time=fix_time_value,initial_constraints=front_constraints,use_coupled_integration=use_coupled_integration)
print("target_achieved",target_achieved)
print("iteration amount",len(sols))
print("final_state",final_state)
print("final_tof dim",final_tof*Tstar)
print("solution_initial_state",solution_initial_state)

solution_delta = solution_initial_state - X0
solution_delta_dim = [*solution_delta[:3]*Lstar,*solution_delta[3:]*Vstar]
print("solution_delta_dim",solution_delta_dim)

DV = solution_delta_dim[3:]
print("DV",DV)
DV_mag = norm(DV)
print("DV_mag",DV_mag)

ax = Plot_CR3BP(sols,mu,tvec[-1]*Tstar,show_initial_pos=False,show_final_pos=False,
           show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6))
ax.plot(xf_target, yf_target, 'go', label="Target Position")
ax.plot(sols[0][-1][0], sols[0][-1][1], 'ro', label="Initial Guess' Final Position")
if target_achieved == False:
    ax.plot(sols[-1][-1][0], sols[-1][-1][1], 'bo', label="Last Iteration's Final Position")


plt.legend()
plt.show()

# # Case a3
# xf_target = -0.4
# yf_target = -0.1
# target_cond = np.array([xf_target,yf_target,None,None,None,None])

# targeter_tolerance = 1e-3
# targeter_max_iters = 300
# fix_time_value = True
# front_constraints = [(0,0),(0,0),(0,0),None,None,(0,0)]
# use_coupled_integration = False
# target_achieved, final_state, final_tof, sols, solution_initial_state = TargetFinalStateFixedTime(X0,mu,target_cond,final_time,timesteps=num_points,tol=targeter_tolerance,max_iters=targeter_max_iters,fix_time=fix_time_value,initial_constraints=front_constraints,use_coupled_integration=use_coupled_integration)
# print("target_achieved",target_achieved)
# print("iteration amount",len(sols))
# print("final_state",final_state)
# print("final_tof dim",final_tof*Tstar)
# print("solution_initial_state",solution_initial_state)

# solution_delta = solution_initial_state - X0
# solution_delta_dim = [*solution_delta[:3]*Lstar,*solution_delta[3:]*Vstar]
# print("solution_delta_dim",solution_delta_dim)

# DV = solution_delta_dim[3:]
# print("DV",DV)
# DV_mag = norm(DV)
# print("DV_mag",DV_mag)

# ax = Plot_CR3BP(sols,mu,tvec[-1]*Tstar,show_initial_pos=False,show_final_pos=False,
#            show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6))
# ax.plot(xf_target, yf_target, 'go', label="Target Position")
# ax.plot(sols[0][-1][0], sols[0][-1][1], 'ro', label="Initial Guess' Final Position")

# plt.legend()
# plt.show()

# # Case b1
# xf_target = -0.3
# yf_target = +0.05
# target_cond = np.array([xf_target,yf_target,None,None,None,None])

# targeter_tolerance = 1e-3
# targeter_max_iters = 1500
# fix_time_value = False
# max_time_value = 6 / Tstar
# front_constraints = [(0,0),(0,0),(0,0),None,None,(0,0)]
# use_coupled_integration = True
# target_achieved, final_state, final_tof, sols, solution_initial_state = TargetFinalStateFixedTime(X0,mu,target_cond,final_time,max_time_value,timesteps=num_points,tol=targeter_tolerance,max_iters=targeter_max_iters,fix_time=fix_time_value,initial_constraints=front_constraints,use_coupled_integration=use_coupled_integration)
# print("target_achieved",target_achieved)
# print("iteration amount",len(sols))
# print("final_state",final_state)
# print("final_tof dim",final_tof*Tstar)
# print("solution_initial_state",solution_initial_state)

# solution_delta = solution_initial_state - X0
# solution_delta_dim = [*solution_delta[:3]*Lstar,*solution_delta[3:]*Vstar]
# print("solution_delta_dim",solution_delta_dim)

# DV = solution_delta_dim[3:]
# print("DV",DV)
# DV_mag = norm(DV)
# print("DV_mag",DV_mag)

# ax = Plot_CR3BP(sols,mu,tvec[-1]*Tstar,show_initial_pos=False,show_final_pos=False,
#            show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6))
# ax.plot(xf_target, yf_target, 'go', label="Target Position")
# ax.plot(sols[0][-1][0], sols[0][-1][1], 'ro', label="Initial Guess' Final Position")
# if target_achieved == False:
#     ax.plot(sols[-1][-1][0], sols[-1][-1][1], 'bo', label="Last Iteration's Final Position")

# plt.legend()
# plt.show()

# # Case b2
# xf_target = -0.1
# yf_target = 1e-4
# target_cond = np.array([xf_target,yf_target,None,None,None,None])

# targeter_tolerance = 1e-3
# targeter_max_iters = 4000
# fix_time_value = False
# max_time_value = 6 / Tstar
# front_constraints = [(0,0),(0,0),(0,0),None,None,(0,0)]
# use_coupled_integration = True
# target_achieved, final_state, final_tof, sols, solution_initial_state = TargetFinalStateFixedTime(X0,mu,target_cond,final_time,max_time_value,timesteps=num_points,tol=targeter_tolerance,max_iters=targeter_max_iters,fix_time=fix_time_value,initial_constraints=front_constraints,use_coupled_integration=use_coupled_integration)
# print("target_achieved",target_achieved)
# print("iteration amount",len(sols))
# print("final_state",final_state)
# print("final_tof dim",final_tof*Tstar)
# print("solution_initial_state",solution_initial_state)

# solution_delta = solution_initial_state - X0
# solution_delta_dim = [*solution_delta[:3]*Lstar,*solution_delta[3:]*Vstar]
# print("solution_delta_dim",solution_delta_dim)

# DV = solution_delta_dim[3:]
# print("DV",DV)
# DV_mag = norm(DV)
# print("DV_mag",DV_mag)

# ax = Plot_CR3BP(sols,mu,tvec[-1]*Tstar,show_initial_pos=False,show_final_pos=False,
#            show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6))
# ax.plot(xf_target, yf_target, 'go', label="Target Position")
# ax.plot(sols[0][-1][0], sols[0][-1][1], 'ro', label="Initial Guess' Final Position")
# if target_achieved == False:
#     ax.plot(sols[-1][-1][0], sols[-1][-1][1], 'bo', label="Last Iteration's Final Position")


# plt.legend()
# plt.show()

# # Case b3
# xf_target = -0.4
# yf_target = -0.1
# target_cond = np.array([xf_target,yf_target,None,None,None,None])

# targeter_tolerance = 1e-3
# targeter_max_iters = 1000
# fix_time_value = False
# max_time_value = 10 / Tstar
# front_constraints = [(0,0),(0,0),(0,0),None,None,(0,0)]
# use_coupled_integration = True
# target_achieved, final_state, final_tof, sols, solution_initial_state = TargetFinalStateFixedTime(X0,mu,target_cond,final_time,max_time_value,timesteps=num_points,tol=targeter_tolerance,max_iters=targeter_max_iters,fix_time=fix_time_value,initial_constraints=front_constraints,use_coupled_integration=use_coupled_integration)
# print("target_achieved",target_achieved)
# print("iteration amount",len(sols))
# print("final_state",final_state)
# print("final_tof dim",final_tof*Tstar)
# print("solution_initial_state",solution_initial_state)

# solution_delta = solution_initial_state - X0
# solution_delta_dim = [*solution_delta[:3]*Lstar,*solution_delta[3:]*Vstar]
# print("solution_delta_dim",solution_delta_dim)

# DV = solution_delta_dim[3:]
# print("DV",DV)
# DV_mag = norm(DV)
# print("DV_mag",DV_mag)

# ax = Plot_CR3BP(sols,mu,tvec[-1]*Tstar,show_initial_pos=False,show_final_pos=False,
#            show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6))
# ax.plot(xf_target, yf_target, 'go', label="Target Position")
# ax.plot(sols[0][-1][0], sols[0][-1][1], 'ro', label="Initial Guess' Final Position")
# if target_achieved == False:
#     ax.plot(sols[-1][-1][0], sols[-1][-1][1], 'bo', label="Last Iteration's Final Position")

# plt.legend()
# plt.show()