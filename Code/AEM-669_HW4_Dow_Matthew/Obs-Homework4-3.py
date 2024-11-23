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

x_0 = 300000 / Lstar
y_0 = 0
z_0 = 0
vx_0 = 0
vy_0 = 0.5 / Vstar
vz_0 = 0.5 / Vstar
X0 = np.array((x_0,y_0,z_0,vx_0,vy_0,vz_0))

# Get time to cross x axis:
t_0 = 0
t_f = 10 / Tstar
num_points = 200

# Case a1
xf_target = 500000 / Lstar
yf_target = -90000 / Lstar
zf_target = 200000 / Lstar

target_cond = np.array([xf_target,yf_target,zf_target,None,None,None])

targeter_tolerance = 1e-3
targeter_max_iters = 1000
fix_time_value = True
max_time = 10
# front_constraints = [0,0,0,False,False,False]
vx_max = 1/Vstar
vy_max = 1/Vstar
vz_max = 1/Vstar
front_constraints = [FIXED,FIXED,FIXED,(0.3/Vstar,3/Vstar),(-0.2/Vstar,0),(0.3,1/Vstar)]
front_constraints = [FIXED,FIXED,FIXED,(0.3/Vstar,5/Vstar),(-0.2/Vstar,0),(0.3,3/Vstar)]
# front_constraints = [FIXED,FIXED,FIXED,FREE,FREE,FREE]
use_coupled_integration = False
target_achieved, final_state, final_tof, sols, solution_initial_state = TargetFinalStateFixedTime(X0,mu,target_cond,t_f,timesteps=num_points,tol=targeter_tolerance,max_iters=targeter_max_iters,fix_time=fix_time_value,initial_constraints=front_constraints,use_coupled_integration=use_coupled_integration)
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

ax = Plot_CR3BP_3D(sols,mu,final_tof*Tstar,show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6))
ax.plot(xf_target, yf_target, zf_target, 'mo', label="Target Position")
ax.plot(sols[0][-1][0], sols[0][-1][1], sols[0][-1][2], 'ro', label="Initial Guess' Final Position")
if target_achieved == False:
    ax.plot(sols[-1][-1][0], sols[-1][-1][1], sols[-1][-1][2], 'yo', label="Last Iteration's Final Position")

plt.legend()
plt.show()