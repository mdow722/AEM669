from ODESolving import *
from ThreeBodyModel import *
from scipy.integrate import solve_ivp
from PlanetaryDataFuncs import *
from Visualization import *
import matplotlib.pyplot as plt

Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
Vstar = Lstar/Tstar

x_0 = 0.488
y_0 = 0.200
z_0 = 0
r_dim = [x_0*Lstar,y_0*Lstar,z_0*Lstar]
print("r_dim",r_dim)

vx_0 = -0.880
vy_0 = 0.200
vz_0 = 0
v_dim = [vx_0*Vstar,vy_0*Vstar,vz_0*Vstar]
print("v_dim",v_dim)

X0 = np.array((x_0,y_0,z_0,vx_0,vy_0,vz_0))

jacobi_constant = GetJacobiConstant([x_0,y_0,z_0],[vx_0,vy_0,vz_0],mu)
print("jacobi_constant",jacobi_constant)

Xgrid,Ygrid,Zgrid = GetZVCGrid_xy(mu,jacobi_constant,resolution=0.001)
t_0 = 0  # nondimensional time
t_f = 10  # nondimensional time
num_points = 100
t_points = np.linspace(t_0, t_f, t_f*num_points)

crossing_x_axis_event = [False, True, False, False, False, False]
yvec, tvec, event_locs = rungekutta4(CR3BP_nondim, X0, t_points, args=(mu,),event_conditions=crossing_x_axis_event, stop_at_events=True)
x_f = yvec[-1,0]
y_f = yvec[-1,1]
rnew = yvec[:, :3]  # nondimensional distance
# y_final_dim = [*yvec[:3]*Lstar,]
print("final_state ref: ",yvec[-1])

STM = STM_finite_forward_diff(X0, t_0, t_f, CR3BP_nondim,(mu,),1e-8)
print("final_STM from FFD:\n",STM)

states, STMs = STM_coupled_integration_vec(X0,tvec,ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
final_state = states[-1]
final_STM_CI = STMs[-1]
print("final_state from CI",final_state)
print("delta final_state",final_state-yvec[-1])
print("final_STM from CI:\n",final_STM_CI)
print("final time",tvec[-1])

Plot_CR3BP([yvec],mu,tvec[-1]*Tstar/86400)

# D1b-i
dx_0 = 0.01*x_0
dy_0 = 0
dvx_0 = 0
dvy_0  = 0
X0_D1b1 = X0+[dx_0,dy_0,0,dvx_0,dvy_0,0]

print("final_state from ref: ",yvec[-1])

dXf1 = final_STM_CI[0,:]*dx_0
print("Xf1: ",dXf1+yvec[-1])

# D1b-ii
dx_0 = 0
dy_0 = 0
dvx_0 = 0
dvy_0  = 0.01*vy_0
X0_D1b2 = X0+[dx_0,dy_0,0,dvx_0,dvy_0,0]

dXf2 = final_STM_CI[4,:]*dvy_0
print("Xf2: ",dXf2+yvec[-1])

# D1b-plot
yvec_D1b1, _, _ = rungekutta4(CR3BP_nondim, X0_D1b1, tvec, args=(mu,))
yvec_D1b2, _, _ = rungekutta4(CR3BP_nondim, X0_D1b2, tvec, args=(mu,))
Plot_CR3BP([yvec,yvec_D1b1,yvec_D1b2],mu,tvec[-1]*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6),
           names=["Reference","Case 1","Case 2"])

# D1c
Xdott = CR3BP_nondim(0,yvec[-1],mu)
dt1 = tvec[-1]*1.01
dt2 = tvec[-1]*1.1
t_points_c1 = np.arange(t_0, dt1, 0.01)
t_points_c2 = np.arange(t_0, dt2, 0.01)
dXfc1 = Xdott*dt1
print("Xfc1: ",dXfc1+yvec[-1])
dXfc2 = Xdott*dt2
print("Xfc2: ",dXfc2+yvec[-1])

yvec_D1c1, _, _ = rungekutta4(CR3BP_nondim, X0, t_points_c1, args=(mu,))
yvec_D1c2, _, _ = rungekutta4(CR3BP_nondim, X0, t_points_c2, args=(mu,))
Plot_CR3BP([yvec,yvec_D1c1,yvec_D1c2],mu,dt2*Tstar/86400,show_initial_pos=False,show_final_pos=False,
           show_bodies=False,show_ZVC=False,xlimits=(-.6,.6),ylimits=(-.6,.6),
           names=["Reference","Case 1 - T+1%","Case 2 - T+10%"])

plt.show()