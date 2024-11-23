import numpy as np
from ODESolving import *
from numpy.linalg import norm
from PlanetaryDataFuncs import *
from Visualization import *
from TwoBodyModel import *

mu = 398600
initial_orbels = [10000,0,0,0,0,0]
initial_cart = classical_to_cartesian(initial_orbels,mu)

Lstar = 6378
Tstar = Lstar/np.sqrt(mu/Lstar)
Vstar = Lstar/Tstar
Mustar = mu
mu_nd = mu/Mustar
print("Tstar",Tstar)
print("mu_nd",mu_nd)

initial_cart_nd = [*initial_cart[:3]/Lstar,*initial_cart[3:]/Vstar]
print("initial cart nd", initial_cart_nd)

orbit_period = GetOrbitPeriod(initial_orbels[0],mu)
orbit_period_nd = orbit_period/Tstar
tvec = np.linspace(0,orbit_period_nd,100)

traj,_,_ = rungekutta4(TwoBody, initial_cart_nd, tvec, args=(mu_nd,))

fig, ax = plt.subplots(figsize=(5,5), dpi=96)
ax.plot(0,0,'bo',label="Earth Center")
ax.plot(traj[:, 0]*Lstar, traj[:, 1]*Lstar,label="RK4")


states, STMs = STM_coupled_integration_vec(initial_cart_nd,tvec,ode=CoupledSTMFunc,ode_args=(mu_nd,))
states = np.array(states)
final_state = states[-1]
final_STM_CI = STMs[-1]
print("final_state from CI",final_state)
print("final_STM from CI:\n",final_STM_CI)
# print("states:",states)
# print("states np:",np.array(states))
ax.plot(states[:,0]*Lstar, states[:,1]*Lstar,linestyle='--',label="Coupled STM")
ax.set(title="Circular Planar Orbit Around Earth With SMA=10,000km",
       xlabel="X [km]",
       ylabel="Y [km]")

eigvals = GetEigenvalues(final_STM_CI)
print("eigvals",eigvals)

ax.set_aspect("equal")
ax.legend()
plt.show()