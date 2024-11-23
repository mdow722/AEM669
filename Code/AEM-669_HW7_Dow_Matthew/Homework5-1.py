import numpy as np
from ODESolving import *
from numpy.linalg import norm
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from Visualization import *

mu = 0.0123

x1 = [0.72,0,0.71,0,0.18,0]
x2 = [0.72,0,-0.71,0,0.18,0]
x3 = [0.72,0,0.71,0,0.18,0]
x4 = [0.72,0,-0.71,0,0.18,0]
x5 = [0.72,0,0.71,0,0.18,0]
dT = 3

X0 = [*x1,*x2,*x3,*x4,*x5]
# X = np.concatenate((x1,x2,x3,x4,x5),axis=0)

Xi = np.array([*X0]).reshape(30,1)

def F_single(Xj,Xjp1_target):
    sol,_,_ = rungekutta4(CR3BP_nondim, Xj, np.linspace(0,dT,100), args=(mu,))
    Xjp1_prop = sol[-1]
    # return (Xjp1_target - Xjp1_prop),sol
    return (Xjp1_prop - Xjp1_target),sol

def F(all_states):
    X1 = [*all_states[0:6].reshape(1,6)[0]]
    X2 = [*all_states[6:12].reshape(1,6)[0]]
    X3 = [*all_states[12:18].reshape(1,6)[0]]
    X4 = [*all_states[18:24].reshape(1,6)[0]]
    X5 = [*all_states[24:30].reshape(1,6)[0]]

    F1,sol1 = F_single(X1,X2)
    F2,sol2 = F_single(X2,X3)
    F3,sol3 = F_single(X3,X4)
    F4,sol4 = F_single(X4,X5)
    sols = [sol1,sol2,sol3,sol4]
    return np.array([*F1,*F2,*F3,*F4]).reshape(6*4,1),sols

def DF(all_states):
    X1 = [*all_states[0:6].reshape(1,6)[0]]
    X2 = [*all_states[6:12].reshape(1,6)[0]]
    X3 = [*all_states[12:18].reshape(1,6)[0]]
    X4 = [*all_states[18:24].reshape(1,6)[0]]
    X5 = [*all_states[24:30].reshape(1,6)[0]]

    _, STMs = STM_coupled_integration_vec(X1,np.linspace(0,dT,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    DX1 = STMs[-1]
    _, STMs = STM_coupled_integration_vec(X2,np.linspace(0,dT,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    DX2 = STMs[-1]
    _, STMs = STM_coupled_integration_vec(X3,np.linspace(0,dT,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    DX3 = STMs[-1]
    _, STMs = STM_coupled_integration_vec(X4,np.linspace(0,dT,100),ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
    DX4 = STMs[-1]

    negeye = -np.identity(6)
    zerarr = np.zeros((6,6))
    DF_final = np.zeros((6*4,6*5))

    # first row
    DF_final[0:6,0:6] = DX1[:,:]
    DF_final[0:6,6:12] = negeye[:,:]
    DF_final[0:6,12:18] = zerarr[:,:]
    DF_final[0:6,18:24] = zerarr[:,:]
    DF_final[0:6,24:30] = zerarr[:,:]

    # second row
    DF_final[6:12,0:6] = zerarr[:,:]
    DF_final[6:12,6:12] = DX2[:,:]
    DF_final[6:12,12:18] = negeye[:,:]
    DF_final[6:12,18:24] = zerarr[:,:]
    DF_final[6:12,24:30] = zerarr[:,:]

    # third row
    DF_final[12:18,0:6] = zerarr[:,:]
    DF_final[12:18,6:12] = zerarr[:,:]
    DF_final[12:18,12:18] = DX3[:,:]
    DF_final[12:18,18:24] = negeye[:,:]
    DF_final[12:18,24:30] = zerarr[:,:]

    # fourth row
    DF_final[18:24,0:6] = zerarr[:,:]
    DF_final[18:24,6:12] = zerarr[:,:]
    DF_final[18:24,12:18] = zerarr[:,:]
    DF_final[18:24,18:24] = DX4[:,:]
    DF_final[18:24,24:30] = negeye[:,:]
    # print("DX",DX3)
    # print("DF_final[12:18,:]",DF_final[12:18,:])

    return DF_final

Fmag = 1
i = 0
max_iters = 50
target_achieved = False
Xvec = []
Fvec = []
solvec = []
tol = 1e-12
while Fmag > tol and i < max_iters:
    print("iteration: ",i)
    Fi,sols = F(Xi)
    # print("Fi",Fi)
    # print("Fi size",Fi.shape)
    Fmag = norm(Fi)
    print("Fmag",Fmag)
    Fvec.append(Fmag)
    solvec.append(sols)
    DFi = DF(Xi)
    print("DFi size",DFi.shape)
    # print("DFi",DFi)
    # if i == 2:
    #     break
        
    Xi = Xi - min_norm(DFi)@Fi
    Xvec.append(Xi)
    i += 1
    if Fmag < tol:
        target_achieved = True

print(len(solvec))
trajvec = []
for j in range(i):
    if j == 0 or j == i-1:
        for traj in solvec[j]:
            trajvec.append(traj)
print(len(trajvec))
trajvec_ig = [*solvec[0]]
trajvec_sol = [*solvec[i-1]]

# ax = Plot_CR3BP(trajvec,mu,dT,show_initial_pos=False,show_final_pos=False,
#            show_bodies=False,show_ZVC=False,xlimits=(0.5,1.0),ylimits=(-.2,.2))
ax = Plot_CR3BP_3D(trajvec_ig,mu,f"Trajectory in the Earth-Moon System over {dT*5} Nondimensional Time Units",show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=False,xlimits=(-0.25,1.0),ylimits=(-.2,.2),names=["Phase 1","Phase 2","Phase 3","Phase 4"])
ax.yaxis.set_major_locator(ticker.LinearLocator(3))
ax1 = Plot_CR3BP_3D(trajvec_sol,mu,f"Trajectory in the Earth-Moon System over {dT*5} Nondimensional Time Units",show_initial_pos=False,show_final_pos=False,
           show_bodies=True,show_ZVC=False,xlimits=(-0.25,1.0),ylimits=(-.2,.2),names=["Phase 1","Phase 2","Phase 3","Phase 4"])
ax1.yaxis.set_major_locator(ticker.LinearLocator(3))

plt.legend()
plt.show()






