from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from ThreeBodyModel import *
from PlanetaryDataFuncs import *
from ODESolving import *
from sympy import *

# These masses represent the Earth-Moon system
# m_1 = 5.974E24  # kg
# m_2 = 7.348E22 # kg
# mu = m_2/(m_1 + m_2)
Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
print("mu",mu)
print("Tstar",Tstar)
Vstar = Lstar/(Tstar*86400)

x_0 = -0.270
y_0 = -0.420
z_0 = 0
r_dim = [x_0*Lstar,y_0*Lstar,z_0*Lstar]
print("r_dim",r_dim)

vx_0 = 0.3
vy_0 = -1.0
# vx_0 = 1.01
# vy_0 = 0.3
vz_0 = 0
v_dim = [vx_0*Vstar,vy_0*Vstar,vz_0*Vstar]
print("v_dim",v_dim)

X0 = np.array((x_0,y_0,z_0,vx_0,vy_0,vz_0))
# X0 = np.array((x_0,y_0,vx_0,vy_0))

jacobi_constant = GetJacobiConstant([x_0,y_0,z_0],[vx_0,vy_0,vz_0],mu)
print("jacobi_constant",jacobi_constant)

Xgrid,Ygrid,Zgrid = GetZVCGrid(mu,jacobi_constant,resolution=0.001)
t_0 = 0  # nondimensional time
t_f = 50  # nondimensional time
num_points = 100
t_points = np.linspace(t_0, t_f, t_f*num_points)
# sol = solve_ivp(CR3BP_nondim, [t_0,t_f], X0, method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)

# Y = sol.y.T
# times = sol.t
# r = Y[:, :3]  # nondimensional distance
# v = Y[:, 3:]  # nondimensional velocity


x_2 = (1 - mu) * np.cos(np.linspace(0, np.pi, 100))
y_2 = (1 - mu) * np.sin(np.linspace(0, np.pi, 100))

# # Plot the orbits
# fig, ax = plt.subplots(figsize=(5,5), dpi=96)
# ax.plot(r[:, 0], r[:, 1], 'r', label="Trajectory")
# ax.axhline(0, color='k')
# ax.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
# ax.plot(-mu, 0, 'bo', label="$m_1$")
# ax.plot(1 - mu, 0, 'go', label="$m_2$")
# ax.plot(x_0, y_0, 'ro')
# ax.set_aspect("equal")
# ax.contour(Xgrid,Ygrid,Zgrid,levels=[0])
# ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
# ax.set(title=f"Spacecraft Trajectory in the Earth-Moon System over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.show()

# # Plot the Jacobi Constant
# JCs = list(map(lambda r, v: GetJacobiConstant([r[0],r[1],0],[v[0],v[1],0],mu),r,v))
# fig1, ax1 = plt.subplots(figsize=(5,5), dpi=96)
# ax1.plot(times*Tstar,JCs-jacobi_constant)
# # ax1.set(xlim=(0, times[-1]), ylim=(-1.5, 1.5))
# ax1.autoscale()
# ax1.set(title=f"Spacecraft Jacobi Constant over {round(t_f*Tstar,1)} Days",
#        xlabel="Days since Epoch", ylabel="Jacobi Constant")
# plt.show()

# xLib,_,_ = Get3BodyColinearLibrationPoints(mu,Lstar)
# U2ders = GetPseudoPotentialSecondDers([xLib[0],0,0],mu)
# eigs = GetEigenvalues(U2ders)
# print(eigs)
# l3 = eigs[2]
# x0dev = 0.01
# x0dev_dim = x0dev *Lstar
# print("x0dev_dim",x0dev_dim)
# y0dev = 0
# vxdev,vydev = GetVdevLinear_OnlyOsc(x0dev,y0dev,l3,U2ders[0],U2ders[1])
# print(vxdev)
# print(vydev)
# X0lin = np.array((x0dev,y0dev,0,vxdev,vydev,0))
# sol2 = solve_ivp(LinearCR3BP, [t_0,t_f], X0lin, method="RK45", t_eval=t_points, args=(U2ders,), rtol=1e-12, atol=1e-12)
# Ylin = sol2.y.T
# timeslin = sol2.t
# rlin = Ylin[:, :3]
# # print(*rlin[:])
# dL1lin = rlin.max()
# print("dL1lin*Lstar",dL1lin*Lstar)
# period = GetLinearPeriod(U2ders[0],U2ders[1])
# print("period",period)
# print("period*Tstar",period*Tstar)
# fig2, ax2 = plt.subplots(figsize=(5,5), dpi=96)
# ax2.plot(rlin[:, 0], rlin[:, 1], 'b', label="Trajectory")
# ax2.set_aspect('equal', 'box')
# ax2.set(title=f"Spacecraft Trajectory (Relative to L1) over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.show()

# X0lin_bary = [*X0lin]
# X0lin_bary[0] += xLib[0]

# jacobi_c2c = GetJacobiConstant(X0lin_bary[:3],X0lin_bary[3:],mu)
# # print("jacobi_c2c",jacobi_c2c)
# # print(jacobi_c2c-jacobi_constant)
# Xgrid2,Ygrid2,Zgrid2 = GetZVCGrid(mu,jacobi_c2c,resolution=0.001)

# Plot the orbits
# fig3, ax3 = plt.subplots(figsize=(5,5), dpi=96)
# ax3.axhline(0, color='k')
# ax3.plot(-mu, 0, 'bo', label="$m_1$")
# ax3.plot(1 - mu, 0, 'go', label="$m_2$")
# ax3.set_aspect("equal")
# ax3.contour(Xgrid,Ygrid,Zgrid,levels=[0])
# ax3.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
# ax3.set(title=f"Zero Velocity Curves for a Jacobi Constant of {jacobi_c2c}",
#        xlabel="X", ylabel="Y")
# plt.show()

# sol3 = solve_ivp(CR3BP_nondim, [t_0,t_f], X0lin_bary, method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin3 = sol3.y.T
# times_nonlin3 = sol3.t
# rnonlin3 = Ynonlin3[:, :3]  # nondimensional distance
# sol4 = solve_ivp(CR3BP_nondim, [t_0,t_f], [X0lin_bary[0]+0.001,*X0lin_bary[1:]], method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin4 = sol4.y.T
# times_nonlin4 = sol4.t
# rnonlin4 = Ynonlin4[:, :3]  # nondimensional distance
# sol5 = solve_ivp(CR3BP_nondim, [t_0,t_f], [X0lin_bary[0]-0.001,*X0lin_bary[1:]], method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin5 = sol5.y.T
# times_nonlin5 = sol5.t
# rnonlin5 = Ynonlin5[:, :3]  # nondimensional distance
# # Plot the orbits
# fig4, ax4 = plt.subplots(figsize=(5,5), dpi=96)
# ax4.plot(rlin[:, 0]+xLib[0], rlin[:, 1], 'r', label="Trajectory")
# ax4.plot(rnonlin3[:, 0], rnonlin3[:, 1], 'g', label="Trajectory")
# ax4.plot(rnonlin4[:, 0], rnonlin4[:, 1], 'm', label="Trajectory")
# ax4.plot(rnonlin5[:, 0], rnonlin5[:, 1], 'y', label="Trajectory")
# ax4.axhline(0, color='k')
# ax4.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
# ax4.plot(-mu, 0, 'bo', label="$m_1$")
# ax4.plot(1 - mu, 0, 'go', label="$m_2$")
# ax4.set_aspect("equal")
# ax4.contour(Xgrid2,Ygrid2,Zgrid2,levels=[0])
# ax4.set(xlim=(xLib[0]-0.2, xLib[0]+0.2), ylim=(-0.2, 0.2))
# ax4.set(title=f"Spacecraft Trajectory in the Earth-Moon System over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.show()


# x0dev = 0.01
# x0dev_dim = x0dev *Lstar
# print("x0dev_dim",x0dev_dim)
# y0dev = 0
# vxdev6,vydev6 = GetVdevLinear_OnlyExp(x0dev,y0dev,eigs[0],U2ders[0],U2ders[1])
# X0lin6 = np.array((x0dev,y0dev,0,vxdev6,vydev6,0))
# sol6 = solve_ivp(LinearCR3BP, [t_0,t_f], X0lin6, method="RK45", t_eval=t_points, args=(U2ders,), rtol=1e-12, atol=1e-12)
# Ylin6 = sol6.y.T
# timeslin6 = sol6.t
# rlin6 = Ylin6[:, :3]
# fig6, ax6 = plt.subplots(figsize=(5,5), dpi=96)
# ax6.plot(rlin6[:, 0], rlin6[:, 1], 'b', label="Trajectory")

# sol7 = solve_ivp(LinearCR3BP, [t_0,t_f], [X0lin6[0]+0.001,*X0lin6[1:]], method="RK45", t_eval=t_points, args=(U2ders,), rtol=1e-12, atol=1e-12)
# Ylin7 = sol7.y.T
# timeslin7 = sol7.t
# rlin7 = Ylin7[:, :3]
# ax6.plot(rlin7[:, 0], rlin7[:, 1], 'g', label="Trajectory")

# sol8 = solve_ivp(LinearCR3BP, [t_0,t_f], [X0lin6[0]-0.001,*X0lin6[1:]], method="RK45", t_eval=t_points, args=(U2ders,), rtol=1e-12, atol=1e-12)
# Ylin8 = sol8.y.T
# timeslin8 = sol8.t
# rlin8 = Ylin8[:, :3]
# ax6.plot(rlin8[:, 0], rlin8[:, 1], 'm', label="Trajectory")

# ax6.set_aspect('equal', 'box')
# ax6.set(title=f"Spacecraft Trajectory (Relative to L1) over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")


# ax6.plot(rlin[:, 0], rlin[:, 1], 'r', label="Trajectory")
# plt.show()

# X0lin_bary6 = [*X0lin6]
# X0lin_bary6[0] += xLib[0]
# sol6 = solve_ivp(CR3BP_nondim, [t_0,t_f], X0lin_bary6, method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin6 = sol6.y.T
# rnonlin6 = Ynonlin6[:, :3]  # nondimensional distance
# # times_nonlin3 = sol3.t
# fig7, ax7 = plt.subplots(figsize=(5,5), dpi=96)
# ax7.plot(rlin6[:, 0]+xLib[0], rlin6[:, 1], 'r', label="Trajectory")
# ax7.plot(rnonlin6[:, 0], rnonlin6[:, 1], 'g', label="Trajectory")
# ax7.axhline(0, color='k')
# ax7.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
# ax7.plot(-mu, 0, 'bo', label="$m_1$")
# ax7.plot(1 - mu, 0, 'go', label="$m_2$")
# ax7.set_aspect("equal")
# ax7.contour(Xgrid2,Ygrid2,Zgrid2,levels=[0])
# ax7.set(xlim=(xLib[0]-0.2, xLib[0]+0.2), ylim=(-0.2, 0.2))
# ax7.set(title=f"Spacecraft Trajectory in the Earth-Moon System over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.show()

# eigs_l4 = GetEigenvaluesEquilateral(mu)
# print(eigs_l4)
# s1 = abs(eigs_l4[0])
# s2 = abs(eigs_l4[2])
# print("s1",s1)
# print("s2",s2)
# period_short = 2*np.pi / s2
# period_long = 2*np.pi / s1
# print("period_long nondim", period_long)
# print("period_long in days", period_long*Tstar)
# print("period_short nondim", period_short)
# print("period_short in days", period_short*Tstar)

# L4,_ = Get3BodyEquilateralLagrangePoints(mu)
# print(L4)
# U2ders_eq = GetPseudoPotentialSecondDers([*L4,0],mu)
# print("U2ders_eq",U2ders_eq)

# ########### C3b
# ICs = GetEquilateralPointOffsets(eigs_l4,U2ders_eq,0.01,0,removal_target="long")
# print("ICs",ICs)
# vxdev_eq = re(ICs.args[0][2])
# vydev_eq = re(ICs.args[0][3])
# print(vxdev_eq)
# print(vydev_eq)

# x0dev = 0.01
# x0dev_dim = x0dev *Lstar
# y0dev = 0
# y0dev_dim = y0dev *Lstar
# vx0dev = vxdev_eq
# vx0dev_dim = vx0dev *Vstar
# vy0dev = vydev_eq
# vy0dev_dim = vy0dev *Vstar
# print("x0dev_dim",x0dev_dim)
# print("y0dev_dim",y0dev_dim)
# print("vx0dev_dim",vx0dev_dim)
# print("vy0dev_dim",vy0dev_dim)
# y0dev = 0
# X0lin9 = np.array((x0dev,y0dev,0,vxdev_eq,vydev_eq,0))
# # X0lin9 = np.array((x0dev,y0dev,0,0,0,0))
# sol9 = solve_ivp(LinearCR3BP, [t_0,t_f], X0lin9, method="RK45", t_eval=t_points, args=(U2ders_eq,), rtol=1e-12, atol=1e-12)
# Ylin9 = sol9.y.T
# timeslin9 = sol9.t
# rlin9 = Ylin9[:, :3]

# period_short = 2*np.pi / s1
# print("period_short in days", period_short*Tstar)

# fig8, ax8 = plt.subplots(figsize=(5,5), dpi=96)
# ax8.plot(rlin9[:, 0], rlin9[:, 1], 'b', label="Trajectory")
# ax8.set(title=f"Trajectory around the Earth-Moon L4 With Long-Period\nOscillations Removed over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.show()

# ########### C3c
# ICs = GetEquilateralPointOffsets(eigs_l4,U2ders_eq,0.01,0,removal_target="short")
# print("ICs",ICs)
# vxdev_eq = re(ICs.args[0][2])
# vydev_eq = re(ICs.args[0][3])
# print(vxdev_eq)
# print(vydev_eq)

# x0dev = 0.01
# x0dev_dim = x0dev *Lstar
# y0dev = 0
# y0dev_dim = y0dev *Lstar
# vx0dev = vxdev_eq
# vx0dev_dim = vx0dev *Vstar
# vy0dev = vydev_eq
# vy0dev_dim = vy0dev *Vstar
# print("x0dev_dim",x0dev_dim)
# print("y0dev_dim",y0dev_dim)
# print("vx0dev_dim",vx0dev_dim)
# print("vy0dev_dim",vy0dev_dim)
# y0dev = 0
# X0lin9 = np.array((x0dev,y0dev,0,vxdev_eq,vydev_eq,0))
# # X0lin9 = np.array((x0dev,y0dev,0,0,0,0))
# sol9 = solve_ivp(LinearCR3BP, [t_0,t_f], X0lin9, method="RK45", t_eval=t_points, args=(U2ders_eq,), rtol=1e-12, atol=1e-12)
# Ylin9 = sol9.y.T
# timeslin9 = sol9.t
# rlin9 = Ylin9[:, :3]

# fig8, ax8 = plt.subplots(figsize=(5,5), dpi=96)
# ax8.plot(rlin9[:, 0], rlin9[:, 1], 'b', label="Trajectory")
# ax8.set(title=f"Trajectory around the Earth-Moon L4 With Short-Period\nOscillations Removed over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.show()

# ########### C3d
# vxdev_eq = 0
# vydev_eq = 0

# x0dev = 0.01
# x0dev_dim = x0dev *Lstar
# y0dev = 0
# y0dev_dim = y0dev *Lstar
# vx0dev = vxdev_eq
# vx0dev_dim = vx0dev *Vstar
# vy0dev = vydev_eq
# vy0dev_dim = vy0dev *Vstar
# print("x0dev_dim",x0dev_dim)
# print("y0dev_dim",y0dev_dim)
# print("vx0dev_dim",vx0dev_dim)
# print("vy0dev_dim",vy0dev_dim)
# y0dev = 0
# X0lin9 = np.array((x0dev,y0dev,0,vxdev_eq,vydev_eq,0))
# # X0lin9 = np.array((x0dev,y0dev,0,0,0,0))
# sol9 = solve_ivp(LinearCR3BP, [t_0,t_f], X0lin9, method="RK45", t_eval=t_points, args=(U2ders_eq,), rtol=1e-12, atol=1e-12)
# Ylin9 = sol9.y.T
# timeslin9 = sol9.t
# rlin9 = Ylin9[:, :3]

# fig8, ax8 = plt.subplots(figsize=(5,5), dpi=96)
# ax8.plot(rlin9[:, 0], rlin9[:, 1], 'b', label="Trajectory")
# ax8.set(title=f"Trajectory around the Earth-Moon L4 With Short-Period\nOscillations Removed over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.show()

# ########### C3e,f
# vxdev_eq = 0
# vydev_eq = 0

# x0dev = 0.01
# x0dev_dim = x0dev *Lstar
# y0dev = 0
# y0dev_dim = y0dev *Lstar
# vx0dev = vxdev_eq
# vx0dev_dim = vx0dev *Vstar
# vy0dev = vydev_eq
# vy0dev_dim = vy0dev *Vstar
# print("x0dev_dim",x0dev_dim)
# print("y0dev_dim",y0dev_dim)
# print("vx0dev_dim",vx0dev_dim)
# print("vy0dev_dim",vy0dev_dim)
# y0dev = 0
# X0lin9 = np.array((x0dev,y0dev,0,vxdev_eq,vydev_eq,0))

# sol9 = solve_ivp(LinearCR3BP, [t_0,t_f], X0lin9, method="RK45", t_eval=t_points, args=(U2ders_eq,), rtol=1e-12, atol=1e-12)
# Ylin9 = sol9.y.T
# timeslin9 = sol9.t
# rlin9 = Ylin9[:, :3]

# X0lin_bary9 = [*X0lin9]
# X0lin_bary9[0] += L4[0]
# X0lin_bary9[1] += L4[1]
# sol9 = solve_ivp(CR3BP_nondim, [t_0,t_f], X0lin_bary9, method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin9 = sol9.y.T
# rnonlin9 = Ynonlin9[:, :3]  # nondimensional distance

# [X0lin_bary9[0]-0.001,*X0lin_bary9[1:]]
# sol10 = solve_ivp(CR3BP_nondim, [t_0,t_f], [X0lin_bary9[0]+0.001,*X0lin_bary9[1:]], method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin10 = sol10.y.T
# rnonlin10 = Ynonlin10[:, :3]  # nondimensional distance
# sol11 = solve_ivp(CR3BP_nondim, [t_0,t_f], [X0lin_bary9[0]-0.001,*X0lin_bary9[1:]], method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin11 = sol11.y.T
# rnonlin11 = Ynonlin11[:, :3]  # nondimensional distance

# jacobi_c2c = GetJacobiConstant(X0lin_bary9[:3],X0lin_bary9[3:],mu)
# print("jacobi_c2c",jacobi_c2c)
# # print(jacobi_c2c-jacobi_constant)
# Xgrid3,Ygrid3,Zgrid3 = GetZVCGrid(mu,jacobi_c2c,resolution=0.001)

# fig4, ax4 = plt.subplots(figsize=(5,5), dpi=96)
# ax4.plot(rlin9[:, 0]+L4[0], rlin9[:, 1]+L4[1], 'r', label="Linear")
# ax4.plot(rnonlin9[:, 0], rnonlin9[:, 1], 'g', label="Nonlinear")
# ax4.plot(rnonlin10[:, 0], rnonlin10[:, 1], 'm', label="Nonlinear, +Offset")
# ax4.plot(rnonlin11[:, 0], rnonlin11[:, 1], 'y', label="Nonlinear, -Offset")
# ax4.axhline(0, color='k')
# ax4.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
# ax4.plot(-mu, 0, 'bo')
# ax4.plot(1 - mu, 0, 'go')
# ax4.set_aspect("equal")
# ax4.contour(Xgrid3,Ygrid3,Zgrid3,levels=[0])
# ax4.set(xlim=(L4[0]-0.2, L4[0]+0.2), ylim=(L4[1]-0.2, L4[1]+0.2))
# ax4.set(title=f"Spacecraft Trajectory in the Earth-Moon System over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.legend()
# plt.show()

########### C3g
Lstar,Mstar,Tstar,mu_PC = Get3BodyCharacteristics("Pluto","Charon")
print("mu_PC",mu_PC)
print("Tstar",Tstar)
eigs_PC = GetEigenvaluesEquilateral(mu_PC)
print(eigs_PC)
s1 = abs(eigs_PC[0])
s2 = abs(eigs_PC[2])
print("s1",s1)
print("s2",s2)
period_short = 2*np.pi / s2
period_long = 2*np.pi / s1
print("period_long nondim", period_long)
print("period_long in days", period_long*Tstar)
print("period_short nondim", period_short)
print("period_short in days", period_short*Tstar)

L4,_ = Get3BodyEquilateralLagrangePoints(mu_PC)
print(L4)
U2ders_eq = GetPseudoPotentialSecondDers([*L4,0],mu_PC)
print("U2ders_eq",U2ders_eq)


ICs = GetEquilateralPointOffsets(eigs_PC,U2ders_eq,0.01,0,removal_target="divergence")
print("ICs",ICs)
vxdev_eq = re(ICs.args[0][2])
vydev_eq = re(ICs.args[0][3])
print(vxdev_eq)
print(vydev_eq)

x0dev = 0.01
x0dev_dim = x0dev *Lstar
y0dev = 0
y0dev_dim = y0dev *Lstar
vx0dev = vxdev_eq
vx0dev_dim = vx0dev *Vstar
vy0dev = vydev_eq
vy0dev_dim = vy0dev *Vstar
print("x0dev_dim",x0dev_dim)
print("y0dev_dim",y0dev_dim)
print("vx0dev_dim",vx0dev_dim)
print("vy0dev_dim",vy0dev_dim)
y0dev = 0
X0lin9 = np.array((x0dev,y0dev,0,vxdev_eq,vydev_eq,0))
# X0lin9 = np.array((x0dev,y0dev,0,0,0,0))
sol9 = solve_ivp(LinearCR3BP, [t_0,t_f], X0lin9, method="RK45", t_eval=t_points, args=(U2ders_eq,), rtol=1e-12, atol=1e-12)
Ylin9 = sol9.y.T
timeslin9 = sol9.t
rlin9 = Ylin9[:, :3]

# fig8, ax8 = plt.subplots(figsize=(5,5), dpi=96)
# ax8.plot(rlin9[:, 0], rlin9[:, 1], 'b', label="Trajectory")
# ax8.plot(0, 0, 'go', label="L4")
# ax8.set(title=f"Trajectory around the Earth-Moon L4 With Short-Period\nOscillations Removed over {round(t_f*Tstar,1)} Days",
#        xlabel="X", ylabel="Y")
# plt.show()

X0lin_bary9 = [*X0lin9]
X0lin_bary9[0] += L4[0]
X0lin_bary9[1] += L4[1]
sol9 = solve_ivp(CR3BP_nondim, [t_0,t_f], X0lin_bary9, method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
Ynonlin9 = sol9.y.T
rnonlin9 = Ynonlin9[:, :3]  # nondimensional distance

# # [X0lin_bary9[0]-0.001,*X0lin_bary9[1:]]
# sol10 = solve_ivp(CR3BP_nondim, [t_0,t_f], [X0lin_bary9[0]+0.001,*X0lin_bary9[1:]], method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin10 = sol10.y.T
# rnonlin10 = Ynonlin10[:, :3]  # nondimensional distance
# sol11 = solve_ivp(CR3BP_nondim, [t_0,t_f], [X0lin_bary9[0]-0.001,*X0lin_bary9[1:]], method="DOP853", t_eval=t_points, args=(mu,), rtol=1e-12, atol=1e-12)
# Ynonlin11 = sol11.y.T
# rnonlin11 = Ynonlin11[:, :3]  # nondimensional distance

jacobi_c3g = GetJacobiConstant(X0lin_bary9[:3],X0lin_bary9[3:],mu)
print("jacobi_c3g",jacobi_c3g)
# print(jacobi_c2c-jacobi_constant)
# Xgrid3,Ygrid3,Zgrid3 = GetZVCGrid(mu,jacobi_c3g,resolution=0.001)

fig4, ax4 = plt.subplots(figsize=(5,5), dpi=96)
ax4.plot(rlin9[:, 0]+L4[0], rlin9[:, 1]+L4[1], 'r', label="Linear")
ax4.plot(rnonlin9[:, 0], rnonlin9[:, 1], 'g', label="Nonlinear")
ax4.axhline(0, color='k')
ax4.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
ax4.plot(-mu, 0, 'bo')
ax4.plot(1 - mu, 0, 'go')
ax4.plot(L4[0], L4[1], 'mo', label="L4")
ax4.set_aspect("equal")
# ax4.contour(Xgrid3,Ygrid3,Zgrid3,levels=[0])
ax4.set(xlim=(L4[0]-0.05, L4[0]+0.05), ylim=(L4[1]-0.05, L4[1]+0.05))
ax4.set(title=f"Spacecraft Trajectory in the Pluto-Charon System over {round(t_f*Tstar,1)} Days",
       xlabel="X", ylabel="Y")
plt.legend()
plt.show()