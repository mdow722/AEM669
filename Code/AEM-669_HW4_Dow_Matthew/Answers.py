from PlanetaryDataFuncs import *
from ThreeBodyModel import *

# Venus_GM = GetPlanetProperty("veNus", "mass",True)
# print(Venus_GM)
Lstar,Mstar,Tstar,mu = Get3BodyCharacteristics("Earth","Moon")
print(Lstar)
print(Mstar)
print(Tstar)
print(mu)

x,g,g_dim=Get3BodyColinearLibrationPoints(mu,Lstar)
print(x)
print(g)
print(g_dim)

# B3b
L4,L5 = Get3BodyEquilateralLagrangePoints(mu)

C1 = GetJacobiConstant([x[0],0,0],[0,0,0],mu)
C2 = GetJacobiConstant([x[1],0,0],[0,0,0],mu)
C3 = GetJacobiConstant([x[2],0,0],[0,0,0],mu)
C4 = GetJacobiConstant([L4[0],L4[1],0],[0,0,0],mu)
C5 = GetJacobiConstant([L5[0],L5[1],0],[0,0,0],mu)
print(C1)
print(C2)
print(C3)
print(C4)
print(C5)

# B3c
x_0 = -0.270
y_0 = -0.420
z_0 = 0
vx_0 = 0.300
vy_0 = -1.00
vz_0 = 0

C = GetJacobiConstant([x_0,y_0,z_0],[vx_0,vy_0,vz_0],mu)
print(C)