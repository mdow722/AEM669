import numpy as np
from scipy.optimize import newton
from sympy import *

def CR3BP_nondim(t, X, mu):
    x, y, z, vx, vy, vz = X

    d = np.sqrt(np.sum(np.square([x+mu,y,z])))
    r = np.sqrt(np.sum(np.square([x-1+mu,y,z])))

    ax = 2*vy + x - ((1-mu)*(x+mu))/(d**3) - (mu*(x-1+mu))/(r**3)
    ay = -2*vx + y - ((1-mu)*y)/(d**3) - (mu*y)/(r**3)
    az = - ((1-mu)*z)/(d**3) - (mu*z)/(r**3)

    return np.array([vx,vy,vz,ax,ay,az])

def CR3BP_2D_nondim(t,X,mu):
    x,y,vx,vy=X

    d = np.sqrt(np.sum(np.square([x+mu,y])))
    r = np.sqrt(np.sum(np.square([x-1+mu,y])))

    ax = 2*vy + x - ((1-mu)*(x+mu))/(d**3) - (mu*(x-1+mu))/(r**3)
    ay = -2*vx + y - ((1-mu)*y)/(d**3) - (mu*y)/(r**3)

    return [vx,vy,ax,ay]

# def LinearCR3BP(t,Xdev,U2ders,Xdev0):
def LinearCR3BP(t,Xdev,U2ders):
    Uxx,Uyy,Uzz,Uxy,Uxz,Uyz = U2ders

    x, y, z, vx, vy, vz = Xdev

    ax = Uxx*x + Uxy*y + Uxz*z + 2*vy
    ay = Uxy*x + Uyy*y + Uyz*z - 2*vx
    az = Uxz*x + Uyz*y + Uzz*z

    return [vx,vy,vz,ax,ay,az]

def ColinearLagrangePointFunction(x, mu):
    return x - (((1-mu)*(x+mu))/np.abs(x+mu)**3) - ((mu*(x-1+mu))/np.abs(x-1+mu)**3)

def Get3BodyColinearLibrationPoints(mu, Lstar):
    x1 = newton(func=ColinearLagrangePointFunction, x0 = 0, args=(mu,))
    g1 = 1 - mu - x1
    g1_dim = g1 * Lstar

    x2 = newton(func=ColinearLagrangePointFunction, x0 = 1, args=(mu,))
    g2 = x2 - 1 + mu
    g2_dim = g2 * Lstar

    x3 = newton(func=ColinearLagrangePointFunction, x0 = -1, args=(mu,))
    g3 = -mu - x3
    g3_dim = g3*Lstar

    return [x1,x2,x3],[g1,g2,g3],[g1_dim,g2_dim,g3_dim]

def Get3BodyEquilateralLagrangePoints(mu):
    x = 0.5 - mu
    y4 = np.sqrt(3)/2
    y5 = -np.sqrt(3)/2 
    return (x,y4),(x,y5)

def GetJacobiConstant(pos,vel,mu):
    d = np.sqrt((pos[0]+mu)**2 + pos[1]**2 + pos[2]**2)
    r = np.sqrt((pos[0]-1+mu)**2 + pos[1]**2 + pos[2]**2)
    v_mag_sqr = vel[0]**2 + vel[1]**2 + vel[2]**2
    return pos[0]**2 + pos[1]**2 + (2*(1-mu)/d) + (2*mu/r) - v_mag_sqr

def GetZVCGrid(mu,current_jacobi,resolution=0.01):
    xlin = np.arange(-1.5,1.5,resolution)
    ylin = np.arange(-1.5,1.5,resolution)
    Xgrid,Ygrid = np.meshgrid(xlin,ylin)
    return Xgrid,Ygrid,list(map(lambda x, y: current_jacobi-GetJacobiConstant([x,y,0],[0,0,0],mu),Xgrid,Ygrid))

def GetPseudoPotential(pos,mu):
    d = np.sqrt((pos[0]+mu)**2 + pos[1]**2 + pos[2]**2)
    r = np.sqrt((pos[0]-1+mu)**2 + pos[1]**2 + pos[2]**2)
    return (1-mu)/d + mu/r + (pos[0]**2 + pos[1]**2)/2

def GetPseudoPotentialFirstDers(pos,mu):
    x, y, z = pos

    # d = np.sqrt(np.sum(np.square([x+mu,y,z])))
    # r = np.sqrt(np.sum(np.square([x-1+mu,y,z])))
    d = np.sqrt((pos[0]+mu)**2 + pos[1]**2 + pos[2]**2)
    r = np.sqrt((pos[0]-1+mu)**2 + pos[1]**2 + pos[2]**2)

    Ux = x - ((1-mu)*(x+mu))/(d**3) - (mu*(x-1+mu))/(r**3)
    Uy = y - ((1-mu)*y)/(d**3) - (mu*y)/(r**3)
    Uz = - ((1-mu)*z)/(d**3) - (mu*z)/(r**3)
    return np.array([[Ux],[Uy],[Uz]])

def GetPseudoPotentialSecondDers(pos,mu):
    x, y, z = pos

    d = np.sqrt((pos[0]+mu)**2 + pos[1]**2 + pos[2]**2)
    r = np.sqrt((pos[0]-1+mu)**2 + pos[1]**2 + pos[2]**2)

    Uxx = 1 - (1-mu)/(d**3) - (mu)/(r**3) + (3*(1-mu)*((x+mu)**2))/(d**5) + (3*mu*((x-1+mu)**2))/(r**5)
    Uyy = 1 - (1-mu)/(d**3) - (mu)/(r**3) + (3*(1-mu)*(y**2))/(d**5) + (3*mu*(y**2))/(r**5)
    Uzz = - (1-mu)/(d**3) - (mu)/(r**3) + (3*(1-mu)*(z**2))/(d**5) + (3*mu*(z**2))/(r**5)

    Uxy = (3*(1-mu)*(x+mu)*y)/(d**5) + (3*mu*(x-1+mu)*y)/(r**5)
    Uxz = (3*(1-mu)*(x+mu)*z)/(d**5) + (3*mu*(x-1+mu)*z)/(r**5)
    Uyz = (3*(1-mu)*y*z)/(d**5) + (3*mu*y*z)/(r**5)

    return Uxx,Uyy,Uzz,Uxy,Uxz,Uyz

def GetEigenvalues(U2ders):
    Uxx,Uyy,Uzz,Uxy,Uxz,Uyz = U2ders
    # return np.linalg.eigvals(np.array([[0,0,0,1,0,0],
    #                            [0,0,0,0,1,0],
    #                            [0,0,0,0,0,1],
    #                            [Uxx,Uxy,Uxz,0,2,0],
    #                            [Uxy,Uyy,Uyz,-2,0,0],
    #                            [Uxz,Uyz,Uzz,0,0,0]]))
    return np.linalg.eigvals(np.array([[0,0,1,0],
                                       [0,0,0,1],
                                       [Uxx,Uxy,0,2],
                                       [Uxy,Uyy,-2,0]]))

def GetEigenvaluesEquilateral(mu):
    g = 1 - 27*mu*(1-mu)
    L1 = 0.5 * (-1 + np.sqrt(g + 0j))
    l1 = +np.sqrt(L1 + 0j)
    l2 = -np.sqrt(L1 + 0j)
    L2 = 0.5 * (-1 - np.sqrt(g + 0j))
    l3 = +np.sqrt(L2 + 0j)
    l4 = -np.sqrt(L2 + 0j)
    return l1,l2,l3,l4

def GetVdevLinear_OnlyOsc(x0dev,y0dev,l3,Uxx,Uyy):
    alpha3 = ((l3**2)-Uxx)/(2*l3)
    delta = np.sqrt(Uyy/Uxx + 0j)
    vxdev = y0dev * l3 / alpha3
    vydev = x0dev*alpha3*l3 + vxdev*alpha3*delta - y0dev*l3*delta
    return np.real(vxdev),np.real(vydev)

def GetVdevLinear_OnlyExp(x0dev,y0dev,l1,Uxx,Uyy):
    alpha1 = ((l1**2)-Uxx)/(2*l1)
    delta = np.sqrt(Uyy/Uxx + 0j)
    vxdev = y0dev * l1 / alpha1
    vydev = x0dev*alpha1*l1 + vxdev*alpha1*delta - y0dev*l1*delta
    return np.real(vxdev),np.real(vydev)


def GetLinearPeriod(Uxx,Uyy):
    print(Uxx)
    print(Uyy)
    b1 = 2 - (Uxx*Uyy)/2
    print(b1)
    b2 = np.sqrt(-Uxx*Uyy)
    print(b2)
    s = np.sqrt(b1 + np.sqrt(b1**2 + b2**2))
    print(s)
    return 2*np.pi/s

def GetLinearEccentricity(Uxx,Uyy):
    print(Uxx)
    print(Uyy)
    b1 = 2 - (Uxx*Uyy)/2
    print(b1)
    b2 = np.sqrt(-Uxx*Uyy)
    print(b2)
    s = np.sqrt(b1 + np.sqrt(b1**2 + b2**2))
    b3 = ((s**2)+Uxx)/(2*s)
    return np.sqrt(1 - (b3**-2))

def GetEquilateralPointCoeffs(eigs,U2ders,symbolic=False):
    if symbolic:
        l1,l2,l3,l4 = symbols('l1:5')
        g1,g2,g3,g4 = symbols('g1:5')
    else:
        Uxx,Uyy,_,Uxy,_,_=U2ders
        l1,l2,l3,l4 = eigs

        g1 = GetEquilateralCoeffRatio(l1,Uxx,Uyy,Uxy)
        g2 = GetEquilateralCoeffRatio(l2,Uxx,Uyy,Uxy)
        g3 = GetEquilateralCoeffRatio(l3,Uxx,Uyy,Uxy)
        g4 = GetEquilateralCoeffRatio(l4,Uxx,Uyy,Uxy)

    
    a1,a2,a3,a4 = symbols('a1:5')
    x1,x2,x3,x4 = symbols('x1:5')


    system = [a1+a2+a3+a4-x1,
              g1*a1+g2*a2+g3*a3+g4*a4-x2,
              l1*a1+l2*a2+l3*a3+l4*a4-x3,
              l1*g1*a1+l2*g2*a2+l3*g3*a3+l4*g4*a4-x4]
    return linsolve(system,a1,a2,a3,a4)

def GetEquilateralPointOffsets(eigs,U2ders,x1,x2,removal_target="long"):
    Uxx,Uyy,_,Uxy,_,_=U2ders
    l1,l2,l3,l4 = eigs

    g1 = GetEquilateralCoeffRatio(l1,Uxx,Uyy,Uxy)
    g2 = GetEquilateralCoeffRatio(l2,Uxx,Uyy,Uxy)
    g3 = GetEquilateralCoeffRatio(l3,Uxx,Uyy,Uxy)
    g4 = GetEquilateralCoeffRatio(l4,Uxx,Uyy,Uxy)

    if removal_target == "long":
        a3=0
        a4=0
        a1,a2 = symbols('a1:3')
    elif removal_target == "short":
        a1=0
        a2=0
        a3,a4 = symbols('a3:5')
    elif removal_target == "divergence":
        a1=0
        a3=0
        a2 = Symbol('a2')
        a4 = Symbol('a4')
    
    x3,x4 = symbols('x3:5')

    system = [a1+a2+a3+a4-x1,
              g1*a1+g2*a2+g3*a3+g4*a4-x2,
              l1*a1+l2*a2+l3*a3+l4*a4-x3,
              l1*g1*a1+l2*g2*a2+l3*g3*a3+l4*g4*a4-x4]
    if removal_target == "long":
        return linsolve(system,a1,a2,x3,x4)
    elif removal_target == "short":
        return linsolve(system,a3,a4,x3,x4)
    elif removal_target == "divergence":
        return linsolve(system,a2,a4,x3,x4)



def GetEquilateralCoeffRatio(lam,Uxx,Uyy,Uxy):
    return ((lam**2) - 2*lam - Uxx + Uxy) / ((lam**2) + 2*lam - Uyy + Uxy)