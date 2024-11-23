import numpy as np
from scipy.optimize import newton
from sympy import *
import ODESolving

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

def GetSTMDer(t,STM,X,mu):
    U2ders = GetPseudoPotentialSecondDers(X[:3],mu)
    Uxx,Uyy,Uzz,Uxy,Uxz,Uyz = U2ders
    return [STM[2],STM[6],STM[10],STM[14],
            STM[3],STM[7],STM[11],STM[15],
            Uxx*STM[0]+Uxy*STM[1]+2*STM[3],
            Uxx*STM[4]+Uxy*STM[5]+2*STM[7],
            Uxx*STM[8]+Uxy*STM[9]+2*STM[11],
            Uxx*STM[12]+Uxy*STM[13]+2*STM[15],
            Uxy*STM[0]+Uyy*STM[1]-2*STM[2],
            Uxy*STM[4]+Uyy*STM[5]-2*STM[6],
            Uxy*STM[8]+Uyy*STM[9]-2*STM[10],
            Uxy*STM[12]+Uyy*STM[13]-2*STM[14]]

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

def GetVelFromJacobiAndPos(pos,jacobi,mu):
    # d = np.sqrt((pos[0]+mu)**2 + pos[1]**2 + pos[2]**2)
    # r = np.sqrt((pos[0]-1+mu)**2 + pos[1]**2 + pos[2]**2)
    def jacobi_func(vel_mag,pos,mu,jacobi):
        return jacobi - GetJacobiConstant(pos,[vel_mag,0,0],mu)
    return newton(func=jacobi_func, x0 = 0.1, args=(pos,mu,jacobi))

def GetZVCGrid_xy(mu,current_jacobi,resolution=0.01,xlimits=[-1.5,1.5],ylimits=[-1.5,1.5]):
    xlin = np.arange(*xlimits,resolution)
    ylin = np.arange(*ylimits,resolution)
    Xgrid,Ygrid = np.meshgrid(xlin,ylin)
    return Xgrid,Ygrid,list(map(lambda x, y: current_jacobi-GetJacobiConstant([x,y,0],[0,0,0],mu),Xgrid,Ygrid))

def GetZVCGrid_xvx(mu,current_jacobi,resolution=0.01,xlimits=[-1.5,1.5],vxlimits=[-1.5,1.5]):
    xlin = np.arange(*xlimits,resolution)
    vxlin = np.arange(*vxlimits,resolution)
    Xgrid,VXgrid = np.meshgrid(xlin,vxlin)
    return Xgrid,VXgrid,list(map(lambda x, vx: current_jacobi-GetJacobiConstant([x,0,0],[vx,0,0],mu),Xgrid,VXgrid))

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

def GetAMatrix_CR3BP(pos, mu, dim=2):
    Uxx,Uyy,Uzz,Uxy,Uxz,Uyz = GetPseudoPotentialSecondDers(pos,mu)

    if dim == 2:
        return np.array([[0,0,1,0],
                [0,0,0,1],
                [Uxx,Uxy,0,2],
                [Uxy,Uyy,-2,0]])
    elif dim == 3:
        return np.array([[0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
                [Uxx,Uxy,Uxz,0,2,0],
                [Uxy,Uyy,Uyz,-2,0,0],
                [Uxz,Uyz,Uzz,0,0,0]])
    else:
        raise ValueError(dim)
    
def CR3BP_CoupledSTMFunc(t,X0,mu,tf):
    xdot0 = CR3BP_nondim(0, X0[:6], mu)
    STM = X0[6:].reshape(6,6)

    sol,_,_ = ODESolving.rungekutta4(CR3BP_nondim, X0[:6], [0,t], args=(mu,))
    Amat = GetAMatrix_CR3BP(sol[-1,:3],mu,dim=3)
    # Amat = GetAMatrix_CR3BP(X0[:3],mu,dim=3)

    phi11 = np.dot(Amat[0,:],STM[:,0])
    phi21 = np.dot(Amat[1,:],STM[:,0])
    phi31 = np.dot(Amat[2,:],STM[:,0])
    phi41 = np.dot(Amat[3,:],STM[:,0])
    phi51 = np.dot(Amat[4,:],STM[:,0])
    phi61 = np.dot(Amat[5,:],STM[:,0])

    phi12 = np.dot(Amat[0,:],STM[:,1])
    phi22 = np.dot(Amat[1,:],STM[:,1])
    phi32 = np.dot(Amat[2,:],STM[:,1])
    phi42 = np.dot(Amat[3,:],STM[:,1])
    phi52 = np.dot(Amat[4,:],STM[:,1])
    phi62 = np.dot(Amat[5,:],STM[:,1])

    phi13 = np.dot(Amat[0,:],STM[:,2])
    phi23 = np.dot(Amat[1,:],STM[:,2])
    phi33 = np.dot(Amat[2,:],STM[:,2])
    phi43 = np.dot(Amat[3,:],STM[:,2])
    phi53 = np.dot(Amat[4,:],STM[:,2])
    phi63 = np.dot(Amat[5,:],STM[:,2])

    phi14 = np.dot(Amat[0,:],STM[:,3])
    phi24 = np.dot(Amat[1,:],STM[:,3])
    phi34 = np.dot(Amat[2,:],STM[:,3])
    phi44 = np.dot(Amat[3,:],STM[:,3])
    phi54 = np.dot(Amat[4,:],STM[:,3])
    phi64 = np.dot(Amat[5,:],STM[:,3])

    phi15 = np.dot(Amat[0,:],STM[:,4])
    phi25 = np.dot(Amat[1,:],STM[:,4])
    phi35 = np.dot(Amat[2,:],STM[:,4])
    phi45 = np.dot(Amat[3,:],STM[:,4])
    phi55 = np.dot(Amat[4,:],STM[:,4])
    phi65 = np.dot(Amat[5,:],STM[:,4])

    phi16 = np.dot(Amat[0,:],STM[:,5])
    phi26 = np.dot(Amat[1,:],STM[:,5])
    phi36 = np.dot(Amat[2,:],STM[:,5])
    phi46 = np.dot(Amat[3,:],STM[:,5])
    phi56 = np.dot(Amat[4,:],STM[:,5])
    phi66 = np.dot(Amat[5,:],STM[:,5])
    # return np.array([*xdot0,
    #         phi11,phi21,phi31,phi41,phi51,phi61,
    #         phi12,phi22,phi32,phi42,phi52,phi62,
    #         phi13,phi23,phi33,phi43,phi53,phi63,
    #         phi14,phi24,phi34,phi44,phi54,phi64,
    #         phi15,phi25,phi35,phi45,phi55,phi65,
    #         phi16,phi26,phi36,phi46,phi56,phi66])
    return np.array([*xdot0,
            phi11,phi12,phi13,phi14,phi15,phi16,
            phi21,phi22,phi23,phi24,phi25,phi26,
            phi31,phi32,phi33,phi34,phi35,phi36,
            phi41,phi42,phi43,phi44,phi45,phi46,
            phi51,phi52,phi53,phi54,phi55,phi56,
            phi61,phi62,phi63,phi64,phi65,phi66])

def CR3BP_CoupledSTMFunc_Simple(t,X0,mu):
    current_state = X0[:6]
    xdot0 = CR3BP_nondim(0, X0[:6], mu)
    # sol,_,_ = ODESolving.rungekutta4(CR3BP_nondim, X0[:6], [0,t], args=(mu,))
    # Uxx,Uyy,Uzz,Uxy,Uxz,Uyz = GetPseudoPotentialSecondDers(sol[-1,:3],mu)
    Uxx,Uyy,Uzz,Uxy,Uxz,Uyz = GetPseudoPotentialSecondDers(current_state[:3],mu)

    STM = X0[6:].reshape(6,6)
    # dSTM = Amat @ STM
    dSTM_11 = STM[3,0]
    dSTM_12 = STM[3,1]
    dSTM_13 = STM[3,2]
    dSTM_14 = STM[3,3]
    dSTM_15 = STM[3,4]
    dSTM_16 = STM[3,5]

    dSTM_21 = STM[4,0]
    dSTM_22 = STM[4,1]
    dSTM_23 = STM[4,2]
    dSTM_24 = STM[4,3]
    dSTM_25 = STM[4,4]
    dSTM_26 = STM[4,5]

    dSTM_31 = STM[5,0]
    dSTM_32 = STM[5,1]
    dSTM_33 = STM[5,2]
    dSTM_34 = STM[5,3]
    dSTM_35 = STM[5,4]
    dSTM_36 = STM[5,5]

    dSTM_41 = Uxx*STM[0,0]+Uxy*STM[1,0]+Uxz*STM[2,0]+2*STM[4,0]
    dSTM_42 = Uxx*STM[0,1]+Uxy*STM[1,1]+Uxz*STM[2,1]+2*STM[4,1]
    dSTM_43 = Uxx*STM[0,2]+Uxy*STM[1,2]+Uxz*STM[2,2]+2*STM[4,2]
    dSTM_44 = Uxx*STM[0,3]+Uxy*STM[1,3]+Uxz*STM[2,3]+2*STM[4,3]
    dSTM_45 = Uxx*STM[0,4]+Uxy*STM[1,4]+Uxz*STM[2,4]+2*STM[4,4]
    dSTM_46 = Uxx*STM[0,5]+Uxy*STM[1,5]+Uxz*STM[2,5]+2*STM[4,5]

    dSTM_51 = Uxy*STM[0,0]+Uyy*STM[1,0]+Uyz*STM[2,0]-2*STM[3,0]
    dSTM_52 = Uxy*STM[0,1]+Uyy*STM[1,1]+Uyz*STM[2,1]-2*STM[3,1]
    dSTM_53 = Uxy*STM[0,2]+Uyy*STM[1,2]+Uyz*STM[2,2]-2*STM[3,2]
    dSTM_54 = Uxy*STM[0,3]+Uyy*STM[1,3]+Uyz*STM[2,3]-2*STM[3,3]
    dSTM_55 = Uxy*STM[0,4]+Uyy*STM[1,4]+Uyz*STM[2,4]-2*STM[3,4]
    dSTM_56 = Uxy*STM[0,5]+Uyy*STM[1,5]+Uyz*STM[2,5]-2*STM[3,5]

    dSTM_61 = Uxz*STM[0,0]+Uyz*STM[1,0]+Uzz*STM[2,0]
    dSTM_62 = Uxz*STM[0,1]+Uyz*STM[1,1]+Uzz*STM[2,1]
    dSTM_63 = Uxz*STM[0,2]+Uyz*STM[1,2]+Uzz*STM[2,2]
    dSTM_64 = Uxz*STM[0,3]+Uyz*STM[1,3]+Uzz*STM[2,3]
    dSTM_65 = Uxz*STM[0,4]+Uyz*STM[1,4]+Uzz*STM[2,4]
    dSTM_66 = Uxz*STM[0,5]+Uyz*STM[1,5]+Uzz*STM[2,5]
    dSTM = np.array([[dSTM_11,dSTM_12,dSTM_13,dSTM_14,dSTM_15,dSTM_16],
                      [dSTM_21,dSTM_22,dSTM_23,dSTM_24,dSTM_25,dSTM_26],
                      [dSTM_31,dSTM_32,dSTM_33,dSTM_34,dSTM_35,dSTM_36],
                      [dSTM_41,dSTM_42,dSTM_43,dSTM_44,dSTM_45,dSTM_46],
                      [dSTM_51,dSTM_52,dSTM_53,dSTM_54,dSTM_55,dSTM_56],
                      [dSTM_61,dSTM_62,dSTM_63,dSTM_64,dSTM_65,dSTM_66]])
    # print("dSTM",dSTM)
    # print("dSTM",dSTM.flatten())
    return np.array([*xdot0,*dSTM.flatten()])


def GetEigenvalues(Amatrix):
    return np.linalg.eigvals(Amatrix)

def GetEigenvaluesAndVectors(Amatrix):
    return np.linalg.eig(Amatrix)

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

