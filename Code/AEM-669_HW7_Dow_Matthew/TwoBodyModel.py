import numpy as np
from numpy.linalg import norm

def normalize(vector):
    magnitude = norm(vector)
    return vector / magnitude

def TwoBody(t,X,mu):
    x, y, z, vx, vy, vz = X
    r = np.sqrt(np.sum(np.square([x,y,z])))
    coeff = -mu / (r**3)

    ax = coeff * x
    ay = coeff * y
    az = coeff * z

    return np.array([vx,vy,vz,ax,ay,az])

def GetAMatrix(pos,mu,dim=2):
    x, y, z = pos
    r = np.sqrt(np.sum(np.square([x,y,z])))
    r3 = r**3
    r5 = r**5
    A21_11 = -mu/r3 + 3*mu*(x**2)/r5
    A21_12 = 3*mu*x*y/r5
    A21_13 = 3*mu*x*z/r5
    A21_22 = -mu/r3 + 3*mu*(y**2)/r5
    A21_23 = 3*mu*y*z/r5
    A21_33 = -mu/r3 + 3*mu*(z**2)/r5
    if dim == 0:
        return A21_11,A21_12,A21_13,A21_22,A21_23,A21_33
    elif dim == 2:
        return np.array([[0,0,1,0],
                [0,0,0,1],
                [A21_11,A21_12,0,0],
                [A21_12,A21_22,0,0]])
    elif dim == 3:
        return np.array([[0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
                [A21_11,A21_12,A21_13,0,0,0],
                [A21_12,A21_22,A21_23,0,0,0],
                [A21_13,A21_23,A21_33,0,0,0]])
    
def CoupledSTMFunc(t,X0,mu):
    current_state = X0[:6]
    xdot0 = TwoBody(0, X0[:6], mu)
    A21_11,A21_12,A21_13,A21_22,A21_23,A21_33 = GetAMatrix(current_state[:3],mu,dim=0)

    STM = X0[6:].reshape(6,6)
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

    dSTM_41 = A21_11*STM[0,0]+A21_12*STM[1,0]+A21_13*STM[2,0]
    dSTM_42 = A21_11*STM[0,1]+A21_12*STM[1,1]+A21_13*STM[2,1]
    dSTM_43 = A21_11*STM[0,2]+A21_12*STM[1,2]+A21_13*STM[2,2]
    dSTM_44 = A21_11*STM[0,3]+A21_12*STM[1,3]+A21_13*STM[2,3]
    dSTM_45 = A21_11*STM[0,4]+A21_12*STM[1,4]+A21_13*STM[2,4]
    dSTM_46 = A21_11*STM[0,5]+A21_12*STM[1,5]+A21_13*STM[2,5]

    dSTM_51 = A21_12*STM[0,0]+A21_22*STM[1,0]+A21_23*STM[2,0]
    dSTM_52 = A21_12*STM[0,1]+A21_22*STM[1,1]+A21_23*STM[2,1]
    dSTM_53 = A21_12*STM[0,2]+A21_22*STM[1,2]+A21_23*STM[2,2]
    dSTM_54 = A21_12*STM[0,3]+A21_22*STM[1,3]+A21_23*STM[2,3]
    dSTM_55 = A21_12*STM[0,4]+A21_22*STM[1,4]+A21_23*STM[2,4]
    dSTM_56 = A21_12*STM[0,5]+A21_22*STM[1,5]+A21_23*STM[2,5]

    dSTM_61 = A21_13*STM[0,0]+A21_23*STM[1,0]+A21_33*STM[2,0]
    dSTM_62 = A21_13*STM[0,1]+A21_23*STM[1,1]+A21_33*STM[2,1]
    dSTM_63 = A21_13*STM[0,2]+A21_23*STM[1,2]+A21_33*STM[2,2]
    dSTM_64 = A21_13*STM[0,3]+A21_23*STM[1,3]+A21_33*STM[2,3]
    dSTM_65 = A21_13*STM[0,4]+A21_23*STM[1,4]+A21_33*STM[2,4]
    dSTM_66 = A21_13*STM[0,5]+A21_23*STM[1,5]+A21_33*STM[2,5]
    dSTM = np.array([[dSTM_11,dSTM_12,dSTM_13,dSTM_14,dSTM_15,dSTM_16],
                      [dSTM_21,dSTM_22,dSTM_23,dSTM_24,dSTM_25,dSTM_26],
                      [dSTM_31,dSTM_32,dSTM_33,dSTM_34,dSTM_35,dSTM_36],
                      [dSTM_41,dSTM_42,dSTM_43,dSTM_44,dSTM_45,dSTM_46],
                      [dSTM_51,dSTM_52,dSTM_53,dSTM_54,dSTM_55,dSTM_56],
                      [dSTM_61,dSTM_62,dSTM_63,dSTM_64,dSTM_65,dSTM_66]])
    return np.array([*xdot0,*dSTM.flatten()])

def cartesian_to_classical(state,mu,sma_or_h="sma"):

    r = state[:3]
    rnormalized = normalize(r)
    v = state[3:]
    vr = np.dot(r,v)/norm(r)

    h = np.cross(r,v)
    hnormalized = normalize(h)
    hmag = norm(h)

    i = np.arccos(hnormalized[2])

    Khat = np.array([0,0,1])
    N = np.cross(Khat,h)
    Nnormalized = normalize(N)

    raan = np.arccos(Nnormalized[0])
    if N[1] < 0:
        raan = 2*np.pi - raan
        
    e = (1/mu)*(np.cross(v,h) - (mu*rnormalized))
    enormalized = normalize(e)
    emag = norm(e)

    aop = np.arccos(np.dot(Nnormalized,enormalized))
    if e[2] < 0:
        aop = 2*np.pi - aop

    true_anomaly = np.arccos(np.dot(enormalized,rnormalized))
    if vr < 0:
        2*np.pi - true_anomaly

    if sma_or_h == "h":
        return np.array([hmag,emag,i,raan,aop,true_anomaly])

    rp = KeplerOrbitEquation(hmag,mu,emag,0)
    ra = KeplerOrbitEquation(hmag,mu,emag,np.pi)
    a = (rp+ra)/2
    return np.array([a,emag,i,raan,aop,true_anomaly])

def classical_to_cartesian(state,mu,sma_or_h="sma"):
    e,i,raan,aop,true_anomaly = state[1:]
    if sma_or_h == "h":
        h = state[0]
    else:
        h = np.sqrt(state[0]*(1-(e**2))*mu)

    r_peri = KeplerOrbitEquation(h,mu,e,true_anomaly)*np.array([np.cos(true_anomaly),np.sin(true_anomaly),0])
    v_peri = mu / h *np.array([-np.sin(true_anomaly),e + np.cos(true_anomaly),0])
    Q_peri2GCE = GetPerifocalToGeocentricEquatorial(i,raan,aop)
    r = Q_peri2GCE @ r_peri
    v = Q_peri2GCE @ v_peri

    return np.array([*r,*v])

def GetOrbitPeriod(a,mu):
    return 2*np.pi * (a**(3/2)) / np.sqrt(mu)

def KeplerOrbitEquation(h,mu,e,theta):
    return ((h**2) / mu) / (1 + e * np.cos(theta))

def GetPerifocalToGeocentricEquatorial(i,raan,aop):
    Q11 = -np.sin(raan) * np.cos(i) * np.sin(aop) + np.cos(raan) * np.cos(aop)
    Q12 = -np.sin(raan) * np.cos(i) * np.cos(aop) - np.cos(raan) * np.sin(aop)
    Q13 = np.sin(raan) * np.sin(i)
    
    Q21 = np.cos(raan) * np.cos(i) * np.sin(aop) + np.sin(raan) * np.cos(aop)
    Q22 = np.cos(raan) * np.cos(i) * np.cos(aop) - np.sin(raan) * np.sin(aop)
    Q23 = -np.cos(raan) * np.sin(i)

    Q31 = np.sin(i) * np.sin(aop)
    Q32 = np.sin(i) * np.cos(aop)
    Q33 = np.cos(i)

    return np.array([Q11,Q12,Q13,Q21,Q22,Q23,Q31,Q32,Q33]).reshape((3,3))

