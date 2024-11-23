import numpy as np
from ThreeBodyModel import *
from numpy.linalg import inv
import scipy.integrate

def rungekutta4(f, y0, t, args=(), event_conditions=None, stop_at_events=False):
    n = len(t)
    y = np.zeros((n, len(y0)))
    tvec = np.zeros(n)
    y[0] = y0
    tvec[0] = t[0]
    event_states = []
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(t[i], y[i], *args)
        k2 = f(t[i] + h / 2., y[i] + k1 * h / 2., *args)
        k3 = f(t[i] + h / 2., y[i] + k2 * h / 2., *args)
        k4 = f(t[i] + h, y[i] + k3 * h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
        tvec[i+1] = t[i+1]

        if event_conditions != None:
            for j in range(len(y0)):
                if event_conditions[j] == True:
                    if y[i+1,j] * y[i,j] < 0:
                        event_states.append(y[i+1])
                        if stop_at_events == True:
                            y = y[0:i+2]
                            tvec = tvec[0:i+2]
                            return y, tvec, event_states
    return y, tvec, event_states

# def arnoldrungekutta45(fname,a,b,alpha,tol,hmin,hmax):
#     if a>=b:
#         print("Input 'a' cannot be larger than Input 'b'")
#         return
    # https://arnoldfw.com/pdf/numerical_methods.pdf

def rungekutta45_fehlberg(fun, t0, y0, tf, tol=1e6, h0=0.1, args=None):
    tvec = [t0]
    yvec = [y0]

    h = h0
    tk = t0
    yk = y0
    while tk < tf:
        print("h",h)
        print("yk",yk)
        a = fun(tk, yk, *args)
        print(a)
        k1 = h * fun(tk, yk, *args)
        print("k1",k1)
        k2 = h * fun(tk + h/4, yk + k1/4, *args)
        print("k2",k2)
        k3 = h * fun(tk + h*3/8, yk + k1*3/32 + k2 *9/32, *args)
        print("k3",k3)
        k4 = h * fun(tk + h*12/13, yk + k1*1932/2197 - k2*7200/2197 + k3*7296/2197, *args)
        print("k4",k4)
        k5 = h * fun(tk + h, yk + k1*439/216 - k2*8 + k3*3680/513 - k4*845/4104, *args)
        print("k5",k5)
        k6 = h * fun(tk + h/2, yk - k1*8/27 + k2*2 - k3*3544/2565 + k4*1859/4104 - k5*11/40, *args)
        print("k6",k6)

        Yk1_rk4 = yk + k1*25/216 + k3*1408/2565 + k4*2197/4101 - k5/5
        Yk1_rk5 = yk + k1*16/135 + k3*6656/12825 + k4*28561/56430 - k5*9/50 + k6*2/55

        # s = ((tol*h)/(2*np.abs(zk1-yk1)))**(1/4)
        # s = 0.84*((tol*h)/(np.abs(zk1-yk1)))**(1/4)
        R = np.abs(Yk1_rk5-Yk1_rk4)/h
        print("R", R)
        if max(R)!=0:
            s = 0.84*(tol/max(R))**(1/4)


        if max(R) <= tol:
            tk = tk + h
            tvec.append(tk)
            yk = Yk1_rk4
            yvec.append(yk)

        h = s * h
        if np.isnan(h):
            raise ValueError(h)

    return tvec,yvec



# def rungekutta45(f, t0, y0, tf, tol=1e-6, args=None):
#     """
#     Adaptive step-size fourth-fifth-order Runge-Kutta-Fehlberg method for solving a first-order ODE.
    
#     Parameters:
#         f: Function representing the derivative dy/dt = f(t, y, args).
#         t0: Initial value of the independent variable.
#         y0: Initial value of the dependent variable.
#         tf: Final value of the independent variable.
#         tol: Tolerance for controlling the error.
        
#     Returns:
#         A list of tuples (t, y) containing the values of the independent
#         and dependent variables at each step.
#     """
#     results = [(t0, y0)]
#     t = t0
#     y = y0
#     h = 0.1  # Initial step size
    
#     while t < tf:
#         # Calculate the fourth-order Runge-Kutta step
#         k1 = h * f(t, y, *args)
#         k2 = h * f(t + h/5, y + k1/5, *args)
#         k3 = h * f(t + 3*h/10, y + 3*k1/40 + 9*k2/40, *args)
#         k4 = h * f(t + 4*h/5, y + 44*k1/45 - 56*k2/15 + 32*k3/9, *args)
#         k5 = h * f(t + 8*h/9, y + 19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729, *args)
#         k6 = h * f(t + h, y + 9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656, *args)
        
#         y4 = y + (9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656)  # Fifth-order
        
#         # Calculate the fifth-order Runge-Kutta step
#         k7 = h * f(t + h/2, y + 35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84, *args)
        
#         y5 = y + (5179*k1/57600 + 7571*k3/16695 + 393*k4/640 - 92097*k5/339200 + 187*k6/2100 + 1*k7/40) * h  # Fourth-order


        
#         # Calculate the error
#         error = np.abs(y5 - y4)
        
#         # Check if the error is within tolerance
#         if (error <= tol).all():
#             # Accept the fifth-order step
#             y = y5
#             t += h
#             results.append((t, y))
        
#         # Adjust the step size
#         if (error == 0).any():
#             error += 1e-15
#         delta = 0.9 * (tol / error) ** 0.2  # Safety factor
#         h *= delta if (delta < 2.0).all() else 2.0   # Limit increase to factor of 2
    
#     return results


def integrate_manual(function, X0, t0, tf, timesteps, method="RK4", args=()):
    Xvec = [X0]
    tvec = [t0]
    Xi = X0
    ti = t0
    dt = (tf-t0)/timesteps
    for i in range(timesteps):
        ti += dt
        tvec.append(ti)
        if method == "RK4":
            Xi1 = rungekutta4(function, Xi, ti, args)
        # elif method == "RK45":
            # Xi1 = runge_kutta_fehlberg(function,)
        else:
            raise ValueError(f"Method '{method}' is not recognized.")
        Xvec.append(Xi1)
    Xvec = np.array(Xvec)
    tvec = np.array(tvec)
    return Xvec,tvec

def runge_kutta_78(f, t0, y0, h, tf):
    """
    Runge-Kutta 7/8 method for solving a first-order ODE.

    Parameters:
        f: function
            The derivative function, dy/dt = f(t, y).
        t0: float
            Initial value of the independent variable.
        y0: float or numpy array
            Initial value(s) of the dependent variable(s).
        h: float
            Step size.
        tf: float
            Final value of the independent variable.

    Returns:
        t: numpy array
            Array of independent variable values.
        y: numpy array
            Array of corresponding dependent variable values.
    """
    t_values = [t0]
    y_values = [y0]

    t = t0
    y = y0

    while t < tf:
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        k5 = h * f(t + h/2, y + k1/2)
        k6 = h * f(t + h, y + k5)
        k7 = h * f(t + h, y + k6)

        y += (k1 + 2*k2 + 2*k3 + k4 + k5 + k6 + k7)/7
        t += h

        t_values.append(t)
        y_values.append(y)

    return np.array(t_values), np.array(y_values)

def GetQuadraticRoots(a,b,c):
    d = (b**2)-(4*a*c)
    root1 = (-b-np.sqrt(d))/(2*a)
    root2 = (-b+np.sqrt(d))/(2*a)
    return root1,root2

def STM_finite_forward_diff(X0,t0,tf,ode=CR3BP_nondim,ode_args=None,epsilon=1.0e-8):
    if epsilon < 1e-9:
        print("To ensure that rounding errors do not impact accuracy, keep the perturbation value >= 1e-9")
    n = len(X0)
    sol,_,_ = rungekutta4(ode, X0, [t0,tf], args=ode_args)
    Xf = sol[-1]
    STM = np.zeros((6,6))
    for i in range(n):
        di = np.zeros(n)
        di[i] = epsilon
        sol,_,_ = rungekutta4(ode, X0+di, [t0,tf], args=ode_args)
        Xfi = sol[-1]
        phii = (Xfi-Xf)/epsilon
        # STM.append(phii)
        STM[:,i] = phii
    return STM

# ####################################################################

def STM_coupled_integration_vec(initial_state,times,ode=CR3BP_CoupledSTMFunc,ode_args=None):
    n_times = len(times)
    n_state_vars = len(initial_state)
    states = [initial_state]
    STMs = [np.identity(n_state_vars)]
    for i in range(1,n_times):
        Yim1 = [*states[-1],*STMs[-1].flatten()]
        # Yim1 = [*states[-1],*np.identity(n_state_vars).flatten()]
        t_points = [times[i-1],times[i]]
        sol,_,_ = rungekutta4(ode, Yim1, t_points, args=ode_args)
        soli = sol[-1]
        Xi = soli[:6]
        states.append(Xi)
        STMi = soli[6:].reshape(6,6)
        STMs.append(STMi)
    return states, STMs

# def newton_raphson_func(f, df, x0, tol=1e-5):
#     x1 = x0 - f(x0)/df(x0)
#     if abs(x1-x0) > tol:
#         return newton_raphson_func(f, df, x1, tol)
#     else:
#         return x1
    
def newton_raphson_func(f, df, x0, tol=1e-5, f_args=None,df_args=None):
    x1 = x0 - f(x0,*f_args)/df(x0,df_args)
    if abs(x1-x0) > tol:
        return newton_raphson_func(f, df, x1, tol,f_args,df_args)
    else:
        return x1
    
def newton_raphson_nonsquare(f, df, x0, tol=1e-5):
    df_inv = np.transpose(df(x0))
    x1 = x0 - (np.transpose(df(x0)))*f(x0)

def min_norm(mat):
    mat_trans = np.transpose(mat)
    return mat_trans @ inv(mat @ mat_trans)
    
    
# f = lambda x: x**2 - 2
# f_prime = lambda x: 2*x
# initial_guess = 1.5
# estimate = newton_raphson(f, f_prime, initial_guess)
# print(estimate)