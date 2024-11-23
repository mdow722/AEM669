import numpy as np
from ThreeBodyModel import *

def rungekutta4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(t[i], y[i], *args)
        k2 = f(t[i] + h / 2., y[i] + k1 * h / 2., *args)
        k3 = f(t[i] + h / 2., y[i] + k2 * h / 2., *args)
        k4 = f(t[i] + h, y[i] + k3 * h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

# def arnoldrungekutta45(fname,a,b,alpha,tol,hmin,hmax):
#     if a>=b:
#         print("Input 'a' cannot be larger than Input 'b'")
#         return
    # https://arnoldfw.com/pdf/numerical_methods.pdf

# def rungekutta45()
def rungekutta45(f, t0, y0, tf, tol=1e-6, args=None):
    """
    Adaptive step-size fourth-fifth-order Runge-Kutta-Fehlberg method for solving a first-order ODE.
    
    Parameters:
        f: Function representing the derivative dy/dt = f(t, y, args).
        t0: Initial value of the independent variable.
        y0: Initial value of the dependent variable.
        tf: Final value of the independent variable.
        tol: Tolerance for controlling the error.
        
    Returns:
        A list of tuples (t, y) containing the values of the independent
        and dependent variables at each step.
    """
    results = [(t0, y0)]
    t = t0
    y = y0
    h = 0.1  # Initial step size
    
    while t < tf:
        # Calculate the fourth-order Runge-Kutta step
        k1 = h * f(t, y, *args)
        k2 = h * f(t + h/5, y + k1/5, *args)
        k3 = h * f(t + 3*h/10, y + 3*k1/40 + 9*k2/40, *args)
        k4 = h * f(t + 4*h/5, y + 44*k1/45 - 56*k2/15 + 32*k3/9, *args)
        k5 = h * f(t + 8*h/9, y + 19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729, *args)
        k6 = h * f(t + h, y + 9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656, *args)
        
        y4 = y + (9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656)  # Fifth-order
        
        # Calculate the fifth-order Runge-Kutta step
        k7 = h * f(t + h/2, y + 35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84, *args)
        
        y5 = y + (5179*k1/57600 + 7571*k3/16695 + 393*k4/640 - 92097*k5/339200 + 187*k6/2100 + 1*k7/40) * h  # Fourth-order


        
        # Calculate the error
        error = np.abs(y5 - y4)
        
        # Check if the error is within tolerance
        if (error <= tol).all():
            # Accept the fifth-order step
            y = y5
            t += h
            results.append((t, y))
        
        # Adjust the step size
        if (error == 0).any():
            error += 1e-15
        delta = 0.9 * (tol / error) ** 0.2  # Safety factor
        h *= delta if (delta < 2.0).all() else 2.0   # Limit increase to factor of 2
    
    return results


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


# def newton_raphson(f, jac, x0, tol=1e-6):
#     if abs(f(x0)) > tol:
#         # x0 = x0 - np.linalg.inv(jac)*
#         # return newton_raphson(f, df, x0 - f(x0)/df(x0), tol)
#     else:
#         return x0
    
    
# f = lambda x: x**2 - 2
# f_prime = lambda x: 2*x
# initial_guess = 1.5
# estimate = newton_raphson(f, f_prime, initial_guess)
# print(estimate)