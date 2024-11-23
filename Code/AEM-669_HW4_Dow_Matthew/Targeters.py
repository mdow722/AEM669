import numpy as np
from ThreeBodyModel import *
from ODESolving import *
from numpy.linalg import inv,norm

FIXED = (0,0)
FREE = None

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def is_square(mat):
    if isinstance(mat,np.ndarray) == False:
        mat = np.array(mat)
    return len(mat.shape) == 2 and mat.shape[0] == mat.shape(1)

def GetSolutionInitialState(sols):
    final_sol = sols[-1]
    initial_state = final_sol[0]
    return initial_state

def TargetFinalStateFixedTime(initial_state_guess,mu,target_cond,time_of_flight_guess,max_tof=None,timesteps=100,tol=1e-3,max_iters=300,fix_time=True,initial_constraints=None,use_coupled_integration=True):
    if fix_time == False:
        assert(max_tof != None)

    sols = []
    yvec, _, _ = rungekutta4(CR3BP_nondim, initial_state_guess, [0,time_of_flight_guess], args=(mu,))
    Amat_baseline = GetAMatrix_CR3BP(yvec[-1,:3],mu,dim=3)
    for cond in target_cond:
        if cond == 0:
            cond = 1e-5
    error_vec = np.ones(len(initial_state_guess)) + tol

    def DiffFromTarget(final_state):
        all_errors = np.zeros(len(final_state))
        for cond_index in range(len(target_cond)):
            if target_cond[cond_index] == None:
                all_errors[cond_index] = tol * 1e-2
            else:
                all_errors[cond_index] = final_state[cond_index] - target_cond[cond_index]
        return np.array([*all_errors]).reshape(6,1)
    
    XT0_baseline = np.array([*initial_state_guess,0]).reshape(7,1)
    XT1 = np.array([*initial_state_guess,0]).reshape(7,1)
    final_state = None
    time_of_flight = time_of_flight_guess
    # baselineSTM = None

    i = 0
    target_achieved = False
    while (error_vec > tol).any() and i < max_iters:
        print("currently on iteration",i)
        # Set initial state of current iteration
        XT0 = XT1
        if i > 0:
            if fix_time == False:
                time_of_flight += XT1[-1][0]
                if time_of_flight > max_tof:
                    time_of_flight = max_tof

            for cons_index in range(len(initial_constraints)):
                if initial_constraints[cons_index] == FIXED:
                    XT0[cons_index,0] = initial_state_guess[cons_index]
                elif initial_constraints[cons_index] != FREE:
                    XT0[cons_index,0] = clamp(XT0[cons_index,0],initial_constraints[cons_index][0],initial_constraints[cons_index][1])

        X0 = [*XT0[:-1].flatten()]        
        # print("X0",X0)
        # if (norm(X0[3:]) > 5).any():
        #     print("norm(X0[3:]) outside limits")
        #     break
        # print("initial vel",X0[3:])

        # Get new tvec
        tvec = np.linspace(0,time_of_flight,timesteps)

        # Propagate state and STM
        if use_coupled_integration == True:    
            # states, _, _ = rungekutta4(CR3BP_nondim, X0, tvec, args=(mu,))
            states,STMs = STM_coupled_integration_vec(X0,tvec,ode=CR3BP_CoupledSTMFunc,ode_args=(mu,Amat_baseline))
            # states,STMs = STM_coupled_integration_vec(X0,tvec,ode=CR3BP_CoupledSTMFunc_Simple,ode_args=(mu,))
            # _,STMs = STM_coupled_integration_vec(X0,[0,time_of_flight],ode=CR3BP_CoupledSTMFunc,ode_args=(mu,))
            final_STM = STMs[-1]
            # if i == 0:
            #     baselineSTM = STMs[-1]
        else:
            states, _, _ = rungekutta4(CR3BP_nondim, X0, tvec, args=(mu,))
            final_STM = STM_finite_forward_diff(X0,tvec[0], tvec[-1], CR3BP_nondim,(mu,),1e-5)
            # if i == 0:
            #     baselineSTM = final_STM
        sols.append(states)
        final_state = states[-1]
        # Get F(XT0)
        Fx = DiffFromTarget(final_state)
        
        # Get Jinv
        Xdott = CR3BP_nondim(0,final_state,mu).reshape(6,1)
        # print("Xdott",Xdott)
        # print("final_STM",final_STM)
        J = np.hstack((final_STM,Xdott))
        # print("J",J)
        if (Xdott > 1e3).any():
            print("Xdott outside limits")
            break
        # J = np.hstack((baselineSTM,Xdott))
        Jtrans = np.transpose(J)
        try:
            Jinv = Jtrans@inv(J@Jtrans)
        except:
            print("Jinv failed")
            break

        # Get XT1
        # dXT0 = XT0 - XT0_baseline
        # dXT1 = dXT0 - Jinv@Fx
        # XT1 = dXT1 + XT0_baseline
        XT1 = XT0 - Jinv@Fx

        # Get Error
        error_vec = abs(XT1 - XT0)
        if (error_vec > tol).any() == False:
            print("Target Achieved")
            target_achieved = True
            print("Final State: ", final_state)
        i = i+1

    solution_initial_state = GetSolutionInitialState(sols)
    return target_achieved, final_state, time_of_flight, sols, solution_initial_state
    
def FixedTimeSingleShooting_3D(free_vars_initial_guess,F,DF,max_iters=500,tol=1e-12):
    Xi = free_vars_initial_guess
    Fmag = tol + 1
    Xvec = []
    Fvec = []

    i = 0
    while Fmag > tol and i < max_iters:
        Fi = np.array(F(Xi))
        Fmag = norm(Fi)
        Fvec.append(Fmag)
        DFi = np.array(DF(Xi))
        if is_square(DFi):
            Jinv = inv(DFi)
        else:
            Jinv = min_norm(DFi)
        Xi = Xi - Jinv@Fi
        Xvec.append(Xi)
        i += 1

    return Xvec,Fvec,i


