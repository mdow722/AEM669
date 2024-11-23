import matplotlib.pyplot as plt
from ThreeBodyModel import GetJacobiConstant,GetZVCGrid
import numpy as np

def Plot_CR3BP(vector_of_trajectories,mu,Duration,show_initial_pos=True,show_final_pos=True,show_bodies=True,show_ZVC=True,xlimits=(-1.5,1.5),ylimits=(-1.5,1.5),names=None):
    fig, ax = plt.subplots(figsize=(5,5), dpi=96)

    ax.axhline(0, color='k')
    if show_bodies:
        ax.plot(-mu, 0, 'bo', label="$m_1$")
        ax.plot(1 - mu, 0, 'go', label="$m_2$")

    i = 0
    for trajectory in vector_of_trajectories:
        if isinstance(trajectory,list):
            trajectory = np.array(trajectory)
        if names == None:
            ax.plot(trajectory[:, 0], trajectory[:, 1])
        else:
            assert(len(names)>=len(vector_of_trajectories))
            ax.plot(trajectory[:, 0], trajectory[:, 1], label=names[i])
        if show_initial_pos:
            x0 = trajectory[0,0]
            y0 = trajectory[0,1]
            ax.plot(x0, y0, 'yo', label="Initial Position")
        if show_final_pos:
            xf = trajectory[-1,0]
            yf = trajectory[-1,1]
            ax.plot(xf, yf, 'ro', label="Final Position")
        if show_ZVC:
            jacobi_constant = GetJacobiConstant(trajectory[0,:3],trajectory[0,3:],mu)
            Xgrid,Ygrid,Zgrid = GetZVCGrid(mu,jacobi_constant,resolution=0.001)
            ax.contour(Xgrid,Ygrid,Zgrid,levels=[0])
        i += 1

    ax.set_aspect("equal")
    ax.set(xlim=xlimits, ylim=ylimits)
    ax.set(title=f"Trajectory in the Earth-Moon System over {round(Duration,3)} Days",
           xlabel="X", ylabel="Y")
    plt.legend()
    plt.draw()
    return ax

def Plot_CR3BP_3D(vector_of_trajectories,mu,Duration,show_initial_pos=True,show_final_pos=True,show_bodies=True,show_ZVC=True,xlimits=(-1.5,1.5),ylimits=(-1.5,1.5),names=None):
    ax = plt.figure().add_subplot(projection='3d')

    if show_bodies:
        ax.plot(-mu, 0, 0, 'bo', label="$m_1$")
        ax.plot(1 - mu, 0, 0, 'go', label="$m_2$")

    i = 0
    for trajectory in vector_of_trajectories:
        if isinstance(trajectory,list):
            trajectory = np.array(trajectory)
        if names == None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        else:
            assert(len(names)>=len(vector_of_trajectories))
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=names[i])
        if show_initial_pos:
            x0 = trajectory[0,0]
            y0 = trajectory[0,1]
            z0 = trajectory[0,2]
            ax.plot(x0, y0, z0, 'yo', label="Initial Position")
        if show_final_pos:
            xf = trajectory[-1,0]
            yf = trajectory[-1,1]
            zf = trajectory[-1,2]
            ax.plot(xf, yf, zf, 'ro', label="Final Position")
        # if show_ZVC:
            # jacobi_constant = GetJacobiConstant(trajectory[0,:3],trajectory[0,3:],mu)
            # Xgrid,Ygrid,Zgrid = GetZVCGrid(mu,jacobi_constant,resolution=0.001)
            # ax.contour(Xgrid,Ygrid,Zgrid,levels=[0])

        i += 1

    ax.set_aspect("equal")
    # ax.set(xlim=xlimits, ylim=ylimits)
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    ax.set(title=f"Trajectory in the Earth-Moon System over {round(Duration,3)} Days",
           xlabel="X", ylabel="Y", zlabel="Z")
    plt.legend()
    plt.draw()
    return ax