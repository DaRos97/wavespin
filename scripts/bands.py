""" Here I compute the bands for the two branches of parameters from LSW.
"""
import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

parameters = importParameters()

if 0:       # Compute PBC bands in frustrated branch
    J1 = 1
    h = 0
    J2s = np.linspace(0,0.5,6,endpoint=True)

    Lx = Ly = 20
    parameters.lat_boundary = 'periodic'
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    resPBC = []
    for J2 in J2s:
        print(J2)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        mySystem = openHamiltonian(parameters)
        resPBC.append(mySystem.dispersion)

    fig = plt.figure(figsize=(15,3))
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    # PBC
    axs = []
    zms = []
    zMs = []
    for i in range(len(J2s)):
        ax = fig.add_subplot(1,6,i+1,projection='3d')
        ax.plot_surface(X,Y,resPBC[i],cmap='plasma')
        zmin,zmax = ax.get_zlim()
        zms.append(zmin)
        zMs.append(zmax)
        axs.append(ax)
        ax.set_title(r"$J_2=$%.2f"%J2s[i],size=15)
    for i in range(len(J2s)):
        axs[i].set_zlim(min(zms),max(zMs))
    fig.tight_layout()
    plt.show()
if 1:       # Compute PBC bands in critical branch
    J1 = 1
    J2 = 0
    Hs = [0,1.5,2,2.5,3]

    Lx = Ly = 20
    parameters.lat_boundary = 'periodic'
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    resPBC = []
    for h in Hs:
        print(h)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        mySystem = openHamiltonian(parameters)
        resPBC.append(mySystem.dispersion)

    fig = plt.figure(figsize=(15,4))
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    # PBC
    axs = []
    zms = []
    zMs = []
    h_title = [r"$h=0$",r"$h=1.5$",r"$h=2$",r"$h=2.5$",r"$h=3$"]
    pane_col = (1.0, 0.973, 0.906)#,0.5)
    lsize = 13
    for i in range(len(Hs)):
        ax = fig.add_subplot(1,5,i+1,projection='3d')
        if i in [3,4]:  #Cube of gap
            cH = np.min(resPBC[i])*0.95
            m = 0
            vertices = np.array([
                [m, m, 0],
                [Lx, m, 0],
                [Lx, Ly, 0],
                [m, Ly, 0],
                [m, m, cH],
                [Lx, m, cH],
                [Lx, Ly, cH],
                [m, Ly, cH]
            ])

            # Define the 6 cube faces (each face is a list of vertices)
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[4], vertices[7], vertices[3], vertices[0]]   # left
            ]

            # Create Poly3DCollection
            cube = Poly3DCollection(
                faces,
                facecolors='sandybrown',
                edgecolors='chocolate',
                lw=0.2,
                alpha=0.3,
                zorder=-1
            )

            ax.add_collection3d(cube)
        ax.plot_surface(X,Y,resPBC[i],cmap='plasma',zorder=10)
        # Limits
        zmin,zmax = ax.get_zlim()
        zms.append(zmin)
        zMs.append(zmax)
        axs.append(ax)
        # Remove background
        ax.set_box_aspect([1, 1, 1])
        ax.xaxis.set_pane_color(pane_col)
        ax.yaxis.set_pane_color(pane_col)
        ax.zaxis.set_pane_color(pane_col)
        ax.grid(False)
        # x-y ticks
        ax.set_xticks([0,Lx//2,Lx],[r"$0$",r"$\pi$",r"$2\pi$"])
        ax.set_yticks([0,Ly//2,Ly],[r"$0$",r"$\pi$",r"$2\pi$"])
        ax.tick_params(
            axis='x',
            pad=0,
            labelsize=lsize,
        )
        ax.tick_params(
            axis='y',
            pad=0,
            labelsize=lsize,
        )
        # x-y grid
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        for n in [0,Lx//2-1,Lx-1]:
            ax.plot([n,n],[ymin,ymax],[0,0],color='gray',lw=0.5,zorder=-1)
            ax.plot([xmin,xmax],[n,n],[0,0],color='gray',lw=0.5,zorder=-1)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        # z-labels
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(
            axis='z',
            labelsize=lsize,
            pad=0
        )
        # Title
        fc = 'coral' if i==2 else 'white'
        ax.set_title(h_title[i],
                     size=20,
                     bbox=dict(facecolor=fc,    # box fill color
                               edgecolor='black',   # border color
                               boxstyle='round,pad=0.3')
                     )
        #ax.set_title(r"$h=$%.2f"%Hs[i],size=15)
    for i in range(len(Hs)):
        axs[i].set_zlim(min(zms),max(zMs))
    #fig.tight_layout()
    plt.subplots_adjust(
    #    bottom = 0.014,
    #    top = 0.944,
        right = 0.98,
        left = 0.,
        wspace=0.025
    )
    plt.show()
