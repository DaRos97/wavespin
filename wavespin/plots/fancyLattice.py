""" My version of plotting a lattice.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from wavespin.tools import pathFinder as pf
from wavespin.tools import functions as fs
from pathlib import Path

def plotSitesGrid(system,**kwargs):
    """ Here we plot the grid structure to see which sites are considered in the calculation.

    Parameters
    ----------
    system : object.
    """
    Lx = system.Lx
    Ly = system.Ly
    offSiteList = system.offSiteList
    indexesMap = system.indexesMap
    printIndices = kwargs.get('indices',True)
    #
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    for ix in range(Lx):
        for iy in range(Ly):
            if (ix,iy) in offSiteList:
                ax.scatter(ix,iy,c='r',marker='x',s=80,zorder=2)
            else:
                ax.scatter(ix,iy,c='b',marker='o',s=80,zorder=2)
                if printIndices:
                    ax.text(ix+0.05,iy+0.15,str(indexesMap.index((ix,iy))),size=20)
            if ix+1<Lx:
                if (ix,iy) in offSiteList or (ix+1,iy) in offSiteList:
                    ax.plot([ix,ix+1],[iy,iy],c='y',ls='--',lw=0.5,zorder=-1)
                else:
                    ax.plot([ix,ix+1],[iy,iy],c='darkgreen',ls='-',lw=2,zorder=-1)
            if iy+1<Ly:
                if (ix,iy) in offSiteList or (ix,iy+1) in offSiteList:
                    ax.plot([ix,ix],[iy,iy+1],c='y',ls='--',lw=0.5,zorder=-1)
                else:
                    ax.plot([ix,ix],[iy,iy+1],c='darkgreen',lw=2,zorder=-1)
    if hasattr(system,'perturbationSite'):
        perturbationSite = system.perturbationSite
        ax.scatter(perturbationSite[0],perturbationSite[1],c='w',edgecolor='m',lw=2,marker='o',s=200,zorder=1)
    ax.set_aspect('equal')
    ax.set_xlabel('x',size=30)
    ax.set_ylabel('y',size=30)
    fig.tight_layout()
    plt.show()

def solutionMC(sim,**kwargs):
    """ Plot theta and phi of each site of the MC solution.
    """
    thetas = np.zeros(sim.Ns)
    phis = np.zeros(sim.Ns)
    for i in range(sim.Ns):
        thetas[i], phis[i] = fs.vector_to_polar_angles(sim.Spins[i])
    fig = plt.figure(figsize=(15,10))
    X,Y = np.meshgrid(np.arange(sim.Lx),np.arange(sim.Ly),indexing='ij')
    # 1
    ax = fig.add_subplot(221,projection='3d')
    ax.plot_surface(X,Y,thetas.reshape(sim.Lx,sim.Ly),cmap='plasma')
    ax.set_title("Polar angle",size=20)
    #2
    ax = fig.add_subplot(222,projection='3d')
    ax.plot_surface(X,Y,phis.reshape(sim.Lx,sim.Ly),cmap='plasma')
    ax.set_title("Azimuthal angle",size=20)
    #3
    ax = fig.add_subplot(223,projection='3d')
    ax.plot_surface(X,Y,
                    abs(thetas-np.pi/2).reshape(sim.Lx,sim.Ly),
                    cmap='plasma',
                    alpha=0.8
                    )
    theta = sim.periodicTheta
    ax.plot_surface(X,Y,
                    np.ones((sim.Lx,sim.Ly))*theta,
                    color='green',
                    alpha=0.3
                    )
    ax.set_title("Polar angle deviations from pi/2",size=20)
    #4
    ax = fig.add_subplot(224,projection='3d')
    func = phis.reshape(sim.Lx,sim.Ly)
    func -= (np.max(func)-np.min(func))/2 + np.min(func)
    ax.plot_surface(X,Y,abs(func)-np.pi/2,cmap='plasma')
    ax.set_title("Azimuthal angle deviations",size=20)

    # Other stuff
    plt.suptitle("Solution of Montecarlo simulation",size=20)
    saveFigure = kwargs.get('saveFigure',False)
    if saveFigure:
        argsFn = ('MC_solution',sim.Lx,sim.Ly,sim.Ns,sim.g1,sim.g2,sim.d1,sim.d2,sim.h)
        figureDn = pf.getHomeDirname(str(Path.cwd()),'Figures/')
        figureFn = pf.getFilename(*argsFn,dirname=figureDn,extension='.png')
        if not Path(figureDn).is_dir():
            print("Creating 'Figures/' folder in home directory.")
            os.system('mkdir '+figureDn)
        fig.savefig(figureFn)

    plt.show()


