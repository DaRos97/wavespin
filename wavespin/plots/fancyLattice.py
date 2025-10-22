""" My version of plotting a lattice.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from wavespin.tools import pathFinder as pf
from wavespin.tools import functions as fs
from pathlib import Path

def plotSitesGrid(lattice,**kwargs):
    """ Here we plot the grid structure to see which sites are considered in the calculation.

    Parameters
    ----------
    lattice : object.
    """
    Lx = lattice.Lx
    Ly = lattice.Ly
    offSiteList = lattice.offSiteList
    indexesMap = lattice.indexToSite
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
    if hasattr(lattice,'perturbationSite'):
        perturbationSite = lattice.perturbationSite
        ax.scatter(perturbationSite[0],perturbationSite[1],c='w',edgecolor='m',lw=2,marker='o',s=200,zorder=1)
    ax.set_aspect('equal')
    ax.set_xlabel('x',size=30)
    ax.set_ylabel('y',size=30)
    fig.tight_layout()
    plt.show()

def plotQuantizationAngles(sim,thetas,phis,**kwargs):
    """ Plot theta and phi of each site of the lattice.
    """
    verbose = kwargs.get('verbose',False)
    Lx,Ly = (sim.Lx,sim.Ly)
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(121,projection='3d')
    func = thetas.reshape(Lx,Ly)
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    ax.plot_surface(X,Y,func,cmap='plasma')
    #
    ax = fig.add_subplot(122,projection='3d')
    for i in range(sim.Ns):
        ix,iy = sim._xy(i)
        if (ix+iy)%2==1:
            thetas[i] += np.pi
    func2 = thetas.reshape(Lx,Ly)
    ax.plot_surface(X,Y,
                    func2,
                    cmap='plasma',
                    alpha=0.8
                    )
    ax.plot_surface(X,Y,
                    np.ones((Lx,Ly))*sim.periodicTheta,
                    color='g',
                    alpha=0.4
                    )
    if sim.p.savePlotSolution:
        argsFn = (sim.txtSim+'_solution',sim.Lx,sim.Ly,sim.Ns,sim.g1,sim.g2,sim.d1,sim.d2,sim.h,sim.boundary)
        figureFn = pf.getFilename(*argsFn,dirname=sim.figureDn,extension='.png')
        if not Path(sim.figureDn).is_dir():
            print("Creating 'Figures/' folder in home directory.")
            os.system('mkdir '+sim.figureDn)
        if verbose:
            print("Saving picture to file: "+figureFn)
        fig.savefig(figureFn)

    showFig = kwargs.get('showFigure',False)
    if showFig:
        plt.show()


