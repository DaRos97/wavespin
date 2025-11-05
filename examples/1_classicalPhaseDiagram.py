""" Code for computing numerically using RMOs the classical phase diagram of magnetic orders in the J1-J2 XY model plus staggered H.
Since there are no other magnetic orders then the canted-Neel and canted-Stripe, an analytical plot is actually better.
"""

import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.classicSpins.RMO import *
from wavespin.tools import pathFinder


J1 = 1  # Positive

J2min = 0
J2max = 1

hmin = 0
hmax = 3

D1 = 0      #Not implemented in RMO
D2 = 0

if 0:
    saveData = True

    nJ2 = 31
    nh = 31
    phaseDiagramParameters = (J1,J2min,J2max,nJ2,hmin,hmax,nh,D1,D2)

    en = computeClassicalGroundState(phaseDiagramParameters,verbose=True,save=saveData)

    plotClassicalPhaseDiagram(en,phaseDiagramParameters,show=True)

    plotClassicalPhaseDiagramParameters(en,phaseDiagramParameters,show=True)

# --- Define colors (RGBA) ---
if 0:
    cNi = np.array([204, 0, 0, 1])/256   # reddish
    cSi = np.array([255, 128, 0, 1])/256   # greenish
    zN  = np.array([181, 253, 255, 1])/256   # uniform bluish
if 0:
    cNi = np.array([59, 130, 246, 1])/256   # reddish
    cSi = np.array([34, 197, 94, 1])/256   # greenish
    zN  = np.array([253, 224, 71, 1])/256   # uniform bluish
if 0:
    cNi = np.array([236, 72, 153, 1])/256   # reddish
    cSi = np.array([251, 146, 60, 1])/256   # greenish
    zN  = np.array([96, 165, 250, 1])/256   # uniform bluish
if 1:
    cNi = np.array([78, 147, 248, 1])/256   # reddish
    cSi = np.array([244, 109, 107, 1])/256   # greenish
    zN  = np.array([144, 210, 83, 1])/256   # uniform bluish
#
cNf = zN
cSf = zN
if 1:
    """ Plot with uniform colors the Delta=0 phase diagram """
    # --- Grid ---
    x = np.linspace(J2min, J2max, 500)
    y = np.linspace(hmin, hmax, 500)
    X, Y = np.meshgrid(x, y)

    # --- Initialize an RGBA array ---
    colors = np.zeros(X.shape + (4,))
    for i in range(x.shape[0]):
        normy = y / y.max()
        if x[i] < J1/2:
            ycrit = 2*(J1*(1-D1) - x[i]*(1-D2))
            mask = y < ycrit
            thetas = np.arccos(y[mask]/ycrit) / np.pi*2
            colors[i,mask] = cNi[None,:] * thetas[:,None] + cNf[None,:] * (1-thetas[:,None])
        if x[i] > J1/2:
            ycrit = 2*x[i]*(1-D2)
            mask = y < ycrit
            thetas = np.arccos(y[mask]/ycrit) / np.pi*2
            colors[i,mask] = cSi[None,:] * thetas[:,None] + cSf[None,:] * (1-thetas[:,None])
        colors[i,~mask] = zN

    colors[..., 3] = 1.0  # alpha channel

    # --- Plot ---
    colors = colors.transpose(1,0,2)
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot()
    ax.imshow(colors, origin='lower', extent=(0,1,0,3), aspect='auto')

    # Draw boundaries
    ax.plot(x[x<J1/2], 2*(1 - x[x<J1/2]), 'k', lw=1)
    ax.plot(x[x>J1/2], 2*x[x>J1/2], 'k', lw=1)
    ax.plot([J1/2,J1/2],[0,1],'k--')

    s_ = 22
    ax.set_xlabel(r"$J_2$",size=s_)
    ax.set_ylabel("h",size=s_,rotation=0,labelpad=15)
    ax.set_xlim(J2min,J2max)
    ax.set_ylim(hmin,hmax)
    s_ = 18
    ax.tick_params(axis='both',direction='in',length=10,width=1,pad=7,labelsize=s_)
    # Colorbars
    s_ = 14
    h_ = 0.8
    w_ = 0.03
    y_ = 0.1
    import matplotlib.colors as mcolors
    cax = fig.add_axes([0.05,y_,w_,h_])
    cmap = mcolors.ListedColormap(colors[y<2*(J1*(1-D1)),0])
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=mcolors.Normalize(vmin=0,vmax=1))
    cbar = fig.colorbar(sm,
                        cax=cax,
                        orientation='vertical')
    cbar.set_ticks([0,0.5,1])
    cbar.set_ticklabels([r"$\pi/2$",r"$\pi/4$",r"$0$"],size=s_)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.set_title(r"$\theta^\text{canted-Néel}$", fontsize=20, pad=10)

    cax = fig.add_axes([0.91,y_,w_,h_])
    cmap = mcolors.ListedColormap(colors[y<2*(x[-1]*(1-D1)),-1])
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=mcolors.Normalize(vmin=0,vmax=1))
    cbar = fig.colorbar(sm,
                        cax=cax,
                        orientation='vertical')
    cbar.set_ticks([0,0.5,1])
    cbar.set_ticklabels([r"$\pi/2$",r"$\pi/4$",r"$0$"],size=s_)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.set_title(r"$\theta^\text{canted-stripe}$", fontsize=20, pad=10)
    #fig.tight_layout()
    plt.subplots_adjust(
        left=0.16,
        right=0.868,
        bottom=y_,
        top=y_+h_
    )
    plt.show()
if 1:
    """ Plot some of te spin configurations """
    import sys, os, argparse
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from wavespin.tools.inputUtils import importParameters
    from wavespin.static.periodic import quantizationAxis
    from wavespin.lattice.lattice import latticeClass
    from wavespin.plots import fancyLattice
    """ Parameters and options """
    parser = argparse.ArgumentParser(description="Static correlator calculation using Holstein-Primakoff formalism")
    parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
    parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
    inputArguments = parser.parse_args()
    verbose = inputArguments.verbose
    parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

    pars = [(0,0),(0,1.5),(0,3),(1,0),(1,1.5)]
    for p in pars:
        J2,h = p
        parameters.dia_Hamiltonian = (J1/2,J2/2,D1,D2,h,0)
        th = quantizationAxis(1/2,(J1/2,J2/2),(D1,D2),h)[0]
        ci = cNi if J2<J1/2 else cSi
        cf = cNf if J2<J1/2 else cSf
        color = ci * th/np.pi*2 + cf * (1-th/np.pi*2)
        color[-1] = 1
        system = latticeClass(parameters)
        system.thetas = np.ones(system.Ns) * th
        fn = "Figures/lattice_"+str(J2)+"_"+str(h)+".png"
        kwargs = {'indices':False,
                  'angles':True,
                  'arrowColor':color,
                  'order':'c-Neel' if J2<J1/2 else 'c-stripe',
                  "savePlot":1,
                  "filename":fn}
        fancyLattice.plotSitesGrid(system,**kwargs)




















