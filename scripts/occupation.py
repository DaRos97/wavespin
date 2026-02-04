""" Here I compute the spin wave occupation to see if it stays low, justifying the HP expansion.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt

parameters = importParameters()

if 0:       # Compute the mean occupation per site in PBC and OBC for different system sizes
    J1 = 1
    J2 = 0
    h = 0
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    Llist = [6,8,10,20,30,40,50,60]
    resPBC = []
    resOBC = []
    for L in Llist:
        print(L)
        Lx = L
        Ly = L
        parameters.lat_Lx = Lx
        parameters.lat_Ly = Ly
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
        mean = np.mean(np.diagonal(G_GS))
        resPBC.append(mean)
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
        mean = np.mean(np.diagonal(G_GS))
        resOBC.append(mean)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.scatter(Llist,resPBC,color='r',label='PBC')
    ax.scatter(Llist,resOBC,color='b',label='OBC')
    ax.axhline(0.06,color='r',ls='dashed',label="Infinite PBC limit (J2=h=0)")
    ax.set_xlabel("System linear size",size=15)
    ax.set_ylabel(r"$\langle n_i \rangle$",size=15)
    ax.legend(fontsize=15)
    plt.show()
if 0:       # Compute site occupation in OBC for different system sizes
    J1 = 1
    J2 = 0
    h = 0
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    Llist = [6,8,10,20,
#             10,10,10,10]
             30,40,50,60]
    meanOBC = []
    siteOBC = []
    for L in Llist:
        print(L)
        Lx = L
        Ly = L
        parameters.lat_Lx = Lx
        parameters.lat_Ly = Ly
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
        siteOBC.append(np.diagonal(G_GS))
        mean = np.mean(np.diagonal(G_GS))
        meanOBC.append(mean)

    fig,axs = plt.subplots(2,4,figsize=(10,6))
    yMin = 10
    yMax = 0
    for i in range(len(Llist)):
        ax = axs[i//4,i%4]
        Ns = Llist[i]**2
        ax.scatter(np.arange(Ns),siteOBC[i],color='b')
        ymin,ymax = ax.get_ylim()
        if ymin<yMin:
            yMin = ymin
        if ymax>yMax:
            yMax = ymax
        ax.axhline(meanOBC[i],color='r',ls='dashed',label='mean')
        if i > 3:
            ax.set_xlabel("Site index",size=15)
        if i in [0,4]:
            ax.set_ylabel(r"$n_i$",size=15)
        ax.set_title("%dx%d lattice"%(Llist[i],Llist[i]),size=15)
        ax.legend(fontsize=15)
    for i in range(len(Llist)):
        axs[i//4,i%4].set_ylim(yMin,yMax)
    plt.show()
if 0:           # Compute position of sites with high occupation
    J1 = 1
    J2 = 0
    h = 0
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    Lx = Ly = 60
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    # OBC
    parameters.lat_boundary = 'open'
    mySystem = openHamiltonian(parameters)
    G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
    sites = np.diagonal(G_GS)
    mask = sites>0.105

    highSites = np.where(mask)[0]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    for i in range(highSites.shape[0]):
        ix, iy = mySystem._xy(highSites[i])
        ax.scatter(ix,iy,color='r')
    ax.set_xlim(-0.5,Lx+0.5)
    ax.set_ylim(-0.5,Ly+0.5)
    plt.show()
if 0:           # Frustrated branch
    J1 = 1
    h = 0
    J2s = np.linspace(0,0.5,50,endpoint=False)

    Lx = Ly = 10
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    resPBC = []
    resOBC = []
    for J2 in J2s:
        print(J2)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
        mean = np.mean(np.diagonal(G_GS))
        resPBC.append(mean)
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
        mean = np.mean(np.diagonal(G_GS))
        resOBC.append(mean)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.scatter(J2s,resPBC,color='r',label='PBC')
    ax.scatter(J2s,resOBC,color='b',label='OBC')
    ax.axhline(0.06,color='r',ls='dashed',label="Infinite PBC limit (J2=h=0)")
    ax.set_xlabel(r"$J_2$",size=15)
    ax.set_ylabel(r"$\langle n_i \rangle$",size=15)
    ax.legend(fontsize=15)
    ax.set_title("%dx%d lattice"%(Lx,Ly),size=15)
    plt.show()
if 0:           # Critical branch
    J1 = 1
    J2 = 0
    Hs = np.linspace(0,3,30,endpoint=False)

    Lx = Ly = 30
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    resPBC = []
    resOBC = []
    for h in Hs:
        print(h)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
        mean = np.mean(np.diagonal(G_GS))
        resPBC.append(mean)
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
        mean = np.mean(np.diagonal(G_GS))
        resOBC.append(mean)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.scatter(Hs,resPBC,color='r',label='PBC')
    ax.scatter(Hs,resOBC,color='b',label='OBC')
    ax.axhline(0.06,color='r',ls='dashed',label="Infinite PBC limit (J2=h=0)")
    ax.axvline(2,color='g',ls='dashed',label="QCP")
    ax.set_xlabel(r"$h$",size=15)
    ax.set_ylabel(r"$\langle n_i \rangle$",size=15)
    ax.legend(fontsize=15)
    ax.set_title("%dx%d lattice"%(Lx,Ly),size=15)
    plt.show()






