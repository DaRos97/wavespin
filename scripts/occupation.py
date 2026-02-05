""" Here I compute the spin wave occupation to see if it stays low, justifying the HP expansion.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm

inp = sys.argv[1]
parameters = importParameters()

if inp=='0':       # Compute the mean occupation per site in PBC and OBC for different system sizes
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
        n_min = np.min(np.diagonal(G_GS))
        n_max = np.max(np.diagonal(G_GS))
        resOBC.append([mean,n_min,n_max])

    resOBC = np.array(resOBC)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.scatter(Llist,resPBC,color='r',label='PBC')
    ax.scatter(Llist,resOBC[:,0],color='b',label='mean OBC')
    ax.fill_between(Llist,resOBC[:,1],resOBC[:,2],color='navy',alpha=0.3,label='range OBC',lw=0,zorder=-1)
    ax.axhline(0.06,color='r',ls='dashed',label="Infinite PBC limit (J2=h=0)")
    ax.set_xlabel("System linear size",size=15)
    ax.set_ylabel(r"$\langle n_i \rangle$",size=15)
    ax.legend(fontsize=15)
    plt.show()
if inp=='1':       # Compute site occupation in OBC for different system sizes
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

    fig2,axs = plt.subplots(2,4,figsize=(10,6))
    for i in range(len(Llist)):
        ax = axs[i//4,i%4]
        zMin = np.min(siteOBC[i])
        zMax = np.max(siteOBC[i])
        X,Y = np.meshgrid(np.arange(Llist[i]),np.arange(Llist[i]),indexing='ij')
        pm = ax.pcolormesh(
            X,Y,
            siteOBC[i].reshape(Llist[i],Llist[i]),
            cmap='bwr',
            vmin=zMin,
            vmax=zMax
        )
        ax.set_title("%dx%d lattice"%(Llist[i],Llist[i]),size=15)
        ax.set_aspect('equal')
    plt.show()
if inp=='2':           # Compute position of sites with high occupation
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
if inp=='3':           # Frustrated branch
    J1 = 1
    h = 0
    J2s = np.linspace(0.,0.5,50,endpoint=False)

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
        n_min = np.min(np.diagonal(G_GS))
        n_max = np.max(np.diagonal(G_GS))
        resOBC.append([mean,n_min,n_max])

    resOBC = np.array(resOBC)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.scatter(J2s,resPBC,color='r',label='PBC')
    ax.scatter(J2s,resOBC[:,0],color='b',label='OBC')
    ax.fill_between(J2s,resOBC[:,1],resOBC[:,2],color='navy',alpha=0.3,label='range OBC',lw=0,zorder=-1)
    ax.axhline(0.06,color='r',ls='dashed',label="Infinite PBC limit (J2=h=0)")
    ax.set_xlabel(r"$J_2$",size=15)
    ax.set_ylabel(r"$\langle n_i \rangle$",size=15)
    ax.set_ylim(0,0.41)
    ax.legend(fontsize=15)
    ax.set_title("%dx%d lattice"%(Lx,Ly),size=15)
    plt.show()
if inp=='4':           # Critical branch
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
        n_min = np.min(np.diagonal(G_GS))
        n_max = np.max(np.diagonal(G_GS))
        resOBC.append([mean,n_min,n_max])

    resOBC = np.array(resOBC)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    ax.scatter(Hs,resPBC,color='r',label='PBC')
    ax.scatter(Hs,resOBC[:,0],color='b',label='OBC')
    ax.fill_between(Hs,resOBC[:,1],resOBC[:,2],color='navy',alpha=0.3,label='range OBC',lw=0,zorder=-1)
    ax.axhline(0.06,color='r',ls='dashed',label="Infinite PBC limit (J2=h=0)")
    ax.axvline(2,color='g',ls='dashed',label="QCP")
    ax.set_xlabel(r"$h$",size=15)
    ax.set_ylabel(r"$\langle n_i \rangle$",size=15)
    ax.legend(fontsize=15)
    ax.set_title("%dx%d lattice"%(Lx,Ly),size=15)
    plt.show()
""" Include temperature: """
""" Spectrum in PBC has maximum at 2*sqrt(2) and minimum approaching 0 in system size. For L=6 minimum is 1. """
if inp=='5':       # Compute the occupation in PBC and OBC for a range of temperatures for different system sizes
    J1 = 1
    J2 = 0
    h = 0
    Tlist = np.linspace(0,2*np.sqrt(2),11)
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    Llist = [6,8,10,20,30,40,50,60]
    resPBC = np.zeros((len(Llist),len(Tlist)))
    resOBC = np.zeros((len(Llist),len(Tlist),3))
    for iL,L in enumerate(Llist):
        print(L)
        Lx = L
        Ly = L
        parameters.lat_Lx = Lx
        parameters.lat_Ly = Ly
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        BF = np.zeros((len(Tlist),mySystem.Ns))
        BF[1:,1:] = 1/(np.exp(mySystem.evals[1:][None,:]/Tlist[1:][:,None])-1)
        for iT in range(len(Tlist)):
            G = np.sum(np.real(mySystem.V_)**2,axis=1) + np.einsum('n,in,in->i',BF[iT],np.real(mySystem.U_)**2,np.real(mySystem.V_)**2,optimize=True)
            mean = np.mean(G)
            resPBC[iL,iT] = mean
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        BF = np.zeros((len(Tlist),mySystem.Ns))
        BF[1:,1:] = 1/(np.exp(mySystem.evals[1:][None,:]/Tlist[1:][:,None])-1)
        for iT in range(len(Tlist)):
            G = np.sum(np.real(mySystem.V_)**2,axis=1) + np.einsum('n,in,in->i',BF[iT],np.real(mySystem.U_)**2,np.real(mySystem.V_)**2,optimize=True)
            mean = np.mean(G)
            resOBC[iL,iT,0] = np.mean(G)
            resOBC[iL,iT,1] = np.min(G)
            resOBC[iL,iT,2] = np.max(G)

    def white_to(color, name):
        return LinearSegmentedColormap.from_list(name, ['white', color])

    cmaps = [
        white_to('red',   'white_red'),
        white_to('green', 'white_green'),
        white_to('blue',  'white_blue'),
        white_to('purple','white_purple'),
    ]
    fig = plt.figure(figsize=(15,8))
    s_ = 15
    ax = fig.add_subplot()
    for iL in range(len(Llist)):
        sc1 = ax.scatter(np.ones(len(Tlist))*Llist[iL],resPBC[iL],c=Tlist,cmap=cmaps[0],zorder=10)
    for iL in range(len(Llist)):
        sc2 = ax.scatter(np.ones(len(Tlist))*Llist[iL],resOBC[iL,:,1],c=Tlist,cmap=cmaps[1])
    for iL in range(len(Llist)):
        sc3 = ax.scatter(np.ones(len(Tlist))*Llist[iL],resOBC[iL,:,0],c=Tlist,cmap=cmaps[2])
    for iL in range(len(Llist)):
        sc4 = ax.scatter(np.ones(len(Tlist))*Llist[iL],resOBC[iL,:,2],c=Tlist,cmap=cmaps[3])
    # Colorbars
    mappables = [sc1,sc2,sc3,sc4]
    labels = ['PBC','OBC min','OBC mean','OBC max']
    divider = make_axes_locatable(ax)
    cbar_axes = [
        divider.append_axes("right", size="3%", pad=0.05 + i*0.03)
        for i in range(4)
    ]
    for sc, cax, lbl in zip(mappables, cbar_axes, labels):
        cbar = fig.colorbar(sc, cax=cax)
        if sc==sc4:
            cbar.set_label('Temperature', fontsize=s_)
            cbar.ax.tick_params(labelsize=s_)
        else:
            cbar.set_ticks([])
    ax.fill_between(Llist,resOBC[:,0,1],resOBC[:,-1,2],color='navy',alpha=0.1,lw=0,zorder=-1)
    ax.axhline(0.06,color='r',ls='dashed',label="Infinite PBC limit (J2=h=0)")
    ax.set_xlabel("System linear size",size=15)
    ax.set_ylabel(r"$\langle n_i \rangle$",size=15)
    # Legend
    legend_handles = [
        Patch(color=cmaps[i](1.0), label=labels[i])
        for i in range(4)
    ]
    ax.legend(
        handles=legend_handles,
        title="Datasets",
        frameon=True,
        fontsize=s_,
        title_fontsize=s_,
        loc='lower right'
    )
    plt.show()
if inp=='6':       # Compute site occupation in OBC for a range of temperatures
    J1 = 1
    J2 = 0
    h = 0
    Tlist = np.linspace(0,2*np.sqrt(2),5)
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    Llist = [10,20,
#             10,10]
             40,60]
    siteOBC = []
    for L in Llist:
        print(L)
        siteOBC.append([])
        Lx = L
        Ly = L
        parameters.lat_Lx = Lx
        parameters.lat_Ly = Ly
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        BF = np.zeros((len(Tlist),mySystem.Ns))
        BF[1:,1:] = 1/(np.exp(mySystem.evals[1:][None,:]/Tlist[1:][:,None])-1)
        for iT in range(len(Tlist)):
            G = np.sum(np.real(mySystem.V_)**2,axis=1) + np.einsum('n,in,in->i',BF[iT],np.real(mySystem.U_)**2,np.real(mySystem.V_)**2,optimize=True)
            siteOBC[-1].append(G.reshape(L,L))

    fig,axs = plt.subplots(len(Llist),len(Tlist),figsize=(10,6))
    zMin = np.min([np.amin(np.array(siteOBC[i])) for i in range(len(Llist))])
    zMax = np.max([np.amax(np.array(siteOBC[i])) for i in range(len(Llist))])
    for i in range(len(Llist)):
        X,Y = np.meshgrid(np.arange(Llist[i]),np.arange(Llist[i]),indexing='ij')
        for t in range(len(Tlist)):
            ax = axs[i,t]
            pm = ax.pcolormesh(
                X,Y,
                siteOBC[i][t],
                cmap='bwr',
                vmin=zMin,
                vmax=zMax
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if i==0:
                ax.set_title("T=%.2f"%Tlist[t],size=15)
            if t==0:
                ax.set_ylabel("%dx%d"%(Llist[i],Llist[i]),size=15)
    plt.show()
if inp=='7':           # Frustrated branch for a range of temperatures
    J1 = 1
    h = 0
    J2s = np.linspace(0.,0.5,50,endpoint=False)
    """ With increasing J2 the spectrum (PBC):
            - the minimum for 10x10 goes from ~0.618 to 0.062 for J2=0.49
            - the maximum goes from 2*sqrt(2) to 2.02 for J2=0.49
        We keep the SAME temperature range for all J2s.
    """
    Tlist = np.linspace(0,2*np.sqrt(2),5)

    Lx = Ly = 10
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    resPBC = np.zeros((len(J2s),len(Tlist)))
    resOBC = np.zeros((len(J2s),len(Tlist),3))
    for i2,J2 in enumerate(J2s):
        print(J2)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        BF = np.zeros((len(Tlist),mySystem.Ns))
        BF[1:,1:] = 1/(np.exp(mySystem.evals[1:][None,:]/Tlist[1:][:,None])-1)
        for iT in range(len(Tlist)):
            G = np.sum(np.real(mySystem.V_)**2,axis=1) + np.einsum('n,in,in->i',BF[iT],np.real(mySystem.U_)**2,np.real(mySystem.V_)**2,optimize=True)
            resPBC[i2,iT] = np.mean(G)
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        BF = np.zeros((len(Tlist),mySystem.Ns))
        BF[1:,1:] = 1/(np.exp(mySystem.evals[1:][None,:]/Tlist[1:][:,None])-1)
        for iT in range(len(Tlist)):
            G = np.sum(np.real(mySystem.V_)**2,axis=1) + np.einsum('n,in,in->i',BF[iT],np.real(mySystem.U_)**2,np.real(mySystem.V_)**2,optimize=True)
            resOBC[i2,iT,0] = np.mean(G)
            resOBC[i2,iT,1] = np.min(G)
            resOBC[i2,iT,2] = np.max(G)

    def white_to(color, name):
        return LinearSegmentedColormap.from_list(name, ['white', color])

    cmaps = [
        white_to('red',   'white_red'),
        white_to('green', 'white_green'),
        white_to('blue',  'white_blue'),
        white_to('purple','white_purple'),
    ]
    fig = plt.figure(figsize=(15,8))
    s_ = 15
    ss = 10
    ax = fig.add_subplot()
    for i2 in range(len(J2s)):
        sc1 = ax.scatter(np.ones(len(Tlist))*J2s[i2],resPBC[i2],c=Tlist,cmap=cmaps[0],zorder=10,s=ss)
        sc2 = ax.scatter(np.ones(len(Tlist))*J2s[i2],resOBC[i2,:,1],c=Tlist,cmap=cmaps[1],zorder=0,s=ss)
        sc3 = ax.scatter(np.ones(len(Tlist))*J2s[i2],resOBC[i2,:,0],c=Tlist,cmap=cmaps[2],zorder=1,s=ss)
        sc4 = ax.scatter(np.ones(len(Tlist))*J2s[i2],resOBC[i2,:,2],c=Tlist,cmap=cmaps[3],zorder=2,s=ss)
    # Colorbars
    mappables = [sc1,sc2,sc3,sc4]
    labels = ['PBC','OBC min','OBC mean','OBC max']
    divider = make_axes_locatable(ax)
    cbar_axes = [
        divider.append_axes("right", size="3%", pad=0.05 + i*0.03)
        for i in range(4)
    ]
    for sc, cax, lbl in zip(mappables, cbar_axes, labels):
        cbar = fig.colorbar(sc, cax=cax)
        if sc==sc4:
            cbar.set_label('Temperature', fontsize=s_)
            cbar.ax.tick_params(labelsize=s_)
        else:
            cbar.set_ticks([])
    ax.fill_between(J2s,resOBC[:,0,1],resOBC[:,-1,2],color='navy',alpha=0.1,lw=0,zorder=-1)
    ax.axhline(0.06,color='r',ls='dashed',label="Infinite PBC limit (J2=h=0)")
    ax.set_xlabel(r"$J_2$",size=15)
    ax.set_ylabel(r"$\langle n_i \rangle$",size=15)
    # Legend
    legend_handles = [
        Patch(color=cmaps[i](1.0), label=labels[i])
        for i in range(4)
    ]
    ax.legend(
        handles=legend_handles,
        title="Datasets",
        frameon=True,
        fontsize=s_,
        title_fontsize=s_,
        loc='upper left'
    )
    plt.show()
if inp=='8':           # Site-dependent frustrated branch for a range of temperatures
    J1 = 1
    h = 0
    J2s = np.linspace(0.4,0.5,10,endpoint=False)
    J2s = [0.4,0.45,0.46,0.465,0.47,0.48,0.49,0.495]
    """ With increasing J2 the spectrum (10x10 PBC):
            - the minimum goes from ~0.618 to 0.062 for J2=0.49
            - the maximum goes from 2*sqrt(2) to 2.02 for J2=0.49
        We keep the SAME temperature range for all J2s.
    """
    Tlist = np.linspace(0,2*np.sqrt(2),4)
    Tlist = np.array([0,0.1,0.5,2*np.sqrt(2)])

    Lx = Ly = 10
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    siteOBC = np.zeros((len(J2s),len(Tlist),Lx,Ly))
    for i2,J2 in enumerate(J2s):
        print(J2)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        BF = np.zeros((len(Tlist),mySystem.Ns))
        BF[1:,1:] = 1/(np.exp(mySystem.evals[1:][None,:]/Tlist[1:][:,None])-1)
        for iT in range(len(Tlist)):
            G = np.sum(np.real(mySystem.V_)**2,axis=1) + np.einsum('n,in,in->i',BF[iT],np.real(mySystem.U_)**2,np.real(mySystem.V_)**2,optimize=True)
            siteOBC[i2,iT] = G.reshape(Lx,Ly)

    fig,axs = plt.subplots(len(Tlist),len(J2s)+1,figsize=(len(J2s)*2-3,len(Tlist)*2),width_ratios=list(np.ones(len(J2s)))+[0.2,])
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    for t in range(len(Tlist)):
        for i2 in range(len(J2s)):
            zMin = np.amin(siteOBC[i2,t])
            zMax = np.amax(siteOBC[i2,t])
            mean = np.mean(siteOBC[i2,t])
            norm = TwoSlopeNorm(vmin=zMin,vcenter=mean,vmax=zMax)
            ax = axs[t,i2]
            pm = ax.pcolormesh(
                X,Y,
                siteOBC[i2,t],
                cmap='bwr',
                norm=norm
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if t==0:
                ax.set_title(r"$J_2=%.3f$"%J2s[i2],size=15)
            if i2==0:
                ax.set_ylabel("T=%.2f"%Tlist[t],size=15)
        ax = axs[t,-1]
        cmap = plt.cm.bwr
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax)
        cbar.set_label("Occupation",fontsize=15)
    fig.tight_layout()
    plt.show()
if inp=='9':           # Site-dependent critical branch in gapless side for a range of temperatures
    J1 = 1
    J2 = 0
    Hs = np.linspace(0,2,10,endpoint=False)
    """ With increasing h<=2 the spectrum (10x10 PBC):
            - the minimum goes from ~0.618 to 0.83 (h=1.9)
            - the maximum goes from 2*sqrt(2) to 2.0
        We keep the SAME temperature range for all Hs.
    """
    Tlist = np.linspace(0,2*np.sqrt(2),3)

    Lx = Ly = 10
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    siteOBC = np.zeros((len(Hs),len(Tlist),Lx,Ly))
    for ih,h in enumerate(Hs):
        print(h)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        BF = np.zeros((len(Tlist),mySystem.Ns))
        BF[1:,1:] = 1/(np.exp(mySystem.evals[1:][None,:]/Tlist[1:][:,None])-1)
        for iT in range(len(Tlist)):
            G = np.sum(np.real(mySystem.V_)**2,axis=1) + np.einsum('n,in,in->i',BF[iT],np.real(mySystem.U_)**2,np.real(mySystem.V_)**2,optimize=True)
            siteOBC[ih,iT] = G.reshape(Lx,Ly)

    fig,axs = plt.subplots(len(Tlist),len(Hs)+1,figsize=(len(Hs)*2-3,len(Tlist)*2),width_ratios=list(np.ones(len(Hs)))+[0.2,])
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    for t in range(len(Tlist)):
        for ih in range(len(Hs)):
            zMin = np.amin(siteOBC[ih,t])
            zMax = np.amax(siteOBC[ih,t])
            mean = np.mean(siteOBC[ih,t])
            norm = TwoSlopeNorm(vmin=zMin,vcenter=mean,vmax=zMax)
            ax = axs[t,ih]
            pm = ax.pcolormesh(
                X,Y,
                siteOBC[ih,t],
                cmap='bwr',
                norm=norm
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if t==0:
                ax.set_title(r"$h=%.2f$"%Hs[ih],size=15)
            if ih==0:
                ax.set_ylabel("T=%.2f"%Tlist[t],size=15)
        ax = axs[t,-1]
        cmap = plt.cm.bwr
        norm = mpl.colors.Normalize(vmin=zMin, vmax=zMax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax)
        cbar.set_label("Occupation",fontsize=15)
    fig.tight_layout()
    plt.show()
if inp=='10':           # Site-dependent critical branch in gapped side for a range of temperatures
    J1 = 1
    J2 = 0
    Hs = np.linspace(2.1,3,10)
    """ With increasing h>=2.1 the spectrum (10x10 PBC):
            - the minimum goes from ~0.64 to 2.23 (h=3)
            - the maximum goes from 2.1 to 3
        We keep the SAME temperature range for all Hs.
    """
    Tlist = np.linspace(0,2*np.sqrt(2),3)

    Lx = Ly = 20
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_plotWf = 0#True
    parameters.dia_saveWf = True

    siteOBC = np.zeros((len(Hs),len(Tlist),Lx,Ly))
    for ih,h in enumerate(Hs):
        print(h)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        BF = np.zeros((len(Tlist),mySystem.Ns))
        BF[1:,1:] = 1/(np.exp(mySystem.evals[1:][None,:]/Tlist[1:][:,None])-1)
        for iT in range(len(Tlist)):
            G = np.sum(np.real(mySystem.V_)**2,axis=1) + np.einsum('n,in,in->i',BF[iT],np.real(mySystem.U_)**2,np.real(mySystem.V_)**2,optimize=True)
            siteOBC[ih,iT] = G.reshape(Lx,Ly)

    fig,axs = plt.subplots(len(Tlist),len(Hs)+1,figsize=(len(Hs)*2-3,len(Tlist)*2),width_ratios=list(np.ones(len(Hs)))+[0.2,])
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    for t in range(len(Tlist)):
        for ih in range(len(Hs)):
            zMin = np.amin(siteOBC[ih,t])
            zMax = np.amax(siteOBC[ih,t])
            mean = np.mean(siteOBC[ih,t])
            norm = TwoSlopeNorm(vmin=zMin,vcenter=mean,vmax=zMax)
            ax = axs[t,ih]
            pm = ax.pcolormesh(
                X,Y,
                siteOBC[ih,t],
                cmap='bwr',
                norm=norm
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            if t==0:
                ax.set_title(r"$h=%.2f$"%Hs[ih],size=15)
            if ih==0:
                ax.set_ylabel("T=%.2f"%Tlist[t],size=15)
        ax = axs[t,-1]
        cmap = plt.cm.bwr
        norm = mpl.colors.Normalize(vmin=zMin, vmax=zMax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax)
        cbar.set_label("Occupation",fontsize=15)
    fig.tight_layout()
    plt.show()




