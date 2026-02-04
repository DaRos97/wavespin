""" Here I compute the decay rate for the two branches of parameters.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

parameters = importParameters()
parameters.dia_plotWf = False
parameters.dia_saveWf = True
parameters.sca_saveVertex = True
parameters.sca_saveRate = True
#parameters.sca_broadening = 0.5

s_ = 15
if 0:       # Compute rate for J2=h=0, different types of first order at various temperatures 
    Lx = Ly = 10
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly

    # Scattering parameters
    sca_types = ('1to2_1','2to2_1','1to3_1')
    parameters.sca_types = sca_types

    # Hamiltonian parameters
    J1 = 1
    J2 = 0
    h = 0
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)

    Ts = [0,0.2,0.4,0.8,1.3,1.9,2.5]
    resPBC1to2 = []
    resPBC2to2 = []
    resPBC1to3 = []
    resOBC1to2 = []
    resOBC2to2 = []
    resOBC1to3 = []
    for temperature in Ts:
        print(temperature)
        parameters.sca_temperature = temperature
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        mySystem.computeRate()
        resPBC1to2.append(mySystem.rates['1to2_1'])
        resPBC2to2.append(mySystem.rates['2to2_1'])
        resPBC1to3.append(mySystem.rates['1to3_1'])
        enPBC = mySystem.evals[1:]
        # OBC
        resOBC = []
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        mySystem.computeRate()
        resOBC1to2.append(mySystem.rates['1to2_1'])
        resOBC2to2.append(mySystem.rates['2to2_1'])
        resOBC1to3.append(mySystem.rates['1to3_1'])
        enOBC = mySystem.evals[1:]
        if 0:
            print("Temperature: ",temperature)
            print("OBC first mode: ",enOBC[0])
            print("OBC last mode: ",enOBC[-1])
            print("PBC first mode: ",enPBC[0])
            print("PBC last mode: ",enPBC[-1])

    resPBC = [resPBC1to2,resPBC2to2,resPBC1to3]
    resOBC = [resOBC1to2,resOBC2to2,resOBC1to3]
    # Colormap
    cmap = plt.cm.cividis
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=Ts[0], vmax=Ts[-1])
    fig = plt.figure(figsize=(15,15))
    for i in range(len(sca_types)):
        axPBC = fig.add_subplot(2,len(sca_types),i+1)
        axOBC = fig.add_subplot(2,len(sca_types),i+1+len(sca_types))
        for t in range(len(Ts)):
            axPBC.scatter(enPBC,resPBC[i][t],color=cmap(norm(Ts[t])))
            axOBC.scatter(enOBC,resOBC[i][t],color=cmap(norm(Ts[t])))
        axOBC.set_xlabel("Mode energy",size=s_)
        axPBC.set_title(sca_types[i],size=s_)
        if i==0:
            axPBC.set_ylabel("PBC",size=s_+10,color='r')
            axOBC.set_ylabel("OBC",size=s_+10,color='r')
    # Cbar
    cax = fig.add_axes([0.93,0.12,0.03,0.75])
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Temperature",size=s_)
    plt.suptitle("%dx%d lattice"%(Lx,Ly), size=s_+5)
    #fig.tight_layout()
    plt.show()
if 0:       # Compute rate for frustrated branch 
    Lx = Ly = 10
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    # Scattering parameters
    sca_types = ('1to2_1','2to2_1','1to3_1')
    parameters.sca_types = sca_types

    # Hamiltonian parameters
    J1 = 1
    h = 0
    J2s = np.linspace(0,0.5,10,endpoint=False)
    fig2 = plt.figure(figsize=(15,6))
    fig3 = plt.figure(figsize=(15,6))
    for i2,J2 in enumerate(J2s):
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        Ts = [0,0.2,0.4,0.8,1.3,1.9,2.5]
        resPBC1to2 = []
        resPBC2to2 = []
        resPBC1to3 = []
        resOBC1to2 = []
        resOBC2to2 = []
        resOBC1to3 = []
        for temperature in Ts:
            print(temperature)
            parameters.sca_temperature = temperature
            # PBC
            parameters.lat_boundary = 'periodic'
            mySystem = openHamiltonian(parameters)
            mySystem.computeRate()
            resPBC1to2.append(mySystem.rates['1to2_1'])
            resPBC2to2.append(mySystem.rates['2to2_1'])
            resPBC1to3.append(mySystem.rates['1to3_1'])
            enPBC = mySystem.evals[1:]
            # OBC
            resOBC = []
            parameters.lat_boundary = 'open'
            mySystem = openHamiltonian(parameters)
            mySystem.computeRate()
            resOBC1to2.append(mySystem.rates['1to2_1'])
            resOBC2to2.append(mySystem.rates['2to2_1'])
            resOBC1to3.append(mySystem.rates['1to3_1'])
            enOBC = mySystem.evals[1:]
        # Figure
        resPBC = [resPBC1to2,resPBC2to2,resPBC1to3]
        resOBC = [resOBC1to2,resOBC2to2,resOBC1to3]
        # Colormap
        cmap = plt.cm.cividis
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=Ts[0], vmax=Ts[-1])
        fig = plt.figure(figsize=(15,15))
        for i in range(len(sca_types)):
            axPBC = fig.add_subplot(2,len(sca_types),i+1)
            axOBC = fig.add_subplot(2,len(sca_types),i+1+len(sca_types))
            for t in range(len(Ts)):
                axPBC.scatter(enPBC,resPBC[i][t],color=cmap(norm(Ts[t])))
                axOBC.scatter(enOBC,resOBC[i][t],color=cmap(norm(Ts[t])))
            axOBC.set_xlabel("Mode energy",size=s_)
            axPBC.set_title(sca_types[i],size=s_)
            if i==0:
                axPBC.set_ylabel("PBC",size=s_+10,color='r')
                axOBC.set_ylabel("OBC",size=s_+10,color='r')
        # Cbar
        cax = fig.add_axes([0.93,0.12,0.03,0.75])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Temperature",size=s_)
        #plt.suptitle("%dx%d lattice"%(Lx,Ly), size=s_+5)
        fig.savefig("Figures/decayFrustratedBranch/J2_%.5f.png"%J2)
        plt.close(fig)
        # Figure 2
        ax2 = fig2.add_subplot(2,5,i2+1)
        for t in range(len(Ts)):
            ax2.scatter(enOBC,resOBC[1][t],color=cmap(norm(Ts[t])))
        if i2>=5:
            ax2.set_xlabel("Mode energy",size=s_)
        ax2.set_title(r"$J_2=$%.3f"%J2,size=s_)
        # Figure 3
        ax3 = fig3.add_subplot(2,5,i2+1)
        for t in range(len(Ts)):
            ax3.scatter(enOBC,resOBC[2][t],color=cmap(norm(Ts[t])))
        if i2>=5:
            ax3.set_xlabel("Mode energy",size=s_)
        ax3.set_title(r"$J_2=$%.3f"%J2,size=s_)
    # Cbar2
    cax = fig2.add_axes([0.93,0.12,0.03,0.75])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig2.colorbar(sm, cax=cax)
    cbar.set_label("Temperature",size=s_)
    fig2.savefig("Figures/decayFrustratedBranch/full_2to2.png")
    # Cbar3
    cax = fig3.add_axes([0.93,0.12,0.03,0.75])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig3.colorbar(sm, cax=cax)
    cbar.set_label("Temperature",size=s_)
    fig3.savefig("Figures/decayFrustratedBranch/full_1to3.png")
if 0:       # Compute rate for critical branch 
    Lx = Ly = 10
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly

    # Scattering parameters
    sca_types = ('1to2_1','2to2_1','1to3_1')
    parameters.sca_types = sca_types

    # Hamiltonian parameters
    J1 = 1
    J2 = 0
    Hs = np.linspace(0,3,7)
    fig2 = plt.figure(figsize=(15,6))
    fig3 = plt.figure(figsize=(15,6))
    for ih,h in enumerate(Hs):
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        Ts = [0,0.2,0.4,0.8,1.3,1.9,2.5]
        resPBC1to2 = []
        resPBC2to2 = []
        resPBC1to3 = []
        resOBC1to2 = []
        resOBC2to2 = []
        resOBC1to3 = []
        for temperature in Ts:
            print(temperature)
            parameters.sca_temperature = temperature
            # PBC
            parameters.lat_boundary = 'periodic'
            mySystem = openHamiltonian(parameters)
            mySystem.computeRate()
            resPBC1to2.append(mySystem.rates['1to2_1'])
            resPBC2to2.append(mySystem.rates['2to2_1'])
            resPBC1to3.append(mySystem.rates['1to3_1'])
            enPBC = mySystem.evals[1:]
            # OBC
            resOBC = []
            parameters.lat_boundary = 'open'
            mySystem = openHamiltonian(parameters)
            mySystem.computeRate()
            resOBC1to2.append(mySystem.rates['1to2_1'])
            resOBC2to2.append(mySystem.rates['2to2_1'])
            resOBC1to3.append(mySystem.rates['1to3_1'])
            enOBC = mySystem.evals[1:]
        # Figure
        resPBC = [resPBC1to2,resPBC2to2,resPBC1to3]
        resOBC = [resOBC1to2,resOBC2to2,resOBC1to3]
        # Colormap
        cmap = plt.cm.cividis
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=Ts[0], vmax=Ts[-1])
        fig = plt.figure(figsize=(15,15))
        for i in range(len(sca_types)):
            axPBC = fig.add_subplot(2,len(sca_types),i+1)
            axOBC = fig.add_subplot(2,len(sca_types),i+1+len(sca_types))
            for t in range(len(Ts)):
                axPBC.scatter(enPBC,resPBC[i][t],color=cmap(norm(Ts[t])))
                axOBC.scatter(enOBC,resOBC[i][t],color=cmap(norm(Ts[t])))
            axOBC.set_xlabel("Mode energy",size=s_)
            axPBC.set_title(sca_types[i],size=s_)
            if i==0:
                axPBC.set_ylabel("PBC",size=s_+10,color='r')
                axOBC.set_ylabel("OBC",size=s_+10,color='r')
        # Cbar
        cax = fig.add_axes([0.93,0.12,0.03,0.75])
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Temperature",size=s_)
        plt.suptitle("%dx%d lattice"%(Lx,Ly), size=s_+5)
        fig.savefig("Figures/decayCriticalBranch/h_%.5f.png"%h)
        plt.close(fig)
        # Figure 2
        ax2 = fig2.add_subplot(2,4,ih+1)
        for t in range(len(Ts)):
            ax2.scatter(enOBC,resOBC[1][t],color=cmap(norm(Ts[t])))
        if ih>=4:
            ax2.set_xlabel("Mode energy",size=s_)
        ax2.set_title(r"$h=$%.3f"%h,size=s_)
        # Figure 3
        ax3 = fig3.add_subplot(2,4,ih+1)
        for t in range(len(Ts)):
            ax3.scatter(enOBC,resOBC[0][t],color=cmap(norm(Ts[t])))
        if ih>=4:
            ax3.set_xlabel("Mode energy",size=s_)
        ax3.set_title(r"$h=$%.3f"%h,size=s_)
    # Cbar2
    cax = fig2.add_axes([0.93,0.12,0.03,0.75])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig2.colorbar(sm, cax=cax)
    cbar.set_label("Temperature",size=s_)
    fig2.savefig("Figures/decayCriticalBranch/full_2to2.png")
    # Cbar3
    cax = fig3.add_axes([0.93,0.12,0.03,0.75])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig3.colorbar(sm, cax=cax)
    cbar.set_label("Temperature",size=s_)
    fig3.savefig("Figures/decayCriticalBranch/full_1to2.png")
if 1:       # Compute rate for J2=h=0, different types of first order at various broadenings 
    Lx = Ly = 10
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly

    # Scattering parameters
    sca_types = ('1to2_1','2to2_1','1to3_1')
    parameters.sca_types = sca_types

    # Hamiltonian parameters
    J1 = 1
    J2 = 0
    h = 0
    temperature = 0.5
    parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
    parameters.sca_temperature = temperature

    Bs = [0.01,0.05,0.1]
    resPBC1to2 = []
    resPBC2to2 = []
    resPBC1to3 = []
    resOBC1to2 = []
    resOBC2to2 = []
    resOBC1to3 = []
    for broadening in Bs:
        parameters.sca_broadening = broadening
        print(broadening)
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        mySystem.computeRate()
        resPBC1to2.append(mySystem.rates['1to2_1'])
        resPBC2to2.append(mySystem.rates['2to2_1'])
        resPBC1to3.append(mySystem.rates['1to3_1'])
        enPBC = mySystem.evals[1:]
        # OBC
        resOBC = []
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        mySystem.computeRate()
        resOBC1to2.append(mySystem.rates['1to2_1'])
        resOBC2to2.append(mySystem.rates['2to2_1'])
        resOBC1to3.append(mySystem.rates['1to3_1'])
        enOBC = mySystem.evals[1:]

    resPBC = [resPBC1to2,resPBC2to2,resPBC1to3]
    resOBC = [resOBC1to2,resOBC2to2,resOBC1to3]
    # Colormap
    cmap = plt.cm.cividis
    norm = Normalize(vmin=Bs[0], vmax=Bs[-1])
    fig = plt.figure(figsize=(15,15))
    for i in range(len(sca_types)):
        axPBC = fig.add_subplot(2,len(sca_types),i+1)
        axOBC = fig.add_subplot(2,len(sca_types),i+1+len(sca_types))
        for b in range(len(Bs)):
            axPBC.scatter(enPBC,resPBC[i][b],color=cmap(norm(Bs[b])))
            axOBC.scatter(enOBC,resOBC[i][b],color=cmap(norm(Bs[b])))
        axOBC.set_xlabel("Mode energy",size=s_)
        axPBC.set_title(sca_types[i],size=s_)
        if i==0:
            axPBC.set_ylabel("PBC",size=s_+10,color='r')
            axOBC.set_ylabel("OBC",size=s_+10,color='r')
    # Cbar
    cax = fig.add_axes([0.93,0.12,0.03,0.75])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Broadening",size=s_)
    plt.suptitle("%dx%d lattice"%(Lx,Ly), size=s_+5)
    #fig.tight_layout()
    #fig.savefig("Figures/broad%.1f.png"%broadening)
    #plt.close()
    plt.show()




