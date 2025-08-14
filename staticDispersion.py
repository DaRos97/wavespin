"""
Here we show how to compute the static ground state from a set of Hamiltonian parameters in a periodic system.
"""

import numpy as np
from wavespin.static.periodic import *
import matplotlib.pyplot as plt
S = 0.5     #spin value
#Lattice parameters
Lx = 401
Ly = 401
#Hamiltonian parameters
nP = 101
listJ1 = np.linspace(0,40,nP)
listJ2 = np.zeros(nP)
listD1 = np.zeros(nP)
listD2 = np.zeros(nP)
listH = np.linspace(30,0,nP)

plotValues = True
plotDispersions = True

dispersions = np.zeros((nP,Lx,Ly))
thetas = np.zeros(nP)
gsEs = np.zeros(nP)
gaps = np.zeros(nP)
for iP in range(nP):
    J = (listJ1[iP],listJ2[iP])
    D = (listD1[iP],listD2[iP])
    h = listH[iP]
    HamiltonianParameters = (J,D,h)
    dispersion, angles, gsE, gap = computeSolution(Lx,Ly,S,HamiltonianParameters)
    dispersions[iP] = dispersion
    thetas[iP] = angles[0]
    gsEs[iP] = gsE
    gaps[iP] = gap

if plotDispersions:
    gridk = momentumGrid(Lx,Ly)
    fig = plt.figure(figsize=(15,10))
    for i,iP in enumerate([10,30,50,60,80,100]):
        ax = fig.add_subplot(2,3,i+1,projection='3d')
        ax.plot_surface(gridk[:,:,0],gridk[:,:,1],dispersions[iP],cmap='plasma')
        ax.set_aspect('equalxy')
        ax.set_title("stop ratio="+"{:.1f}".format(iP/nP))
        n_i = 6
        ax.set_xticks([ik*2*np.pi/n_i for ik in range(n_i+1)],["{:.2f}".format(ik*2*np.pi/n_i) for ik in range(n_i+1)],size=8)
        ax.set_yticks([ik*2*np.pi/n_i for ik in range(n_i+1)],["{:.2f}".format(ik*2*np.pi/n_i) for ik in range(n_i+1)],size=8)
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
    plt.show()

if plotValues:
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot()
    xAxis = np.arange(nP)
    l1 = ax.plot(xAxis,thetas,'b*-',label=r'$\theta$')
    ax.set_yticks([i/6*np.pi/2 for i in range(7)],["{:.1f}".format(i/6*90)+'°' for i in range(7)],size=15,color='b')

    ax_r = ax.twinx()
    l2 = ax_r.plot(xAxis,gsEs,'r*-',label=r'$E_{GS}$')
    ax_r.tick_params(axis='y',colors='r')

    ax_r = ax.twinx()
    l3 = ax_r.plot(xAxis,gaps,'g*-',label='Gap')
    ax_r.tick_params(axis='y',colors='g')

    ax.set_xlabel("stop ratio (time)",size=20)
    #Legend
    labels = [l.get_label() for l in l1+l2+l3]
    ax.legend(l1+l2+l3,labels,fontsize=20,loc=(0.4,0.1))

    plt.show()
