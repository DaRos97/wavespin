""" Check where modes scatter to.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
import wavespin.tools.pathFinder as pf
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

parameters = importParameters()
parameters.dia_Hamiltonian = (10,0,0,0,0,0)
sca_types = (
#    '1to2_1','1to2_2',
    '2to2_1',
#    '2to2_2',
#    '1to3_1','1to3_2','1to3_3',
)
parameters.sca_types = sca_types
parameters.sca_broadening = 0.5
parameters.sca_saveVertex = True
parameters.sca_temperature = 8

parsLattices = {
    '24-diamond': [
        6,
        6,
        ( (0,0), (0,1), (0,4), (0,5), (1,0), (1,5), (4,0), (4,5), (5,0), (5,1), (5,4), (5,5) )
    ],
    '60-diamond': [
        10,
        10,
        ( (0,0), (0,1), (0,2), (0,3), (0,6), (0,7), (0,8), (0,9), (1,0), (1,1), (1,2), (1,7), (1,8), (1,9), (2,0), (2,1), (2,8), (2,9), (3,0), (3,9), (6,0), (6,9), (7,0), (7,1), (7,8), (7,9), (8,0), (8,1), (8,2), (8,7), (8,8), (8,9), (9,0), (9,1), (9,2), (9,3), (9,6), (9,7), (9,8), (9,9) )
    ],
    '60-diamond2': [
        10,
        10,
        (
            (0,0), (0,1), (0,2), (0,7), (0,8), (0,9),
            (1,0), (1,1), (1,8), (1,9),
            (2,0), (2,9),
            (7,0), (7,9),
            (8,0), (8,1), (8,8), (8,9),
            (9,0), (9,1), (9,2), (9,7), (9,8), (9,9)
        )
    ],
    '60-diamond3': [
        10,
        10,
        (
            (0,0), (0,1), (0,8), (0,9),
            (1,0), (1,9),
            (8,0), (8,9),
            (9,0), (9,1), (9,8), (9,9)
        )
    ],
    '60-diamond4': [
        10,
        10,
        (
            (0,0), (0,9),
            (9,0), (9,9)
        )
    ],
    '10x10-rectangle': [
        10,
        10,
        (),
    ],
    '9x9-rectangle-hole': [
        9,
        9,
        (
            (3,3), (3,4), (3,5),
            (4,3), (4,4), (4,5),
            (5,3), (5,4), (5,5),
        ),
    ],
    '7x8-rectangle': [
        7,
        8,
        (),
    ],
    '12x12-rectangle': [
        12,
        12,
        (),
    ],
    '41-sharp4': [
        9,
        9,
        (
            (0,0), (0,1), (0,2), (0,3), (0,5), (0,6), (0,7), (0,8),
            (1,0), (1,1), (1,2), (1,6), (1,7), (1,8),
            (2,0), (2,1), (2,7), (2,8),
            (3,0), (3,8),
            (5,0), (5,8),
            (6,0), (6,1), (6,7), (6,8),
            (7,0), (7,1), (7,2), (7,6), (7,7), (7,8),
            (8,0), (8,1), (8,2), (8,3), (8,5), (8,6), (8,7), (8,8),
        )
    ],
    '50-sharp2': [
        10,
        9,
        (
            (0,0), (0,1), (0,2), (0,3), (0,5), (0,6), (0,7), (0,8),
            (1,0), (1,1), (1,2), (1,6), (1,7), (1,8),
            (2,0), (2,1), (2,7), (2,8),
            (3,0), (3,8),
            (6,0), (6,8),
            (7,0), (7,1), (7,7), (7,8),
            (8,0), (8,1), (8,2), (8,6), (8,7), (8,8),
            (9,0), (9,1), (9,2), (9,3), (9,5), (9,6), (9,7), (9,8),
        )
    ],
    '3corner': [
        9,
        9,
        (
            (0,0), (0,1), (0,2), (0,6), (0,7), (0,8),
            (1,0), (1,1), (1,7), (1,8),
            (2,0), (2,8),
            (6,0), (6,8),
            (7,0), (7,1), (7,7), (7,8),
            (8,0), (8,1), (8,2), (8,6), (8,7), (8,8),
        )
    ],
    '97-diamond': [
        13,
        13,
        (
            (0,0), (0,1), (0,2), (0,3), (0,4), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12),
            (1,0), (1,1), (1,2), (1,3), (1,8), (1,9), (1,10), (1,11), (1,12),
            (2,0), (2,1), (2,2), (2,9), (2,10), (2,11), (2,12),
            (3,0), (3,1), (3,10), (3,11), (3,12),
            (4,0), (4,11), (4,12),
            (5,12),
            (7,0),
            (8,0), (8,1), (8,12),
            (9,0), (9,1), (9,2), (9,11), (9,12),
            (10,0), (10,1), (10,2), (10,3), (10,10), (10,11), (10,12),
            (11,0), (11,1), (11,2), (11,3), (11,4), (11,9), (11,10), (11,11), (11,12),
            (12,0), (12,1), (12,2), (12,3), (12,4), (12,5), (12,8), (12,9), (12,10), (12,11), (12,12)
        )
    ],
}

save = True
lattices = (
#    '24-diamond',
#    '60-diamond',
#    '60-diamond2',
#    '60-diamond3',
#    '60-diamond4',
#    '10x10-rectangle',
#    '9x9-rectangle-hole',
#    '41-sharp4',
#    '50-sharp2',
#    '3corner',
#    '97-diamond',
#    '7x8-rectangle',
    '12x12-rectangle',
)
if 1:       # Check elements of vertex
    fig=plt.figure(figsize=(10,10))
    for ic,lattice in enumerate(lattices):
        parameters.lat_Lx = parsLattices[lattice][0]
        parameters.lat_Ly = parsLattices[lattice][1]
        parameters.lat_offSiteList = parsLattices[lattice][2]
        parameters.lat_boundary = 'periodic'
        Ns = parameters.lat_Lx*parameters.lat_Ly - len(parameters.lat_offSiteList)
        print(Ns," sites")
        #parameters.dia_plotWf = True
        #parameters.lat_plotLattice = True
        system = openHamiltonian(parameters)
        system.computeVertex('2to2')
        V = system.vertex2to2

        if 0:
            ax = fig.add_subplot(projection='3d')
            nmin = 15
            nmax = 30
            X,Y = np.meshgrid(np.arange(nmin,nmax),np.arange(nmin,nmax),indexing='ij')
            ax.plot_surface(
                X,Y,
                np.sum(V**2,axis=(2,3))[nmin:nmax,nmin:nmax],
                cmap='plasma'
            )
            #ax.set_zscale('log')
        elif 1:
            from matplotlib.colors import LogNorm
            #ax = fig.add_subplot()
            nmin = 1
            nmax = Ns
            Z = (np.sum(V**2,axis=(2,3))[nmin:nmax,nmin:nmax] )**(1/1)
            X,Y = np.meshgrid(np.arange(nmin,nmax),np.arange(nmin,nmax))#,indexing='ij')
            plt.scatter(
                X.ravel(),
                Y.ravel(),
                c=Z.ravel(),
                cmap='plasma',
                norm=LogNorm(vmin=Z[Z > 0].min(), vmax=Z.max())
            )
            plt.colorbar()
        else:
            from matplotlib.colors import LogNorm
            #ax = fig.add_subplot()
            x1 = 10
            nmin = 1
            nmax = Ns
            Z = (np.sum(V**2,axis=(2,3))[nmin:nmax,nmin:nmax] )**(1/1)
            X,Y = np.meshgrid(np.arange(nmin,nmax),np.arange(nmin,nmax))#,indexing='ij')
            plt.scatter(
                X.ravel(),
                Y.ravel(),
                c=Z.ravel(),
                cmap='plasma',
                norm=LogNorm(vmin=Z[Z > 0].min(), vmax=Z.max())
            )
            plt.colorbar()

        fig.tight_layout()
        plt.show()
    exit()

if 0:
    fig=plt.figure(figsize=(13,8))
    for ic,lattice in enumerate(lattices):
        ax = fig.add_subplot()
        parameters.lat_Lx = parsLattices[lattice][0]
        parameters.lat_Ly = parsLattices[lattice][1]
        parameters.lat_offSiteList = parsLattices[lattice][2]
        Ns = parameters.lat_Lx*parameters.lat_Ly - len(parameters.lat_offSiteList)
        print(Ns," sites")
        parameters.dia_plotWf = True
        #parameters.lat_plotLattice = True
        system = openHamiltonian(parameters)
        exit()
        #continue
        system.computeRate('2to2_1')
        gamma = system.rates['2to2_1']
        ax.scatter(
#            system.evals[1:],
            np.arange(1,Ns),
            gamma,
            label=lattice
        )
        ymin,ymax = ax.get_ylim()
        dy = (ymax-ymin)/20
        for i in range(1,Ns):
            ax.text(
                #system.evals[i],
                i,
                gamma[i-1]+dy,
                "%d"%i,
                size=10
            )
        ax.set_title(lattice)
        if ic==0:
            ymin,ymax = ax.get_ylim()
        else:
            ax.set_ylim(ymin,ymax)
    plt.show()
    exit()
#exit()
if 1:
    system.computeRate('2to2_1')
    gamma = system.rates['2to2_1']
    fig=plt.figure(figsize=(13,8))
    if 0:
        ax = fig.add_subplot(121)
        ax.scatter(
            system.evals[1:],
            gamma,
        )
        ax = fig.add_subplot(122)
    else:
        ax = fig.add_subplot()
    ax.scatter(
        np.arange(1,Ns),
        gamma,
        color='orange'
    )
    ymin,ymax = ax.get_ylim()
    dy = (ymax-ymin)/20
    for i in range(1,Ns):
        ax.text(
            i,gamma[i-1]+dy,
            "%d"%i,
            size=10
        )
    plt.show()
    exit()

system.computeVertex('2to2')
V = system.vertex2to2
ind = 9     #of the plot

print("Scattering of index %d"%ind)
for i2 in range(Ns-1):
    print("\nWith index %d:"%i2)
    for i3 in range(Ns-1):
        for i4 in range(Ns-1):
            if V[ind,i2,i3,i4]>5e-1:
                print("\tresults into %d and %d with amplitude %.5f"%(i3,i4,V[ind,i2,i3,i4]))

    input()
#print(V.shape)
