""" Here we compute rates of decay for sweeps of temperature, amplitude and in different geometries.
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
    '1to2_1','1to2_2',
    '2to2_1','2to2_2',
    '1to3_1','1to3_2','1to3_3',
)
parameters.sca_types = sca_types
parameters.sca_broadening = 0.5
parameters.sca_saveVertex = True

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
    '7x8-rectangle': [
        7,
        8,
        (),
    ],
    '80-diamond': [
        7,      #not finished
        8,
        (),
    ],
    '144-diamond': [
        16,      #not finished
        16,
        (
            (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (0,15),
            (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,10), (1,11), (1,12), (1,13), (1,14), (1,15),
            (2,0), (2,1), (2,2), (2,3), (2,4), (2,11), (2,12), (2,13), (2,14), (2,15),
            (3,0), (3,1), (3,2), (3,3), (3,12), (3,13), (3,14), (3,15),
            (4,0), (4,1), (4,2), (4,13), (4,14), (4,15),
            (5,0), (5,1), (5,14), (5,15),
            (6,0), (6,15),
            (9,0), (9,15),
            (10,0), (10,1), (10,14), (10,15),
            (11,0), (11,1), (11,2), (11,13), (11,14), (11,15),
            (12,0), (12,1), (12,2), (12,3), (12,12), (12,13), (12,14), (12,15),
            (13,0), (13,1), (13,2), (13,3), (13,4), (13,11), (13,12), (13,13), (13,14), (13,15),
            (14,0), (14,1), (14,2), (14,3), (14,4), (14,5), (14,10), (14,11), (14,12), (14,13), (14,14), (14,15),
            (15,0), (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14), (15,15),
        )
    ],
}

save = True
lattices = (
#    '24-diamond',
#    '60-diamond',
    '60-diamond2',
#    '144-diamond',
#    '7x8-rectangle',
)
#amps = (0,0.5,1,1.5,2,12)
amps = tuple(np.linspace(0.5,4,8))
energy1 = -0.50
energies = tuple(np.linspace(-0.54,-0.15,10))
amp1 = 1

dataFn = pf.getFilename(*('ampSweep',lattices,amps,energy1,energies,amp1),dirname='Data/',extension='.pkl')
enFn = pf.getFilename(*('energySweep',lattices),dirname='Data/',extension='.pkl')
if 0 and Path(dataFn).is_file() and Path(enFn).is_file():
    with open(dataFn,"rb") as f:
        gammaAmp, gammaEn = pickle.load(f)
    with open(enFn,"rb") as f:
        evalsDic = pickle.load(f)
else:
    evalsDic = {}
    gammaAmp = {}
    for lattice in lattices:
        parameters.lat_Lx = parsLattices[lattice][0]
        parameters.lat_Ly = parsLattices[lattice][1]
        parameters.lat_offSiteList = parsLattices[lattice][2]
        parameters.lat_plotLattice = True
        system = openHamiltonian(parameters)
        evalsDic[lattice] = system.evals
        system.p.sca_temperature = system._temperature(energy1)
        #
        Phi = system.Phi
        evals = system.evals[1:]/10
        Ns = system.Ns
        system.computeRate()
        data = (
            system.rates['1to2_1']+system.rates['1to3_1']+system.rates['2to2_1'],
            system.rates['1to2_2']+system.rates['1to3_2']+system.rates['2to2_2'],
            system.rates['1to3_3'],
        )
        if lattice=='7x8-rectangle' and 0:
            fig=plt.figure(figsize=(12,5))
            ax = fig.add_subplot(131)
            ax.plot(
                np.arange(1,system.Ns),
                data[0],
            )
            ax = fig.add_subplot(132)
            ax.plot(
                np.arange(1,system.Ns),
                data[1],
            )
            ax = fig.add_subplot(133)
            ax.plot(
                np.arange(1,system.Ns),
                data[2],
            )
            plt.show()
            exit()
        gammaAmp[lattice] = np.zeros((len(amps),system.Ns-1))
        for ia in range(len(amps)):
            A = amps[ia]
            gammaAmp[lattice][ia] = data[0] + data[1]*(A/2)**2 + data[2]*(A/2)**4
    gammaEn = {}
    for lattice in lattices:
        parameters.lat_Lx = parsLattices[lattice][0]
        parameters.lat_Ly = parsLattices[lattice][1]
        parameters.lat_offSiteList = parsLattices[lattice][2]
        system = openHamiltonian(parameters)
        Ns = system.Ns
        gammaEn[lattice] = np.zeros((len(energies),system.Ns-1))
        for ie,E in enumerate(energies):
            system.p.sca_temperature = system._temperature(energies[ie])
            system.computeRate()
            gamma1 = system.rates['1to2_1']+system.rates['1to3_1']+system.rates['2to2_1']
            gamma2 = system.rates['1to2_2']+system.rates['1to3_2']+system.rates['2to2_2']
            gamma3 = system.rates['1to3_3']
            gammaEn[lattice][ie] = gamma1 + gamma2*(amp1/2)**2 + gamma3*(amp1/2)**4
    if save:
        with open(dataFn,"wb") as f:
            pickle.dump((gammaAmp,gammaEn),f)
        with open(enFn,"wb") as f:
            pickle.dump(evalsDic,f)

colors = ['navy','orange','r']
fig = plt.figure(figsize=(20,8))
for il,lattice in enumerate(lattices):
    for ia in range(len(amps)):
        ax = fig.add_subplot(len(lattices),len(amps),1+il*len(amps)+ia)
        Ns = parsLattices[lattice][0]*parsLattices[lattice][1] - len(parsLattices[lattice][2])
        ax.scatter(
            np.arange(1,Ns),
            gammaAmp[lattice][ia],
            color=colors[il],
            label=lattice
        )
        if ia==0:
            ax.legend()
            ax.set_ylabel(r"$\Gamma$",size=15)
        if il==0:
            ax.set_title("Amplitude: %.2f"%amps[ia],size=20)
        if il==1:
            ax.set_xlabel("Mode index",size=15)
fig.tight_layout()

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
cmap = plt.cm.plasma
norm = Normalize(vmin=energies[0], vmax=energies[-1])
fig = plt.figure(figsize=(15,7))
for il,lattice in enumerate(lattices):
    ax = fig.add_subplot(1,len(lattices),il+1)
    for ie in range(len(energies)):
        Ns = parsLattices[lattice][0]*parsLattices[lattice][1] - len(parsLattices[lattice][2])
        ax.scatter(
            np.arange(1,Ns),
            gammaEn[lattice][ie],
            color=cmap(norm(energies[ie])),
            label=lattice
        )
    ax.set_title(lattice,size=20)
    ax.set_xlabel("Mode index",size=15)
    ax.set_ylabel(r"$\Gamma$",size=15)
cax = fig.add_axes([0.92,0.1,0.02,0.8])
sm = ScalarMappable(norm=norm, cmap=cmap)
fig.colorbar(sm, cax=cax, label="Bath energy")
#fig.tight_layout()
plt.show()








