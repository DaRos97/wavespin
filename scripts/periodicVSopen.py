""" Here I compute the decay rate of modes comparing between periodic and open boundary conditions.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
import wavespin.tools.pathFinder as pf
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
from pathlib import Path

parameters = importParameters()
parameters.dia_Hamiltonian = (10,0,0,0,0,0)
sca_types = ('2to2_1',)

parameters.sca_types = sca_types
parameters.sca_broadening = 0.5
parameters.sca_saveVertex = True
parameters.sca_temperature = 8

sizes = [6,8,10,12]
gamma = []
evals = []
for iL in range(len(sizes)):
    Lx = sizes[iL]
    Ly = sizes[iL]
    print("%dx%d"%(Lx,Ly))
    evals.append( np.zeros((2,Lx*Ly-1)) )
    gamma.append( np.zeros((2,Lx*Ly-1)) )
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.lat_plotLattice = False

    #Open
    print("Computing OBC")
    parameters.lat_boundary = 'open'
    system = openHamiltonian(parameters)
    system.computeRate()
    gamma[-1][0] = system.rates[sca_types[0]]
    evals[-1][0] = system.evals[1:]

    #Periodic
    print("Computing PBC")
    parameters.lat_boundary = 'periodic'
    system = openHamiltonian(parameters)
    system.computeRate()
    gamma[-1][1] = system.rates[sca_types[0]]
    evals[-1][1] = system.evals[1:]

fig = plt.figure(figsize=(10,5))
ax0 = fig.add_subplot(1,2,1)
ax1 = fig.add_subplot(1,2,2)

cmap = plt.cm.cividis
from matplotlib.colors import Normalize
norm = Normalize(vmin=sizes[0], vmax=sizes[-1])
for iL in range(len(sizes)):
    Lx = sizes[iL]
    Ly = sizes[iL]
    ax0.scatter(
        evals[iL][0],
        gamma[iL][0],
        color=cmap(norm(Lx)),
        label="Size:%dx%d"%(Lx,Ly)
    )
    ax1.scatter(
        evals[iL][1],
        gamma[iL][1],
        color=cmap(norm(Lx)),
    )

ax0.legend(fontsize=15)
# y limit
ax0.set_title("Open system",size=20)
ymin0,ymax0 = ax0.get_ylim()
ax1.set_title("Periodic system",size=20)
ymin1,ymax1 = ax1.get_ylim()

ax0.set_ylim(min(ymin0,ymin1),max(ymax0,ymax1))
ax1.set_ylim(min(ymin0,ymin1),max(ymax0,ymax1))

# labels
ax0.set_xlabel("Energy",size=15)
ax1.set_xlabel("Energy",size=15)
ax0.set_ylabel(r"$\Gamma(2\rightarrow 2)$",size=15)
ax1.set_ylabel(r"$\Gamma(2\rightarrow 2)$",size=15)


#
fig.tight_layout()
plt.show()




















