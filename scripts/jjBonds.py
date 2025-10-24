""" Here we compute the jj correlator for square lattices.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openSystem, openRamp

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation in OBC")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

""" Define the parameters of the system at different 'times' """
parameters.dia_Hamiltonian = (10,0,0,0,0,0)
magModes = [(1,2,3),(1,),(2,),(3,)]
ens = [-0.52,]
data = [[] for i in range(len(magModes))]
for im in range(len(magModes)):
    parameters.cor_magnonModes = magModes[im]
    parameters.cor_energy = ens[0]
    sys0 = openSystem(parameters)
    sys0.realSpaceCorrelatorBond(verbose=verbose)
    # Extract bonds: specific for fig.3 -> perturbation 2,2 in 6x6 chip
    # In order: hor bond A, ver bond up, hor bond, ver bond down
    data[im] = [-sys0.correlatorXT_h[2,2], -sys0.correlatorXT_v[3,2], sys0.correlatorXT_h[2,3], sys0.correlatorXT_v[2,2] ]

nTimes = np.arange(sys0.nTimes) / 10

# Plot
import matplotlib.pyplot as plt
colors = ['purple','k','grey','silver']
fig,axs = plt.subplots(1,4,sharey=True,figsize=(15,7))
title = ["all","1","2","3"]
for im in range(len(magModes)):
    ax = axs[im]
    ax.set_title("Magnons: "+title[im])
    bonds = data[im]
    tot = np.zeros_like(bonds[0])
    for i in range(4):
        ax.plot(nTimes,
                np.imag(bonds[i]),
                color=colors[i],
                ls='dashed' if i != 0 else '-',
                lw=1 if i != 0 else 3,
                zorder = 8-2*i,
                )
        tot += bonds[i]
    ax.plot(nTimes,
            np.imag(tot),
            color='teal',
            ls='-',
            lw=3,
            zorder=7,
            )

    ax.set_xlim(0,3)
plt.show()















