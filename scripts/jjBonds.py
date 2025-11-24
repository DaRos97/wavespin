""" Here we compute the jj correlator for square lattices.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openSystem, openRamp

""" Parameters and options """
energies = np.linspace(-0.55,-0.25,4)
Lx = 6
Ly = 6
offSiteList = ()
pS = (2,2)

parameters = importParameters()
parameters.lat_Lx = Lx
parameters.lat_Ly = Ly
parameters.lat_offSiteList = offSiteList
parameters.lat_plotLattice = 0#True
parameters.cor_correlatorType = 'jj'
parameters.cor_perturbationSite = pS

""" Define the parameters of the system at different 'times' """
parameters.dia_Hamiltonian = (10,0,0,0,0,0)
magModes = [(1,2,3),(1,),(2,),(3,)]
data = np.zeros((len(energies),4,4,401),dtype=complex)
for ie in range(len(energies)):
    for im in range(len(magModes)):
        parameters.cor_magnonModes = magModes[im]
        sys0 = openSystem(parameters)
        if ie == 0:
            energies[0] = sys0.GSE
        sys0.p.cor_energy = energies[ie]
        sys0.realSpaceCorrelatorBond()
        # Extract bonds: specific for fig.3 -> perturbation 2,2 in 6x6 chip
        # In order: hor bond A, ver bond up, hor bond, ver bond down
        data[ie,im] = np.array([-sys0.correlatorXT_h[2,2], -sys0.correlatorXT_v[3,2], sys0.correlatorXT_h[2,3], sys0.correlatorXT_v[2,2] ])

nTimes = np.arange(sys0.nTimes) / 10

# Plot
import matplotlib.pyplot as plt
colors = ['purple','k','darkgrey','silver']
fig,axs = plt.subplots(4,4,sharey=True,sharex=True,figsize=(18,15))
title_m = ["all","1","2","3"]
s_ = 20
for ie in range(len(energies)):
    for im in range(len(magModes)):
        ax = axs[ie,im]
        if ie==0:
            ax.set_title("Magnons: "+title_m[im],size=s_)
        if im==0:
            ax.set_ylabel("E = %.2f"%energies[ie],size=s_)
        bonds = data[ie,im]
        tot = np.zeros_like(bonds[0])
        for i in range(4):
            ax.plot(nTimes,
                    np.imag(bonds[i]),
                    color=colors[i],
                    ls='dashed' if i != 0 else '-',
                    lw=2 if i != 0 else 3,
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

fig.tight_layout()
plt.show()















