""" Here we compute the jj correlator for square lattices.
"""

import numpy as np
import matplotlib.pyplot as plt

from wavespin.tools import SimParams, LatticeParams, DiagParams, CorrelatorParams
from wavespin.static.open import openCorrelators

energies = np.linspace(-0.55, -0.25, 4)
Lx = 6
Ly = 6
pS = (2, 2)

magModes = [(1, 2, 3), (1,), (2,), (3,)]
nE = len(energies)
nM = len(magModes)
data = np.zeros((nE, nM, 4, 401), dtype=complex)

for ie in range(nE):
    for im in range(nM):
        params = SimParams(
            lattice=LatticeParams(Lx=Lx, Ly=Ly, boundary='open'),
            diag=DiagParams(Hamiltonian=(10, 0, 0, 0, 0, 0)),
            correlator=CorrelatorParams(
                correlatorType='jj',
                perturbationSite=pS,
                magnonOrder=magModes[im],
            ),
        )
        s = openCorrelators(params)
        if ie == 0:
            energies[0] = s.GSE
        s.p.correlator.energy = energies[ie]
        s.realSpaceCorrelatorBond()
        data[ie, im] = np.array([
            -s.correlatorXT_h[pS[0], pS[1]],
            -s.correlatorXT_v[pS[0] + 1, pS[1]],
            s.correlatorXT_h[pS[0], pS[1] + 1],
            s.correlatorXT_v[pS[0], pS[1]],
        ])

nTimes = np.arange(s.nTimes) / 10

colors = ['purple', 'k', 'darkgrey', 'silver']
fig, axs = plt.subplots(nE, nM, sharey=True, sharex=True, figsize=(18, 15))
title_m = ['all', '1', '2', '3']
s_ = 20

for ie in range(nE):
    for im in range(nM):
        ax = axs[ie, im]
        if ie == 0:
            ax.set_title(f'Magnons: {title_m[im]}', size=s_)
        if im == 0:
            ax.set_ylabel(f'E = {energies[ie]:.2f}', size=s_)
        bonds = data[ie, im]
        tot = np.zeros_like(bonds[0])
        for i in range(4):
            ax.plot(nTimes, np.imag(bonds[i]),
                    color=colors[i],
                    ls='dashed' if i > 0 else '-',
                    lw=2 if i > 0 else 3,
                    zorder=8 - 2 * i)
            tot += bonds[i]
        ax.plot(nTimes, np.imag(tot),
                color='teal', ls='-', lw=3, zorder=7)
        ax.set_xlim(0, 3)

fig.tight_layout()
fig.savefig('data/figures/jjBonds.png', dpi=150)
print('Saved data/figures/jjBonds.png')
