"""Open-boundary jj bond-correlator time traces.

Computes the bond-current correlator :math:`\\langle[J_i(t), J_j(0)]\\rangle`
on the four bonds surrounding the perturbation site on a :math:`6\\times 6`
open lattice at :math:`g_1 = 10, h = 0`, for four energy values and four
magnon-order settings, following the setup of ``scripts/jjBonds.py``.

Usage
-----
    python 4_openCorrelatorTime.py
"""

import numpy as np
import matplotlib.pyplot as plt

from wavespin.tools import SimParams, LatticeParams, DiagParams, CorrelatorParams
from wavespin.static.open import openCorrelators

Lx, Ly = 6, 6
pS = (2, 2)
magnon_orders = [(1, 2, 3), (1,), (2,), (3,)]
nM = len(magnon_orders)

data = np.zeros((nM, 4, 401), dtype=complex)

for im in range(nM):
    params = SimParams(
        lattice=LatticeParams(Lx=Lx, Ly=Ly, boundary='open'),
        diag=DiagParams(Hamiltonian=(10, 0, 0, 0, 0, 0)),
        correlator=CorrelatorParams(
            correlatorType='jj',
            perturbationSite=pS,
            magnonOrder=magnon_orders[im],
            energy=-100,
        ),
    )
    s = openCorrelators(params)
    s.p.correlator.energy = s.GSE
    s.realSpaceCorrelatorBond()

    data[im] = np.array([
        -s.correlatorXT_h[pS[0], pS[1]],
        -s.correlatorXT_v[pS[0] + 1, pS[1]],
        s.correlatorXT_h[pS[0], pS[1] + 1],
        s.correlatorXT_v[pS[0], pS[1]],
    ])

nTimes = s.nTimes
times = np.arange(nTimes) / 10

fig, axes = plt.subplots(1, nM, sharey=True, sharex=True, figsize=(18, 5))
title_m = ['all', '1', '2', '3']
colors = ['purple', 'k', 'darkgrey', 'silver']

for im in range(nM):
    ax = axes[im]
    ax.set_title(f'Magnons: {title_m[im]}', fontsize=20)

    bonds = data[im]
    tot = np.zeros_like(bonds[0])
    for i in range(4):
        ax.plot(times, np.imag(bonds[i]),
                color=colors[i],
                ls='dashed' if i > 0 else '-',
                lw=2 if i > 0 else 3,
                zorder=8 - 2 * i)
        tot += bonds[i]
    ax.plot(times, np.imag(tot),
            color='teal', ls='-', lw=3, zorder=7)

    ax.set_xlim(0, 3)
    ax.set_xlabel('Time')

axes[0].set_ylabel(r'$\mathrm{Im}\langle[J_i(t),J_j(0)]\rangle$')

fig.tight_layout()
fig.savefig('data/figures/4_jj_bonds.png', dpi=150)
print('Saved data/figures/4_jj_bonds.png')
plt.show()
