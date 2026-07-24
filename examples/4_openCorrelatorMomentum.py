"""Open-boundary zz correlator in momentum space (DCT).

Computes the dynamical longitudinal correlator :math:`\\langle[Z_i(t), Z_j(0)]\\rangle`
on a :math:`7\\times 8` open lattice at :math:`g_1 = 1` for four
staggered-field values (:math:`h = 0, 2, 4, 6`), then transforms to
momentum space via the discrete cosine transform.

Each panel shows :math:`|\\chi_{zz}(k_x, k_y)|`.

Usage
-----
    python 4_openCorrelatorMomentum.py
"""

import numpy as np
import matplotlib.pyplot as plt

from wavespin.tools import SimParams, LatticeParams, DiagParams, CorrelatorParams
from wavespin.static.open import openCorrelators

g1 = 1
h_values = [0, 2, 4, 6]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ih, h in enumerate(h_values):
    params = SimParams(
        lattice=LatticeParams(Lx=7, Ly=8, boundary='open'),
        diag=DiagParams(Hamiltonian=(g1, 0, 0, 0, h, 0)),
        correlator=CorrelatorParams(
            correlatorType='zz',
            transformType='dct',
            perturbationSite=(3, 4),
            energy=-100,
        ),
    )
    s = openCorrelators(params)
    s.realSpaceCorrelator()
    s.momentumSpaceCorrelator()

    ax = axes[ih]
    im = ax.imshow(
        np.abs(s.correlatorKW),
        origin='lower', aspect='auto', cmap='inferno',
    )
    ax.set_title(f'zz, h={h}')
    ax.set_xlabel(r'$k_x$ index')
    ax.set_ylabel(r'$k_y$ index')
    fig.colorbar(im, ax=ax, fraction=0.046, label=r'$|\chi_{zz}|$')

fig.suptitle(r'ZZ correlator — momentum space (DCT), $g_1=1$, 7×8', fontsize=13)
fig.tight_layout()
fig.savefig('data/figures/4_zz_momentum.png', dpi=150)
print('Saved data/figures/4_zz_momentum.png')
plt.show()
