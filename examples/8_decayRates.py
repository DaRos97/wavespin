"""Magnon decay rates on diamond and rectangular lattices.

Computes Fermi Golden Rule decay rates at :math:`g_1 = 1` for two
staggered-field values (:math:`h = 0, 2`) and two state energies
(ground state and ground state plus :math:`g_1/10`).

Three first-order processes are evaluated: ``1to2_1``, ``2to2_1``,
``1to3_1``.  Separate figures are produced for the 40-diamond preset
and a :math:`7\\times 8` open rectangle.

Usage
-----
    python 8_decayRates.py
"""

import numpy as np
import matplotlib.pyplot as plt

from wavespin.tools import SimParams, DiagParams, ScatteringParams, LatticeParams
from wavespin.lattice import get_preset
from wavespin.static.open import openHamiltonian

plt.rcParams.update({'text.usetex': False})

g1 = 1
h_values = [0, 2]
processes = ['1to2_1', '2to2_1', '1to3_1']
delta_energy = g1 / 10


def compute_rates(lat_params, label_text):
    columns = [
        (f'h={h}  GS', h, None) for h in h_values
    ] + [
        (f'h={h}  GS+g1/10', h, delta_energy) for h in h_values
    ]

    fig, axes = plt.subplots(len(processes), len(columns),
                             figsize=(3 * len(columns), 3 * len(processes)))

    for jcol, (col_label, h, delta) in enumerate(columns):
        for irow, proc in enumerate(processes):
            params = SimParams(
                lattice=lat_params,
                diag=DiagParams(Hamiltonian=(g1, 0, 0, 0, h, 0)),
                scattering=ScatteringParams(types=(proc,)),
            )
            s = openHamiltonian(params)
            if delta is None:
                T = 0
            else:
                T = s._temperature(s.GSE + delta)
            s.p.scattering.temperature = T
            s.computeRate(verbose=False)

            ax = axes[irow, jcol]
            ax.scatter(np.arange(1, s.Ns), s.rates[proc],
                       s=12, c='steelblue', alpha=0.7)
            ax.set_xlabel('Mode index')
            ax.set_ylabel(r'$\Gamma$')
            if irow == 0:
                ax.set_title(col_label, fontsize=9)
            if jcol == 0:
                ax.set_ylabel(f'{proc}\n' + r'$\Gamma$')

    fig.suptitle(f'Decay rates — {label_text}, ' r'$g_1=1$', fontsize=13)
    fig.tight_layout()
    fname = f'data/figures/8_decayRates_{label_text.replace(" ", "_")}.png'
    fig.savefig(fname, dpi=150)
    print(f'Saved {fname}')


# ---- 40-diamond ---------------------------------------------------------
compute_rates(get_preset('40-diamond').p, '40-diamond')

# ---- 7x8 rectangle ------------------------------------------------------
rect = LatticeParams(Lx=7, Ly=8, boundary='open')
compute_rates(rect, '7x8 rectangle')

plt.show()

