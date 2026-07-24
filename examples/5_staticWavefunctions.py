"""Bogoliubov wavefunctions on a diamond-shaped lattice.

Computes the real-space wavefunctions :math:`\\Phi_{i,n}` for the 40-diamond
preset on an :math:`8\\times8` grid with open boundaries.  Each column of
``Phi`` is a mode; the first eight modes are plotted as 2D colormaps for
four staggered-field values (:math:`h = 0, 2, 4, 6`) at :math:`g_1 = 1`.

Usage
-----
    python 5_staticWavefunctions.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from wavespin.tools import SimParams, DiagParams
from wavespin.lattice import get_preset
from wavespin.static.open import openHamiltonian

g1 = 1
h_values = [0, 2, 4, 6]
lat = get_preset('40-diamond')
n_modes = lat.Ns

fig, axes = plt.subplots(
    len(h_values), n_modes,
    figsize=(0.6 * n_modes, 1.5 * len(h_values)),
)

for irow, h in enumerate(h_values):
    params = SimParams(
        lattice=lat.p,
        diag=DiagParams(Hamiltonian=(g1, 0, 0, 0, h, 0)),
    )
    system = openHamiltonian(params)

    for imode in range(n_modes):
        ax = axes[irow, imode]
        wf = lat.patchFunction(system.Phi[:, imode])
        norm = TwoSlopeNorm(vcenter=0)
        im = ax.imshow(wf.T, origin='lower', cmap='RdBu_r', norm=norm,
                       extent=(0, lat.Lx, 0, lat.Ly))
        ax.set_xticks([])
        ax.set_yticks([])
        if irow == 0:
            ax.set_title(f"Mode {imode + 1}", fontsize=10)
        if imode == 0:
            ax.set_ylabel(f"h={h}", fontsize=11)

fig.suptitle(
    f"Bogoliubov wavefunctions — 40-diamond, " r"$g_1=1$",
    fontsize=14,
)
fig.tight_layout()
fig.savefig("data/figures/5_staticWavefunctions.png", dpi=150)
print("Saved data/figures/5_staticWavefunctions.png")
plt.show()
