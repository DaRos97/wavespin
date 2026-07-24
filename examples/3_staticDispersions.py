"""Dispersion relation for periodic and open boundary conditions.

Computes the spin-wave dispersion on a 10x10 lattice at g1=1 for three
staggered-field values (h = 0, 1.5, 3).  The left column shows the 2D
momentum-space dispersion for periodic boundary conditions; the right column
shows the quasiparticle eigenvalues (sorted) for open boundary conditions.

Usage
-----
    python 3_staticDispersions.py
"""

import numpy as np
import matplotlib.pyplot as plt

from wavespin.tools import SimParams, LatticeParams, DiagParams
from wavespin.static.periodic import periodicHamiltonian
from wavespin.static.open import openHamiltonian

g1 = 1
h_values = [0, 2, 4, 6]
Lx, Ly = 18, 20

fig = plt.figure(figsize=(12, 18))

for irow, h in enumerate(h_values):
    # ---- Periodic (3D surface) ---------------------------------------------
    params_pbc = SimParams(
        lattice=LatticeParams(Lx=Lx, Ly=Ly, boundary='periodic'),
        diag=DiagParams(Hamiltonian=(g1, 0, 0, 0, h, 0)),
    )
    per = periodicHamiltonian(params_pbc)

    n_rows = len(h_values)
    ax = fig.add_subplot(n_rows, 2, 2 * irow + 1, projection='3d')
    KX, KY = np.meshgrid(per.gridk[:, 0, 0], per.gridk[0, :, 1], indexing='ij')
    ax.plot_surface(KX, KY, per.dispersion, cmap='plasma')
    ax.set_title(f"Periodic  (g1={g1}, h={h})")
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_zlabel(r'$\varepsilon(k)$')
    ax.set_aspect('equalxy')

    # ---- Open (eigenvalues) ------------------------------------------------
    params_obc = SimParams(
        lattice=LatticeParams(Lx=Lx, Ly=Ly, boundary='open'),
        diag=DiagParams(Hamiltonian=(g1, 0, 0, 0, h, 0)),
    )
    ope = openHamiltonian(params_obc)

    ax = fig.add_subplot(n_rows, 2, 2 * irow + 2)
    ax.scatter(np.arange(1, ope.Ns), ope.evals[1:], s=15, c='steelblue')
    ax.set_title(f"Open  (g1={g1}, h={h})")
    ax.set_xlabel('Mode index')
    ax.set_ylabel(r'$\varepsilon_n$')

fig.tight_layout()
fig.savefig("data/figures/3_staticDispersions.png", dpi=150)
print("Saved data/figures/3_staticDispersions.png")
plt.show()
