"""Open-boundary zz correlator in momentum space (DCT).

Computes the dynamical longitudinal correlator :math:`\\langle[Z_i(t), Z_j(0)]\\rangle`
on a :math:`18\\times 20` open lattice at :math:`g_1 = 1` for four
staggered-field values (:math:`h = 0, 2, 4, 6`), then transforms to
momentum space via the discrete cosine transform.

Each panel shows the binned intensity :math:`|\\chi_{zz}(|k|, \\omega)|`
with frequency on the y-axis and momentum magnitude on the x-axis.

Usage
-----
    python 4_openCorrelatorMomentum.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fftfreq

from wavespin.tools import SimParams, LatticeParams, DiagParams, CorrelatorParams
from wavespin.static.open import openCorrelators

g1 = 1
h_values = [0, 2, 4, 6]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ih, h in enumerate(h_values):
    params = SimParams(
        lattice=LatticeParams(Lx=18, Ly=20, boundary='open'),
        diag=DiagParams(Hamiltonian=(g1, 0, 0, 0, h, 0)),
        correlator=CorrelatorParams(
            correlatorType='zz',
            transformType='dct',
            perturbationSite=(9, 10),
            energy=-100,
        ),
    )
    s = openCorrelators(params)
    s.realSpaceCorrelator()
    s.momentumSpaceCorrelator()

    k_flat = np.sqrt(s.momentum[:, 0]**2 + s.momentum[:, 1]**2)
    corr = np.abs(s.correlatorKW)
    freqs = fftshift(fftfreq(s.nOmega, s.fullTimeMeasure / s.nTimes))

    num_k_bins = 50
    k_bins = np.linspace(0, np.sqrt(2) * np.pi, num_k_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    K_mesh, W_mesh = np.meshgrid(k_centers, freqs, indexing='ij')

    P_k_omega = np.zeros((num_k_bins, s.nOmega))
    for i in range(num_k_bins):
        mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i + 1])
        if np.any(mask):
            P_k_omega[i, :] = np.mean(corr[mask, :], axis=0)

    ax = axes[ih]
    ax.set_facecolor('black')
    ylim = 7
    mesh = ax.pcolormesh(K_mesh, W_mesh, P_k_omega,
                         shading='auto', cmap='Blues')
    ax.set_title(f'zz, h={h}')
    ax.set_xlabel(r'$|k|$')
    ax.set_ylabel(r'$\omega$')
    ax.set_ylim(-ylim, ylim)

fig.suptitle(r'ZZ correlator — momentum space (DCT), $g_1=1$, 7×8', fontsize=13)
fig.tight_layout()
fig.savefig('data/figures/4_zz_momentum.png', dpi=150)
print('Saved data/figures/4_zz_momentum.png')
plt.show()
