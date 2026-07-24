# Decay Rates

Magnon decay and scattering rates are computed using Fermi's Golden Rule
with interaction vertices obtained from the cubic and quartic terms of the
Holstein-Primakoff expansion.

## Overview

The computation has two stages:

1. **Vertex construction** — `computeVertex()` builds the interaction
   tensor :math:`V_{nlm\ldots}` from the Bogoliubov matrices :math:`U, V`
   and the bond-dependent coupling factors :math:`f_{ij}`.
2. **Rate evaluation** — `computeRate()` loops over the requested
   processes and dispatches to the corresponding function in
   :mod:`wavespin.static.decayProcesses`, which evaluates the
   Golden Rule sum.

Both results are cached to disk under ``data/``.

## Available Processes

| Code | Physical Channel | Vertex shape | Notes |
|------|-----------------|-------------|-------|
| `'1to2_1'` | 1 magnon → 2 magnons, 1st order | `(Ns,Ns,Ns)` | Reverse channel :math:`2\\to 1` included |
| `'1to2_2'` | 1 magnon → 2 magnons, 2nd order | `(Ns,Ns,Ns)` | Single channel :math:`2\\to 1` |
| `'2to2_1'` | 2 magnons → 2 magnons, 1st order | `(Ns,Ns,Ns,Ns)` | Vanishes at T = 0 |
| `'2to2_2'` | 2 magnons → 2 magnons, 2nd order | `(Ns,Ns,Ns,Ns)` | Single channel :math:`2n\\to l+m` |
| `'1to3_1'` | 1 magnon → 3 magnons, 1st order | `(Ns,Ns,Ns,Ns)` | Reverse channel :math:`3\\to 1` included |
| `'1to3_2'` | 1 magnon → 3 magnons, 2nd order | `(Ns,Ns,Ns,Ns)` | Single channel :math:`3\\to 1` |
| `'1to3_3'` | 1 magnon → 3 magnons, 3rd order | `(Ns,Ns,Ns,Ns)` | Single channel :math:`3\\to 1` |
| `'2to2_1_sc'` | 2→2 self-consistent | `(Ns,Ns,Ns,Ns)` | Iterative broadening |

## Usage

```python
from wavespin.static import openHamiltonian
from wavespin.tools import SimParams

params = SimParams()
params.lattice.Lx = 10
params.lattice.Ly = 10
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)
params.scattering.types = ('1to2_1', '2to2_1')
params.scattering.temperature = 0.0
params.scattering.broadening = 0.5

system = openHamiltonian(params)    # diagonalization runs automatically
system.computeRate()                # vertex is computed on demand
# system.rates[process]             # array of shape (Ns,), per-mode rate
```

.. note::
   ``computeVertex()`` is called automatically by ``computeRate()`` for
   each needed vertex type — you rarely need to call it yourself.

## Vertex Computation

The interaction vertices :math:`V_{n\ldots}` are built by contracting the
Bogoliubov matrices :math:`U, V` with the bond-dependent coupling factors:

.. math::
    f_{ij}^{(1)} = \\frac{g_{ij}}{\\sqrt{2S}\\, 4S} \\, (1 - D_{ij})
    \\, \\sin(2\\theta_i)

.. math::
    f_{ij}^{(2,3)} \\propto \\cos^2\\theta_i,\\ \\sin^2\\theta_i

Each vertex type is computed via :func:`numpy.einsum` contractions
(6 terms for 1→2, 22 terms for 2→2, 20 terms for 1→3), then symmetrized
over the outgoing-mode indices and doubled for the :math:`i\\leftrightarrow j`
bond counterpart.

### Memory

The 2→2 and 1→3 vertices are rank-4 dense tensors of shape
:math:`(N_s, N_s, N_s, N_s)` — roughly :math:`8 N_s^4` bytes.
For a practical lattice:

| :math:`N_s` | Vertex size |
|------------|-------------|
| 40 (40-diamond) | ~20 MB |
| 56 (7×8 rect) | ~80 MB |
| 100 | ~800 MB |
| 400 | ~204 GB |

## Rate Formula

For a 1→2 process at first order:

.. math::
    \\Gamma_n = 2\\pi \\sum_{l,m} |V_{nlm}|^2
    \\, \\delta_\\eta(E_n - E_l - E_m)
    \\, (1 + n_l + n_m)
    \\;+\\;
    \\text{reverse } (2\\to 1)

The energy-conserving :math:`\\delta`-function is broadened with a
Lorentzian of width :math:`\\eta = \\texttt{broadening} \\times
\\langle\\Delta E\\rangle`.

Bose-Einstein factors :math:`n_i = (e^{E_i/T} - 1)^{-1}` are included
when :math:`T > 0`.  At :math:`T = 0` all :math:`n_i = 0` and only
spontaneous decay contributes.

## Self-consistent Rate

``rate_2to2_1_sc`` iterates the 2→2 rate formula, using the rate itself
as the broadening width, until the mode-dependent rate converges.
Useful for studying the breakdown of the Fermi Golden Rule at strong
coupling.
