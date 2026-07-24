# Correlators

Dynamical spin-spin correlators are computed via Wick contractions of the
Holstein-Primakoff expansion using the Bogoliubov propagators.

The computation is handled by :class:`~wavespin.static.open.openCorrelators`,
which extends :class:`~wavespin.static.open.openHamiltonian`.

## Correlator Types

| Type | Operator | Physical Meaning |
|------|----------|------------------|
| `'zz'` | :math:`S^z_i S^z_j` | Longitudinal spin-spin correlator |
| `'xx'` | :math:`S^x_i S^x_j` | Transverse spin-spin correlator |
| `'ee'` | :math:`S^+_i S^-_j + \\text{h.c.}` | Excitation correlator |
| `'jj'` | :math:`J_i J_j` | Bond current correlator |
| `'ze'` | :math:`S^z_i (S^+_j - S^-_j)` | Mixed longitudinal-transverse |
| `'ez'` | :math:`(S^+_i - S^-_i) S^z_j` | Mixed transverse-longitudinal |

## Real-Space Correlators

Computed by ``openCorrelators.realSpaceCorrelator()``:

```python
from wavespin.static import openCorrelators
from wavespin.tools import SimParams

params = SimParams()
params.lattice.Lx = 10
params.lattice.Ly = 10
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)
params.correlator.correlatorType = 'zz'
params.correlator.perturbationSite = (5, 5)

sys = openCorrelators(params)
sys.realSpaceCorrelator()  # populates sys.correlatorXT (Ns × nTimes)
```

Both ground-state (T=0) and finite-temperature correlators are computed
using Bose-Einstein factors from ``p.correlator.energy``.

### How It Works

1. **Bogoliubov propagators** — four tensors of shape :math:`(N_s, N_s, N_\\text{times})` are built:

   .. math::
        A_{ij}(t) &= \\sum_k e^{-i E_k t} V_{ik} U_{jk} \\\\
        B_{ij}(t) &= \\sum_k e^{-i E_k t} U_{ik} V_{jk} \\\\
        G_{ij}(t) &= \\sum_k e^{-i E_k t} V_{ik} V_{jk} \\\\
        H_{ij}(t) &= \\sum_k e^{-i E_k t} U_{ik} U_{jk}

   At finite temperature, corrections using Bose-Einstein factors
   :math:`n_k = (e^{E_k/T} - 1)^{-1}` are added.

2. **Wick contractions** — for each site :math:`i`, the spin operator is
   expanded into bosonic terms via the Holstein-Primakoff transformation
   (e.g., :math:`S^z \\approx S - a^\\dagger a`,
   :math:`S^x \\approx \\sqrt{S/2}\\,(a + a^\\dagger)`).
   All pairings are enumerated and evaluated using the propagators.

3. **Magnon order** — the parameter ``p.correlator.magnonOrder`` controls
   which terms of the HP expansion are kept:
   ``(1,)`` = only 1-magnon contributions,
   ``(1,2)`` = 1- and 2-magnon, etc.

## Bond Correlators

``realSpaceCorrelatorBond()`` computes correlators of J-operators on
horizontal and vertical bonds, producing ``correlatorXT_h`` and
``correlatorXT_v``.

## Momentum-Space Correlators

Fourier-transforms the real-space correlators to :math:`(k, \\omega)` space:

```python
sys.momentumSpaceCorrelator()  # populates sys.correlatorKW
```

Available transforms (via ``CorrelatorParams.transformType``):

| Transform | Description | Boundary |
|-----------|-------------|----------|
| `'fft'` | 2D Fast Fourier Transform | Periodic |
| `'dct'` | 2D Discrete Cosine Transform | Open (Neumann-like) |
| `'dst'` | 2D Discrete Sine Transform | Open (Dirichlet-like) |
| `'dat'` | Uses Bogoliubov eigenfunctions as basis | Open |
