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

Transforms the real-space correlator :math:`\\chi_{ij}(t)` to
:math:`\\chi(k, \\omega)`:

```python
sys.momentumSpaceCorrelator()
# stores: sys.correlatorKW  — shape (Ns, nOmega)
#         sys.momentum      — shape (Ns, 2), (kx, ky) per mode
```

The transformation is done by first applying a spatial transform
(DCT/FFT/DST) to each time slice, then an FFT along the time axis per
momentum point.

### Available Transforms

| Key | Function | Description |
|-----|----------|-------------|
| `'fft'` | 2D FFT | Standard 2D FFT on Lx×Ly grid (periodic) |
| `'dct'` | DCT-II | Discrete cosine transform — open boundaries, Neumann-like. ``ortho`` normalised |
| `'dst'` | DST-I | Discrete sine transform — open boundaries, Dirichlet-like. ``ortho`` normalised |
| `'dat'` | Bogoliubov basis | Projects onto Bogoliubov eigenfunctions |
| `'dat2'` | same, alternative implementation | |

### Result structure

After ``momentumSpaceCorrelator()``:

- ``correlatorKW`` — shape ``(Ns, nOmega)``.  Each row *i* is the
  frequency-domain correlator at momentum :math:`\\mathbf{k}_i`.
- ``momentum`` — shape ``(Ns, 2)``.  Each row is :math:`(k_x, k_y)`
  for the corresponding mode.

For ``'fft'`` the result has shape ``(Lx, Ly, nOmega)`` instead (the
full 2D k-grid is preserved).

### Plotting

To produce a :math:`\\omega`-vs-:math:`|k|` colormap (as in
:func:`~wavespin.plots.rampPlots.plotRampKW`):

1. Compute :math:`|k| = \\sqrt{k_x^2 + k_y^2}` from ``momentum``
2. Bin the ``correlatorKW`` rows by :math:`|k|` magnitude
3. Compute the frequency axis via :func:`scipy.fft.fftfreq`
4. Use ``pcolormesh`` to plot the binned intensity

```python
|k| = sqrt(momentum[:,0]**2 + momentum[:,1]**2)
corr = abs(correlatorKW)
freqs = fftshift(fftfreq(nOmega, fullTimeMeasure / nTimes))

k_bins = linspace(0, sqrt(2)*pi, num_k_bins + 1)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

P = zeros((num_k_bins, nOmega))
for i in range(num_k_bins):
    mask = (|k| >= k_bins[i]) & (|k| < k_bins[i+1])
    if mask.any():
        P[i] = mean(corr[mask], axis=0)

pcolormesh(k_centers, freqs, P, ...)
```

### Transform implementations

.. seealso::
   :mod:`wavespin.static.momentumTransformation` for the full source.
