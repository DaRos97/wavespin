# Correlators

Dynamical spin-spin correlators are computed via Wick contractions of the
Holstein-Primakoff expansion using the Bogoliubov propagators.

## Correlator Types

| Type | Operator | Physical Meaning |
|------|----------|------------------|
| `'zz'` | \\( S^z_i S^z_j \\) | Longitudinal spin-spin correlator |
| `'xx'` | \\( S^x_i S^x_j \\) | Transverse spin-spin correlator |
| `'ee'` | \\( S^+_i S^-_j + \text{h.c.} \\) | Excitation correlator |
| `'jj'` | \\( J_i J_j \\) | Bond current correlator |
| `'ze'` | \\( S^z_i (S^+_j - S^-_j) \\) | Mixed longitudinal-transverse |
| `'ez'` | \\( (S^+_i - S^-_i) S^z_j \\) | Mixed transverse-longitudinal |

## Real-Space Correlators

Computed by `openSystem.realSpaceCorrelator()`:

```python
from wavespin.static import openSystem
from wavespin.tools import SimParams

params = SimParams()
params.lattice.Lx = 10
params.lattice.Ly = 10
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)
params.correlator.correlatorType = 'zz'
params.correlator.perturbationSite = (5, 5)

sys = openSystem(params)
sys.diagonalize()
sys.realSpaceCorrelator()  # returns <[A_i(t), B_j(0)]> for all i,j,t
```

The time grid is specified by the `measureTimeList` attribute. Both ground-state
(T=0) and finite-temperature correlators are computed using Bose-Einstein factors.

## Bond Correlators

`realSpaceCorrelatorBond()` computes correlators of bond variables (J-operators)
on each bond type (NN, NNN).

## Momentum-Space Correlators

Fourier-transforms the real-space correlators to \\( (k, \omega) \\) space:

```python
sys.momentumSpaceCorrelator()  # dispatches to the transform type
```

Available transforms (via `CorrelatorParams.transformType`):

| Transform | Description | Boundary |
|-----------|-------------|----------|
| `'fft'` | 2D Fast Fourier Transform | Periodic |
| `'dct'` | 2D Discrete Cosine Transform | Open (Neumann-like) |
| `'dst'` | 2D Discrete Sine Transform | Open (Dirichlet-like) |
| `'dat'` | Uses Bogoliubov eigenfunctions as basis | Open |

## Implementation: Wick Contractions

The Holstein-Primakoff expansion expresses spin operators as series in bosonic
operators (e.g., \\( S^x \approx \sqrt{S/2} \\, a + \ldots \\)). For a correlator
like \\( \langle S^z_i(t) S^z_j(0) \rangle \\), each operator is expanded, all
cross-terms are multiplied, and Wick's theorem reduces each term to a sum over
pairings of the Bogoliubov propagators \\( A, B, G, H \\).
