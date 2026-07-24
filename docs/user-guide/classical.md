# Classical Ground State

The classical ground state determines the quantization axes for the spin-wave expansion.

## Methods

### RMO (Regular Magnetic Order)

The 5-angle variational ansatz from `wavespin.classicSpins.RMO`:

```python
from wavespin.classicSpins import computeClassicalGroundState, plotClassicalPhaseDiagram
from wavespin.tools import SimParams

params = SimParams()
params.lattice.boundary = 'periodic'

result = computeClassicalGroundState(params, j2_values, h_values)
plotClassicalPhaseDiagram(result)
```

This sweeps \\( J_2 \\) and \\( h \\) to find the phase boundaries between the
canted-Néel and canted-stripe phases.

### Monte Carlo

Simulated annealing from `wavespin.classicSpins.montecarlo`:

```python
from wavespin.lattice import latticeClass
from wavespin.classicSpins import XXZJ1J2MC
from wavespin.tools import LatticeParams

lat = latticeClass(LatticeParams(Lx=10, Ly=10))
mc = XXZJ1J2MC(lat, J1=1.0, J2=0.0)
mc.anneal(T_max=10.0, T_min=0.01, n_sweeps=10000)
```

Uses Metropolis updates + overrelaxation sweeps. Measures magnetization,
staggered magnetization, and structure factor.

### OBC Angle Minimization

For open boundaries, site-dependent canting angles are obtained by minimizing
the classical energy:

```python
from wavespin.classicSpins import classicMagnetization

angles = classicMagnetization(system)  # system is an openHamiltonian instance
```

Uses `scipy.optimize.minimize` (Nelder-Mead) over the angles \\( \theta_i \\).
