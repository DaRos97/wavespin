# wavespin

Mean-field simulations of quantum simulators using the Holstein-Primakoff
(spin-wave) approximation for XY-like Hamiltonians.

The code simulates spin systems on 2D square lattices undergoing a quench
from a staggered magnetic field into a symmetry-broken phase. It computes
classical ground states, Bogoliubov quasiparticle spectra, dynamical spin
correlators, and magnon decay/scattering rates.

## Installation

Requires Python >= 3.11.

```bash
git clone https://github.com/DaRos97/wavespin
cd wavespin
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[docs]"
```

## Quick Start

```python
from wavespin.tools import SimParams
from wavespin.static import openHamiltonian

params = SimParams()
params.lattice.Lx = 20
params.lattice.Ly = 20
params.lattice.boundary = 'periodic'
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)

system = openHamiltonian(params.lattice)
system.diagonalize()
print("Quasiparticle energies:", system.evals[:10])
```

## Package Overview

| Module | Purpose |
|--------|---------|
| `wavespin.lattice` | Lattice geometry, site indexing, NN/NNN construction, preset shapes |
| `wavespin.classicSpins` | Classical ground-state determination (RMO, Monte Carlo, OBC angles) |
| `wavespin.static` | Bogoliubov diagonalization, dynamical correlators, decay rates |
| `wavespin.tools` | Parameter system, input parsing, file paths, math utilities |
| `wavespin.plots` | Lattice plots, dispersion maps, correlator visualization, rate curves |

## Workflow

```
Lattice construction → Classical ground state → Bogoliubov diagonalization
  → Quasiparticle energies & wavefunctions → Real-space correlators
  → Momentum-space correlators → Decay/scattering rates → Plotting
```
