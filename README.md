# wavespin

Mean-field simulations of quantum simulators using the Holstein-Primakoff (spin-wave) approximation for XY-like Hamiltonians.

The code simulates spin systems on 2D square lattices undergoing a quench from a staggered magnetic field into a symmetry-broken phase. It computes classical ground states, Bogoliubov quasiparticle spectra, dynamical spin correlators, and magnon decay/scattering rates.

## Installation

Requires Python >= 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development with tests:

```bash
pip install -e ".[test]"
pytest tests/
```

## Quick Start

```python
from wavespin.tools import SimParams, LatticeParams
from wavespin.static import openHamiltonian, openSystem
from wavespin.lattice import get_preset

# Use a predefined diamond-shaped lattice
system = openHamiltonian(get_preset('60-diamond', plotLattice=True))

# Or build a custom lattice
params = SimParams()
params.lattice.Lx = 20
params.lattice.Ly = 20
params.lattice.boundary = 'periodic'
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)  # J1/2=5, J2=0, D1=0, D2=0, h=0
system = openHamiltonian(params.lattice)
system.diagonalize()                           # Bogoliubov diagonalization
print("Eigenvalues:", system.evals[:10])       # quasiparticle energies
```

Run example scripts from the command line:

```bash
python examples/3_staticDispersion.py examples/input_3.txt
```

## Project Structure

```
wavespin/
  lattice/              Lattice geometry, site indexing, NN/NNN construction, preset shapes
    lattice.py            latticeClass — base class for all simulations
    presets.py            Predefined diamond-shaped lattices (12 to 144 sites)
  classicSpins/         Classical ground-state determination
    RMO.py                Regular Magnetic Order (5-angle variational ansatz)
    montecarlo.py         Monte Carlo simulated annealing
    anglesOBC.py          Site-dependent angle minimization for open boundaries
  static/               Quantum (spin-wave) calculations — the core of the package
    open.py               openHamiltonian — real-space Bogoliubov diagonalization
                          openSystem — adds correlator computation on top
                          openRamp — container for parameter sweeps
    periodic.py           periodicHamiltonian — analytical k-space Bogoliubov theory
    correlators.py        Wick-contraction-based dynamical spin correlators
    decayProcesses.py     Fermi Golden Rule magnon decay/scattering rates
    momentumTransformation.py  Fourier transforms (FFT, DCT, DST, DAT)
  tools/                Utilities
    inputUtils.py         SimParams nested dataclass and input file parser
    pathFinder.py         Deterministic file naming and path management
    functions.py          Math utilities (Lorentzians, rotation matrices, diffusion modes)
  plots/                Visualization
    latticePlots.py       Lattice structure plots (sites, bonds, quantization angles)
    classicPlots.py       Classical phase diagram and spin configuration plots
    rampPlots.py          Dispersion maps, (k,ω) correlator maps, wavefunction plots
examples/               Example scripts demonstrating usage
scripts/                Research scripts for specific calculations
tests/                  Unit tests (lattice, inputs, plots)
```

## Parameters

Parameters are defined as nested dataclasses in `wavespin.tools.inputUtils`:

```
SimParams
  ├── lattice: LatticeParams       Lx, Ly, offSiteList, boundary, plotLattice
  ├── diag: DiagParams             Hamiltonian, excludeZeroMode, uniformQA, saveWf, ...
  ├── correlator: CorrelatorParams  correlatorType, transformType, perturbationSite, ...
  └── scattering: ScatteringParams  types, temperature, broadening, saveVertex, ...
```

Each sub-dataclass validates its own fields. Parameters can be loaded from a key-value input file:

```python
from wavespin.tools import importParameters
params = importParameters('input.txt')
```

## Workflow

The typical simulation pipeline:

```
Lattice construction (latticeClass / presets)
  → Classical ground state (RMO or Monte Carlo) → quantization angles θ
  → Real-space Bogoliubov Hamiltonian construction
  → Bogoliubov diagonalization → quasiparticle energies & wavefunctions
  → Real-space dynamical correlators → momentum-space correlators
  → Magnon decay/scattering rates (Fermi Golden Rule)
  → Plotting (dispersion, correlator maps, rate curves)
```

Data is cached to a `data/` directory with deterministic filenames based on parameter tuples.

## Hamiltonian

The primary model is the J1-J2 XY Hamiltonian with staggered field:

```
H = J1 Σ⟨i,j⟩ (SˣᵢSˣⱼ + SʸᵢSʸⱼ + D1 SᶻᵢSᶻⱼ)
  + J2 Σ⟨⟨i,j⟩⟩ (SˣᵢSˣⱼ + SʸᵢSʸⱼ + D2 SᶻᵢSᶻⱼ)
  + h Σᵢ (-1)ˣ⁺ʸ Sᶻᵢ
```

Where J1 (J2) are nearest-neighbor (next-nearest-neighbor) couplings, D1, D2 are ZZ anisotropies, h is a staggered magnetic field, and S = 1/2 throughout.

In the code, the Hamiltonian is specified as a 6-tuple `(g1, g2, d1, d2, h, h_disorder)` where g = J/2.

## Citing

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
