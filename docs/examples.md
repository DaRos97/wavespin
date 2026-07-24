# Examples

Example scripts are in the `examples/` directory and demonstrate
specific calculations.

## Running Examples

Each example is a self-contained script that builds a system, runs a calculation,
and generates plots or output.

From the project root:

```bash
python examples/0_plotLattice.py
python examples/1_classicalPhaseDiagram.py
python examples/3_staticDispersion.py examples/input_3.txt
```

## Available Examples

| File | Description |
|------|-------------|
| `0_plotLattice.py` | Visualize a preset lattice geometry |
| `1_classicalPhaseDiagram.py` | Compute and plot the classical phase diagram |
| `2_classicalMontecarlo.py` | Monte Carlo simulated annealing |
| `3_staticDispersion.py` | Compute and plot the spin-wave dispersion |
| `4_staticOpenCorrelators.py` | Real-space dynamical correlators (OBC) |
| `5_staticWavefunctions.py` | Plot Bogoliubov wavefunctions |
| `6_staticPeriodicCorrelator.py` | Momentum-space correlators (periodic BC) |
| `7_classicalThetaMinimization.py` | Site-dependent canting angle optimization |
| `8_magnonLifetime.py` | Magnon decay rate computation |
| `9_plotContributions.py` | ZZ correlator broken into magnon contributions |
| `10_plotOpenDispersions.py` | Dispersion with OBC (DCT-extracted momenta) |

## Input Files

Some examples accept input files with `key: value` pairs:

```
# examples/input_3.txt
Lx: 20
Ly: 20
Hamiltonian: (5, 0, 0, 0, 0, 0)
boundary: periodic
plotLattice: False
```
