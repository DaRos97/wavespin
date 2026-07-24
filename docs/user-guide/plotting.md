# Plotting

Visualization functions for lattice geometry, dispersion relations, correlator maps,
wavefunctions, and decay rates.

## Lattice Plots

```python
from wavespin.lattice import get_preset
from wavespin.plots import latticePlots

lat = get_preset('60-diamond', plotLattice=False)
latticePlots.plotLattice(lat)        # sites, bonds, off-sites
```

## Classical Phase Diagram

```python
from wavespin.classicSpins import computeClassicalGroundState, plotClassicalPhaseDiagram

result = computeClassicalGroundState(params, j2_values, h_values)
plotClassicalPhaseDiagram(result)     # colored phase map in (J2, h) plane
```

## Dispersion & Correlator Maps

From `wavespin.plots.rampPlots`:

| Function | Description |
|----------|-------------|
| `plotRampKW()` | Frequency-vs-|k| intensity colormaps |
| `plotRampDispersions()` | 3D dispersion surfaces |
| `plotWf2D()` / `plotWf3D()` | 2D / 3D Bogoliubov wavefunction plots |
| `plotWfCos()` | Comparison of Bogoliubov modes with cosine functions |
| `plotBogoliubovMomenta()` | Scatter plot of mode momenta colored by energy |
| `plotRate()` | Decay rate vs. mode number (auto-generated LaTeX labels) |

Plots use matplotlib with Computer Modern (LaTeX-rendered) fonts.
