# Parameters

All simulation parameters are defined as nested dataclasses in `wavespin.tools.inputUtils`.

## The `SimParams` Hierarchy

```
SimParams
  ├── lattice: LatticeParams
  ├── diag: DiagParams
  ├── correlator: CorrelatorParams
  └── scattering: ScatteringParams
```

### `LatticeParams`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Lx` | `int` | `7` | Grid width |
| `Ly` | `int` | `9` | Grid height |
| `offSiteList` | `tuple` | `()` | Sites to exclude (x,y) pairs |
| `boundary` | `str` | `'open'` | `'open'` or `'periodic'` |
| `plotLattice` | `bool` | `False` | Plot lattice on construction |

### `DiagParams`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Hamiltonian` | `tuple` | `(0,0,0,0,0,0)` | 6-tuple `(g1, g2, d1, d2, h, h_disorder)` |
| `excludeZeroMode` | `bool` | `True` | Exclude the Goldstone zero mode |
| `uniformQA` | `bool` | `True` | Use uniform canting angle (otherwise site-dependent) |
| `saveWf` | `bool` | `False` | Cache wavefunctions to disk |
| `plotWf` | `bool` | `False` | Plot wavefunctions after diagonalization |

### `CorrelatorParams`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `correlatorType` | `str` | `'zz'` | `'zz'`, `'xx'`, `'ee'`, `'jj'`, `'ze'`, `'ez'` |
| `transformType` | `str` | `'dct'` | `'fft'`, `'dst'`, `'dct'`, `'dat'`, `'dat2'` |
| `perturbationSite` | `tuple` | `(0,0)` | Site where perturbation is applied |
| `magnonModes` | `tuple` | `(1,2,3,4)` | Magnon expansion terms to include |
| `energy` | `float` | `-100` | Energy offset for correlator evaluation |

### `ScatteringParams`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `types` | `tuple` | `('2to2_1',)` | Rate processes: `'1to2_1'`, `'2to2_1'`, `'1to3_1'`, etc. |
| `temperature` | `float` | `0.0` | Temperature for Bose factors |
| `broadening` | `float` | `0.5` | Lorentzian broadening of energy delta |

## Creating Parameters

### Programmatically

```python
from wavespin.tools import SimParams, LatticeParams, DiagParams

params = SimParams()
params.lattice.Lx = 20
params.lattice.Ly = 20
params.lattice.boundary = 'periodic'
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)
```

### From an Input File

```python
from wavespin.tools import importParameters

params = importParameters('examples/input_3.txt')
```

Input files use `key: value` format. Lines starting with `#` are comments:

```
# Lattice
Lx: 20
Ly: 20
boundary: periodic
# Hamiltonian
Hamiltonian: (5, 0, 0, 0, 0, 0)
```

## Backward Compatibility

`myParameters` is an alias for `SimParams`:

```python
from wavespin.tools.inputUtils import myParameters  # same as SimParams
```
