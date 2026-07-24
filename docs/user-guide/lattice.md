# Lattice & Geometry

The lattice is the foundation of all simulations. It defines the 2D grid of sites,
the boundary conditions, and precomputes neighbor lists.

## `latticeClass`

Every simulation class inherits from `latticeClass`. It stores:

| Attribute | Type | Description |
|-----------|------|-------------|
| `Lx`, `Ly` | `int` | Grid dimensions |
| `Ns` | `int` | Number of active sites (excluding `offSiteList`) |
| `NN[i]` | `list[int]` | Nearest-neighbor indices of site *i* |
| `NNN[i]` | `list[int]` | Next-nearest-neighbor indices of site *i* |
| `indexToSite` | `list[tuple]` | Maps index → (x, y) coordinate |
| `siteToIndex` | `dict` | Maps (x, y) → index (O(1) lookup) |
| `boundary` | `str` | `'open'` or `'periodic'` |

## Basic Usage

```python
from wavespin.tools import LatticeParams
from wavespin.lattice import latticeClass

params = LatticeParams(Lx=10, Ly=10, boundary='open', plotLattice=True)
lat = latticeClass(params)
print(f"Number of sites: {lat.Ns}")
```

## Non-Rectangular Lattices

Exclude sites with `offSiteList` to create non-rectangular shapes:

```python
params = LatticeParams(
    Lx=8, Ly=8,
    offSiteList=((0, 0), (0, 7), (7, 0), (7, 7)),  # corners removed
    boundary='open'
)
lat = latticeClass(params)
```

## Preset Geometries

Predefined diamond-shaped lattices are available:

```python
from wavespin.lattice import get_preset

lat = get_preset('60-diamond')        # 60-site diamond on 10x10 grid
lat = get_preset('97-tilted-diamond') # 97-site tilted diamond on 13x13 grid
```

Valid preset names: `'12-diamond'`, `'24-diamond'`, `'40-diamond'`, `'60-diamond'`,
`'84-diamond'`, `'97-tilted-diamond'`, `'112-diamond'`, `'144-diamond'`.

## Periodic Boundaries

Periodic boundary conditions require even `Lx` and `Ly` and an empty `offSiteList`.
Under periodic BCs, neighbors wrap around the grid edges.
