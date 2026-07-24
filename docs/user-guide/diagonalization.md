# Diagonalization

The core computation: building the Bogoliubov-de Gennes Hamiltonian and
diagonalizing it to obtain quasiparticle energies and wavefunctions.

## `openHamiltonian`

The main class for open boundary conditions. Extends `latticeClass`.

```python
from wavespin.tools import LatticeParams, SimParams
from wavespin.static import openHamiltonian

# Option 1: from LatticeParams only (Hamiltonian set later)
system = openHamiltonian(LatticeParams(Lx=10, Ly=10))

# Option 2: from SimParams (all parameters at once)
params = SimParams()
params.lattice.Lx = 10
params.lattice.Ly = 10
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)
system = openHamiltonian(params)
```

## Key Attributes After Diagonalization

```python
system.diagonalize()
```

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `evals` | `(Ns,)` | Quasiparticle energies \\( E_n \\) |
| `U_` | `(Ns, Ns)` | Bogoliubov transformation matrix |
| `V_` | `(Ns, Ns)` | Bogoliubov transformation matrix |
| `Phi` | `(Ns, Ns)` | Wavefunctions (\\( \Phi = U - V \\)) |

## Diagonalization Algorithm

1. Build the \\( 2N_s \times 2N_s \\) real-space Hamiltonian matrix
2. Extract \\( A \\) (upper-left) and \\( B \\) (upper-right) blocks
3. Cholesky decomposition: \\( K = \text{chol}(A - B) \\)
4. Solve: \\( K (A + B) K^T \\)
5. Eigenvalues: \\( \omega_n^2 \\), eigenvectors: \\( \chi_n \\)
6. Quasiparticle energies: \\( E_n = \sqrt{\omega_n^2} \\)
7. Wavefunctions: \\( \phi_n = K^T \chi_n / \sqrt{E_n} \\), \\( \psi_n = (A+B) \phi_n / E_n \\)
8. Bogoliubov matrices: \\( U_n = (\phi_n + \psi_n) / 2 \\), \\( V_n = (\phi_n - \psi_n) / 2 \\)

## `periodicHamiltonian`

For periodic boundary conditions, the diagonalization is done analytically
in momentum space:

```python
from wavespin.static import periodicHamiltonian
from wavespin.tools import SimParams

params = SimParams()
params.lattice.Lx = 20
params.lattice.Ly = 20
params.lattice.boundary = 'periodic'
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)

system = periodicHamiltonian(params)
# Results available directly:
system.E(kx, ky)     # dispersion relation
system.r_k(kx, ky)   # Bogoliubov rotation angle
```

## Parameter Sweeps: `openRamp` / `periodicRamp`

Container classes for sweeping over Hamiltonian parameters:

```python
from wavespin.static import openRamp, openSystem
from wavespin.tools import SimParams

ramp = openRamp()
for h in [0, 1, 2, 3]:
    params = SimParams()
    params.lattice.Lx = 20
    params.lattice.Ly = 20
    params.diag.Hamiltonian = (5, 0, 0, 0, h, 0)
    ramp.addSystem(openSystem(params))

ramp.correlatorsXT()   # compute all real-space correlators
ramp.correlatorsKW()   # compute and plot momentum-space correlators
```
