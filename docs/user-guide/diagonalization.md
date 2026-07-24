# Diagonalization

The core computation: building the Bogoliubov-de Gennes Hamiltonian and
diagonalizing it to obtain quasiparticle energies and wavefunctions.

There are two distinct approaches depending on the boundary conditions:

- **Open boundaries** — `openHamiltonian` : real-space Bogoliubov
  diagonalization via Cholesky decomposition.
- **Periodic boundaries** — `periodicHamiltonian` : analytic
  momentum-space theory.

## `openHamiltonian`

The main class for open boundary conditions.  Extends `latticeClass`.

Construction triggers the full computation chain — lattice setup,
quantization-axis determination, Holstein-Primakoff coefficient
computation, and Bogoliubov diagonalization — automatically at
`__init__` time.

```python
from wavespin.static import openHamiltonian
from wavespin.tools import SimParams, LatticeParams, DiagParams

params = SimParams(
    lattice=LatticeParams(Lx=10, Ly=10),
    diag=DiagParams(Hamiltonian=(5, 0, 0, 0, 0, 0)),
)
system = openHamiltonian(params)
```

### Key Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `evals` | `(Ns,)` | Quasiparticle energies \(E_n\) (zero mode is `evals[0]`) |
| `U_`, `V_` | `(Ns, Ns)` | Bogoliubov transformation matrices |
| `Phi` | `(Ns, Ns)` | Real-space wavefunctions (\(\Phi = U - V\)) |
| `thetas` | `(Ns,)` | Per-site canting angles \(\theta_i\) |
| `ts` | `(Ns, 3, 3)` | Rotation vectors per site |
| `Ps` | `(2, Ns, Ns, 3, 3)` | HP expansion coefficients for NN (index 0) and NNN (index 1) |
| `GSE` | float | Ground-state energy per bond |
| `g1, g2, d1, d2, h` | float | Hamiltonian parameters |
| `order` | str | `'canted-Neel'` or `'canted-stripe'` |

### Quantization Axis Angles

By default (`uniformQA=True`), the base canting angle \(\theta\) is
computed from the averaged Hamiltonian parameters via
`quantizationAxis`, then spatially patterned across the lattice
according to the magnetic order:

- **canted-Néel** (\(J_2 \le J_1/2\)): checkerboard alternation,
  \(\theta_i = \theta + \pi\cdot(x_i+y_i)\bmod 2\).
- **canted-stripe** (\(J_2 > J_1/2\)): site-dependent pattern with sign
  flips and \(\pi\)-shifts based on x,y parity.

### Diagonalization Algorithm

1. Build the \(2N_s \times 2N_s\) real-space Bogoliubov-de Gennes matrix
   from the \(P\) coefficients.
2. Extract \(A\) (upper-left) and \(B\) (upper-right) blocks.
3. Cholesky decomposition: \(K = \text{chol}(A - B)\).
4. Diagonalize \(K (A + B) K^T\) → eigenvalues \(\omega_n^2\),
   eigenvectors \(\chi_n\).
5. \(E_n = \sqrt{\omega_n^2}\),
   \(\phi_n = K^T \chi_n / \sqrt{E_n}\),
   \(\psi_n = (A+B) \phi_n / E_n\).
6. \(U_n = (\phi_n + \psi_n) / 2\),
   \(V_n = (\phi_n - \psi_n) / 2\),
   \(\Phi = U - V\).

Results are cached to disk at `data/*.npz`.

### Decay Rates & Vertex Computation

After diagonalization, magnon decay/scattering rates can be computed:

```python
params.scattering.types = ('1to2_1', '2to2_1')
system.computeRate()
print(system.rates['1to2_1'])   # decay rate per mode
```

## `periodicHamiltonian`

For periodic boundary conditions, the diagonalization is done
analytically in momentum space. All couplings are uniform (scalars).

```python
from wavespin.static import periodicHamiltonian
from wavespin.tools import SimParams, LatticeParams, DiagParams

params = SimParams(
    lattice=LatticeParams(Lx=10, Ly=10, boundary='periodic'),
    diag=DiagParams(Hamiltonian=(5, 0, 0, 0, 0, 0)),
)
per = periodicHamiltonian(params)
```

### Key Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `gridk` | `(Lx, Ly, 2)` | Brillouin-zone k-points `(kx, ky)` |
| `gamma` | `(Gamma1, Gamma2)` | Dispersion factors \(\Gamma_1(k), \Gamma_2(k)\) |
| `dispersion` | `(Lx, Ly)` | Spin-wave dispersion \(\varepsilon(k_x, k_y)\) |
| `gsEnergy` | float | Ground-state energy per site |
| `theta`, `phi` | float | Canting and azimuthal angles |
| `ts` | `(2, 3, 3)` | Rotation vectors (sublattice A and B) |
| `rk` | `(Lx, Ly)` | Bogoliubov rotation angle \(r_k\) |
| `phik` | `(Lx, Ly)` | Phase factor \(e^{i\varphi_k}\) |
| `g1, g2, d1, d2, h` | float | Hamiltonian parameters |

All results are pre-computed at construction time — no separate
`diagonalize()` call needed.

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
