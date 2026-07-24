# Decay Rates

Magnon decay and scattering rates are computed using Fermi's Golden Rule
with interaction vertices from the higher-order Holstein-Primakoff expansion.

## Available Processes

| Code | Physical Process | Order |
|------|-----------------|-------|
| `'1to2_1'` | 1 magnon → 2 magnons | 1st order |
| `'1to2_2'` | 1 magnon → 2 magnons | 2nd order |
| `'2to2_1'` | 2 magnons → 2 magnons | 1st order |
| `'2to2_2'` | 2 magnons → 2 magnons | 2nd order |
| `'1to3_1'` | 1 magnon → 3 magnons | 1st order |
| `'1to3_2'` | 1 magnon → 3 magnons | 2nd order |
| `'1to3_3'` | 1 magnon → 3 magnons | 3rd order |

## Usage

```python
from wavespin.static import openHamiltonian
from wavespin.tools import SimParams

params = SimParams()
params.lattice.Lx = 10
params.lattice.Ly = 10
params.diag.Hamiltonian = (5, 0, 0, 0, 0, 0)
params.scattering.types = ('1to2_1',)
params.scattering.temperature = 0.0
params.scattering.broadening = 0.5

system = openHamiltonian(params)
system.diagonalize()
system.computeVertex()  # compute interaction tensors
system.computeRate()    # compute decay rates
# system.rates[mode]    # decay rate per mode
```

## Method

1. **Vertex computation**: Interaction tensors are computed via Einstein summation
   (\\( \texttt{np.einsum} \\)) over the Bogoliubov matrices \\( U, V \\).
2. **Rate formula**: For a 1→2 process at mode \\( m \\):

   \[
   \Gamma_m = \frac{\pi}{2} \sum_{n,l} |V_{mnl}|^2
   \delta(E_m - E_n - E_l) (1 + n_n + n_l)
   \]

   where \\( V_{mnl} \\) is the vertex tensor and \\( n_i \\) are Bose-Einstein factors.
   Energy-conserving delta functions are broadened with Lorentzians.

3. **Self-consistency**: The `2to2_1_sc` rate includes iterative self-consistent
   broadening (see `wavespin.static.decayProcesses`).
