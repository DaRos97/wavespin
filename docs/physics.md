# Physics Background

## Hamiltonian

The primary model is the J1-J2 XY Hamiltonian with a staggered magnetic field:

\[
\begin{aligned}
H &= J_1 \sum_{\langle i,j \rangle} \left( S_i^x S_j^x + S_i^y S_j^y + D_1 S_i^z S_j^z \right) \\
  &+ J_2 \sum_{\langle\langle i,j \rangle\rangle} \left( S_i^x S_j^x + S_i^y S_j^y + D_2 S_i^z S_j^z \right) \\
  &+ h \sum_i (-1)^{x_i + y_i} S_i^z
\end{aligned}
\]

Where:

- \\( J_1 \\), \\( J_2 \\) — nearest-neighbor and next-nearest-neighbor XY couplings
- \\( D_1 \\), \\( D_2 \\) — ZZ anisotropies (D = 1 for Heisenberg, D = 0 for pure XY)
- \\( h \\) — staggered magnetic field strength
- \\( S = 1/2 \\) throughout

In the code, the Hamiltonian is specified as a 6-tuple `(g1, g2, d1, d2, h, h_disorder)` where \\( g = J/2 \\).

## Holstein-Primakoff Expansion

The spin operators are expanded around the classical ground state using the Holstein-Primakoff
transformation. Spins are rotated to align with their local quantization axis (determined by the
canting angle \\( \theta_i \\)), then bosonized:

\[
\tilde{S}_i^z = S - a_i^\dagger a_i, \quad
\tilde{S}_i^- \approx \sqrt{2S} \, a_i, \quad
\tilde{S}_i^+ \approx \sqrt{2S} \, a_i^\dagger
\]

Higher-order terms (\\( a^\dagger a a \\), etc.) are kept when computing interaction vertices
for decay/scattering rates.

## Bogoliubov Diagonalization

The quadratic bosonic Hamiltonian is written as a \\( 2N_s \times 2N_s \\) Bogoliubov-de Gennes matrix:

\[
H_{\text{quad}} = \frac{1}{2} \begin{pmatrix} \mathbf{a}^\dagger & \mathbf{a} \end{pmatrix}
\begin{pmatrix} A & B \\ B^* & A^* \end{pmatrix}
\begin{pmatrix} \mathbf{a} \\ \mathbf{a}^\dagger \end{pmatrix}
\]

For open boundary conditions, this is diagonalized numerically via Cholesky decomposition,
yielding quasiparticle energies \\( E_n \\) and wavefunctions \\( \Phi_{i,n} \\).

For periodic boundary conditions, the Hamiltonian is diagonalized analytically in momentum space.

## Classical Ground State

The system supports two magnetically ordered phases:

- **Canted-Néel** (checkerboard): stable for \\( J_2 \le J_1/2 \\)
- **Canted-stripe** (columnar): stable for \\( J_2 > J_1/2 \\)

The canting angle \\( \theta \\) is determined by minimizing the classical energy
with respect to the staggered field \\( h \\).
