# Roadmap

## Lattice

### ☐ P1
- Add other lattice types (triangular, kagome, honeycomb)
- Add third-neighbor bond lists and coupling parameters

---

## Classical Spins

### ☐ P1
- Implement Monte Carlo
- Add legends to classic phase diagram plots
- Re-derive and document the RMO 5-angle variational ansatz

### ☐ P2
- Does the ZZ anisotropy change the symmetry of the ground state?
- What are the limits of the classical calculation?

---

## Static

### ☐ P0 — Must have
- Fix remaining 9 broken examples (old flat API: `dia_Hamiltonian` → `diag.Hamiltonian`, etc.)
- Fix `openRamp` flat-API references (`cor_saveXTbonds` → `correlator.saveXTbonds`, `cor_plotKW` → `correlator.plotKW`)
- Add tests for `openHamiltonian.diagonalize()`, `periodicHamiltonian`, correlators, and decay processes
- Fix remaining scripts with broken imports (lineWidths.py, etc.)

### ☐ P1 — Soon
- Implement non-uniform quantization angles via `classicMagnetization` (`uniformQA=False`)
- HDF5 output for compact data storage (replace `.npz` / `.npy`)
- Pipeline the momentum-space correlator computation — skip explicit real-space storage
- Make the staggered-field disorder implementation consistent across modules
- Clarify and document the `Phi` re-rotation convention for canted-Néel vs canted-stripe
- Add NNN bond contributions (g2, D2) to vertex f-factors in `computeVertex()`
- Address memory explosion for 2to2/1to3 vertex tensors (O(Ns⁴) dense arrays; explore sparse or factorised storage)
- Verify the cubic HP terms for X and Y in `computeCombinations` in `correlators.py` — currently commented out because they dramatically change the correlator results. The expansion coefficients need physics review
- Verify and fix the finite-temperature correlator computation (Bose-Einstein factors in `realSpaceCorrelator`)
- Fix the Discrete Awesome Transform (DAT/DAT2) in `momentumTransformation` — uses Bogoliubov U and V matrices to project real-space correlators to momentum space; needs verification and cleanup

### ☐ P2 — Nice to have
- Clean up the save-correlator-bonds logic (currently used only for saving, not plotting)
- Support input of experimental parameters (real device coupling maps)
- In `diagonalize()`: use `LAPACK` `dsygv` instead of the explicit Cholesky+eigh route

---

## Dynamics and Magnon Interactions

### ☐ P2
- Check and document the decay-rate notes / derivations
- Add the azimuthal angle `phi` to the quantization axis (needed for dynamic ramp)
- Check the quantum rotor description

---

## Infrastructure

### ☐ P1 — Soon
- Complete the documentation website (`mkdocs serve` → deploy)
- Fix remaining 9 broken examples and make them self-contained demonstrations

### ☐ P2
- Add `rampPlots` to `wavespin/plots/__init__.py`
- Clean up dead-code blocks (`if 0:` patterns) in examples and scripts
