# To-do list

- Change the `__init__.py` files to have a clearer package structure.
- Documentation website
- Make examples real examples and move else to scripts
- Make cleare in README.md the folder structure and where files are saved (setup)

## Classical

- Add legends to plots
- Limits of calculation? 
- ZZ anisotropy -> changes the symmetry?
- Go back to RMO derivation, there are no notes on it

### Montecarlo
- Implement integration with RMO
- Implement parameters path calculation for the ramp
- Add parent class of lattice and then derived classes for classical, open, periodic etc..
- Same for input parameters

## Static
- Pass to hdf5 for saving data compactly
- Put together periodic and open for real space computation
- Remove ramp class?
- Make clear Phi: needs different transformation for c-Neel and c-Stripe
- Make clear g and h in README.md what they refer to
- Make consistent h disorder

### Periodic boundary
- This is just for the momentum space formulas -> can implement it in the open one
- Implement site-dependent theta when using OBC -> from MC

### Open boundary
- Input of experimental parameters
- Pipeline momentum space correlator computation so one does not need to firts explicitly compute the real space one.
- Handle the save correlator bonds -> I'm not plotting so its used just for saving data
- In diagonalization -> since we exclude the zero mode, reduce dimensionaliti of U and V a-priori?

## Dynamic
- Check notes
- Add angle phi to quantization axis
- Check rotor description
