# To-do list

- Change the `__init__.py` files to have a clearer package structure.
- Write tests -> should be easy, just use the example files.
- Remove plotSites from input arguments -> it's specific for the calculation.

## Classical

- Add legends to plots
- Limits of calculation? 
- ZZ anisotropy -> changes the symmetry
- Go back to RMO derivation, there are no notes on it

### Montecarlo
- Implement integration with RMO
- Implement parameters path calculation for the ramp
- Add parent class of lattice and then derived classes for classical, open, periodic etc..
- Same for input parameters

## Static
- Pass to hdf5 for saving data compactly
- Put together periodic and open for real space computation -> that's what we care about anyway

### Periodic boundary
- Analysis on spin size S
- Implement site-dependent theta when using OBC -> from MC
- Need documentation for N11, N12, computePs, computeTs, computeEpsilon, computeGsE, computeE0
- offsitelist in computePs goes in conflict with computeTs
- computePs and computeTs need to be extended for canted-Stripe order

### Open boundary
- Input of experimental parameters
- Implement the non-rectangular geometry
- Pipeline momentum space correlator computation so one does not need to firts explicitly compute the real space one.
- Switch correlators from x,y to ind=x*Ly+y
- Handle the save correlator bonds -> I'm not plotting so its used just for saving data
- In diagonalization -> since we exclude the zero mode, reduce dimensionaliti of U and V a-priori

## Dynamic
