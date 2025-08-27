""" Here we compute the classical arrangement of spins in a lattice using a Montecarlo simulation.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importClassicParameters as importParameters
from wavespin.classicSpins.montecarlo import *
from wavespin.plots import fancyLattice

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation in OBC")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

if parameters.plotSites:
    simulation = XXZJ1J2MC(parameters)
    fancyLattice.plotSitesGrid(simulation)

""" Hamiltonian parameters """
gInitial = 0
gFinal = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
hInitial = 15
hFinal = 0
pValue = 0.3
g_p = (1-pValue)*gInitial + pValue*gFinal
h_p = (1-pValue)*hInitial + pValue*hFinal
parametersHamiltonian = (g_p,0,0,0,h_p)

""" Initialize and run Montecarlo simulation """
sim = XXZJ1J2MC(parameters,parametersHamiltonian)
hist = sim.anneal(verbose=verbose)
E_over_N = sim.total_energy() / sim.Ns
m = sim.magnetization()
ms = sim.staggered_mz()
print("Final results:")
print(f"E/N = {E_over_N:.8f}")
print(f"m = {m}")
print(f"m_staggered_z = {ms:.6f}")

kwargs = {'saveFigure':parameters.savePlotSolution}
fancyLattice.solutionMC(sim,**kwargs)






