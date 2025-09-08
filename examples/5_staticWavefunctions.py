""" Example script to compute and plot the Bogoliubov wavefunctions.
Use with input_5.txt
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importOpenParameters as importParameters
from wavespin.static.open import openHamiltonian
from wavespin.lattice.lattice import latticeClass
from wavespin.plots import fancyLattice
from wavespin.plots import rampPlots

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation using Holstein-Primakoff formalism")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

if parameters.plotSites:
    lattice = latticeClass(parameters)
    fancyLattice.plotSitesGrid(lattice)

""" Initialie system """
""" Hamiltonian parameters """
gInitial = 0
gFinal = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
hInitial = 15
hFinal = 0
for pValue in np.linspace(0.1,1,10):
    print('p_value = %.1f'%pValue)
    g_p = (1-pValue)*gInitial + pValue*gFinal
    h_p = (1-pValue)*hInitial + pValue*hFinal
    parametersHamiltonian = (g_p,0,0,0,h_p)
    system = openHamiltonian(parameters,parametersHamiltonian)

    """ Compute Bogoliubov wavefunctions """
    system.diagonalize(verbose=verbose)

    if parameters.plotWf:
        """ Plot wavefunctions """
        rampPlots.plotWf(system,nModes=16)
        """ Plot wavefunctions compared to cosines """
        rampPlots.plotWfCos(system,nModes=6)
        """ Plot extracted momentum points from Bogoliubov modes """
        rampPlots.plotBogoliubovMomenta(system)




