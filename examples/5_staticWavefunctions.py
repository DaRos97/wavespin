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
g1 = 20
h = 0
system = openHamiltonian(parameters,(g1,0,0,0,h))

""" Compute Bogoliubov wavefunctions """
system.diagonalize(verbose=verbose)

if parameters.plotWf:
    """ Plot wavefunctions """
    rampPlots.plotWf(system,nModes=16)
    """ Plot wavefunctions compared to cosines """
    rampPlots.plotWfCos(system,nModes=6)
    """ Plot extracted momentum points from Bogoliubov modes """
    rampPlots.plotBogoliubovMomenta(system)




