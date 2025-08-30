""" Here we show how to perform the quantization angle minimization for a planar system.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importClassicParameters as importParameters
from wavespin.classicSpins.minimization import *
from wavespin.plots import fancyLattice

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation in OBC")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

if parameters.plotSites:
    lattice = latticeClass(parameters)
    fancyLattice.plotSitesGrid(lattice)

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

    sim = minHam(parameters,parametersHamiltonian)

    res = sim.minimization(verbose=verbose)

    thetas = res
    phis = np.zeros(sim.Ns)
    options = {'showFigure':False,
               'verbose':verbose
               }
    fancyLattice.plotQuantizationAngles(sim,thetas,phis,**options)






