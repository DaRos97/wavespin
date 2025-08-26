""" Here we compute correlators in a system with PBC.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importPeriodicParameters as importParameters
from wavespin.static.periodic import periodicSystem, periodicRamp
from wavespin.plots import fancyLattice
from wavespin.plots import rampPlots

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation in PBC")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

if parameters.plotSites:
    simulation = openSystem(parameters)
    fancyLattice.plotSitesGrid(simulation)

""" Define the parameters of the system at different 'times' """
nP = 10     #number of parameters computed in the "ramp" -> analogue to stop ratio
gInitial = 0
gFinal = 40      #factor of 2 from experiment due to s^xs^x -> s^+s^-
hInitial = 30
hFinal = 0
pValues = np.linspace(0.1,1,nP)
g_p = (1-pValues)*gInitial + pValues*gFinal
h_p = (1-pValues)*hInitial + pValues*hFinal

""" Initialize all the systems and store them in a ramp object """
ramp = periodicRamp()
for i in range(nP):
    termsHamiltonian = (g_p[i],0,0,0,h_p[i])
    ramp.addSystem(periodicSystem(parameters,termsHamiltonian))

""" Compute correlator XT and KW for all systems in the ramp """
ramp.correlatorsXT(verbose=verbose)
ramp.correlatorsKW(verbose=verbose)

if parameters.plotCorrelatorKW:
    """ Plot the Fourier-transformed correlators of the ramp """
    rampPlots.plotRampKW(ramp,
                         kwargs={
                             'numKbins' : 50,
                             'ylim' : 70,            #MHz
                             'saveFigure' : False,
                             'showFigure' : True,
                         }
                         )













