""" Example script to compute and plot the zz correlator of a rectangular lattice.
Use with input_4.txt
"""


import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importOpenParameters as importParameters
from wavespin.static.open import openSystem, openRamp
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
    simulation = openSystem(parameters)
    fancyLattice.plotSitesGrid(simulation)

""" Define the parameters of the system at different 'times' """
nP = 10     #number of parameters computed in the "ramp" -> analogue to stop ratio
gInitial = 0
gFinal = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
hInitial = 15
hFinal = 0
pValues = np.linspace(0.1,1,nP)
g_p = (1-pValues)*gInitial + pValues*gFinal
h_p = (1-pValues)*hInitial + pValues*hFinal

""" Initialize all the systems and store them in a ramp object """
ramp = openRamp()
for i in range(nP):
    termsHamiltonian = (g_p[i],0,0,0,h_p[i])
    ramp.addSystem(openSystem(parameters,termsHamiltonian))

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



