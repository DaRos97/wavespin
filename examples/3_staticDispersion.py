""" Here we compute the dispersion of a periodic system, together with quantization axis canting, GS energy and gap.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.periodic import periodicHamiltonian, periodicRamp
from wavespin.plots import fancyLattice
from wavespin.plots import rampPlots

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static dispersion calculation in PBC")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

""" Define the parameters of the system at different 'times' """
nP = 6#101     #number of parameters computed in the "ramp" -> analogue to stop ratio
gInitial = 0
gFinal = 10      #factor of 2 from experiment due to s^xs^x -> s^+s^-
hInitial = 15
hFinal = 0
#pValues = np.linspace(0.1,1,nP)
pValues = np.array([0.1,0.3,3/7,0.45,0.6,1])#np.linspace(3/7-0.06,3/7+0.06,nP)
g_p = (1-pValues)*gInitial + pValues*gFinal
h_p = (1-pValues)*hInitial + pValues*hFinal

""" Initialize all the systems and store them in a ramp object """
ramp = periodicRamp()
for i in range(nP):
    parameters.dia_Hamiltonian = (g_p[i],0,0,0,h_p[i],0)
    ramp.addSystem(periodicHamiltonian(parameters))
#    print(ramp.rampElements[i].theta/np.pi*180)
#    print(ramp.rampElements[i].gsEnergy /g_p[i]/2)

if 0:
    """ Plot interesting values of the ramp """
    rampPlots.plotRampValues(ramp)

if 1:
    """ Plot dispersions of the ramp """
    rampPlots.plotRampDispersions(ramp)

