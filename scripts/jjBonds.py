""" Here we compute the jj correlator for square lattices.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importOpenParameters as importParameters
from wavespin.static.open import openSystem, openRamp
from wavespin.plots import fancyLattice
from wavespin.plots import rampPlots

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation in OBC")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

if parameters.plotSites:
    simulation = openSystem(parameters,(0,0,0,0,0))
    fancyLattice.plotSitesGrid(simulation,**{'indices':False})

""" Define the parameters of the system at different 'times' """
nP = 1     #number of parameters computed in the "ramp" -> analogue to stop ratio
gInitial = 0
gFinal = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
hInitial = 15
hFinal = 0
pValues = np.array([1,])#np.linspace(1,1,nP)
g_p = (1-pValues)*gInitial + pValues*gFinal
h_p = (1-pValues)*hInitial + pValues*hFinal

data = []
Lx = parameters.Lx
Ly = parameters.Ly
for ip,pS in enumerate([(0,0),(Lx-2,0)]):
    parameters.perturbationSite=pS
    """ Initialize all the systems and store them in a ramp object """
    ramp = openRamp()
    for i in range(nP):
        termsHamiltonian = (g_p[i],0,0,0,h_p[i])
        ramp.addSystem(openSystem(parameters,termsHamiltonian))
#    ramp.rampElements[0].perturbationDirection = 'h' if ip==0 else 'v'

    """ Compute correlator XT and KW for all systems in the ramp """
    ramp.correlatorsXT(verbose=verbose)

    data.append(ramp.rampElements[0].correlatorXT_h)

time = np.arange(ramp.rampElements[0].nTimes)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot()
ax.plot(time,np.imag(data[0][1,0,:]),color='r')
ax.plot(time,np.imag(data[1][Lx-3,0,:]),color='b')
plt.show()
