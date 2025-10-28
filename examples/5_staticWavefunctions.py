""" Example script to compute and plot the Bogoliubov wavefunctions.
Use with input_5.txt
"""

import numpy as np
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation using Holstein-Primakoff formalism")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

""" Chose the parameters """
gInitial = 0
gFinal = 10
hInitial = 15
hFinal = 0
stopRatios = [1,]#np.linspace(0.1,1,10)
Nr = len(stopRatios)

for ir in range(Nr):
    stopRatio = stopRatios[ir]
    g_p = (1-stopRatio)*gInitial + stopRatio*gFinal
    h_p = (1-stopRatio)*hInitial + stopRatio*hFinal
    parameters.dia_Hamiltonian = (g_p,0,0,0,h_p,0)
    system = openHamiltonian(parameters)

    """ Compute Bogoliubov wavefunctions """
    system.diagonalize(verbose=verbose)


if 0:   # Save just the amazing functions to share
    U_ = system.U_
    V_ = system.V_
    phi = np.real(U_ - V_)
    for i in range(system.Ns):
        ix,iy = system._xy(i)
        phi[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    np.save("Data/amazing.npy",phi)
    np.save("Data/evals.npy",system.evals)
