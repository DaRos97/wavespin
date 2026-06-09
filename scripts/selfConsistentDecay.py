""" Here I compute the decay rates self-consistently starting from the usual way and looping the decay rates for each mode.
"""
import numpy as np
import argparse
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt

parameters = importParameters()
parameters.dia_saveWf = True
parameters.sca_saveVertex = True
parameters.sca_saveRate = False


# Lattice parameters
Lx = 8
Ly = 6
parameters.lat_Lx = Lx
parameters.lat_Ly = Ly
parameters.lat_boundary = 'open'

# Hamiltonian parameters
J1 = 1
J2 = 0
h = 0
parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)

# Scattering parameters
sca_types = ('2to2_1',)
parameters.sca_types = sca_types
parameters.sca_temperature = 0.25    # About b/w mode 2 and 3
parameters.sca_broadening = 0.5

# Computation
mySystem = openHamiltonian(parameters)
mySystem.computeRate(verbose=True)          # Need to go change the dic_processes to _sc in wavespin/static/decayProcesses.py
resOBC1to2.append(mySystem.rates['1to2_1'])
enOBC = mySystem.evals[1:]
