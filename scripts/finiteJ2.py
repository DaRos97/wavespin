""" Here I compute the ZZ correlator and scattering rates of increasing J2 BELOW the transition.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
from wavespin.static.open import openSystem, openRamp
import matplotlib.pyplot as plt

parameters = importParameters()
Lx = 10
Ly = 10

parameters.lat_Lx = Lx
parameters.lat_Ly = Ly
parameters.lat_boundary = 'open'
parameters.dia_plotWf = False
parameters.dia_saveWf = True
# Correlator parameters
parameters.cor_correlatorType = 'zz'
parameters.cor_transformType = 'dct'
parameters.cor_perturbationSite = (Lx//2,Ly//2)
parameters.cor_plotKW = True
# Scattering parameters
parameters.sca_broadening = 0.5
parameters.sca_saveVertex = True
parameters.sca_temperature = 8
#parameters.sca_types = ('1to2_1','2to2_1','1to3_1')
parameters.sca_types = ('2to2_1',)

J1 = 1
J2s = np.linspace(0,J1/2,6)
nP = len(J2s)

if 0:   #ZZ commutator
    ramp = openRamp()
    for i2 in range(len(J2s)):
        parameters.dia_Hamiltonian = (J1/2,J2s[i2]/2,0,0,0,0)
        mySystem = openSystem(parameters)
        ramp.addSystem(mySystem)
    ramp.correlatorsXT()
    ramp.correlatorsKW()


if 1:   # Scattering rates
    gamma = np.zeros((2,nP,Lx*Ly-len(parameters.lat_offSiteList)-1))
    evals = np.zeros((2,nP,Lx*Ly-len(parameters.lat_offSiteList)-1))
    for ib in range(2):
        parameters.lat_boundary = 'open' if ib==0 else 'periodic'
        print(parameters.lat_boundary)
        for i2 in range(nP):
            print("J2=%.2f"%J2s[i2])
            parameters.dia_Hamiltonian = (J1/2,J2s[i2]/2,0,0,0,0)
            mySystem = openSystem(parameters)
            mySystem.computeRate()
            for sca_type in parameters.sca_types:
                gamma[ib,i2] += mySystem.rates[sca_type]
            evals[ib,i2] = mySystem.evals[1:]
    #
    fig,axs = plt.subplots(2,3,figsize=(12,8))
    col = ['darkorange','blue']
    for i in range(nP):
        ax = axs[i//3,i%3]
        for ib in range(2):
            ax.scatter(
                evals[ib,i],
                gamma[ib,i],
                color=col[ib],
                alpha=0.5,
                lw=0
            )
        if i>=3:
            ax.set_xlabel("Energy",size=15)
        if i in [0,3]:
            ax.set_ylabel(r"$\Gamma$",size=15)
        ax.set_title(r"$J_2=%.2f$"%J2s[i],size=20)
    plt.suptitle("Lattice %dx%d"%(Lx,Ly),size=25)
    plt.show()
#

