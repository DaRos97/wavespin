""" Here I compute the decay rate of modes comparing between periodic and open boundary conditions.
"""

import numpy as np
import argparse
from wavespin.tools.inputUtils import importParameters
import wavespin.tools.pathFinder as pf
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
from pathlib import Path

parameters = importParameters()
parameters = importParameters()
parameters.dia_plotWf = False
parameters.dia_saveWf = False#True
parameters.sca_saveVertex = False#True
parameters.sca_saveRate = False#True
parameters.sca_broadening = 0.5

s_ = 15
if 1:       # Compare rates for frustrated branch 
    Lx = 8
    Ly = 6
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    # Scattering parameters
    sca_types = ('2to2_1',)
    parameters.sca_types = sca_types
    temperatures = [0.3697, 0.2994, 0.2112]

    # Hamiltonian parameters
    J1 = 1
    h = 0
    NJ2 = 1
    J2s = [0,0.2,0.4]#np.linspace(0.2,0.5,NJ2,endpoint=False)
    fig1 = plt.figure(figsize=(12,4))
    fig2 = plt.figure(figsize=(12,4))
    for i2,J2 in enumerate(J2s):
        temperature = temperatures[i2]
        print(J2,temperature)
        parameters.sca_temperature = temperature
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        mySystem.computeRate()
        resPBC2to2 = mySystem.rates['2to2_1']
        enPBC = mySystem.evals[1:]
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        mySystem.computeRate()
        resOBC2to2 = mySystem.rates['2to2_1']
        enOBC = mySystem.evals[1:]

        # Figure
        ax1 = fig1.add_subplot(1,3,i2+1)
        ax1.scatter(enPBC,resPBC2to2,color='r',label='PBC')
        ax1.scatter(enOBC,resOBC2to2,color='b',label='OBC')
        ax1.set_xlabel("Mode energy",size=s_)
        ax2 = fig2.add_subplot(1,3,i2+1)
        ax2.scatter(np.arange(1,48),resPBC2to2,color='r',label='PBC')
        ax2.scatter(np.arange(1,48),resOBC2to2,color='b',label='OBC')
        ax2.set_xlabel("Mode number",size=s_)
        #
        ax1.set_title(r"$J_2=$%.1f, $T=$%.4f"%(J2,temperature),size=s_)
        ax2.set_title(r"$J_2=$%.1f, $T=$%.4f"%(J2,temperature),size=s_)
    #
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
if 0:       # Compare rates for critical branch 
    Lx = 8
    Ly = 6
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    # Scattering parameters
    sca_types = ('2to2_1',)
    parameters.sca_types = sca_types
    temperature = 0.5
    parameters.sca_temperature = temperature

    # Hamiltonian parameters
    J1 = 1
    J2 = 0
    Nh = 1
    Hs = np.linspace(1.,1.5,Nh,endpoint=True)
    fig = plt.figure(figsize=(16,4))
    for ih,h in enumerate(Hs):
        print(h)
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
        # PBC
        parameters.lat_boundary = 'periodic'
        mySystem = openHamiltonian(parameters)
        mySystem.computeRate()
        resPBC2to2 = mySystem.rates['2to2_1']
        enPBC = mySystem.evals[1:]
        # OBC
        parameters.lat_boundary = 'open'
        mySystem = openHamiltonian(parameters)
        mySystem.computeRate()
        resOBC2to2 = mySystem.rates['2to2_1']
        enOBC = mySystem.evals[1:]

        # Figure
        ax = fig.add_subplot(1+Nh//6,Nh,ih+1)
        ax.scatter(enPBC,resPBC2to2,color='r',label='PBC')
        ax.scatter(enOBC,resOBC2to2,color='b',label='OBC')
        if ih>=5:
            ax.set_xlabel("Mode energy",size=s_)
        ax.set_title(r"$h=$%.3f"%h,size=s_)
    #
    fig.tight_layout()
    plt.show()


