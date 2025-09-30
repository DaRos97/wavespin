""" Here I compute the <XX> value at different temperatures """

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
from wavespin.plots import rampPlots

from time import time
import matplotlib.pyplot as plt
from pathlib import Path

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation using Holstein-Primakoff formalism")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})
Lx = parameters.lat_Lx
Ly = parameters.lat_Ly
Ns = Lx*Ly

g_p,_,d_p,_,h_p,disorder = parameters.dia_Hamiltonian
""" Initialize and diagonalize system """
parametersHamiltonian = (g_p,0,d_p,0,h_p,disorder)
system = openHamiltonian(parameters)
system.diagonalize(verbose=verbose)
U = np.real(system.U_)
V = np.real(system.V_)
epsilon = system.evals
S = system.S
NN_terms = system._NNterms(1)
Nbonds = np.sum(NN_terms)//2      #(Lx-1)*Ly + (Ly-1)*Lx
EGS = -3/2 + np.sum(epsilon)/Nbonds/g_p
print(Lx,Ly,' energy of GS: ',EGS)

Tlist = np.linspace(0,12,300)
result = np.zeros(len(Tlist))
for iT,T in enumerate(Tlist):
    if abs(T)<1e-5:
        FactorBose_T = np.zeros(Ns-1)
    else:
        FactorBose_T = 1/(np.exp(epsilon[1:]/T)-1)      #size Ns-1
    result[iT] = EGS + 2*np.sum(epsilon[1:]*FactorBose_T)/Nbonds/g_p

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot()
ax.plot(Tlist,result,marker='^',label='<XX>(T)')
ax.axhline(-0.5,color='r')
bestT = Tlist[np.argmin(np.absolute(0.5+result))]
ax.axvline(bestT,color='g')
for k in range(1,Ns):
    ax.axvline(epsilon[k],color='lime')
ax.set_xlim(0,Tlist[-1])
ax.set_xticks([0,]+list(epsilon[(epsilon<Tlist[-1])&(epsilon>1)]),["0",] + ["mag #%s"%i for i in range(1,len(epsilon[(epsilon<Tlist[-1])]))],size=12)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.3,0.7,r"T $\sim$ %.2f MHz"%bestT,transform=ax.transAxes,bbox=props,size=20)
ax.set_xlabel('Temperature (MHz)',size=20)
ax.set_ylabel("energy (g)")
plt.show()















