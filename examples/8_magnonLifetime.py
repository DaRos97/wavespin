""" Example script to compute the decay rate and scattering amplitude of single modes in the OBC setting.
Use with input_8.txt
"""

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

""" Initialize and diagonalize system """
system = openHamiltonian(parameters)
best_modes = np.array([17,18,19,20,24,25,26,27,30,31,32,34])
kwargs = {'best_modes':best_modes}
system.diagonalize(verbose=verbose,**kwargs)

#rampPlots.plotWf(system)
#rampPlots.plotBogoliubovMomenta(system)

system.decayRates(temperature=parameters.sca_temperature,verbose=verbose)

if 1:
    """ Decay vs amplitude """
    data = system.dataScattering
    evals = system.evals *2*np.pi / 1e3     #in GHz
    maxK = 20
    As = np.array([0.5,1,2,4,6,8])
    contA = np.linspace(As[0],As[-1],100)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot()

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.cividis
    norm = mcolors.Normalize(vmin=evals[1], vmax=evals[maxK+1])
    from scipy.optimize import curve_fit
    def amp(x,a1,a2):
        return a1 + a2*x**2
    for n in range(1,maxK+1):
        color = cmap(norm(evals[n]))
        modeD =( data['2to2a'][n-1] + As**2/4 * data['2to2b'][n-1] ) / 1e3    #in GHz
        ax.scatter(As, modeD,
                  color=color, marker='o')
        popt,pcov = curve_fit(amp, As, modeD)
        ax.plot(contA,amp(contA,*popt),color=color,ls='--',zorder=-1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Amplitude",size=20)
    ax.set_title(r"$\Gamma^{2\leftrightarrow2}$",size=20)
    #ax.set_ylabel("Decay rate (GHz)",size=20)
    ax.set_xlim(As[0]-(As[1]-As[0])/3,As[-1]+(As[-1]-As[-2]))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm,ax=ax)
    cbar.set_label("Mode freq. (GHz)",size=20)
    plt.show()

if 0:
    """ Compute multiple temperatures """
    Ts = np.linspace(2,20,10)
#    Ts = [8,]
    dataT = np.zeros((len(Ts),system.Ns-1))

    system.p.sca_plotVertex = False
    for it,T in enumerate(Ts):
        """ Scattering vertices """
        system.decayRates(temperature=T,verbose=verbose)
        dataT[it] = system.dataScattering['2to2a']

    """ Plotting """
    title = "2 to 2 scattering process"
    logPlot = False
    maxK = 50
    s_ = 20

    g_p, _, d_p, _, h_p, disorder = parameters.dia_Hamiltonian
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(131)
    for it,T in enumerate(Ts):
        Gamma_n = dataT[it]
        if logPlot:
            ax.loglog(np.arange(1,maxK),Gamma_n[:maxK-1],marker='o',markersize=10,label=r"T = %.2f MHz"%T)
        else:
            ax.scatter(np.arange(1,maxK),Gamma_n[:maxK-1],marker='o',label=r"T = %.2f MHz"%T)
            for i in best_modes:
                ax.scatter(i,Gamma_n[i-1],marker='*',color='r')
    #ax.legend(fontsize=s_-10)
    ax.set_xlabel("Mode number",size=s_)
    ax.set_ylabel("Decay rate (MHz)",size=s_)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.03,0.86,"g=%s, H=%.1f\n"%(g_p/2,h_p)+r"$\Delta$=%.1f, $h_{dis}$=%.1f"%(d_p,disorder)+'\n'+r"$\eta$=%.2f"%parameters.sca_broadening,
            size=s_-3,transform=ax.transAxes, bbox=props)

    ax = fig.add_subplot(132)
    for it,T in enumerate(Ts):
        Gamma_n = dataT[it]
        if logPlot:
            ax.loglog(system.evals[1:maxK],Gamma_n[:maxK-1],marker='o',markersize=10,label=r"T = %.2f MHz"%T)
        else:
            ax.scatter(system.evals[1:maxK],Gamma_n[:maxK-1],marker='o',label=r"T = %.2f MHz"%T)
            for i in best_modes:
                ax.scatter(system.evals[i],Gamma_n[i-1],marker='*',color='r')
    ax.legend(fontsize=s_-10)
    ax.set_xlabel("Energy (MHz)",size=s_)

    ax = fig.add_subplot(133)
    for i in range(1,maxK):
        Gamma_t = dataT[:,i-1]
        if logPlot:
            ax.loglog(Ts,Gamma_t,marker='o',markersize=10,label="mode # %s"%i)
        else:
            ax.plot(Ts,Gamma_t,marker='o',markersize=10,label="mode # %s"%i)
    ax.legend(fontsize=s_-10)
    ax.set_xlabel("Temperature (MHz)",size=s_)


    plt.suptitle(title,size=s_+5)
    fig.tight_layout()

    plt.show()


