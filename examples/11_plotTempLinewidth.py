""" Here we check the linewidth of the ZZ response by increasing the temperature.
Use with input4.txt
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
from wavespin.static.open import openSystem, openRamp
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fftfreq
from scipy.optimize import curve_fit

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation in OBC")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

""" Initialize all the systems and store them in a ramp object """
energies = np.linspace(-0.55,-0.5,3)
ramp = openRamp()
for en in energies:
    parameters.cor_energy = en if en != energies[0] else -100
    parameters.dia_Hamiltonian = (10,0,0,0,0,0)
    mySystem = openSystem(parameters)
    ramp.addSystem(mySystem)
""" Compute correlator XT and KW for all systems in the ramp """
ramp.correlatorsXT(verbose=verbose)
ramp.correlatorsKW(verbose=verbose)

sys0 = ramp.rampElements[0]
Ns = sys0.Ns
nOmega = sys0.nOmega
indMin = nOmega//2 + int(nOmega/500*0)
indMax = nOmega//2 + int(nOmega/500*70)
freqs = fftshift(fftfreq(nOmega,sys0.fullTimeMeasure/sys0.nTimes)) [indMin:indMax] / 10
xline = np.linspace(freqs[0],freqs[-1],1000)
def lorentzian(x, x0, gamma, A, y0):
    return y0 + (abs(A) / np.pi) * (0.5 * abs(gamma)) / ((x - x0)**2 + (0.5 * gamma)**2)

if 1: # Plot single k to check fitting
    fig = plt.figure(figsize=(25,20))
    for ii,indK in enumerate(range(0,63,1)):
        ax = fig.add_subplot(7,9,ii+1)
        ikx, iky = sys0._xy(indK)
        for ie,en in enumerate(energies):
            zz = np.abs(ramp.rampElements[ie].correlatorKW[ikx,iky][indMin:indMax])
            popt, pcov = curve_fit(lorentzian, freqs, zz, p0=[freqs[np.argmax(zz)],2,10,np.max(zz)])
            if 1:
                print(abs(popt[1]))
                sc = ax.scatter(freqs,zz,
                                label="Energy: %.2f"%en if en != energies[0] else "Energy: GS")
                ax.plot(xline, lorentzian(xline,*popt), color=sc.get_facecolors(),label="Width: %.5f"%popt[1])
        #
        ax.set_xlim(popt[0]-1,popt[0]+1)
        #ax.legend()
    fig.tight_layout()
    plt.show()
    exit()

def weighted_stdev(x, amplitude, w_mean = None, threshold = None):
    """ To compute the weighted standard deviation.
    """
    if w_mean == None:
        w_mean = weighted_mean(x, amplitude, threshold)
    if threshold != None:
        t = threshold * np.max(amplitude)  # 10% threshold
        mask = amplitude > t
        x = x[mask]
        if len(x) < 2:
            return(np.nan)
        amplitude = amplitude[mask]
    return np.sqrt(np.sum(amplitude * (x - w_mean) ** 2) / np.sum(amplitude))
def weighted_mean(x, amplitude, threshold = None):
    """ To compute the weighted mean.
    """
    if threshold != None:
        t = threshold * np.max(amplitude)  # 10% threshold
        mask = amplitude > t
        x = x[mask]
        amplitude = amplitude[mask]
    return np.sum(x * amplitude) / np.sum(amplitude)
# Plot linewidth over energy
print("Computing LW")
LW = np.zeros((Ns,len(energies)))
threshold = 0.25
for ik in range(1,Ns):
    ikx, iky = sys0._xy(ik)
    for ie,en in enumerate(energies):
        zz = np.abs(ramp.rampElements[ie].correlatorKW[ikx,iky][indMin:indMax])
        if 1: # std
            w_mean = weighted_mean(freqs, zz, threshold)
            LW[ik,ie] = weighted_stdev(freqs, zz,
                                       w_mean = w_mean,
                                       threshold = threshold)
        else: # Fit 
            popt, pcov = curve_fit(lorentzian, freqs, zz, p0=[freqs[np.argmax(zz)],0.2,1,1])
            if abs(popt[1]) < 2:
                LW[ik,ie] = abs(popt[1])
            else:
                LW[ik,ie] = np.nan
        if 0:
            fig = plt.figure(figsize=(12,5))
            ax = fig.add_subplot()
            sc = ax.scatter(freqs,zz,
                            label="Energy: %.2f"%en if en != -100 else "Energy: GS")
            ax.plot(xline, lorentzian(xline,*popt), color=sc.get_facecolors(),label="Width: %.5f"%popt[1])
            #ax.set_xlim(popt[0]-1,popt[0]+1)
            ax.legend()
            plt.show()

# Plot
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot()
if 1:
    for ik in range(1,Ns):
        ax.plot(energies,LW[ik],'-o')
mean = np.zeros(len(energies))
for ie in range(len(energies)):
    mean[ie] = np.mean(LW[:,ie])
ax.plot(energies,mean,'k-o',lw=5)
#ax.set_ylim(-0.1,2)
plt.show()

