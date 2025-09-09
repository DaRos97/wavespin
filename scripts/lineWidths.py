""" Example script to compute and plot the zz correlator of a rectangular lattice.
Use with input_4.txt
"""


import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importOpenParameters as importParameters
from wavespin.lattice.lattice import latticeClass
from wavespin.static.open import openSystem, openRamp
from scipy.fft import fftshift, fftfreq

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation in OBC")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

""" Define the parameters of the system at different 'times' """
nP = 5     #number of parameters computed in the "ramp" -> analogue to stop ratio
gInitial = 0
gFinal = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
hInitial = 15
hFinal = 0
pValues = np.array([0.1,0.2,0.23,0.25,0.26,0.27,0.28,0.29,0.3,0.35,0.4,0.6,0.8,1])
#np.linspace(0.1,0.5,nP)
nP = pValues.shape[0]
g_p = (1-pValues)*gInitial + pValues*gFinal
h_p = (1-pValues)*hInitial + pValues*hFinal

""" Which k to extract """
kValue = 0.5*np.pi
kx = np.pi * np.arange(0, parameters.Lx ) / (parameters.Lx )
ky = np.pi * np.arange(0, parameters.Ly ) / (parameters.Ly )
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K_mag = np.sqrt(KX**2 + KY**2)
K_flat = K_mag.ravel()
kIndex = np.argmin(abs(kValue-K_flat))
lattice = latticeClass(parameters,boundary='open')
kxInd,kyInd = lattice._xy(kIndex)
data = []
for im, magnonModes in enumerate([(1,),(2,),(1,2)]):
    parameters.magnonModes = magnonModes
    """ Initialize all the systems and store them in a ramp object """
    ramp = openRamp()
    for i in range(nP):
        termsHamiltonian = (g_p[i],0,0,0,h_p[i])
        ramp.addSystem(openSystem(parameters,termsHamiltonian))
        ramp.rampElements[i].nOmega = ramp.rampElements[i].nTimes

    """ Compute correlator XT and KW for all systems in the ramp """
    ramp.correlatorsXT(verbose=verbose)
    ramp.correlatorsKW(verbose=verbose)

    """ Extract single k-value and compute linewidth for each parameter set. """
    data.append([])
    for iP in range(nP):
        data[im].append(np.absolute(ramp.rampElements[iP].correlatorKW[kxInd,kyInd,:]))

system = ramp.rampElements[0]
freqs = fftshift(fftfreq(system.nOmega,system.fullTimeMeasure/system.nTimes))
""" Compute widths """
from scipy.optimize import curve_fit
def lorentz(x,x0,gamma,c):
    return 1/np.pi*gamma/((x-x0)**2+gamma**2) + c
def gauss(x,x0,sigma,c):
    return np.exp(-(x-x0)**2/2/sigma**2)/np.sqrt(2*np.pi*sigma**2)+c
func = lorentz
def extractWidth(signal):
    yvals = signal[signal.shape[0]//2:]
    xvals = freqs[signal.shape[0]//2:]
    xcenter = xvals[np.argmax(yvals)]
    gamma0 = 0.5
    popt,pcov = curve_fit(func,xvals,yvals,p0=(xcenter,gamma0,0))
    return popt
widths = []
centers = []
offs = []
for im in range(3):
    widths.append([])
    centers.append([])
    offs.append([])
    for iP in range(nP):
        try:
            fit = extractWidth(data[im][iP])
        except:
            print(str(iP)+" not fitting")
            fit = [np.nan,np.nan,np.nan]
        widths[im].append(fit[1])
        centers[im].append(fit[0])
        offs[im].append(fit[2])

""" Plot results """
import matplotlib.pyplot as plt
title = ['1-magnon','2-magnon','1 and 2 magnon']
fig = plt.figure(figsize=(20,15))
n0 = data[0][0].shape[0]//2
f0 = freqs[n0:]
cmap = plt.cm.viridis
colors = [cmap(i) for i in np.linspace(0,1,nP)]
for im in [0,1,2]:#range(3):
    # Cut plots
    ax = fig.add_subplot(2,3,im+1)
    ax.set_title(title[im],size=20)
    for iP in [0,nP//2,nP-1]:
        ax.plot(f0,data[im][iP][n0:],label='stop ratio:%.4f'%pValues[iP],c=colors[iP])
        ax.plot(f0,func(f0,centers[im][iP],widths[im][iP],offs[im][iP]),ls='dashed',c=colors[iP])
    ax.set_xlim(0,50)
    ax.legend()

# width plot over stop ratio
ax = fig.add_subplot(2,1,2)
ax.set_title('with over stop ratio',size=20)
for im in [0,2]:#range(3):
    ax.plot(pValues,widths[im],label='magnon modes::%s'%title[im])
ax.axvline(0.275,c='r',ls='dashed')
ax.set_xlabel("stop ratio")
ax.set_ylabel("lorentzian width")
ax.legend()

plt.show()
















