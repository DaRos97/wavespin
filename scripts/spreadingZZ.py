""" Here we check the linewidth of the ZZ response by increasing the temperature.
To do it we need to implement finite-temperature expectation values using a magnon bath.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
from wavespin.static.open import openSystem, openRamp
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fftfreq
from scipy.fft import dctn, fft
from scipy.optimize import curve_fit

parameters = importParameters()

""" Initialize all the systems and store them in a ramp object """
energies = np.linspace(-0.55,-0.25,4)
Lx = 7#11
Ly = 9#12
offSiteList = ()
pS = (Lx//2,Ly//2)

parameters.lat_Lx = Lx
parameters.lat_Ly = Ly
parameters.lat_offSiteList = offSiteList
parameters.lat_plotLattice = 0#True
parameters.cor_plotKW = True
parameters.cor_correlatorType = 'zz'
parameters.cor_perturbationSite = pS
parameters.cor_magnonModes = (1,2,3,4,5,6,)
ramp = openRamp()
for en in energies:
    parameters.dia_Hamiltonian = (10,0,0,0,0,0)
    mySystem = openSystem(parameters)
    mySystem.p.cor_energy = en if en != energies[0] else mySystem.GSE
    ramp.addSystem( mySystem )

""" Compute correlator XT for all systems in the ramp """
if 0:
    for iP in range(ramp.nP):
        mySys = ramp.rampElements[iP]
        pI = mySys.perturbationIndex
        U = mySys.U_[:,1:].copy()
        V = mySys.V_[:,1:].copy()
        for i in range(mySys.Ns):
            x,y = mySys._xy(i)
            U[i,:] *= (-1)**(x+y)
            V[i,:] *= (-1)**(x+y)
        evals = mySys.evals[1:]
        times = mySys.measureTimeList
        temperature = mySys._temperature(mySys.p.cor_energy)
        exp_e = np.exp(1j*2*np.pi*times[:,None]*evals[None,:])
        exp_e_c = exp_e.conj()
        corr = np.zeros((mySys.Ns,mySys.nTimes),dtype=complex)
        #
        corr += np.einsum('tn,in,n->it',exp_e,U+V,U[pI,:]+V[pI,:],optimize=True)
        #
        if temperature != 0:
            Bn = 1/(np.exp(evals/temperature)-1)
            corr += np.einsum('n,tn,in,n->it',Bn,exp_e_c,U+V,U[pI,:]+V[pI,:],optimize=True)
            corr += np.einsum('n,tn,in,n->it',Bn,exp_e,U+V,U[pI,:]+V[pI,:],optimize=True)
        #
        ramp.rampElements[iP].correlatorXT = -corr
else:
    ramp.correlatorsXT()

if 0:   #Plot time traces
    fig = plt.figure(figsize=(15,15))
    px,py = parameters.cor_perturbationSite
    for ix in range(Lx):
        for iy in range(Ly):
            ax = fig.add_subplot(Lx,Ly,1+iy+Ly*ix)
            for ie in range(len(energies)):
                idx = ramp.rampElements[ie]._idx(ix,iy)
                corr = ramp.rampElements[ie].correlatorXT[idx,:100]
                times = ramp.rampElements[ie].measureTimeList[:100]
                ax.plot(times,np.real(corr),label='real E=%.3f'%energies[ie])
                #ax.plot(times,np.imag(corr),label='imag E=%.3f'%energies[ie])
            ax.set_title("Site: %d,%d"%(ix,iy))
            if ix==0 and iy==0:
                ax.legend()
    fig.tight_layout()
    plt.show()
#exit()

""" Compute correlator KW for all systems in the ramp """
if 0:
    for iP in range(ramp.nP):
        mySys = ramp.rampElements[iP]
        corrXT = np.imag(mySys.correlatorXT)
        nOmega = mySys.nOmega
        kx = np.pi * np.arange(0, Lx ) / (Lx )
        ky = np.pi * np.arange(0, Ly ) / (Ly )
        nTimes = mySys.nTimes
        corrKW = np.zeros((mySys.Ns,nOmega),dtype=complex)
        temp = np.zeros((Lx,Ly,nTimes),dtype=complex)
        momentum = np.zeros((mySys.Ns,2))
        for it in range(nTimes):
            temp[:,:,it] = dctn(corrXT[:,it].reshape(Lx,Ly),
                                type=2,
                                #norm='ortho'
                                )
        for ind in range(mySys.Ns):
            x,y = mySys._xy(ind)
            corrKW[ind] = fftshift(fft(temp[x,y],n=nOmega))
            momentum[ind] = np.array([kx[x],ky[y]])
        mySys.correlatorKW = corrKW
        mySys.momentum = momentum
else:
    ramp.correlatorsKW()

#exit()

""" Fit peak to see if it spreads """
sys0 = ramp.rampElements[0]
Ns = sys0.Ns
nOmega = sys0.nOmega
indMin = nOmega//2 + int(nOmega/500*0)
indMax = nOmega//2 + int(nOmega/500*70)
freqs = fftshift(fftfreq(nOmega,sys0.fullTimeMeasure/sys0.nTimes)) [indMin:indMax] / 10
xline = np.linspace(freqs[0],freqs[-1],1000)
def lorentzian(x, x0, gamma, A, y0):
    return y0 + (abs(A) / np.pi) * (0.5 * abs(gamma)) / ((x - x0)**2 + (0.5 * gamma)**2)

if 0: # Plot single k to check fitting
    fig,axs = plt.subplots(1,2,figsize=(20,15))
    ymax = -1
    ymin = 10
    for iax,ii in enumerate([52,100]):
        mySys0 = ramp.rampElements[0].correlatorKW[ii]
        ax = axs[iax]
        for ie,en in enumerate(energies):
            mySys = ramp.rampElements[ie].correlatorKW[ii]
            zz = np.abs(mySys)[indMin:indMax]
            popt, pcov = curve_fit(lorentzian, freqs, zz, p0=[freqs[np.argmax(zz)],2,10,np.max(zz)])
            if 1:
                print(ii,en,abs(popt[1]))
                sc = ax.scatter(freqs,zz,
                                label="Energy: %.2f"%en if en != energies[0] else "Energy: GS")
                ax.plot(xline, lorentzian(xline,*popt), color=sc.get_facecolors(),label="Width: %.5f"%popt[1])
                ym,yM = ax.get_ylim()
                ymin = min(ym,ymin)
                ymax = max(yM,ymax)
        #
        ax.set_xlim(popt[0]-1,popt[0]+1)
        ax.set_title(ii)
        ax.legend()
    for iax,ii in enumerate([52,100]):
        ax = axs[iax]
        ax.set_ylim(ymin,ymax)
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
LW = np.zeros((Ns-1,len(energies)))
threshold = 0.10
for ik in range(1,Ns):
    for ie,en in enumerate(energies):
        zz = np.abs(ramp.rampElements[ie].correlatorKW[ik][indMin:indMax])
        if 1: # std
            w_mean = weighted_mean(freqs, zz, threshold)
            LW[ik-1,ie] = weighted_stdev(
                freqs, zz,
                w_mean = w_mean,
                threshold = threshold
            )
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
if 0:
    #ks = [4,5,17,22,31]    #11x12
    ks = [4,14,26,34,47]
    import matplotlib
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cmap = matplotlib.colormaps.get_cmap('plasma')
    norm = Normalize(vmin=0.2, vmax=0.5)
    for ik in range(1,Ns):
        v = np.linalg.norm(ramp.rampElements[0].momentum[ik]) / np.pi/2
        ax.plot(
            energies,
            LW[ik-1],
            '-o',
            color=cmap(norm(v)),
            label="%.3f"%v,
        )
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=r"$\vert k\vert$")
    #ax.legend()
mean = np.zeros(len(energies))
for ie in range(len(energies)):
    mean[ie] = np.mean(LW[:,ie])
ax.plot(energies,mean,'k-o',lw=5)
#ax.set_ylim(-0.1,2)
plt.show()

