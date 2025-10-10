""" Example script to plot the open dispersion over stop ratio and the correlator spread.
Use with input_5.txt
"""

import numpy as np
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
from wavespin.static.open import openSystem, openRamp

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
Nr = 20
stopRatios = np.linspace(0.1,1,Nr)
#Nr = 6
#stopRatios = np.linspace(3/11-0.03,3/11+0.03,Nr)

def funcLin(vIn,vFin,ratios):
    return (1-ratios)*vIn + ratios*vFin
def funcQuad(vIn,vFin,ratios):
    return (1-ratios)**2*vIn
def funcSqrt(vIn,vFin,ratios):
    return (1-ratios)**0.5*vIn
fG = funcLin
fH = funcQuad

gs = fG(gInitial,gFinal,stopRatios)
#hs = funciLin(hInitial,hFinal,stopRatios)
hs = fH(hInitial,hFinal,stopRatios)

# Critical ratio when h=4g
criticalRatio = np.linspace(0,1,200)[np.argmin(np.absolute(4*fG(gInitial,gFinal,np.linspace(0,1,200)) - fH(hInitial,hFinal,np.linspace(0,1,200))))]
print("Critical ratio at alpha=%.3f"%criticalRatio)

Lx = parameters.lat_Lx
Ly = parameters.lat_Ly
Ns = Lx*Ly
ramp = openRamp()
for ir,stopRatio in enumerate(stopRatios):
    """ Actual computation """
    print('stop ratio = %.2f'%stopRatio)
    parameters.dia_Hamiltonian = (gs[ir],0,0,0,hs[ir],0)
    ramp.addSystem(openSystem(parameters))

""" Compute correlator XT and KW for all systems in the ramp """
ramp.correlatorsXT(verbose=verbose)
ramp.correlatorsKW(verbose=verbose)

if 1:
    """ Plot ZZ correlator peak center over stop ratio for each mode """
    from scipy.fft import fftshift, fftfreq
    # Collect data
    peak = np.zeros((Ns,Nr))
    sys0 = ramp.rampElements[0]
    freqs = fftshift(fftfreq(sys0.nOmega,sys0.fullTimeMeasure/sys0.nTimes))
    for ik in range(Ns):
        ikx, iky = sys0._xy(ik)
        for ir in range(Nr):
            peak[ik,ir] = np.absolute(freqs[np.argmax(ramp.rampElements[ir].correlatorKW[ikx,iky])])
    # Plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot()
    colors = cm.Blues(np.linspace(0.1,1,Ns))
    for ik in range(1,Ns):
        ax.plot(stopRatios,peak[ik],'o-',color=colors[ik],label='peak of ZZ' if ik==Ns-1 else '')
    ax.axvline(criticalRatio,linestyle='dashed',color='k',alpha=0.8)
    ax.legend(loc='upper left',fontsize=20)
    ax.set_xticks(np.linspace(stopRatios[0],stopRatios[-1],10),["%.1f"%i for i in np.linspace(stopRatios[0],stopRatios[-1],10)],size=15)
    ax.set_xlabel(r"$\alpha$",size=20)
    ax.set_ylabel("Frequency (MHz)",size=20)
    ax.tick_params(axis='y', labelsize=15)
    # Colorbars
    cax1 = fig.add_axes([0.136, 0.78, 0.14, 0.02])
    cb1 = plt.colorbar(cm.ScalarMappable(cmap=cm.Blues), label='|k|',cax=cax1,orientation='horizontal')
    cb1.set_ticks(np.linspace(0,1,6))
    cb1.set_ticklabels(["%.2f"%i for i in np.linspace(0,1,6)/np.sqrt(2)])

    # Inset
    if 1:
        """ Plot ramp parameters """
        ax = fig.add_axes([0.4,0.65,0.2,0.2])
        ax.plot(stopRatios,gs,'-o',label=r"$g_{XY}$",color="green")
        ax.plot(stopRatios,hs,'-o',label=r"$h_z$",color="blue")
        ax.axvline(criticalRatio,linestyle='dashed',color='k',alpha=0.8)
        ax.legend(fontsize=15)
        ax.set_xlabel(r'$\alpha$',size=15)
    plt.show()
if 0:
    """ Plot dispersion over stop ratio for all modes """
    # Collect data
    peak = np.zeros((Ns,Nr))    # For each mode (Ns) energy at each stop ratio (Nr)
    for ik in range(Ns):
        for ir in range(Nr):
            peak[ik,ir] = ramp.rampElements[ir].evals[ik]
    # Plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot()
    colors = cm.Blues(np.linspace(0.1,1,Ns))
    colors2 = cm.Reds(np.linspace(0.1,1,Ns))
    factors = np.ones(Nr)
    mask = stopRatios<criticalRatio
    factors[mask] *= 2
    for ik in range(1,Ns):
        ax.plot(stopRatios,peak[ik]*factors,'o-',color=colors[ik],alpha=0.9,zorder=-1,label='2*dispersion in gapped phase' if ik==Ns-1 else '')
        ax.plot(stopRatios,peak[ik],'o-',color=colors2[ik],alpha=0.4,zorder=0,label='1*disersion in gapped phase' if ik==Ns-1 else '')
    ax.axvline(criticalRatio,linestyle='dashed',color='k',alpha=0.8)
    ax.legend(loc='upper center',fontsize=20)
    ax.set_xticks(np.linspace(stopRatios[0],stopRatios[-1],10),["%.1f"%i for i in np.linspace(stopRatios[0],stopRatios[-1],10)],size=15)
    ax.set_xlabel(r"$\alpha$",size=20)
    ax.set_ylabel("Frequency (MHz)",size=20)
    ax.tick_params(axis='y', labelsize=15)
    # Colorbars
    cax1 = fig.add_axes([0.136, 0.78, 0.14, 0.02])
    cb1 = plt.colorbar(cm.ScalarMappable(cmap=cm.Blues), label='|k|',cax=cax1,orientation='horizontal')
    cb1.set_ticks(np.linspace(0,1,6))
    cb1.set_ticklabels(["%.2f"%i for i in np.linspace(0,1,6)/np.sqrt(2)])
    cax2 = fig.add_axes([0.136, 0.83, 0.14, 0.02])
    cb2 = plt.colorbar(cm.ScalarMappable(cmap=cm.Reds),cax=cax2,orientation='horizontal')
    cb2.set_ticks([])
    plt.show()
if 0:
    """ Plot correlator with different colors depending on the magnon contribution """
    # Compute other magnon contributions
    ramps = []
    for m in range(2):
        parameters.cor_magnonModes = (m+1,)
        ramps.append(openRamp())
        for ir,stopRatio in enumerate(stopRatios):
            """ Actual computation """
            print('stop ratio = %.2f'%stopRatio)
            g_p = (1-stopRatio)*gInitial + stopRatio*gFinal
            h_p = (1-stopRatio)*hInitial + stopRatio*hFinal
            parameters.dia_Hamiltonian = (g_p,0,0,0,h_p,0)
            ramps[m].addSystem(openSystem(parameters))

        """ Compute correlator XT and KW for all systems in the ramp """
        ramps[m].correlatorsXT(verbose=verbose)
        ramps[m].correlatorsKW(verbose=verbose)
    # Compute axis grid
    from scipy.fft import fftfreq, fftshift
    sys0 = ramp.rampElements[0]
    transformType = sys0.p.cor_transformType
    nOmega = sys0.nOmega
    fullTimeMeasure = sys0.fullTimeMeasure
    nTimes = sys0.nTimes
    #Axis
    kx = np.pi * np.arange(0, Lx ) / (Lx )
    ky = np.pi * np.arange(0, Ly ) / (Ly )
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2)
    K_flat = K_mag.ravel()
    freqs = fftshift(fftfreq(nOmega,fullTimeMeasure/nTimes))
    # Define k bins
    num_k_bins = 50             ##################################
    k_bins = np.linspace(0,np.sqrt(2)*np.pi, num_k_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    K_mesh, W_mesh = np.meshgrid(k_centers, freqs, indexing='ij')
    # average over k in the bins
    data1 = np.zeros((Nr,num_k_bins,nOmega))
    data2 = np.zeros((Nr,num_k_bins,nOmega))
    for iP in range(Nr):
        corr_flat1 = ramps[0].rampElements[iP].correlatorKW.reshape(Lx*Ly, nOmega)
        corr_flat2 = ramps[1].rampElements[iP].correlatorKW.reshape(Lx*Ly, nOmega)
        for i in range(num_k_bins):
            mask = (K_flat >= k_bins[i]) & (K_flat < k_bins[i+1])
            if np.any(mask):
                data1[iP, i, :] = np.mean(np.abs(corr_flat1[mask, :]), axis=0)
                data2[iP, i, :] = np.mean(np.abs(corr_flat2[mask, :]), axis=0)
    # Figure
    from wavespin.plots.rampPlots import createFigure, SqrtNorm
    import matplotlib.pyplot as plt
    fig, axes, rows, cols = createFigure(Nr,subplotSize=(4,4),nRows=1,nCols=Nr)
    title = "ZZ with different magnon contributions"
    plt.suptitle(title,fontsize=20)
    ylim = 70
    for iP in range(Nr):
        mag1 = data1[iP]
        mag2 = data2[iP]
        vmax1 = np.max(mag1)
        vmax2 = np.max(mag2)
        vmax = max(vmax1,vmax2)
        ax = axes[iP]
        ax.set_facecolor('white')
        stopRatio = stopRatios[iP]
        mask1 = mag1 > mag2
        mask2 = ~mask1
        mesh1 = ax.pcolormesh(K_mesh, W_mesh,
                              np.ma.masked_where(~mask1,mag1),
                              shading='auto',
                              cmap='Reds',
                              norm=SqrtNorm(vmin=0,vmax=vmax),
                              #alpha=0.8
                        )
        mesh2 = ax.pcolormesh(K_mesh, W_mesh,
                              np.ma.masked_where(~mask2,mag2),
                              shading='auto',
                              cmap='Blues',
                              norm=SqrtNorm(vmin=0,vmax=vmax),
                              #alpha=0.5
                            )
        ax.set_ylim(-ylim,ylim)
        ax.set_xlim(0,np.sqrt(2)*np.pi)
        if vmax==vmax1:
            cbar1 = fig.colorbar(mesh1, ax=ax)
        else:
            cbar2 = fig.colorbar(mesh2, ax=ax)
        if iP in [cols*i for i in range(0,rows)]:
            ax.set_ylabel(r'$\omega$',fontsize=15)
        if iP in np.arange((rows-1)*cols,rows*cols):
            ax.set_xlabel(r'$|k|$',fontsize=15)
        ax.set_title(r"$\alpha=$%.3f"%stopRatio,size=20)
    for i in range(Nr,len(axes)):       #set to blank extra plots
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()






















































