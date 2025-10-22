""" Example script to make fancy plots about magnon contributions.
Use with input_4.txt
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
Nr = 70
stopRatios = np.linspace(0.1,1,Nr)
#stopRatios = np.array([0.01,0.2,0.22,0.26,0.27,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#stopRatios = np.array([0.1,0.2,0.25,0.26,0.27,0.28,0.29,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#Nr = 6
#stopRatios = np.linspace(3/11-0.03,3/11+0.03,Nr)[1:]
Nr = stopRatios.shape[0]

def funcLin(vIn,vFin,ratios):
    return (1-ratios)*vIn + ratios*vFin
def funcQuad(vIn,vFin,ratios):
    return (1-ratios)**2*vIn
def funcSqrt(vIn,vFin,ratios):
    return (1-ratios)**0.5*vIn
fG = funcLin
fH = funcLin

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
    #print('stop ratio = %.2f'%stopRatio)
    parameters.dia_Hamiltonian = (gs[ir],0,0,0,hs[ir],0)
    ramp.addSystem(openSystem(parameters))

""" Compute correlator XT and KW for all systems in the ramp """
ramp.correlatorsXT(verbose=verbose)
ramp.correlatorsKW(verbose=verbose)

import matplotlib.pyplot as plt
from scipy.fft import fftshift, fftfreq
if 0:
    """ Plot ZZ correlator peak center over stop ratio for each mode """
    from scipy.fft import fftshift, fftfreq
    # Collect data
    peak = np.zeros((Ns,Nr))
    sys0 = ramp.rampElements[0]
    freqs = fftshift(fftfreq(sys0.nOmega,sys0.fullTimeMeasure/sys0.nTimes)) / 10
    for ik in range(Ns):
        ikx, iky = sys0._xy(ik)
        for ir in range(Nr):
            peak[ik,ir] = np.absolute(freqs[np.argmax(ramp.rampElements[ir].correlatorKW[ikx,iky])])
    # Plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    colors = cm.plasma_r(np.linspace(0.1,1,Ns))
    for ik in range(1,Ns):
        ax.plot(stopRatios,peak[ik],'o-',color=colors[ik],label='peak of ZZ' if ik==Ns-1 else '')
    ax.axvline(criticalRatio,linestyle='dashed',color='k',alpha=0.8)
    #ax.legend(loc='upper left',fontsize=20)
    ax.set_xticks(np.linspace(stopRatios[0],stopRatios[-1],10),["%.1f"%i for i in np.linspace(stopRatios[0],stopRatios[-1],10)],size=15)
    ax.set_xlabel(r"$\alpha$",size=20)
    ax.set_ylabel("Mean frequency (g)",size=20)
    ax.tick_params(axis='y', labelsize=15)
    # Colorbars
    cax1 = fig.add_axes([0.4, 0.8, 0.2, 0.05])  #left-bottom-width-height
    cb1 = plt.colorbar(cm.ScalarMappable(cmap=cm.plasma_r),cax=cax1,orientation='horizontal')
    cb1.set_label(r'|k| ($\pi$)',fontsize=20)
    cb1.set_ticks(np.linspace(0.1,0.9,3))
    cb1.set_ticklabels(["%.2f"%i for i in np.linspace(0.1,0.9,3)/np.sqrt(2)],fontsize=15)

    # Inset
    if 0:
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
    freqs = fftshift(fftfreq(nOmega,fullTimeMeasure/nTimes)) / 10
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
    #plt.suptitle(title,fontsize=20)
    ylim = 7
    data1 /= 10
    data2 /= 10
    vmax1 = np.max(data1)
    vmax2 = np.max(data2)
    vmax = max(vmax1,vmax2)
    for iP in range(Nr):
        mag1 = data1[iP]
        mag2 = data2[iP]
        vmax1 = np.max(mag1)
        vmax2 = np.max(mag2)
        vmaxl = max(vmax1,vmax2)
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
        if iP==0:
            ax.set_ylabel(r"$\omega(g)$",size=20)
            #ax.set_yticks()
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-ylim,ylim)
        ax.set_xlim(0,np.sqrt(2)*np.pi)
        ax.yaxis.set_ticks_position('both')  # place ticks on both sides
        ax.xaxis.set_ticks_position('both')  # place ticks on both sides
        ax.tick_params(axis='both',direction='in',length=7,width=1,pad=2,labelsize=15)
        if iP in np.arange((rows-1)*cols,rows*cols):
            ax.set_xlabel(r'$|k|$',fontsize=15)
        ax.set_title(r"$\alpha=$%.3f"%stopRatio,size=20)
    if 1:   # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.75])  # [left, bottom, width, height]
        cbar = fig.colorbar(mesh1, cax=cbar_ax)
        cbar.set_label(r"$\vert\chi_{ZZ}(\omega)\vert$",fontsize=20)
    fig.subplots_adjust(left=0.052, right=0.9, top=0.9, bottom=0.14, wspace=0.15)
    #plt.tight_layout()
    plt.show()
if 0:
    """ Plot ZZ correlator spread over stop ratio for each mode with different magnon contributions"""
    if 1:# Compute other two contributions
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
    # Collect data
    spread = np.zeros((Ns,Nr))
    spread1 = np.zeros((Ns,Nr))
    spread2 = np.zeros((Ns,Nr))
    sys0 = ramp.rampElements[0]
    freqs = fftshift(fftfreq(sys0.nOmega,sys0.fullTimeMeasure/sys0.nTimes))
    mask1to15 = (freqs > 1) & (freqs < 15)
    for ik in range(Ns):
        ikx, iky = sys0._xy(ik)
        for ir in range(Nr):
            zz = np.abs(ramp.rampElements[ir].correlatorKW[ikx,iky])
            zz /= np.sum(zz)
            spread[ik,ir] = np.mean(zz[mask1to15])
            if stopRatios[ir]>3/11:
                zz1 = np.abs(ramps[0].rampElements[ir].correlatorKW[ikx,iky])
                zz1 /= np.sum(zz1)
                spread1[ik,ir] = np.mean(zz1[mask1to15])
            else:
                spread1[ik,ir] = np.nan
            zz2 = np.abs(ramps[1].rampElements[ir].correlatorKW[ikx,iky])
            zz2 /= np.sum(zz2)
            spread2[ik,ir] = np.mean(zz2[mask1to15])
    # Plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot()
    Nlist = [30, 33, 38, 41, 42, 45, 47, 48, 51, 53, 54, 55, 59, 60, 62]
    avg = np.mean(spread[Nlist,:],axis=0)
    avg1 = np.mean(spread1[Nlist,:],axis=0)
    avg2 = np.mean(spread2[Nlist,:],axis=0)
    for ik in Nlist:
        ax.plot(stopRatios,spread[ik],'-',color='chartreuse',zorder=3)
        ax.plot(stopRatios,spread1[ik],'-',color='skyblue',zorder=1)
        ax.plot(stopRatios,spread2[ik],'-',color='khaki',zorder=2)
    ax.plot(stopRatios,avg,'o-',color='g',zorder=6,lw=3,label='full correlator')
    ax.plot(stopRatios,avg1,'o-',color='b',zorder=5,lw=3,label='1 magnon')
    ax.plot(stopRatios,avg2,'o-',color='goldenrod',zorder=4,lw=3,label='2 magnon')
    ax.legend(fontsize=20)
    ax.axvline(criticalRatio,linestyle='dashed',color='k',alpha=0.8)
    ax.set_xticks(np.linspace(stopRatios[0],stopRatios[-1],10),["%.1f"%i for i in np.linspace(stopRatios[0],stopRatios[-1],10)],size=15)
    ax.set_xlabel(r"$\alpha$",size=20)
    ax.set_ylabel(r"$\bar{\vert\chi\vert}(\omega)_{1-15\text{MHz}}$",size=20)
    ax.tick_params(axis='y', labelsize=15)

    plt.show()

if 0:
    """ Plot std over stop ratio """
    sys0 = ramp.rampElements[0]
    if 1:# Compute other two contributions
        ramps = []
        for m in range(2):
            parameters.cor_magnonModes = (m+1,)
            ramps.append(openRamp())
            for ir,stopRatio in enumerate(stopRatios):
                """ Actual computation """
                g_p = (1-stopRatio)*gInitial + stopRatio*gFinal
                h_p = (1-stopRatio)*hInitial + stopRatio*hFinal
                parameters.dia_Hamiltonian = (g_p,0,0,0,h_p,0)
                ramps[m].addSystem(openSystem(parameters))

            """ Compute correlator XT and KW for all systems in the ramp """
            ramps[m].correlatorsXT(verbose=verbose)
            ramps[m].correlatorsKW(verbose=verbose)
    threshold = 0.25        # Consider only peaks which are at least 25% of maximum
    indMin = sys0.nOmega//2 + int(sys0.nOmega/500*5)
    indMax = sys0.nOmega//2 + int(sys0.nOmega/500*80)
    freqs = fftshift(fftfreq(sys0.nOmega,sys0.fullTimeMeasure/sys0.nTimes)) [indMin:indMax] / 10
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

    std = np.zeros((Ns,Nr))
    std1 = np.zeros((Ns,Nr))
    std2 = np.zeros((Ns,Nr))
    # Compute std
    for ik in range(1,Ns):
        ikx, iky = sys0._xy(ik)
        for ir in range(Nr):
            zz = np.abs(ramp.rampElements[ir].correlatorKW[ikx,iky][indMin:indMax])
            w_mean = weighted_mean(freqs, zz, threshold)
            std[ik,ir] = weighted_stdev(freqs, zz,
                                        w_mean = w_mean,
                                        threshold = threshold)
            zz1 = np.abs(ramps[0].rampElements[ir].correlatorKW[ikx,iky][indMin:indMax])
            w_mean1 = weighted_mean(freqs, zz1, threshold)
            std1[ik,ir] = weighted_stdev(freqs, zz1,
                                         w_mean = w_mean1,
                                         threshold = threshold)
            zz2 = np.abs(ramps[1].rampElements[ir].correlatorKW[ikx,iky][indMin:indMax])
            w_mean2 = weighted_mean(freqs, zz2, threshold)
            std2[ik,ir] = weighted_stdev(freqs, zz2,
                                         w_mean = w_mean2,
                                         threshold = threshold)
            if 0:
                fig = plt.figure(figsize=(15,8))
                ax = fig.add_subplot()
                ax.plot(freqs,zz,label="{:.7f}".format(std[ik,ir]))
                ax.axvline(w_mean)
                ax.legend()
                plt.show()

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    for ik in range(Ns):
        continue
        ax.plot(stopRatios,std[ik])
    mean = np.zeros(Nr)
    mean1 = np.zeros(Nr)
    mean2 = np.zeros(Nr)
    indKi = Ns-15
    indKf = Ns
    for ir in range(Nr):
        mean[ir] = np.mean(std[indKi:indKf,ir])
        mean1[ir] = np.mean(std1[indKi:indKf,ir])
        mean2[ir] = np.mean(std2[indKi:indKf,ir])
    ax.plot(stopRatios,mean,'-o',color='k',lw=5,label='full')
    ax.plot(stopRatios,mean1,'-o',color='r',lw=3,label='1-magnon',alpha=0.7)
    y_min, y_max = ax.get_ylim()
    ax.plot(stopRatios,mean2,'-o',color='b',lw=3,label='2-magnon',alpha=0.7)
    ax.axvline(criticalRatio,linestyle='dashed',color='k',alpha=0.8)
    #
    ax.set_ylabel(r"Spectral width ($g$)",size=20)
    ax.set_xlabel(r"$\alpha$",size=20)
    ax.set_ylim(y_min,y_max)
    ax.legend(fontsize=20)
    plt.show()
















































