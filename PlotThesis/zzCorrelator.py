""" Here I plot the zz correlator for the thesis.
"""
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.static.open import openHamiltonian, openSystem
from wavespin.plots.rampPlots import SqrtNorm
from wavespin.tools.inputUtils import importParameters
import wavespin.tools.pathFinder as pf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fftshift
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

final = True
if final:
    plt.rcParams.update({
        "text.usetex": True,              # Use LaTeX for all text
        "font.family": "serif",           # Set font family
        "font.serif": ["Computer Modern"], # Default LaTeX font
    })

save = True
Lxs = (7,20)
Lys = (9,20)
stopRatios = (0.2,3/11,0.3,0.6,1)
pertSite = ((3,4),(10,10))

### Data
dataFn = pf.getFilename(*('zzCorrelator',Lxs,Lys,stopRatios),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    corr = np.load(dataFn)['corr']      # (2,5,...) #size,stop ratio, actual size
    K_ms = np.load(dataFn)['K_ms']      # (2, ..)
    W_ms = np.load(dataFn)['W_ms']      # (2, ..)
else:
    # Colored plots
    gFinal = 10
    hInitial = 15
    parameters = importParameters()
    parameters.cor_correlatorType = 'zz'
    parameters.cor_transformType = 'dct'
    parameters.dia_excludeZeroMode = True
    parameters.cor_magnonModes = (1,2,3,4)
    num_k_bins = 50
    k_bins = np.linspace(0,np.sqrt(2)*np.pi, num_k_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    corr = []
    K_ms = []
    W_ms = []
    for i in range(2):
        parameters.cor_perturbationSite = pertSite[i]
        parameters.lat_Lx = Lxs[i]
        parameters.lat_Ly = Lys[i]
        temp = []
        for ir in range(5):         #stop ratio
            print("ratio #",ir)
            alpha = stopRatios[ir]
            parameters.dia_Hamiltonian = (gFinal*alpha,0,0,0,hInitial*(1-alpha),0)
            sys = openSystem(parameters)
            sys.diagonalize()
            sys.realSpaceCorrelator()
            sys.momentumSpaceCorrelator()
            if ir==0:
                K_flat = np.sqrt(sys.momentum[:,0]**2 + sys.momentum[:,1]**2)
                freqs = fftshift(fftfreq(sys.nOmega,sys.fullTimeMeasure/sys.nTimes))
                K_m, W_m = np.meshgrid(k_centers, freqs, indexing='ij')
                K_ms.append(K_m)
                W_ms.append(W_m)
            # Compute actually plottable values
            corr_flat = sys.correlatorKW
            P_k_omega = np.zeros((num_k_bins,sys.nOmega))
            for i in range(num_k_bins):
                mask = (K_flat >= k_bins[i]) & (K_flat < k_bins[i+1])
                if np.any(mask):
                    P_k_omega[i, :] = np.mean(np.abs(corr_flat[mask, :]), axis=0)
            #
            temp.append(P_k_omega)
        corr.append(temp)
    if save:
        np.savez(dataFn,
                 corr=corr,
                 K_ms=K_ms,
                 W_ms=W_ms,
                 )

# Figure
fig = plt.figure(figsize=(4.65,2.5))
gs = fig.add_gridspec(
    2,5,
    wspace=0.05,
    hspace=0.05
)

s_norm = 10
s_small = 9

for i_s in range(2):     #rows
    axes_2d = [fig.add_subplot(gs[i_s,j]) for j in range(5)]
    min1 = []
    max1 = []
    for ir in range(5):
        min1.append(np.min(corr[i_s][ir]))
        max1.append(np.max(corr[i_s][ir]))
    vmin = np.min(np.array(min1)) / 10
    vmax = np.max(np.array(max1)) / 10

    for ir in range(5):
        ax = axes_2d[ir]
        zz = corr[i_s][ir]/10
        mesh = ax.pcolormesh(
            K_ms[i_s],
            W_ms[i_s]/10,
            zz,
            cmap='Blues',
            norm=SqrtNorm(vmin=vmin,vmax=vmax),
            rasterized=True
        )
        ax.set_ylim(-7,7)
        if i_s==0:
            ax.set_title(r"$\alpha=%.3f$"%stopRatios[ir],size=s_norm)
        #ax.yaxis.set_ticks_position('both')  # place ticks on both sides
        #ax.xaxis.set_ticks_position('both')  # place ticks on both sides
        ax.tick_params(
            axis='both',
            direction='in',
            left=True,
            right=True,
            top=True,
            bottom=True,
            length=4,
            width=0.8,
            pad=3,
            labelsize=s_small
        )
        ax.set_xticks(np.arange(0,5,1),[str(i) for i in np.arange(0,5,1)])
        ax.set_yticks(list(np.linspace(-6,6,5)),[r"$%d$"%i for i in np.linspace(-6,6,5)])
        if ir == 0:
            ax.set_ylabel(r"$\omega(g)$",size=s_norm)
        else:
            ax.set_yticklabels([])
        if i_s==1:
            ax.set_xlabel(
                r'$|k|$',
                fontsize=s_norm
            )
        else:
            ax.set_xticklabels([])

    #Blue cb
    xb = 0.92
    yb = 0.54 if i_s==0 else 0.14
    hb = 0.38
    wb = 0.02
    cbar_ax = fig.add_axes([xb, yb, wb, hb])  # [left, bottom, width, height]
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label(r"$\vert \chi_{ZZ}(\omega)\vert$",size=s_norm)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    cbar.ax.tick_params(
        width=0.8,
        labelsize=7
    )
    #cbar.ax.yaxis.set_label_position('left')


plt.subplots_adjust(
    bottom = 0.138,
    top = 0.92,
#    right = 0.951,
    left = 0.05
)

if final:
    plt.savefig(
        "Figures/zzCorrelators.pdf",
        bbox_inches="tight",
        dpi=600
    )
















































