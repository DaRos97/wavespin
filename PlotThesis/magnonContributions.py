""" Here I plot the magnon contributions' figure for the thesis.
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

final = True
if final:
    plt.rcParams.update({
        "text.usetex": True,              # Use LaTeX for all text
        "font.family": "serif",           # Set font family
        "font.serif": ["Computer Modern"], # Default LaTeX font
    })

save = True
Lx = 7 if not final else 20
Ly = 9 if not final else 20
stopRatios = np.linspace(3/11-0.03,3/11+0.03,6)[1:]

### Data
dataFn = pf.getFilename(*('magnonContributions',Lx,Ly),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    corr = np.load(dataFn)['corr']
    K_m = np.load(dataFn)['K_m']
    W_m = np.load(dataFn)['W_m']
else:
    # Colored plots
    gFinal = 10
    hInitial = 15
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.cor_correlatorType = 'zz'
    parameters.cor_transformType = 'dct'
    parameters.dia_excludeZeroMode = True
    parameters.cor_perturbationSite = (3,4)
    num_k_bins = 50
    k_bins = np.linspace(0,np.sqrt(2)*np.pi, num_k_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    corr = []
    for ir in range(5):         #stop ratio
        print("ratio #",ir)
        temp = []
        alpha = stopRatios[ir]
        parameters.dia_Hamiltonian = (gFinal*alpha,0,0,0,hInitial*(1-alpha),0)
        for im in range(2):     #magnon modes
            print("Magnon ",im+1)
            parameters.cor_magnonModes = (im+1,)
            sys = openSystem(parameters)
            sys.diagonalize()
            sys.realSpaceCorrelator()
            sys.momentumSpaceCorrelator()
            if ir==0 and im==0:
                K_flat = np.sqrt(sys.momentum[:,0]**2 + sys.momentum[:,1]**2)
                freqs = fftshift(fftfreq(sys.nOmega,sys.fullTimeMeasure/sys.nTimes))
                K_m, W_m = np.meshgrid(k_centers, freqs, indexing='ij')
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
                 K_m=K_m,
                 W_m=W_m)

# Figure
fig = plt.figure(figsize=(4.65,1.5))
gs = fig.add_gridspec(
    1,5,
    wspace=0.05
)
axes_2d = [fig.add_subplot(gs[j]) for j in range(5)]

s_norm = 10
s_small = 9

min1 = []
max1 = []
for i in range(5):
    for j in range(2):
        min1.append(np.min(corr[i][j]))
        max1.append(np.max(corr[i][j]))
vmin = np.min(np.array(min1)) / 10
vmax = np.max(np.array(max1)) / 10

mesh1 = []
mesh2 = []
for i in range(5):
    ax = axes_2d[i]
    mag1 = corr[i][0]/10
    mag2 = corr[i][1]/10
    mask1 = mag1 > mag2
    mask2 = ~mask1
    mesh1.append( ax.pcolormesh(K_m,
                  W_m/10,
                  np.ma.masked_where(~mask1,mag1),
                  cmap='Reds',
                  norm=SqrtNorm(vmin=vmin,vmax=vmax),
                  rasterized=True
                  )
                 )
    mesh2.append( ax.pcolormesh(K_m,
                  W_m/10,
                  np.ma.masked_where(~mask2,mag2),
                  cmap='Blues',
                  norm=SqrtNorm(vmin=vmin,vmax=vmax),
                  rasterized=True
                  )
                 )
    ax.set_ylim(-4,4)
    ax.set_title(r"$\alpha=%.3f$"%stopRatios[i],size=s_norm)
    if i == 0:
        ax.set_ylabel(r"$\omega(g)$",size=s_norm)
    else:
        ax.set_yticklabels([])
    ax.yaxis.set_ticks_position('both')  # place ticks on both sides
    ax.xaxis.set_ticks_position('both')  # place ticks on both sides
    ax.tick_params(
        axis='both',
        direction='in',
        length=4,
        width=0.8,
        pad=3,
        labelsize=s_small
    )
    ax.set_xticks(np.arange(0,5,1),[str(i) for i in np.arange(0,5,1)])
    ax.set_xlabel(
        r'$|k|$',
        fontsize=s_norm
    )


#Blue cb
xb = 0.95
yb = 0.14
hb = 0.78
wb = 0.02
cbar_ax2 = fig.add_axes([xb, yb, wb, hb])  # [left, bottom, width, height]
cbar = fig.colorbar(mesh2[-1], cax=cbar_ax2)
cbar.set_label("$2$-magnon",size=s_norm)
cbar.ax.yaxis.set_label_position('left')
cbar.set_ticks([])
cbar_ax1 = fig.add_axes([xb+wb, yb, wb, hb])  # [left, bottom, width, height]
cbar = fig.colorbar(mesh1[-1], cax=cbar_ax1)
cbar.set_label("$1$-magnon",size=s_norm)
cbar.ax.yaxis.set_label_position('right')
cbar.set_ticks([])


plt.subplots_adjust(
    bottom = 0.138,
    top = 0.92,
#    right = 0.951,
    left = 0.05
)


if final:
    plt.savefig(
        "Figures/magnonContributions.pdf",
        bbox_inches="tight",
        dpi=600
    )
















































