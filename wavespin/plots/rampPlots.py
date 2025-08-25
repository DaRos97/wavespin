""" Plotting of KW correlator.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.fft import fftfreq, fftshift
import math
import os
from pathlib import Path

class SqrtNorm(mcolors.Normalize):
    def __call__(self, value, clip=None):
        return (super().__call__(value, clip))**(1/2)

def plotRampKW(ramp, **kwargs):
    """ Plot frequency over mod k for the different stop ratios.
    """
    sys0 = ramp.rampElements[0]
    transformType = sys0.transformType
    nP = ramp.nP
    Lx = sys0.Lx
    Ly = sys0.Ly
    nOmega = sys0.nOmega
    fullTimeMeasure = sys0.fullTimeMeasure
    nTimes = sys0.nTimes
    #Axis
    if transformType=='fft':
        kx = fftshift(fftfreq(Lx,d=1)*2*np.pi)
        ky = fftshift(fftfreq(Ly,d=1)*2*np.pi)
    elif transformType=='dst':
        kx = np.pi * np.arange(1, Lx + 1) / (Lx + 1)
        ky = np.pi * np.arange(1, Ly + 1) / (Ly + 1)
    elif transformType in ['dct','dat']:
        kx = np.pi * np.arange(0, Lx ) / (Lx )
        ky = np.pi * np.arange(0, Ly ) / (Ly )
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2)
    K_flat = K_mag.ravel()
    freqs = fftshift(fftfreq(nOmega,fullTimeMeasure/nTimes))
    # Define k bins
    num_k_bins = kwargs.get('numKbins',50)
    k_bins = np.linspace(0,np.sqrt(2)*np.pi, num_k_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    K_mesh, W_mesh = np.meshgrid(k_centers, freqs, indexing='ij')
    P_k_omega_p = np.zeros((nP,num_k_bins,nOmega))
    for iP in range(nP):
        corr_flat = ramp.rampElements[iP].correlatorKW.reshape(Lx*Ly, nOmega)
        for i in range(num_k_bins):
            mask = (K_flat >= k_bins[i]) & (K_flat < k_bins[i+1])
            if np.any(mask):
                P_k_omega_p[iP, i, :] = np.mean(np.abs(corr_flat[mask, :]), axis=0)
    # Figure
    fig, axes, rows, cols = createFigure(nP,subplotSize=(4,4))
    txtMagnon = ''
    for i in sys0.magnonModes:
        txtMagnon += str(i)
        if not i==sys0.magnonModes[-1]:
            txtMagnon += '-'
    title = 'Commutator: ' + sys0.correlatorType + ', momentum transform: ' + transformType + ', magnons mode(s): ' + txtMagnon
    plt.suptitle(title,fontsize=20)
    ylim = kwargs.get('ylim',70)
    vmax = np.max(P_k_omega_p)
    for iP in range(nP):
        P_k_omega = P_k_omega_p[iP]
        vmax = np.max(P_k_omega)
        ax = axes[iP]
        ax.set_facecolor('black')
        mesh = ax.pcolormesh(K_mesh, W_mesh, P_k_omega,
                             shading='auto',
                             cmap='inferno',
                             norm=SqrtNorm(vmin=0,vmax=vmax)
                            )
        ax.set_ylim(-ylim,ylim)
        ax.set_xlim(0,np.sqrt(2)*np.pi)
        cbar = fig.colorbar(mesh, ax=ax)
        if iP in [cols*i-1 for i in range(1,rows+1)]:
            cbar.set_label(transformType,fontsize=15)
        if iP in [cols*i for i in range(0,rows)]:
            ax.set_ylabel(r'$\omega$',fontsize=15)
        if iP in np.arange((rows-1)*cols,rows*cols):
            ax.set_xlabel(r'$|k|$',fontsize=15)
    for i in range(nP,len(axes)):
        axes[i].set_axis('off')
    #
    plt.tight_layout()
    #
    saveFigure = kwargs.get('saveFigure',False)
    if saveFigure:
        argsFn = ('fig_correlatorKW_rs',self.correlatorType,self.transformType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy)
        figureDn = pf.getHomeDirname(str(Path.cwd()),'Figure/')
        figureFn = pf.getFilename(*argsFn,dirname=figureDn,extension='.png')
        if not Path(figureDn).is_dir():
            print("Creating 'Figure/' folder in home directory.")
            os.system('mkdir '+dataDn)
        fig.savefig(figureFn)
    showFigure = kwargs.get('showFigure',True)
    if showFigure:
        plt.show()

def createFigure(n_subplots, subplotSize=(4, 4)):
    """Create a figure with n_subplots, keeping each subplot the same size.

    Parameters
    ----------
    n_subplots (int): number of subplots
    subplotSize (tuple): (width, height) of each subplot in inches
    """
    # choose rows and cols as close as possible to a square
    cols = math.ceil(math.sqrt(n_subplots)) if n_subplots!=10 else 5
    rows = math.ceil(n_subplots / cols) if n_subplots!=10 else 2

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * subplotSize[0], rows * subplotSize[1])
    )

    # If only one subplot, axes is a single Axes object
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

    return fig, axes, rows, cols


