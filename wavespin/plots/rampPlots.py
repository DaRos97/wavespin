""" Plotting of KW correlator.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.fft import fftfreq, fftshift
import math
import os
from pathlib import Path
from wavespin.static.momentumTransformation import extractMomentum

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

def plotWf(system,nModes=16):
    """ Here we plot just the wavefunctions (first n modes)
    """
    Lx = system.Lx
    Ly = system.Ly
    Ns = system.Ns
    U_ = system.U_
    V_ = system.V_
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
    phi_ik = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system.indexesMap[i]
        phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    fig, axes, rows, cols = createFigure(nModes,plot3D=True)
    for ik in range(nModes):
        kx, ky = system._xy(ik)
        ax = axes[ik]
        ax.plot_surface(X,Y,
                        phi_ik[:,ik].reshape(Lx,Ly).T,
                        cmap='plasma'
                        )
        ax.set_title("Mode: "+str(ik))
    for ik in range(nModes,len(axes)):
        axes[ik].axis('off')
    plt.suptitle("Modes from bogoliubov transformation",size=20)
    plt.show()

def plotWfCos(system,nModes=6):
    """ Here we plot the wavefunctions next to cosine functions (first n modes)
    """
    Lx = system.Lx
    Ly = system.Ly
    Ns = system.Ns
    U_ = system.U_
    V_ = system.V_
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
    phi_ik = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system.indexesMap[i]
        phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    fig, axes, rows, cols = createFigure(nModes,plot3D=True,nRows=2,nCols=nModes)
    for ik in range(nModes):
        kx, ky = system._xy(ik)
        ax = axes[ik]
        ax.plot_surface(X,Y,
                        phi_ik[:,ik].reshape(Lx,Ly).T,
                        cmap='plasma'
                        )
        ax.set_title("Mode: "+str(ik))
        ax = axes[ik+nModes]
        kx, ky = extractMomentum(phi_ik[:,ik].reshape(Lx,Ly))
        ax.plot_surface(X,Y,
                        np.cos(np.pi*kx*(2*X+1)/(2*Lx))*np.cos(np.pi*ky*(2*Y+1)/(2*Ly)),
                        cmap='plasma'
                        )
        ax.set_title("Momentum: (%d,%d)"%(kx,ky))
    plt.suptitle("Comparison of modes and cosine functions",size=20)
    plt.show()

def plotBogoliubovMomenta(system):
    """ Here we plot the momenta obtained from the modes of the Bogoliubov tranformation.
    """
    Lx = system.Lx
    Ly = system.Ly
    Ns = system.Ns
    U_ = system.U_
    V_ = system.V_
    fig = plt.figure(figsize=(20,15))
    phi_ik = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system.indexesMap[i]
        phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    ax = fig.add_subplot()
    ks = []
    for k in range(Ns):
        kx,ky = extractMomentum(phi_ik[:,k].reshape(Lx,Ly))
        ax.scatter(kx,ky,color='r',alpha=0.3,s=100)
        ax.text(kx,ky+0.2,"{:.3f}".format(system.evals[k]))
    ax.set_xlabel("Kx",size=20)
    ax.set_ylabel("Ky",size=20)
    ax.set_aspect('equal')
    ax.grid(True)
    plt.suptitle("Momenta obtained from Bogoliubov modes and their energies",size=20)
    plt.show()

def createFigure(n_subplots, subplotSize=(4, 4), plot3D=False, nRows=-1, nCols=-1):
    """Create a figure with n_subplots, keeping each subplot the same size.

    Parameters
    ----------
    n_subplots (int): number of subplots
    subplotSize (tuple): (width, height) of each subplot in inches
    """
    if nRows==-1 and nCols==-1: # choose rows and cols as close as possible to a square
        cols = math.ceil(math.sqrt(n_subplots)) if n_subplots!=10 else 5
        rows = math.ceil(n_subplots / cols) if n_subplots!=10 else 2
    else:
        cols = nCols
        rows = nRows

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * subplotSize[0], rows * subplotSize[1]),
        subplot_kw={"projection": "3d" if plot3D else "2d"}
    )

    # If only one subplot, axes is a single Axes object
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

    return fig, axes, rows, cols


