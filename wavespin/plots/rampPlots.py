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
from wavespin.tools import pathFinder as pf

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
        subplot_kw={"projection": "3d"} if plot3D else {}
    )

    # If only one subplot, axes is a single Axes object
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

    return fig, axes, rows, cols

class SqrtNorm(mcolors.Normalize):
    def __call__(self, value, clip=None):
        return (super().__call__(value, clip))**(1/2)

def plotRampKW(ramp, **kwargs):
    """ Plot frequency over mod k for the different ramp parameters.
    """
    sys0 = ramp.rampElements[0]
    transformType = sys0.p.transformType
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
    if hasattr(sys0,'magnonModes'):
        txtMagnon = ', magnons mode(s): '
        for i in sys0.magnonModes:
            txtMagnon += str(i)
            if not i==sys0.magnonModes[-1]:
                txtMagnon += '-'
    else:
        txtMagnon = ''
    title = 'Commutator: ' + sys0.p.correlatorType + ', momentum transform: ' + transformType + txtMagnon
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
    plt.tight_layout()
    #
    if transformType=='dat':
        fig2, axes2, rows2, cols2 = createFigure(nP,subplotSize=(4,4))
        for iP in range(nP):
            kwargs = {'newFigure':False,'axis':axes2[iP],'showFigureMomentum':False,'printTitle':False,'printEnergies':False}
            plotBogoliubovMomenta(ramp.rampElements[iP],**kwargs)
    plt.tight_layout()
    #
    saveFigure = kwargs.get('saveFigure',False)
    if saveFigure:
        argsFn = ('fig_correlatorKW_rs',sys0.correlatorType,sys0.transformType,sys0.Lx,sys0.Ly,sys0.Ns)
        figureFn = pf.getFilename(*argsFn,dirname=self.figureDn,extension='.png')
        if not Path(self.figureDn).is_dir():
            print("Creating 'Figures/' folder in home directory.")
            os.system('mkdir '+self.figureDn)
        fig.savefig(figureFn)
        if transformType=='dat':
            argsFn = ('fig_correlatorKW_rs_momenta',sys0.correlatorType,sys0.transformType,sys0.Lx,sys0.Ly,sys0.Ns)
            figureFn = pf.getFilename(*argsFn,dirname=self.figureDn,extension='.png')
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

def plotBogoliubovMomenta(system,**kwargs):
    """ Here we plot the momenta obtained from the modes of the Bogoliubov tranformation.
    """
    Lx = system.Lx
    Ly = system.Ly
    Ns = system.Ns
    U_ = system.U_
    V_ = system.V_
    phi_ik = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system.indexesMap[i]
        phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    #
    printEnergies = kwargs.get('printEnergies',True)
    printIndexes = kwargs.get('printIndexes',True)
    best_modes = kwargs.get('best_modes',[])
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(121)
    ks = []
    ens = np.zeros((Lx,Ly))
    for k in range(Ns):
        kx,ky = extractMomentum(phi_ik[:,k].reshape(Lx,Ly))
        ks.append([kx,ky])
        ens[kx,ky] = system.evals[k]
        ax.scatter(kx,ky,color='orange',alpha=0.6,s=100)
        if printEnergies:
            ax.text(kx,ky+0.2,"{:.3f}".format(system.evals[k]))
        if printIndexes:
            ax.text(kx,ky-0.2,str(k))
        if k in best_modes:
            ax.scatter(kx,ky,color='r',alpha=0.8,s=100)
    ax.set_xlabel("Kx",size=20)
    ax.set_ylabel("Ky",size=20)
    ax.set_aspect('equal')
    ax.grid(True)
    printTitle = kwargs.get('printTitle',True)
    if printTitle:
        ax.set_title("Momenta obtained from Bogoliubov modes and their energies",size=20)
    #
    KX,KY = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    ax = fig.add_subplot(122,projection='3d')
    ax.plot_surface(KX,KY,ens,cmap='viridis',alpha=0.5,zorder=0)
    #
    for i in best_modes:
        ax.scatter(ks[i][0],ks[i][1],system.evals[i],color='r',marker='*',s=60,zorder=1)

    plt.show()

def plotRampDispersions(ramp, **kwargs):
    """ Plot the dispersions of each system in the ramp.
    """
    nP = ramp.nP
    fig, axes, rows, cols = createFigure(nP,plot3D=True)
    for iP in range(ramp.nP):
        gridk = ramp.rampElements[iP].gridk
        dispersion = ramp.rampElements[iP].dispersion
        ax = axes[iP]
        ax.plot_surface(gridk[:,:,0],gridk[:,:,1],dispersion,cmap='plasma')
        ax.set_aspect('equalxy')
        n_i = 6
        ax.set_xticks([ik*2*np.pi/n_i for ik in range(n_i+1)],["{:.2f}".format(ik*2*np.pi/n_i) for ik in range(n_i+1)],size=8)
        ax.set_yticks([ik*2*np.pi/n_i for ik in range(n_i+1)],["{:.2f}".format(ik*2*np.pi/n_i) for ik in range(n_i+1)],size=8)
        if iP in [cols*i for i in range(0,rows)]:
            ax.set_ylabel(r'$k_y$',fontsize=15)
        if iP in np.arange((rows-1)*cols,rows*cols):
            ax.set_xlabel(r'$k_x$',fontsize=15)
    plt.suptitle("Dispersion relation of periodic system",size=20)
    plt.show()

def plotRampValues(ramp, **kwargs):
    """ Plot the values of the systems along the ramp.
    """
    nP = ramp.nP
    thetas = np.zeros(nP)
    gsEnergies = np.zeros(nP)
    gaps = np.zeros(nP)
    for iP in range(nP):
        thetas[iP] = ramp.rampElements[iP].theta
        gsEnergies[iP] = ramp.rampElements[iP].gsEnergy
        gaps[iP] = np.min(ramp.rampElements[iP].dispersion)
    #
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot()
    xAxis = np.arange(nP)
    l1 = ax.plot(xAxis,thetas,'b*-',label=r'$\theta$')
    ax.set_yticks([i/6*np.pi/2 for i in range(7)],["{:.1f}".format(i/6*90)+'°' for i in range(7)],size=15,color='b')

    ax_r = ax.twinx()
    l2 = ax_r.plot(xAxis,gsEnergies,'r*-',label=r'$E_{GS}$')
    ax_r.tick_params(axis='y',colors='r')

    ax_r = ax.twinx()
    l3 = ax_r.plot(xAxis,gaps,'g*-',label='Gap')
    ax_r.tick_params(axis='y',colors='g')

    ax.set_xlabel("Ramp evolution",size=20)
    #Legend
    labels = [l.get_label() for l in l1+l2+l3]
    ax.legend(l1+l2+l3,labels,fontsize=20,loc=(0.4,0.1))

    plt.show()

def plotVertex(system,**kwargs):
    """ Plot decay vertex at the end of openHamiltonian.decayRates()
    """
    title = {
        '1to2':"1 to 2 decay process",
        '1to3':"1 to 3 decay process",
        '2to2a':"2 to 2: linear process",
        '2to2b':"2 to 2: quadratic process",
    }
    best_modes = kwargs.get('best_modes',None)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    g_p,_,d_p,_,h_p,disorder = system.p.dia_Hamiltonian
    T = system.p.sca_temperature
    s_ = 20
    types = system.p.sca_types
    data = system.dataScattering
    print(types)
    if '2to2a' in types and '2to2b' in types:
        types.append('2to2all')
        Amplitudes = [0.5,]
        data['2to2all'] = []
        for Amplitude in Amplitudes:
            data['2to2all'].append(data['2to2a'] + Amplitude**2/4 * data['2to2b'])
        title['2to2all'] = 'full 2 to 2'
    nS = len(types)
    nRows = 1 if nS<4 else 2
    nCols = nS if nS<4 else nS//2 + 1
    fig = plt.figure(figsize=(6*nCols,6*nRows))
    col = ['navy','orange']
    col2 = ['blue','red']
    for st, scatteringType in enumerate(types):
        Gamma_n = data[scatteringType]
        ax = fig.add_subplot(nRows,nCols,st+1)
        if not scatteringType=='2to2all':
            ax.scatter(np.arange(1,system.Ns),Gamma_n,marker='o',color=col[0],s=70)
        else:
            for iA in range(len(Amplitudes)):
                ax.scatter(np.arange(1,system.Ns),Gamma_n[iA],marker='o',color=col[iA],s=70,label="A=%.2f"%Amplitudes[iA])
        ax.set_xlabel("Mode number",size=s_)
        ax.set_title(title[scatteringType],size=s_+5)
        # best modes
        if not best_modes is None and scatteringType!='2to2all':
            ax.scatter(best_modes,Gamma_n[best_modes-1],marker='o',color=col2[0],s=70)
        if st==0:
            ax.text(0.58,0.09,"g=%s, H=%.1f\n"%(g_p/2,h_p)+r"$\Delta$=%.1f, $h_{dis}$=%.1f"%(d_p,disorder)+'\n'+r"$\gamma$=%.2f mEd"%system.p.sca_broadening,
                    size=s_-3,transform=ax.transAxes, bbox=props)
            txtT = "{:.2f}".format(T)+" MHz" if T!=0 else "inf"
            ax.text(0.03,0.88,"T="+txtT,
                    size=s_-3,transform=ax.transAxes, bbox=props)
            ax.set_ylabel("Decay rate (MHz)",size=s_)
        # Print mode number
        if not scatteringType=='2to2all':
            fac = (np.max(Gamma_n) - np.min(Gamma_n)) / 40
            for i in range(1,system.Ns):
                ax.text(i-0.6,Gamma_n[i-1]+fac,str(i))
        else:   #numbers just on the first amplitude
            fac = (np.max(Gamma_n[0]) - np.min(Gamma_n[0])) / 40
            for i in range(1,system.Ns):
                ax.text(i-0.6,Gamma_n[0][i-1]+fac,str(i))
        if scatteringType=='2to2all':
            ax.legend()

    plt.suptitle("Grid: %d x %d"%(system.Lx,system.Ly),size=s_)
    fig.tight_layout()
    plt.show()






