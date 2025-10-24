""" Plotting of KW correlator.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.fft import fftfreq, fftshift
import math
import os
from pathlib import Path
from wavespin.static.momentumTransformation import extractMomentum, extractMomentum2
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
    transformType = sys0.p.cor_transformType
    if transformType == 'dat2':
        plotRampDAT(ramp,**kwargs)
        return
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
    #fig, axes, rows, cols = createFigure(nP,subplotSize=(4,4),nRows=1,nCols=nP)
    fig, axes = plt.subplots(1,5,figsize=(14,4))
    rows = 1
    cols = 5
    if hasattr(sys0,'magnonModes'):
        txtMagnon = ', magnons mode(s): '
        for i in sys0.magnonModes:
            txtMagnon += str(i)
            if not i==sys0.magnonModes[-1]:
                txtMagnon += '-'
    else:
        txtMagnon = ''
    title = 'Commutator: ' + sys0.p.cor_correlatorType + ', momentum transform: ' + transformType + txtMagnon
    #plt.suptitle(title,fontsize=20)
    ylim = kwargs.get('ylim',7)
    P_k_omega_p /= 10
    W_mesh /= 10
    vmax = np.max(P_k_omega_p)
    for iP in range(nP):
        P_k_omega = P_k_omega_p[iP]
        #vmax = np.max(P_k_omega)
        ax = axes[iP]
        ax.set_facecolor('black')
        mesh = ax.pcolormesh(K_mesh, W_mesh, P_k_omega,
                             shading='auto',
                             cmap='Blues',
                             norm=SqrtNorm(vmin=0,vmax=vmax)
                            )
        if iP==0:
            ax.set_ylabel(r"$\omega(g)$",size=20)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-ylim,ylim)
        ax.set_xlim(0,np.sqrt(2)*np.pi)
        ax.yaxis.set_ticks_position('both')  # place ticks on both sides
        ax.xaxis.set_ticks_position('both')  # place ticks on both sides
        ax.tick_params(axis='both',direction='in',length=7,width=1,pad=2,labelsize=15)
        if iP in np.arange((rows-1)*cols,rows*cols):
            ax.set_xlabel(r'$|k|$',fontsize=15)
        if 0:
            if iP in [cols*i-1 for i in range(1,rows+1)]:
                cbar.set_label(transformType,fontsize=15)
            if iP in [cols*i for i in range(0,rows)]:
                ax.set_ylabel(r'$\omega$',fontsize=15)
        if iP in np.arange((rows-1)*cols,rows*cols):
            ax.set_xlabel(r'$|k|$',fontsize=15)
        stopRatio = ramp.rampElements[iP].g1 / 10
        ax.set_title(r"$\alpha=$%.2f"%stopRatio,size=20)
    for i in range(nP,len(axes)):       #set to blank extra plots
        axes[i].axis('off')

    if 1:   # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.75])  # [left, bottom, width, height]
        cbar = fig.colorbar(mesh, cax=cbar_ax)
        cbar.set_label(r"$\vert\chi_{ZZ}(\omega)\vert$",fontsize=20)
    fig.subplots_adjust(left=0.052, right=0.9, top=0.9, bottom=0.14, wspace=0.15)
    #plt.tight_layout()
    #
    if transformType=='dat' and 0:
        fig2, axes2, rows2, cols2 = createFigure(nP,subplotSize=(4,4))
        for iP in range(nP):
            kwargs = {'newFigure':False,'axis':axes2[iP],'showFigureMomentum':False,'printTitle':False,'printEnergies':False}
            plotBogoliubovMomenta(ramp.rampElements[iP],**kwargs)
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

def plotWf3D(system,nModes=16):
    """ Here we plot just the wavefunctions (first n modes)
    """
    Lx = system.Lx
    Ly = system.Ly
    Ns = system.Ns
    U_ = system.U_
    V_ = system.V_
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    phi = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system._xy(i)
        phi[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    fig, axes, rows, cols = createFigure(nModes,plot3D=True)
    for n in range(nModes):
        ax = axes[n]
        if len(system.p.lat_offSiteList)==0:
            formattedPhi = phi[:,n].reshape(Lx,Ly)
        else:
            formattedPhi = np.zeros((Lx,Ly))
            for ix in range(Lx):
                for iy in range(Ly):
                    if (ix,iy) in system.offSiteList:
                        formattedPhi[ix,iy] = np.nan
                    else:
                        formattedPhi[ix,iy] = phi[system._idx(ix,iy),n]
        ax.plot_surface(X,Y,
                        formattedPhi,
                        cmap='plasma'
                        )
        ax.set_title("Mode: "+str(n))
    for ik in range(nModes,len(axes)):
        axes[ik].axis('off')
    plt.suptitle("Modes from bogoliubov transformation",size=20)
    plt.show()

def plotWf2D(system,nModes=25):
    """ Here we plot just the wavefunctions (first n modes)
    """
    Lx = system.Lx
    Ly = system.Ly
    Ns = system.Ns
    U_ = system.U_
    V_ = system.V_
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    phi = np.real(U_ - V_)
    for i in range(Ns):
        ix,iy = system._xy(i)
        phi[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    fig, axes, rows, cols = createFigure(nModes)
    for n in range(nModes):
        ax = axes[n]
        if len(system.p.lat_offSiteList)==0:
            formattedPhi = phi[:,n].reshape(Lx,Ly)
        else:
            formattedPhi = np.zeros((Lx,Ly))
            for ix in range(Lx):
                for iy in range(Ly):
                    if (ix,iy) in system.offSiteList:
                        formattedPhi[ix,iy] = np.nan
                    else:
                        formattedPhi[ix,iy] = phi[system._idx(ix,iy),n]
        ax.pcolormesh(X,Y,
                      formattedPhi,
                      cmap='bwr'
                      )
        ax.set_title("Mode: "+str(n))
        ax.set_aspect('equal')
    for ik in range(nModes,len(axes)):
        axes[ik].axis('off')
    plt.suptitle("Modes from bogoliubov transformation",size=20)
    fig.tight_layout()
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
    fig, axes, rows, cols = createFigure(nP,plot3D=True,nCols=3,nRows=2)
    for iP in range(ramp.nP):
        gridk = ramp.rampElements[iP].gridk
        dispersion = ramp.rampElements[iP].dispersion
        ax = axes[iP]
        ax.plot_surface(gridk[:,:,0],gridk[:,:,1],dispersion,cmap='plasma')
        ax.set_aspect('equalxy')
        n_i = 2
        ax.set_xticks([ik*2*np.pi/n_i for ik in range(n_i+1)],["{:.2f}".format(ik*2*np.pi/n_i) for ik in range(n_i+1)],size=8)
        ax.set_yticks([ik*2*np.pi/n_i for ik in range(n_i+1)],["{:.2f}".format(ik*2*np.pi/n_i) for ik in range(n_i+1)],size=8)
        #if iP in [cols*i for i in range(0,rows)]:
        #ax.set_ylabel(r'$k_y$',fontsize=15)
        #if iP in np.arange((rows-1)*cols,rows*cols):
        #ax.set_xlabel(r'$k_x$',fontsize=15)

        stopRatio = ramp.rampElements[iP].g1 / 10
        ax.set_title(r"$\alpha=$%.2f"%stopRatio,size=20)
    plt.suptitle("Dispersion relation of periodic system",size=20)
    plt.tight_layout()
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
        gsEnergies[iP] = ramp.rampElements[iP].gsEnergy / ramp.rampElements[-1].g1 / 2
        gaps[iP] = np.min(ramp.rampElements[iP].dispersion) / ramp.rampElements[-1].g1 / 2
    #
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot()
    xvals = np.linspace(0.1,1,10)
    xAxis = np.linspace(xvals[0],xvals[-1],nP)
    l1 = ax.plot(xAxis,thetas,'b*-',label=r'$\theta$')
    ax.set_yticks([i/6*np.pi/2 for i in range(7)],["{:.1f}".format(i/6*90)+'°' for i in range(7)],size=15,color='b')

    ax_r = ax.twinx()
    l2 = ax_r.plot(xAxis,gsEnergies,'r*-',label=r'$E_{GS}$')
    ax_r.tick_params(axis='y',colors='r',pad=20,width=2,length=6)

    ax_r = ax.twinx()
    l3 = ax_r.plot(xAxis,gaps,'g*-',label='Gap')
    ax_r.tick_params(axis='y',colors='g')

    ax.set_xticks([i for i in xvals],["{:.1f}".format(i) for i in xvals],size=15)
    ax.set_xlabel("Stop ratio",size=20)
    #Legend
    labels = [l.get_label() for l in l1+l2+l3]
    ax.legend(l1+l2+l3,labels,fontsize=20,loc=(0.5,0.1))

    plt.show()

def plotVertex(system,**kwargs):
    """ Plot decay vertex at the end of openHamiltonian.decayRates()
    """
    title = {
        '1to2_1':r"$\Gamma^{1\leftrightarrow2}_1$",
        '1to2_2':r"$\Gamma^{1\leftrightarrow2}_2$",
        '1to3_1':r"$\Gamma^{1\leftrightarrow3}_1$",
        '1to3_2':r"$\Gamma^{1\leftrightarrow3}_2$",
        '1to3_3':r"$\Gamma^{1\leftrightarrow3}_3$",
        '2to2_1':r"$\Gamma^{2\leftrightarrow2}_1$",
        '2to2_2':r"$\Gamma^{2\leftrightarrow2}_2$",
    }
    best_modes = kwargs.get('best_modes',None)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    g_p,_,d_p,_,h_p,disorder = system.p.dia_Hamiltonian
    T = system.p.sca_temperature
    s_ = 20
    types = system.p.sca_types
    data = system.dataScattering
    nS = len(types)
    fig, axes, nRows, nCols = createFigure(nS,nRows=2,nCols=4)            #plt.figure(figsize=(6*nCols,6*nRows))
    col_1 = ['navy','orange']
    col_2 = ['blue','red']
    for st, scatteringType in enumerate(types):
        Gamma_n = data[scatteringType]        #in MHz
        ax = axes[st]
        ax.scatter(np.arange(1,system.Ns),Gamma_n/1e3,marker='o',color=col_1[0],s=70)
        ax.set_xlabel("Mode number",size=s_)
        ax.set_title(title[scatteringType],size=s_+5)
        # best modes
        if not best_modes is None:
            ax.scatter(best_modes,Gamma_n[best_modes-1]/1e3,marker='o',color=col_2[0],s=70)
        if st==0:
            tx,ty = (0.58,0.09) if scatteringType=='2to2a' else (0.03,0.6)
            ax.text(tx,ty,"g=%s, H=%.1f\n"%(g_p,h_p)+r"$\Delta$=%.1f, $h_{dis}$=%.1f"%(d_p,disorder)+'\n'+r"$\gamma$=%.2f mEd"%system.p.sca_broadening,
                    size=s_-3,transform=ax.transAxes, bbox=props)
            txtT = "{:.2f}".format(T)+" MHz" if T!=0 else "inf"
            ax.text(0.03,0.88,"T="+txtT,
                    size=s_-3,transform=ax.transAxes, bbox=props)
            ax.set_ylabel("Decay rate (GHz)",size=s_)
        # Print mode number
        fac = (np.max(Gamma_n) - np.min(Gamma_n)) / 40 / 1e3
        for i in range(1,system.Ns):
            ax.text(i-0.6,Gamma_n[i-1]/1e3+fac,str(i))

    #plt.suptitle("Grid: %d x %d"%(system.Lx,system.Ly),size=s_)
    fig.tight_layout()
    plt.show()

def plotRampDAT(ramp, **kwargs):
    """ Plot frequency over mod k for the different ramp parameters for the DAT transform.
    """
    sys0 = ramp.rampElements[0]
    transformType = sys0.p.cor_transformType
    nP = ramp.nP
    Lx = sys0.Lx
    Ly = sys0.Ly
    nOmega = sys0.nOmega
    fullTimeMeasure = sys0.fullTimeMeasure
    nTimes = sys0.nTimes
    #
    freqs = fftshift(fftfreq(nOmega,fullTimeMeasure/nTimes))
    # Define k bins
    num_k_bins = kwargs.get('numKbins',50)
    k_bins = np.linspace(0,np.sqrt(2)*np.pi, num_k_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    K_mesh, W_mesh = np.meshgrid(k_centers, freqs, indexing='ij')
    P_k_omega_p = np.zeros((nP,num_k_bins,nOmega))
    for iP in range(nP):
        vecK = extractMomentum2(ramp.rampElements[iP])
        K_flat = vecK[:,0] + vecK[:,1]
        corr_flat = ramp.rampElements[iP].correlatorKW      #shape: Ns, nOmega
        for i in range(num_k_bins):
            mask = (K_flat >= k_bins[i]) & (K_flat < k_bins[i+1])
            if np.any(mask):
                P_k_omega_p[iP, i, :] = np.mean(np.abs(corr_flat[mask, :]), axis=0)
    # Figure
    fig, axes, rows, cols = createFigure(nP,subplotSize=(4,4))#,nRows=1,nCols=nP)
    if hasattr(sys0,'magnonModes'):
        txtMagnon = ', magnons mode(s): '
        for i in sys0.magnonModes:
            txtMagnon += str(i)
            if not i==sys0.magnonModes[-1]:
                txtMagnon += '-'
    else:
        txtMagnon = ''
    title = 'Commutator: ' + sys0.p.cor_correlatorType + ', momentum transform: ' + transformType + txtMagnon
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
        stopRatio = ramp.rampElements[iP].g1 / 10
        ax.set_title(r"$\alpha=$%.2f"%stopRatio,size=20)
    for i in range(nP,len(axes)):       #set to blank extra plots
        axes[i].axis('off')
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




