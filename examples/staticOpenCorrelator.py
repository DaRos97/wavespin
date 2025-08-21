import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.static.open import *
from wavespin.tools.inputUtils import importOpenParameters as importParameters
from wavespin.tools.pathFinder import *

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation using Holstein-Primakoff formalism")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})
(correlatorType, fourierType,
 excludeZeroMode,
 Lx, Ly,
 offSiteList, perturbationSite,
 includeList,
 plotSites,
 saveWf,plotWf,saveCorrelator,plotCorrelator,saveFig,
 ) = parameters.values()

""" Plot circuit """
if plotSites:
    plotSitesGrid(Lx,Ly,offSiteList,perturbationSite,indexesMap)

""" Parameters """
Ns = Lx*Ly-len(offSiteList)
S = 0.5     #spin value
nP = 11     #number of parameters computed in the "ramp" -> analogue to stop ratio
gFinal = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
hInitial = 15
site0 = 0 #if h_t_i[0,0,0]<0 else 1     #decide sublattice A and B of reference lattice site
fullTimeMeasure = 0.8     #measure time in ms
nTimes = 401        #time steps after ramp for the measurement
measureTimeList = np.linspace(0,fullTimeMeasure,nTimes)
nOmega = 2000

""" Bogoliubov transformation"""
U_, V_, evals = bogoliubovTransformation(Lx,Ly,Ns,nP,gFinal,hInitial,S,offSiteList,**{'saveWf':saveWf,'excludeZeroMode':excludeZeroMode,'verbose':verbose})

""" Plot wavefunction """
if plotWf: # Plot wavefunction
    import matplotlib.pyplot as plt
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
    indexesMap = mapSiteIndex(Lx,Ly,offSiteList)
    perturbationIndex = indexesMap.index(perturbationSite) #site_j[1] + site_j[0]*Ly
    if 1:   #Plot phi
        i_sr = 10
        A_ik = np.real(U_[i_sr] - V_[i_sr])
        B_ik = np.real(U_[i_sr] + V_[i_sr])
        Bj_k = B_ik[perturbationIndex]
        phi_ik = A_ik #/ Bj_k[None,:]
        for i in range(Ns):
            ix,iy = indexesMap[i]
            phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
        fig = plt.figure(figsize=(20,15))
        for ik in range(10):
            k = ik#Ns-ik-1
            kx = k//Ly
            ky = k%Ly
            ax = fig.add_subplot(3,4,ik+1,projection='3d')
            ax.plot_surface(X,Y,
#                            np.sin(np.pi*(kx+1)*(X+1)/(Lx+1))*np.sin(np.pi*(ky+1)*(Y+1)/(Ly+1)),
                            extendFunction(
                                phi_ik[:,k],
                                #B_ik[:,k],
                                Lx,Ly,offSiteList,indexesMap).T,
                            cmap='plasma'
                            )
        plt.show()

        exit()
    if 0:   #Plot dctn of phi
        from scipy.fft import dctn
        i_sr = 9
        A_ik = np.real(U_[i_sr] - V_[i_sr])
        B_ik = np.real(U_[i_sr] + V_[i_sr])
        Bj_k = B_ik[perturbationIndex]
        phi_ik = A_ik #/ Bj_k[None,:]
        for i in range(Ns):
            ix,iy = indexesMap[i]
            phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
        fig = plt.figure(figsize=(20,15))
        for ik in range(15):
            k = ik
            ax = fig.add_subplot(3,5,k+1,projection='3d')
            ps = dctn(phi_ik[:,ik].reshape(Lx,Ly),type=2)
            ax.plot_surface(X,Y,
                            np.absolute(ps).T,
                            cmap='plasma'
                            )
        plt.show()

        exit()
    if 1:       #Plot momenta
        fig = plt.figure(figsize=(20,15))
        for i_sr in range(9,10):
            A_ik = np.real(U_[i_sr] - V_[i_sr])
            B_ik = np.real(U_[i_sr] + V_[i_sr])
            Bj_k = B_ik[perturbationIndex]
            phi_ik = A_ik #/ Bj_k[None,:]
            for i in range(Ns):
                ix,iy = indexesMap[i]
                phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
            ax = fig.add_subplot()#2,5,i_sr+1)
            ks = []
            for k in range(Ns):
#                f_in = fs.extendFunction(phi_ik[:,k],Lx,Ly,offSiteList,indexesMap)
                #k_abs = fs.get_momentum_Bogoliubov_Laplacian(phi_ik[:,k],Lx,Ly)
                kx,ky = fs.get_momentum_Bogoliubov3(phi_ik[:,k].reshape(Lx,Ly))
                ax.scatter(kx,ky,color='r',alpha=0.3,s=100)
                ax.text(kx,ky+np.random.rand()/2,"{:.3f}".format(evals[i_sr,k]))
                #ax.text(kx,ky+0.1*np.random.rand(),str(k))
            #ax.set_aspect('equal')
            #ax.set_ylim(Ly-1,0)
        fig.tight_layout()
        plt.show()
        exit()

""" Computation of correlator"""
correlator = computeCorrelator(Lx,Ly,Ns,nP,measureTimeList,gFinal,hInitial,S,U_,V_,evals,offSiteList,site0,perturbationSite,includeList,**{'saveCorrelator':saveCorrelator,'correlatorType':correlatorType,'excludeZeroMode':excludeZeroMode,'verbose':verbose})

""" Plot real space correlator """
if 0:
    i_sr = 10
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(Lx*Ly):
        ix, iy = i//Ly, i%Ly
        ax.plot(measureTimeList,np.imag(correlator[i_sr,ix,iy]))

    plt.show()
    exit()

""" Fourier transform """
args = (nOmega,U_,V_,perturbationSite,offSiteList)
correlator_kw = fourierTransform[fourierType](correlator,*args)

""" Figure """
title = 'Commutator '+correlatorType+', fourier: '+fourierType

correlatorPlot(
    correlator_kw[1:],
    n_bins=100,
    fourier_type=fourierType,
    title=title,
    figname='',
    showfig=True,
)






