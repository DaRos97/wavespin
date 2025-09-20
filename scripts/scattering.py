""" Example script to compute the decay rate and scattering amplitude of single modes in the OBC setting.
Use with inputScattering.txt
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importOpenParameters as importParameters
from wavespin.static.open import openHamiltonian
from wavespin.lattice.lattice import latticeClass
from wavespin.plots import fancyLattice
from wavespin.plots import rampPlots

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation using Holstein-Primakoff formalism")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

""" Initialie system """
""" Hamiltonian parameters """
g_p = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
d_p = 0.
h_p = 2
parametersHamiltonian = (g_p,0,d_p,0,h_p)
system = openHamiltonian(parameters,parametersHamiltonian)

""" Compute Bogoliubov wavefunctions """
system.diagonalize(verbose=verbose)

rampPlots.plotBogoliubovMomenta(system)

import matplotlib.pyplot as plt
print("Computing vertex ")

""" Compute parameters """
if parameters.scatteringType == '1to2':
    p_xz = system.J_i[0] * np.sqrt(system.S/8) * (1-system.D_i[0]) * np.sin(2 * system.thetas)

    """ Compute Vertex """
    N = parameters.Lx*parameters.Ly
    U = np.real(system.U_)
    V = np.real(system.V_)
    Vn_lm = 2 * np.einsum('ij,il,im,jn->nlm',p_xz,U,V,U+V,optimize=True)
    Vn_lm += 2 * np.einsum('ij,il,in,jm->nlm',p_xz,U,U,U+V,optimize=True)
    Vn_lm += 2 * np.einsum('ij,in,im,jl->nlm',p_xz,V,V,U+V,optimize=True)

    evals = system.evals
    edif = evals[1:]-evals[:-1]
    eta = 3*np.mean(edif)

    arg = evals[:,None,None] - evals[None,:,None] - evals[None,None,:]
    delta_vals = (1.0/np.pi) * (eta / (arg**2 + eta**2))

    Gamma3_n = np.pi * np.sum(Vn_lm**2*delta_vals,axis=(1,2))

    if 1:
        best_modes = [23,24,27,28,30,31,33,34,35,36]
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot()
        ax.scatter(np.arange(1,N),Gamma3_n[1:],marker='o',color='orange',s=70)
        best_modes = [23,24,27,28,30,31,33,34,35,36]
        ax.scatter(best_modes,Gamma3_n[best_modes],marker='o',color='red',s=70)
        for i in range(1,N):
            ax.text(i,Gamma3_n[i]+0.005,str(i))
        s_ = 20
        ax.set_xlabel("Mode number",size=s_)
        ax.set_ylabel("Decay rate",size=s_)
        ax.set_title("1 to 2 decay process",size=s_+5)
        ax.text(0.03,0.8,r"g=%s, H=%.1f, $\Delta$=%.1f"%(g_p/2,h_p,d_p),size=s_,transform=ax.transAxes)
        plt.show()

if 1:       # Plot where the branched modes are in the BZ with the energy contours
    """ Compute approximate momenta from fft with cosines """
    from wavespin.static.momentumTransformation import extractMomentum
    Lx = system.Lx
    Ly = system.Ly
    Ns = system.Ns
    phi_ik = np.real(U - V)
    for i in range(Ns):
        ix,iy = system.indexesMap[i]
        phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
    ks = []
    ens = np.zeros((parameters.Lx,parameters.Ly))
    for k in range(Ns):
        kx,ky = extractMomentum(phi_ik[:,k].reshape(Lx,Ly))
        ks.append([kx,ky])
        ens[kx,ky] = evals[k]
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(projection='3d')
    KX,KY = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    ens[ens==0] = np.nan
    ens[0,0] = 0
    ax.plot_surface(KX,KY,ens,cmap='viridis',alpha=0.5,zorder=0)
    #
    for i in best_modes:
        ax.scatter(ks[i][0],ks[i][1],evals[i],color='r',marker='*',s=60,zorder=1)
    plt.show()


if parameters.scatteringType == '1to3':
    pass















