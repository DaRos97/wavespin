""" Example script to compute the decay rate and scattering amplitude of single modes in the OBC setting.
Use with input_8.txt
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian

import matplotlib.pyplot as plt
from pathlib import Path

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation using Holstein-Primakoff formalism")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,**{'verbose':verbose})

""" Initialize and diagonalize system """
parameters.dia_Hamiltonian = (10,0,0,0,0,0)
sca_types = (
    '1to2_1','1to2_2',
    '2to2_1','2to2_2',
    '1to3_1','1to3_2','1to3_3',
)
parameters.sca_types = sca_types
listTs = [8,]
data = []
for T in listTs:
    parameters.sca_temperature = T
    system = openHamiltonian(parameters)
    if T==listTs[0]:
        Phi = system.Phi
        evals = system.evals[1:]/10
        Ns = system.Ns
        maxN = np.zeros(Ns-1)
        for n in range(1,Ns):
            maxN[n-1] = np.max(np.absolute(Phi[:,n]))
    system.computeRate(verbose=verbose)
    data.append(
        (
            system.rates['1to2_1']+system.rates['1to3_1']+system.rates['2to2_1'],
            system.rates['1to2_2']+system.rates['1to3_2']+system.rates['2to2_2'],
            system.rates['1to3_3'],
         )
    )

if 0:   #plot all different scatterings separately
    fig = plt.figure(figsize=(18,10))
    lr = len(sca_types)
    for i in range(lr):
        ax = fig.add_subplot(2,4,1+i%4+4*(i//4))
        ax.scatter(np.arange(1,system.Ns),system.rates[sca_types[i]])
        ax.set_title(sca_types[i],size=20)
    fig.tight_layout()
    plt.show()

if 1:
    fig = plt.figure(figsize=(21,12))
    s_ = 10
    ss_ = 15
    sss_ = 20
    colors = ['orange','navy','forestgreen']
    As = [0,0.5,1]
    for ia in range(len(As)):
        ax = fig.add_subplot(2,len(As),1+ia)
        for it in range(len(listTs)):
            gamma = data[it][0].copy()
            gamma += data[it][1] * (As[ia] / 2 )**2
            gamma += data[it][2] * (As[ia] / 2 )**4
            ax.scatter(
                np.arange(1,Ns),
                gamma,
                marker='o',
                color=colors[it],
                label="T: %.1f"%listTs[it]
            )
            for n in range(Ns-1):
                ax.text(
                    n+1,
                    gamma[n],
                    str(n+1),
                    va='bottom',
                    size=s_
                )
        #ax.legend()
        ax.set_title("Amplitude: %.1f"%As[ia],size=ss_)
        ax.set_xlabel("Mode number",size=ss_)
        ax.set_ylabel(r"$\Gamma$",size=ss_)
    for ia in range(len(As)):
        ax = fig.add_subplot(2,len(As),1+len(As)+ia)
        for it in range(len(listTs)):
            gamma = data[it][0].copy()
            gamma += data[it][1] * (As[ia] / 2 )**2
            gamma += data[it][2] * (As[ia] / 2 )**4
            ax.scatter(
                evals,
                gamma,
                marker='o',
                color=colors[it],
                label="T: %.1f"%listTs[it]
            )
            Mg = np.max(gamma)
            mg = np.min(gamma)
            for n in range(Ns-1):
                ax.text(
                    evals[n],
                    gamma[n]+1/abs(evals[n]-evals[(n+2)%(Ns-1)])/50 + np.random.rand()*(Mg-mg)/15,
                    str(n+1),
                    va='bottom',
                    size=s_
                )
        #ax.legend()
        ax.set_xlabel("Energy (g)",size=ss_)
        ax.set_ylabel(r"$\Gamma$",size=ss_)
    #fig.tight_layout()
    #plt.suptitle("7x8 rectangle",size=sss_)
    plt.suptitle("58-sites diamond",size=sss_)
    plt.show()

if 0:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    sc = np.zeros((3,system.Ns-1))
    lr = len(sca_types)
    for i in range(lr):
        ty = sca_types[i]
        sc[int(ty[-1])-1] += system.rates[ty]
    As = [0,]
    ls = []
    for ia in range(len(As)):
        A = As[ia]
        data = sc[0]+A**2/2*sc[1]+A**4/4*sc[2]
        # insert mode 12 at position 8
        data[7], data[8], data[9], data[10], data[11] = data[11], data[7], data[8], data[9], data[10]
        ls.append(ax.plot(
            np.arange(1,system.Ns),
            data,
            marker='*',
            label="A:%.1f"%A,
            markersize=10
        )[0])
    ax_r = ax.twinx()
    data = system.evals[1:]
    data[7], data[8], data[9], data[10], data[11] = data[11], data[7], data[8], data[9], data[10]
    ls.append(ax_r.plot(
        np.arange(1,system.Ns),
        data,
        marker='o',
        color='r',
        label='Energies'
    )[0])
    ax_r = ax.twinx()
    from wavespin.tools.functions import solve_diffusion_eigenmodes_xy
    sites = []
    for ix in range(system.Lx):
        for iy in range(system.Ly):
            sites.append((ix,iy))
    for i in system.offSiteList:
        sites.remove(i)
    diffEvals, diffEvecs = solve_diffusion_eigenmodes_xy(sites)
    data = diffEvals[1:]
    data[7], data[8], data[9], data[10], data[11] = data[11], data[7], data[8], data[9], data[10]
    ls.append( ax_r.plot(
        np.arange(1,system.Ns),
        data,
        marker='o',
        color='g',
        label=r'$k^2$'
    )[0])
    ax.legend(ls,[lll.get_label() for lll in ls],fontsize=20)
    plt.show()

if 0:
    """ Decay vs amplitude """
    data = system.dataScattering
    evals = system.evals *2*np.pi / 1e3     #in GHz
    maxK = 20
    As = np.array([0.5,1,2,4,6,8])
    contA = np.linspace(As[0],As[-1],100)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot()

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.cividis
    norm = mcolors.Normalize(vmin=evals[1], vmax=evals[maxK+1])
    from scipy.optimize import curve_fit
    def amp(x,a1,a2):
        return a1 + a2*x**2
    for n in range(1,maxK+1):
        color = cmap(norm(evals[n]))
        modeD =( data['2to2a'][n-1] + As**2/4 * data['2to2b'][n-1] ) / 1e3    #in GHz
        ax.scatter(As, modeD,
                  color=color, marker='o')
        popt,pcov = curve_fit(amp, As, modeD)
        ax.plot(contA,amp(contA,*popt),color=color,ls='--',zorder=-1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Amplitude",size=20)
    ax.set_title(r"$\Gamma^{2\leftrightarrow2}$",size=20)
    #ax.set_ylabel("Decay rate (GHz)",size=20)
    ax.set_xlim(As[0]-(As[1]-As[0])/3,As[-1]+(As[-1]-As[-2]))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm,ax=ax)
    cbar.set_label("Mode freq. (GHz)",size=20)
    plt.show()


if 0:
    def computeEs(system,Tlist):
        """ Initialize and diagonalize system """
        g_p = system.g1
        epsilon = system.evals
        S = system.S
        NN_terms = system._NNterms(1)
        Nbonds = np.sum(NN_terms)//2      #(Lx-1)*Ly + (Ly-1)*Lx
        EGS = -3/2 + np.sum(epsilon) / Nbonds / g_p /2
        print(system.Lx,system.Ly,' energy of GS: ',EGS)
        result = np.zeros(len(Tlist))
        for iT,T in enumerate(Tlist):
            if abs(T)<1e-5:
                FactorBose_T = np.zeros(Ns-1)
            else:
                FactorBose_T = 1/(np.exp(epsilon[1:]/T)-1)      #size Ns-1
            result[iT] = EGS + np.sum(epsilon[1:]*FactorBose_T)/Nbonds/g_p
        return result,EGS
    """ Compute multiple temperatures """
    Ts = np.linspace(1,20,10)[:8]
#    Ts = [8,]
    dataT = np.zeros((len(Ts),system.Ns-1))

    system.p.sca_plotVertex = False
    for it,T in enumerate(Ts):
        """ Scattering vertices """
        system.decayRates(temperature=T,verbose=verbose)
        dataT[it] = system.dataScattering['2to2_1']

    """ Plotting """
    title = r"$\Gamma_1^{2\leftrightarrow2}$"
    maxK = 50
    s_ = 20

    g_p, _, d_p, _, h_p, disorder = parameters.dia_Hamiltonian
    fig = plt.figure(figsize=(8,8))
    if 0:
        ax = fig.add_subplot(131)
        for it,T in enumerate(Ts):
            Gamma_n = dataT[it]
            ax.scatter(np.arange(1,maxK),Gamma_n[:maxK-1],marker='o',label=r"T = %.2f MHz"%T)
            for i in best_modes:
                ax.scatter(i,Gamma_n[i-1],marker='*',color='r')
        #ax.legend(fontsize=s_-10)
        ax.set_xlabel("Mode number",size=s_)
        ax.set_ylabel("Decay rate (MHz)",size=s_)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.03,0.86,"g=%s, H=%.1f\n"%(g_p/2,h_p)+r"$\Delta$=%.1f, $h_{dis}$=%.1f"%(d_p,disorder)+'\n'+r"$\eta$=%.2f"%parameters.sca_broadening,
                size=s_-3,transform=ax.transAxes, bbox=props)

        ax = fig.add_subplot(132)
        for it,T in enumerate(Ts):
            Gamma_n = dataT[it]
            ax.scatter(system.evals[1:maxK],Gamma_n[:maxK-1],marker='o',label=r"T = %.2f MHz"%T)
            for i in best_modes:
                ax.scatter(system.evals[i],Gamma_n[i-1],marker='*',color='r')
        ax.legend(fontsize=s_-10)
        ax.set_xlabel("Energy (MHz)",size=s_)

    kInds = [1,2,3,5,9,13]
    Es,EGS = computeEs(system,Ts)
    ax = fig.add_subplot()
    for i in kInds:
        Gamma_t = dataT[:,i-1]
        ax.plot(Es-EGS,Gamma_t,marker='o',markersize=10,label="mode # %s"%i)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=s_-10)
    ax.set_xlabel("State Energy (MHz)",size=s_)


    plt.suptitle(title,size=s_+5)
    fig.tight_layout()

    plt.show()

