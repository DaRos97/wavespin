""" Script to follow the evolution of eigenvales along the ramp to reconstruct the indices of the eigenvectors.
Use with inputRec.txt
"""
import numpy as np
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt

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
Lx = parameters.lat_Lx
Ly = parameters.lat_Ly
Ns = Lx*Ly

if 0:   #Do DAT on modes at stop ratios to track changes -> does not work
    def extractMomentumDCTN(f_in,f_ref=None,k_n=None):
        """ We get the momentum associated with a given Bogoliubov mode f_in wrt functions f_ref.
        We do this by computing the peak of the transform of the input function (mode) to extract the momentum.
        f_in has shape Ns.  -> i
        f_ref has shape (Ns,Ns) -> i and k
        """
        Lx,Ly = f_in.shape
        f_in = f_in.T
        X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
        #Renormalize the awesome functions by the cosine
        abs_f = np.absolute(f_in)
        res = np.zeros((Lx,Ly))
        for kx in range(Lx):
            for ky in range(Ly):
                cosx = np.cos(np.pi*kx*(2*X+1)/2/Lx)
                cosy = np.cos(np.pi*ky*(2*Y+1)/2/Ly)
                abs_cosx = np.absolute(cosx)
                abs_cosx[abs_cosx==0] = 1
                abs_cosy = np.absolute(cosy)
                abs_cosy[abs_cosy==0] = 1
                res[kx,ky] = np.sum(f_in*abs_f*cosx/abs_cosx*cosy/abs_cosy)
        ind = np.argmax(np.absolute(res))
        kx,ky = ind//Ly, ind%Ly
        return kx,ky
    def extractMomentumRamp(f_in,f_ref,n_ref):
        """ We get the momentum associated with a given Bogoliubov mode f_in wrt functions f_ref.
        We do this by computing the peak of the transform of the input function (mode) to extract the momentum.
        f_in has shape Ns.  -> i
        f_ref has shape (Ns,Ns) -> i and k
        """
        Ns = f_in.shape[0]
        newN = np.zeros(Ns)
        for n in range(Ns):
            DAT = abs( np.sum(f_in[:,n,None] * f_ref[:,:], axis=0) )
            newN[n] = n_ref[ np.argmax(DAT) ]
        return newN

    refK = np.zeros((Ns,2))
    refN = np.zeros((Nr,Ns),dtype=int)
    refPhi = np.zeros((Nr,Ns,Ns))
    for ir in range(Nr-1,-1,-1):
        # Compute evals and evecs
        stopRatio = stopRatios[ir]
        print('stop ratio = %.3f'%stopRatio)
        g_p = (1-stopRatio)*gInitial + stopRatio*gFinal
        h_p = (1-stopRatio)*hInitial + stopRatio*hFinal
        parameters.dia_Hamiltonian = (g_p,0,0,0,h_p,0)
        system = openHamiltonian(parameters)
        system.diagonalize(verbose=verbose)

        # Compute Amazing function
        refPhi[ir] = np.real(system.U_ - system.V_)
        for i in range(Ns):
            ix,iy = system._xy(i)
            refPhi[ir,i,:] *= 2/np.pi*(-1)**(ix+iy+1)
        for n in range(1,Ns):
            refPhi[ir,:,n] /= np.max(np.absolute(refPhi[ir,:,n]))
        if ir==Nr-1:
            # Use DCTN for first stop ratio
            for n in range(Ns):
                refK[n] = extractMomentumDCTN(refPhi[ir,:,n].reshape(Lx,Ly))
                refN[ir,n] = n
        else:
            # Use reference phi to get new K[n]
            refN[ir] = extractMomentumRamp(refPhi[ir],refPhi[ir+1],refN[ir+1])
            print(refN[ir]-refN[ir+1])
            print(refN[ir])
            input()
        dataEvals[ir] = system.evals


if 0:
    from scipy.signal import find_peaks
    from pathlib import Path
    from tqdm import tqdm
    """ First, extract all the swappings """
    # Compute bunch of stop ratios
    finalStopRatio = 0.3#3/11     #final stop ratio we want
    stopRatios = np.linspace(0.4,finalStopRatio,10000)
    Nr = len(stopRatios)
    dataFn = "Data/bog_%.2f_%.2f_%d.npy"%(stopRatios[0],stopRatios[-1],Nr)
    if Path(dataFn).is_file():
        dataEvals = np.load(dataFn)
    else:
        dataEvals = np.zeros((Nr,Ns-1))
        for ir in tqdm(range(Nr),desc="Computing evals"):
            # Compute evals and evecs
            stopRatio = stopRatios[ir]
            g_p = (1-stopRatio)*gInitial + stopRatio*gFinal
            h_p = (1-stopRatio)*hInitial + stopRatio*hFinal
            parameters.dia_Hamiltonian = (g_p,0,0,0,h_p,0)
            system = openHamiltonian(parameters)
            system.diagonalize(verbose=verbose)
            dataEvals[ir] = system.evals[1:]
        np.save(dataFn,dataEvals)
    # Starting from top band, compute all crossings by:
    #   - find_peak
    # Save each swap with n
    swaps = []
    for n in range(Ns-3,0,-1):
        peak_indices, _ = find_peaks(dataEvals[:,n]/dataEvals[:,n+1], height = 1-1e-4)
        if 0:
            print("band: %d,\t indices: "%n,peak_indices)
            fig = plt.figure(figsize=(10,15))
            ax = fig.add_subplot()
            ax.scatter(stopRatios,dataEvals[:,n]/dataEvals[:,n+1])
            for ind in peak_indices:
                ax.axvline(stopRatios[ind])
            # Save to file for checking                                     ########################################    
            plt.show()
        for ind in peak_indices:
            swaps.append(np.array([ind,n],dtype=int))
    # Sort by swap ratio the ful list
    swaps = np.array(swaps,dtype=int)
    sorted_swaps = swaps[swaps[:,0].argsort()]
    diffs = abs(sorted_swaps[1:,0] - sorted_swaps[:-1,0])
    if min(diffs) == 0:
        print("Multiple swaps at same stop ratio")
        print(sorted_swaps[1:][diffs==0])
        print(sorted_swaps[:-1][diffs==0])
    # Starting from 1 -> Ns-1, do the swaps of band n with band n-1
    Norder = np.zeros((Nr,Ns-1), dtype=int)
    Norder[0] = np.arange(Ns-1)
    inS = 0     #initial index of swapping
    checkSwap = True
    print(sorted_swaps)
    for ir in range(1,Nr):
        Norder[ir] = Norder[ir-1]
        if checkSwap:
            if ir == sorted_swaps[inS,0]:
                n = sorted_swaps[inS,1]
                Norder[ir,n], Norder[ir,n+1] = Norder[ir,n+1], Norder[ir,n]
                inS += 1
                if inS == sorted_swaps.shape[0]:
                    checkSwap = False
                    continue
                if ir == sorted_swaps[inS,0]:       #Multiple swaps at same ir
                    n = sorted_swaps[inS,1]
                    Norder[ir,n], Norder[ir,n+1] = Norder[ir,n+1], Norder[ir,n]
                    inS += 1
                print("swap: ",sorted_swaps[inS-1])
                print(Norder[ir])
                input()

    # Plot
    print(Norder[-1])
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot()
    Nmin = 1
    Nmax = Ns-1
    colors = plt.get_cmap('viridis')(np.linspace(0,1,Ns-1))
    En = np.zeros_like(dataEvals)
    for ir in range(Nr):
        En[ir] = dataEvals[ir,Norder[ir]]
    for n in range(Ns-1):
        ax.plot(np.arange(Nr),En[:,n])
    plt.show()
    exit()
    for ir in range(0,Nr,20):
        vals = Norder[ir][Nmin:Nmax]
        col = vals-min(vals)
        ax.scatter(
            np.ones(Nmax-Nmin)*stopRatios[ir],
            dataEvals[ir,Nmin:Nmax],#/dataEvals[ir,Nmax-1],
#            c=colors[vals],
        )
    ax.invert_xaxis()
    plt.show()




































