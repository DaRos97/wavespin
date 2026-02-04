""" Script to check dct of awesome functions.
Use with inputRec.txt
"""
import numpy as np
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

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

stopRatio = 3/11

g_p = (1-stopRatio)*gInitial + stopRatio*gFinal
h_p = (1-stopRatio)*hInitial + stopRatio*hFinal
parameters.dia_Hamiltonian = (g_p,0,0,0,h_p,0)
system = openHamiltonian(parameters)
system.diagonalize(verbose=verbose)
U_,V_ = system.U_, system.V_
phi = np.real(U_ - V_)
for i in range(Ns):
    ix,iy = system._xy(i)
    phi[i,:] *= 2/np.pi*(-1)**(ix+iy+1)

X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
from scipy.fft import dctn
vecK = np.zeros((Ns,3))
for n in range(1,Ns):
    #Renormalize the awesome functions by the cosine
    if 0:   # Normalized manual dct
        f_in = phi[:,n].reshape(Lx,Ly)
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
    else:
        f_in = phi[:,n].reshape(Lx,Ly)
        res = np.absolute(dctn(f_in))
        res[0,0] = 0
        res /= np.sum(res)
        #
        input(X)
        kx = np.sum(res * X) / Lx
        ky = np.sum(res * Y) / Ly
        vecK[n,0] = kx * np.pi
        vecK[n,1] = ky * np.pi
        vecK[n,2] = (kx+ky) * np.pi

    if 0:   # Plot sigle DCT
        fig = plt.figure(figsize=(22,7))
        ax = fig.add_subplot(131,projection='3d')
        ax.plot_surface(X,Y,phi[:,n].reshape(Lx,Ly),cmap='plasma')
        #ax.plot_surface(X,Y,normPhi,cmap='plasma')
        ax = fig.add_subplot(132,projection='3d')
        ax.plot_surface(X,Y,res,cmap='plasma')
        print(kxlist[-1],kylist[-1])
        ax = fig.add_subplot(133,projection='3d')
        if 0:
            maxk = np.argmax(np.absolute(res))
            maxX, maxY = maxk//Ly, maxk%Ly
            cos = np.cos(np.pi*maxX*(2*X+1)/2/Lx) * np.cos(np.pi*maxY*(2*Y+1)/2/Ly)
            ax.plot_surface(X,Y,cos,cmap='plasma')
        fig.tight_layout()
        plt.show()

#print(vecK)
print(np.sort(np.sqrt(vecK[:,0]**2+vecK[:,1]**2)))
print(np.sort(vecK[:,2]))
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(vecK[1:,0],vecK[1:,1],system.evals[1:],color='r',zorder=3)
fig.tight_layout()
plt.show()







