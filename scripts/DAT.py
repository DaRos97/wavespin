""" Script for Discrete Awesome Transform (DAT).
In time we always use fft.
"""

import numpy as np
from scipy.fft import fftshift, fft

dataFilename = "7x9_zz_example.npy"     #stop ratio = 1
wavefunctionFilename = "7x9_wavefunction.npy"        #stop ratio = 1
correlatorXT = np.load(dataFilename)       #For me it has shape Lx,Ly,nTimes
U_ = np.load(wavefunctionFilename)['awesomeU']
V_ = np.load(wavefunctionFilename)['awesomeV']
evals = np.load(wavefunctionFilename)['evals']
Lx,Ly,nTimes = correlatorXT.shape
Ns = Lx*Ly
nOmega = 2000

def _xy(i):
    return i // Ly, i % Ly

def extractMomentum(f_in,ik=0,dctType=2):
    """ We get the momentum associated with a given Bogoliubov mode.
    We do this by computing the peak of the dctn of the input function (mode) to extract the momentum.
    f_in has shape (Lx,Ly).
    """
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

# Compute DAT

correlatorXT = correlatorXT.reshape(Lx*Ly,nTimes)
correlatorKW = np.zeros((Lx,Ly,nOmega),dtype=complex)
temp = np.zeros((Lx,Ly,nTimes),dtype=complex)
phi_ik = np.real(U_ - V_)
for i in range(Ns):
    ix,iy = _xy(i)
    phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
ks = []
ens = np.zeros((Lx,Ly))
for k in range(Ns):
    kx,ky = extractMomentum(phi_ik[:,k].reshape(Lx,Ly))
    ks.append((kx,ky))
    ens[kx,ky] = evals[k]
    temp[kx,ky] = np.sum(phi_ik[:,k,None]*correlatorXT[:,:],axis=0)
    correlatorKW[kx,ky] = fftshift(fft(temp[kx,ky],n=nOmega))

if 1:       #Here if you want you can check (by plotting) which momenta were identified by the procedure above
    import matplotlib.pyplot as plt
    printEnergies = True
    printIndexes = True
    printTitle = True
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(121)
    for k in range(Ns):
        kx,ky = ks[k]
        ax.scatter(kx,ky,color='orange',alpha=0.6,s=100)
        if printEnergies:
            ax.text(kx,ky+0.2,"{:.3f}".format(evals[k]))
        if printIndexes:
            ax.text(kx,ky-0.2,str(k))
    ax.set_xlabel("Kx",size=20)
    ax.set_ylabel("Ky",size=20)
    ax.set_aspect('equal')
    ax.grid(True)
    if printTitle:
        ax.set_title("Momenta obtained from Bogoliubov modes and their energies",size=20)
    #
    KX,KY = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    ax = fig.add_subplot(122,projection='3d')
    ax.plot_surface(KX,KY,ens,cmap='viridis',alpha=0.5,zorder=0)

    plt.show()

# Now you can plot your DAT correlator

