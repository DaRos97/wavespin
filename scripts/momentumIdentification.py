""" Here we try to derive the momentum from the wavefunctions.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
from wavespin.lattice.lattice import latticeClass
from wavespin.tools.functions import solve_diffusion_eigenmodes_xy as sde
import wavespin.tools.functions as fs
import matplotlib.pyplot as plt
from pathlib import Path

Lx = 6
Ly = 6
offSiteList = [
    (0,0),
    (0,1),
    (0,4),
    (0,5),
    (1,0),
    (1,5),
    (4,0),
    (4,5),
    (5,0),
    (5,1),
    (5,4),
    (5,5)
]
#Lx = 7
#Ly = 8
#offSiteList = []

plotWf = 1
debug = 0

## Options
# Sites
sites = []
for ix in range(Lx):
    for iy in range(Ly):
        sites.append((ix,iy))
for ir in offSiteList:
    sites.remove(ir)

# Lattice object
parameters = importParameters()
parameters.lat_Lx = Lx
parameters.lat_Ly = Ly
parameters.lat_offSiteList = offSiteList
parameters.lat_plotLattice = False
lattice = latticeClass(parameters)
Ns = lattice.Ns
X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
#parameters.dia_plotWf = True
parameters.dia_Hamiltonian = (10,0,0,0,0,0)
system = openHamiltonian(parameters)

### Function for Laplacian
def laplacian_masked(f,direction='xy'):
    """
    Discrete Laplacian with open (Neumann) boundary conditions,
    implemented by padding the array with edge values (constant padding).
    """
    lap = np.zeros_like(f)
    for n in range(f.shape[0]):
        x,y = sites[n]
        # Start with center point
        list_nn_dic = {'x': [(x+1,y),(x-1,y)], 'y':[(x,y+1),(x,y-1)], 'xy':[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]}
        list_nn = list_nn_dic[direction]
        val = -len(list_nn) * f[n]
        for nn in list_nn:
            if nn in sites:
                val += f[sites.index(nn)]
            else:
                val += f[n]
        lap[n] = val
    return lap

diffEvals, diffEvecs = sde(sites,dx=1,dy=1+1e-5,direction='xy')
#diffEvecs, groups, block_info = fs.fix_degeneracy_with_rotation(diffEvals,diffEvecs,Lx,Ly)
#print(block_info)

ks_dif = np.zeros((Ns,2))
ks_bog = np.zeros((Ns,2))
ks = [ks_dif,ks_bog]
for n in range(1,Ns):
    func_dif = diffEvecs[:,n]
    func_bog = system.Phi[:,n]
    # Compute Laplacian numerically
    for i,func in enumerate([func_bog]):
        lapx = laplacian_masked(func,direction='x')
        lapy = laplacian_masked(func,direction='y')
        lapxy = laplacian_masked(func,direction='xy')
        #
        print(np.sort(abs(func)))
        mask = np.absolute(func)>np.max(np.absolute(func))/5
        evx = np.sum( lapx[mask]/func[mask] ) / func[mask].shape[0]
        evy = np.sum( lapy[mask]/func[mask] ) / func[mask].shape[0]
        print(n,evx,evy)
        #evx = min(abs(evx),4)
        #evy = min(abs(evy),4)
        #
        kx = np.arcsin(np.sqrt(abs(evx)/4)) /np.pi * 2 * Lx
        ky = np.arcsin(np.sqrt(abs(evy)/4)) /np.pi * 2 * Ly
        print("Derived kx,ky: %.3f,%.3f"%(kx,ky))
        ks[i][n,0] = kx
        ks[i][n,1] = ky
        if debug:
            fig = plt.figure(figsize=(20,5))
            for ii,f in enumerate([func,lapx,lapy,lapxy]):
                ax = fig.add_subplot(1,4,ii+1)
                patched = system.patchFunction(f)
                pm = ax.pcolormesh(
                    X,Y,
                    patched,
                    cmap='bwr'
                )
                ax.set_aspect('equal')
                fig.colorbar(pm)
            fig.tight_layout()
            plt.show()
    lambda_discrete = 4*(np.sin(np.pi/Lx/2*kx)**2+np.sin(np.pi/Ly/2*ky)**2)
    ev_dif = diffEvals[n]
    print("Eval and derived: %.3f,%.3f"%(ev_dif,lambda_discrete))
    print("------------------------")

if plotWf:
    fig = plt.figure(figsize=(15,15))
    Nmax = 14
    xdim = int(np.sqrt(Nmax)) + 1
    ydim = xdim
    ax = fig.add_subplot(xdim,ydim,1)
    ax.axis('off')
    #ax.text(0,0.5,"Diffusion equations solutions",size=20)
    ax.text(0,0.5,"Mean-field solutions",size=20)
    for n in range(1,Nmax+1):
        #func = diffEvecs[:,n]
        func = system.Phi[:,n]
        ax = fig.add_subplot(xdim,ydim,n+1)
        patchedFunc = lattice.patchFunction(func)
        pm = ax.pcolormesh(
            X,Y,
            patchedFunc,
            cmap='bwr'
        )
        ax.set_title(r"mode $\#$ %d"%n,size=20)
        ax.set_aspect('equal')
    fig.tight_layout()

# Plot BZ
fig = plt.figure(figsize=(20,9))
ax = fig.add_subplot(121)
for n in range(Ns):
    ax.scatter(
        ks_dif[n,0],ks_dif[n,1],
        marker='o',
        facecolors='none',
        edgecolors='r',
        s=500,
        lw=2,
    )
    ax.text(
        ks_dif[n,0],ks_dif[n,1],
        "%d"%n,
        color='r',
        ha='center',
        va='bottom',
        size=10
    )
    ax.scatter(
        ks_bog[n,0],ks_bog[n,1],
        facecolors='none',
        edgecolors='b',
        s=300,
        lw=2,
    )
    ax.text(
        ks_bog[n,0],ks_bog[n,1],
        "%d"%n,
        color='b',
        ha='center',
        va='center',
        size=10
    )
ax.set_aspect('equal')

ax = fig.add_subplot(122)
ax.scatter(np.arange(1,Ns),diffEvals[1:],label='evals diff',c='r')
ax.set_ylabel("evals diff")
ax.legend(loc='upper left')
ax = ax.twinx()
ax.scatter(np.arange(1,Ns),system.evals[1:],label='evals Bog',c='b')
ax.set_ylabel("evals Bog")
ax.legend(loc='upper right')







plt.show()


