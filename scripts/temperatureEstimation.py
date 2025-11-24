""" Here I compute the system's energy as a funtion of temperature.
I do it both with the eigenvalues and with the expectation value of <XX+YY>.
Different geometries are considered.
"""

import numpy as np
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

#lat = '60-diamond'
lat = '7x8-rectangle'

parameters = importParameters()
if lat == '24-diamond':
    Lx = 6
    Ly = 6
    offSiteList = ( (0,0), (0,1), (0,4), (0,5), (1,0), (1,5), (4,0), (4,5), (5,0), (5,1), (5,4), (5,5) )
    TMax = 15
    Eref = -0.4
elif lat == '60-diamond':
    Lx = 10
    Ly = 10
    offSiteList = ( (0,0), (0,1), (0,2), (0,3), (0,6), (0,7), (0,8), (0,9), (1,0), (1,1), (1,2), (1,7), (1,8), (1,9), (2,0), (2,1), (2,8), (2,9), (3,0), (3,9), (6,0), (6,9), (7,0), (7,1), (7,8), (7,9), (8,0), (8,1), (8,2), (8,7), (8,8), (8,9), (9,0), (9,1), (9,2), (9,3), (9,6), (9,7), (9,8), (9,9) )
    TMax = 15
    Eref = -0.5
elif lat == '7x8-rectangle':
    Lx = 7
    Ly = 8
    offSiteList = ()
    TMax = 15
    Eref = -0.5
g1 = 10
h = 0

""" Initialize and diagonalize system """
parameters.lat_Lx = Lx
parameters.lat_Ly = Ly
parameters.lat_offSiteList = offSiteList
parameters.dia_Hamiltonian = (g1,0,0,0,h,0)
system = openHamiltonian(parameters)
Ns = system.Ns
U = np.real(system.U_)[:,1:]
V = np.real(system.V_)[:,1:]
evals = system.evals[1:]
print("Gapless mode: ",system.evals[0])
S = system.S
Nbonds = np.sum(system._NNterms(1)) // 2      #number of bonds in the system: (Lx-1)*Ly + (Ly-1)*Lx

# GS energy from evals
evGS = -3/2 + np.sum(evals)/Nbonds/g1/2
print('Energy of GS from evals: %.3f and %.3f'%(evGS,system.GSE))

# GS energy from XX+YY explicit
exGS = 0
if 1:
    xx = - 3 + (np.sum(V**2+U**2,axis=1)[:,None] + np.sum(V**2+U**2,axis=1)[None,:])
    yy = - np.sum((U-V)[:,None,:]*(V-U)[None,:,:],axis=2)
    for i in tqdm(range(system.Ns)):
        for j in system.NN[i]:
            ix,iy = system._xy(i)
            jx,jy = system._xy(j)
            if jx < ix or jy < iy:
                continue
            exGS += (xx[i,j]+yy[i,j]) / 2 / Nbonds
else:
    exGS = -3/2
    #exGS += 4/Nbonds*np.sum(V**2)
    for i in tqdm(range(system.Ns)):
        for j in system.NN[i]:
            ix,iy = system._xy(i)
            jx,jy = system._xy(j)
            if jx < ix or jy < iy:
                continue
            exGS += 1/(2*Nbonds) * np.sum(V[i,:]**2 + U[i,:]**2)
            exGS += 1/(2*Nbonds) * np.sum(V[j,:]**2 + U[j,:]**2)
            exGS -= 1/(2*Nbonds) * np.sum(U[i,:]*V[j,:])
            exGS += 1/(2*Nbonds) * np.sum(U[i,:]*U[j,:])
            exGS += 1/(2*Nbonds) * np.sum(V[i,:]*V[j,:])
            exGS -= 1/(2*Nbonds) * np.sum(V[i,:]*U[j,:])
print('Energy of GS from explicit XX+YY: %.3f'%exGS)

Tlist = np.linspace(1e-1,TMax,100)

# E(T) using eigenvalues
evT = np.zeros(len(Tlist))
for iT,T in enumerate(Tlist):
    FactorBose_T = 1/(np.exp(evals/T)-1)      #size Ns-1
    evT[iT] = evGS + np.sum(evals*FactorBose_T)/Nbonds/g1

# E(T) using T-dependent <XX+YY>
exT = np.zeros(len(Tlist))
xx0 = - 3 + (np.sum(V**2+U**2,axis=1)[:,None] + np.sum(V**2+U**2,axis=1)[None,:])
yy0 = - np.sum((U-V)[:,None,:]*(V-U)[None,:,:],axis=2)
for iT,T in tqdm(enumerate(Tlist)):
    BT = 1/(np.exp(evals/T)-1)
    BTsum = np.sum(BT)*0
    for i in range(system.Ns):
        for j in system.NN[i]:
            ix,iy = system._xy(i)
            jx,jy = system._xy(j)
            if jx < ix or jy < iy:
                continue
            xxt = xx0[i,j] + 2*np.sum(BT*(U[i,:]**2+V[i,:]**2+U[j,:]**2+V[j,:]**2)) + BTsum * np.sum(U[i,:]**2+V[i,:]**2+U[j,:]**2+V[j,:]**2)
            yyt = yy0[i,j] - 2*np.sum(BT*(U-V)[i,:]*(V-U)[j,:]) - BTsum * np.sum((U-V)[i,:]*(V-U)[j,:])
            exT[iT] += (xxt+yyt)/2/Nbonds

# Figure
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot()
ax.plot(Tlist,evT,marker='^',label=r'$E(T)$')
ax.plot(Tlist,exT,marker='*',label=r'$<XX+YY>_T$',alpha=0.5)
ax.legend(fontsize=20)
ax.axhline(Eref,color='r')
bestT = Tlist[np.argmin(np.absolute(-Eref+evT))]
ax.axvline(bestT,color='g')
for k in range(Ns-1):
    ax.axvline(evals[k],color='lime')
ax.set_xlim(0,Tlist[-1])
ax.set_xticks([0,]+list(evals[(evals<Tlist[-1])&(evals>1)]),["0",] + [r"mag \#%s"%i for i in range(len(evals[(evals<Tlist[-1])]))],size=12)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(
    0.3,0.7,
    r"T $\sim$ %.2f MHz"%bestT,
    transform=ax.transAxes,
    bbox=props,
    size=20
)
ax.set_xlabel('Temperature [MHz]',size=20)
ax.set_ylabel("Energy (g)",size=20)
ax.set_title(lat,size=25)
plt.show()















