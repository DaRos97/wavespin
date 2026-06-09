""" 
Here we compute the sitew occupation of single magnon modes
"""

import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt

parameters = importParameters()

J1 = 20
J2 = 0
h = 0
parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)
parameters.dia_plotWf = False
parameters.dia_saveWf = True

resSquare = []
Lx = 11
Ly = 10
parameters.lat_Lx = Lx
parameters.lat_Ly = Ly
# OBC
parameters.lat_boundary = 'open'
mySystem = openHamiltonian(parameters)
G_GS = np.real(np.einsum('ik,jk->ij',mySystem.V_,mySystem.V_,optimize=True))
resSquare = np.diagonal(G_GS)

fig, axs = plt.subplots(5,6,figsize=(24,20))
for n in range(30):
    ax = axs[n//6,n%6]
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')
    resN = resSquare - np.real(mySystem.V_)[:,n]**2 + np.real(mySystem.U_)[:,n]**2
    pm = ax.pcolormesh(
        X,Y,
        resN.reshape(Lx,Ly),
        cmap='bwr',
    )
    print("%d, %.4f occupation"%(n,np.sum(resN)))
    print("occupation per site: %.4f"%(np.sum(resN)/Lx/Ly))
    print("occupation per site minus GS: %.4f"%(np.sum(resN-resSquare)))
    print("-"*20)
    fig.colorbar(pm,ax=ax)
    ax.set_title("Mode %d, occupation per site=%.3f"%(n,np.sum(resN)/Lx/Ly),size=10)
fig.tight_layout()
plt.show()
