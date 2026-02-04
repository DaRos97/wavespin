""" Just plot of theta with h for J2=0.
"""

import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "text.usetex": True,              # Use LaTeX for all text
    "font.family": "serif",           # Set font family
    "font.serif": ["Computer Modern"], # Default LaTeX font
})
Lx = Ly =10
hmin = 0
hmax = 3
nh = 49
hlist = np.linspace(hmin,hmax,nh)
J1 = 1
J2 = 0

gap = True

dataFn = 'dataFigPresentation.npz'
if Path(dataFn).is_file():
    ths = np.load(dataFn)['theta']
    gaps = np.load(dataFn)['gaps']
else:
    from wavespin.static.open import openHamiltonian
    from wavespin.tools.inputUtils import importParameters
    parameters = importParameters()
    parameters.lat_boundary = 'periodic'
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    ths = np.zeros(nh)
    gaps = np.zeros(nh)
    for i in range(nh):
        print("h = %.3f"%hlist[i])
        parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,hlist[i],0)
        ham = openHamiltonian(parameters)
        ths[i] = ham.theta
        gaps[i] = ham.gap
    np.savez(dataFn,theta=ths,gaps=gaps)



fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
s_ = 50

ax.plot(
    hlist,
    ths,
    color='r',
    lw=5
)
if gap:
    ax_r = ax.twinx()
    ax_r.plot(
        hlist,
        gaps,
        color='g',
        lw=5
    )
    ax_r.set_ylabel(
        r"Gap, $\Delta$",
        size=s_,
        color='g'
    )
    ax_r.set_yticks([],[],size=s_)

ax.set_xlabel(
    "h",
    size=s_
)

ax.set_ylabel(
    r"Canting angle, $\theta$",
    size=s_,
    color='r'
)
ax.set_xticks([0,1,2,3],['0','1','2','3'],size=s_)
ax.set_yticks([0,np.pi/2],['0',r'$\frac{\pi}{2}$'],size=s_,color='r')


plt.show()
