""" Here I plot the state energy wrt T in 7x8 system.
"""

import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.static.open import openHamiltonian
from wavespin.tools.inputUtils import importParameters
import wavespin.tools.pathFinder as pf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

final = True
if final:
    plt.rcParams.update({
        "text.usetex": True,              # Use LaTeX for all text
        "font.family": "serif",           # Set font family
        "font.serif": ["Computer Modern"], # Default LaTeX font
    })

save = True
Lx = 7
Ly = 8
Ns = Lx*Ly
Tmax = 11
nT = 200
Tlist = np.linspace(0,Tmax,nT)
g1 = 10
h = 0

# Data
dataFn = pf.getFilename(*('Tempeature',Lx,Ly,Ns,g1,h,Tmax,nT),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    ETs = np.load(dataFn)['ETs']
    evals = np.load(dataFn)['evals']
else:
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_Hamiltonian = (g1,0,0,0,h,0)
    system = openHamiltonian(parameters)
    Nbonds = np.sum(system._NNterms(1)) // 2
    evals = system.evals
    GSE = system.get_GSE()
    ETs = np.zeros(nT)
    for i in range(nT):
        B = 1/(np.exp(evals[1:]/Tlist[i])-1)
        ETs[i] = GSE + np.sum(evals[1:]*B) / Nbonds / g1
    if save:
        np.savez(dataFn,evals=evals,ETs=ETs)

fig = plt.figure(figsize=(4.65/2,4.65/2))
ax = fig.add_subplot()
s_norm = 10
s_small = 9

ax.plot(
    Tlist,
    ETs,
    lw=1,
    color='firebrick',
    label="$E(T)$"
)
ax.set_xlabel(
    r"$T$ [MHz]",
    size=s_small
)
ax.set_ylabel(
    "State energy (g)",
    size=s_small
)
xmin,xmax=ax.get_xlim()
col = ['b','dodgerblue','navy']
for i in range(1,4):
    ax.axvline(
        evals[i],
        lw=1,
        ls='--',
        color=col[i%3],
        label=r'mode $n=%d$'%i
    )
ax.set_xlim(xmin,xmax)
ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
if 0:
    ax.axvline(
        ETs[np.argmin(abs(-0.5-ETs))],
        color='r',
        label=r"$T=8$"
    )

ax.axhline(
    -0.5,
    color='r'
)

ax.legend(
    fontsize=s_small,
    handlelength=1,
    handletextpad=1
)

if final:
    plt.savefig(
        "Figures/temperature.pdf",
        bbox_inches="tight"
    )

