""" Here I plot the decay rates at second order.
"""

import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.static.open import openHamiltonian
from wavespin.tools.inputUtils import importParameters
import wavespin.tools.pathFinder as pf
from pathlib import Path
import matplotlib.pyplot as plt
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
types = ('1to2_2','1to3_2','2to2_2','1to3_3')
T = 8
broadening = 0.5
g1 = 10
stopRatios = (1,0.9)

### Data
dataFn = pf.getFilename(*('VerticesOrder2',Lx,Ly,Ns,g1,stopRatios,T,broadening),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    rates = np.load(dataFn)['rates']     #(Type,sr,Ns)
    evals = np.load(dataFn)['evals']
    momenta = np.load(dataFn)['momenta']
else:
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.sca_types = types
    parameters.sca_broadening = broadening
    rates = np.zeros((len(types),len(stopRatios),Ns))
    parameters.sca_temperature = T
    for ia,alpha in enumerate(stopRatios):
        parameters.dia_Hamiltonian = (g1*alpha,0,0,0,15*(1-alpha),0)
        system = openHamiltonian(parameters)
        system.computeRate()
        for ir in range(len(types)):
            rates[ir,ia] = system.rates[parameters.sca_types[ir]]
        if alpha==1:
            from wavespin.static.momentumTransformation import extractMomentum
            evals = system.evals
            momenta = np.zeros((Ns,2))
            for ik in range(1,Ns):
                momenta[ik] = extractMomentum(system.Phi[:,ik].reshape(Lx,Ly))
    if save:
        np.savez(dataFn,rates=rates,evals=evals,momenta=momenta)

# Figure
fig = plt.figure(figsize=(4.65,1.2))
gs = fig.add_gridspec(
    1,4,
    wspace=0.12
)
title = {
    '1to2_2':r"$\Gamma^{1\leftrightarrow2}_2$",
    '1to3_2':r"$\Gamma^{1\leftrightarrow3}_2$",
    '2to2_2':r"$\Gamma^{2\leftrightarrow2}_2$",
    '1to3_3':r"$\Gamma^{1\leftrightarrow3}_3$",
}
s_norm = 10
s_small = 9
s_verysmall = 7
s_extrasmall = 6
colors = ['navy', 'dodgerblue', 'lightblue']
bm = np.array([17,18,19,20,24,25,26,27,30,31,32,34])
for ir in range(4):
    ax = fig.add_subplot(gs[ir])
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 2 decimals
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    for ia,alpha in enumerate(stopRatios):
        if ir==0 or alpha==1:
            ax.scatter(
                evals[1:]/10,
                rates[ir,ia,1:],
                marker='o',
                color=colors[ia],
                label=r'$\alpha=%.1f$'%alpha if ir==0 else '',
                s=2
            )
        if (ir==2 and alpha==1):
            ax.scatter(
                evals[bm]/10,
                rates[ir,ia,bm],
                marker='o',
                facecolors='none',   # Empty inside
                edgecolors='red',
                color='r',
                lw=0.5,
                s=15,
                zorder=-1,
            )
    if ir==0:
        ax.set_ylabel("Decay rate [MHz]",size=s_small-1)
    #ax.set_xlabel("Mode number",size=s_label)
    ax.set_xlabel("Energy(g)",size=s_small-1)
    ax.set_title(title[types[ir]],
                 size=s_norm,
                 x=0.2,
                 #y=1.03
                 y=1.08
                 )
    ax.ticklabel_format(
        style='sci',
        axis='y',
        scilimits = (0,0)
    )
    ax.tick_params(axis='both',
                   which='both',      # apply to both major and minor ticks
                   direction='in',    # ticks point inward
                   top=True,          # show ticks on top
                   bottom=True,       # show ticks on bottom
                   left=True,       # show ticks on bottom
                   labelsize=s_extrasmall,
                   pad=2,
                   length=3,
                   )          # tick length (optional)
    ax.yaxis.get_offset_text().set_fontsize(s_extrasmall)
    Ebm = np.mean(evals[bm]/10)
    ymin,ymax = ax.get_ylim()
    ven = [Ebm/2,Ebm/3,Ebm,Ebm/3]
    lab = ["MSE/2","MSE/3","MSE","MSE/3"]
    col = ['orange','y','r','y']
    ax.axvline(
        ven[ir],
        ls='--',
        color=col[ir],
        label=lab[ir],
        lw=0.8
    )
    ax.set_ylim(ymin,ymax)
    ax.legend(
        fontsize=5,
        handletextpad=0.7,
        handlelength=1
    )

if 0:
    # Adjust figure
    plt.subplots_adjust(
    #    bottom = 0.062,
        #top = 0.926,
        top = 0.847,
        right = 0.992,
        left = 0.054,
        wspace=0.15,
    )

if final:
    plt.savefig(
        "Figures/secondOrderDecay.pdf",
        bbox_inches="tight"
    )
