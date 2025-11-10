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
hvals = (0,0.5,1)

### Data
dataFn = pf.getFilename(*('VerticesOrder2',Lx,Ly,Ns,g1,hvals,T,broadening),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    rates = np.load(dataFn)['rates']     #(Type,h,Ns)
    evals = np.load(dataFn)['evals']
    momenta = np.load(dataFn)['momenta']
else:
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.sca_types = types
    parameters.sca_broadening = broadening
    rates = np.zeros((len(types),len(hvals),Ns))
    parameters.sca_temperature = T
    for ih,h in enumerate(hvals):
        parameters.dia_Hamiltonian = (g1,0,0,0,h,0)
        system = openHamiltonian(parameters)
        system.computeRate()
        for ir in range(len(types)):
            rates[ir,ih] = system.rates[parameters.sca_types[ir]]
        if h==0:
            from wavespin.static.momentumTransformation import extractMomentum
            evals = system.evals
            momenta = np.zeros((Ns,2))
            for ik in range(1,Ns):
                momenta[ik] = extractMomentum(system.Phi[:,ik].reshape(Lx,Ly))
    if save:
        np.savez(dataFn,rates=rates,evals=evals,momenta=momenta)

# Figure
fig = plt.figure(figsize=(8.27*2,6))
title = {
    '1to2_2':r"$\Gamma^{1\leftrightarrow2}_2$",
    '1to3_2':r"$\Gamma^{1\leftrightarrow3}_2$",
    '2to2_2':r"$\Gamma^{2\leftrightarrow2}_2$",
    '1to3_3':r"$\Gamma^{1\leftrightarrow3}_3$",
}
s_label = 20
s_ticklabel = 15
s_legend = 15
s_title = 25
s_text = 25
#colors = ['khaki','y','olive']
colors = ['navy', 'dodgerblue', 'lightblue']
bm = np.array([17,18,19,20,24,25,26,27,30,31,32,34])
for ir in range(4):
    ax = fig.add_subplot(1,4,1 + ir)
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 2 decimals
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    for ih,h in enumerate(hvals):
        if ir==0 or ih==0:
            ax.scatter(
                #np.arange(1,Ns),
                evals[1:]/10,
                       rates[ir,ih,1:],
                       marker='o',
                       color=colors[ih],
                       label='h=%.1f'%h if ir==0 else '',
                       s=70
                       )
        if (ir==2 and ih==2):
            ax.scatter(
                evals[bm]/10,
                   rates[ir,ih,bm],
                   marker='o',
                   facecolors='none',   # Empty inside
                   edgecolors='red',
                   color='r',
                   lw=2,
                   s=200,
                   zorder=-1,
                   )
    if ir==0:
        ax.set_ylabel("Decay rate [MHz]",size=s_label)
        ax.legend(fontsize=s_legend)
    #ax.set_xlabel("Mode number",size=s_label)
    ax.set_xlabel("Energy(g)",size=s_label)
    ax.set_title(title[types[ir]],
                 size=s_title,
                 x=0.2,
                 #y=1.03
                 y=1.08
                 )
    ax.tick_params(axis='both',
                   which='both',      # apply to both major and minor ticks
                   direction='in',    # ticks point inward
                   top=True,          # show ticks on top
                   bottom=True,       # show ticks on bottom
                   left=True,       # show ticks on bottom
                   labelsize=s_ticklabel,
                   #pad=2,
                   length=5)          # tick length (optional)
    if ir!=2:
        ax.ticklabel_format(style='sci', axis='y', scilimits = (0,0))
    if ir==2:
        Ebm = np.mean(evals[bm]/10)
        ymin,ymax = ax.get_ylim()
        ax.axvline(Ebm,ls='--',color='r',label='MSE')
        #ax.fill_between([np.min(evals[bm]/10),np.max(evals[bm]/10)],ymin,ymax,color='r',alpha=0.2,lw=0)
        ax.set_ylim(ymin,ymax)
        ax.legend(fontsize=s_legend)
    if ir==0:
        Ebm = np.mean(evals[bm]/10)
        ymin,ymax = ax.get_ylim()
        ax.axvline(Ebm/2,ls='--',color='orange',label="half of MSE")
        #ax.fill_between([np.min(evals[bm]/10)/2,np.max(evals[bm]/10)/2],ymin,ymax,color='orange',alpha=0.2,lw=0)
        ax.set_ylim(ymin,ymax)
        ax.legend(fontsize=s_legend)
    if ir==1 or ir==3:
        Ebm = np.mean(evals[bm]/10)
        ymin,ymax = ax.get_ylim()
        ax.axvline(Ebm/3,ls='--',color='y',label="third of MSE")
        #ax.fill_between([np.min(evals[bm]/10)/2,np.max(evals[bm]/10)/2],ymin,ymax,color='orange',alpha=0.2,lw=0)
        ax.set_ylim(ymin,ymax)
        ax.legend(fontsize=s_legend)


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
    fig.savefig("Figures/SecondOrderDecay.png")

plt.show()
