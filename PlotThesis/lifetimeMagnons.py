""" Here I plot the amplitude dependence of magnon lifetime.
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
types = ('1to2_1','1to2_2','1to3_1','1to3_2','1to3_3','2to2_1','2to2_2')
Ein = -0.55
Efin = -0.3
nE = 10
T = 8
broadening = 0.5
g1 = 10
h = 0

### Data
dataFn = pf.getFilename(*('AllVertices',Lx,Ly,Ns,g1,h,T,broadening,Ein,Efin,nE),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    rates = np.load(dataFn)['rates']     #(Type,h,Ns)   -> for first two plots
    evals = np.load(dataFn)['evals']    # -> for first 2 plots
    momenta = np.load(dataFn)['momenta']
    rates2 = np.load(dataFn)['rates2']      #(nE,Ns)    -> for last plot -> just 2to2_1
    GSE = np.load(dataFn)['GSE']        # Ground state energy for last plot
    Es=np.load(dataFn)['Es']
else:
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.sca_types = types
    parameters.sca_broadening = broadening
    rates = np.zeros((len(types),Ns))
    parameters.sca_temperature = T
    parameters.dia_Hamiltonian = (g1,0,0,0,h,0)
    system = openHamiltonian(parameters)
    system.computeRate()
    for ir in range(len(types)):
        rates[ir] = system.rates[parameters.sca_types[ir]]
    #
    from wavespin.static.momentumTransformation import extractMomentum
    evals = system.evals/g1
    momenta = np.zeros((Ns,2))
    for ik in range(1,Ns):
        momenta[ik] = extractMomentum(system.Phi[:,ik].reshape(Lx,Ly))
    #
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.sca_types = ('2to2_1',)
    parameters.sca_broadening = broadening
    parameters.dia_Hamiltonian = (g1,0,0,0,h,0)
    rates2 = np.zeros((nE,Ns))
    system0 = openHamiltonian(parameters)
    GSE = system0.get_GSE()
    Es = np.logspace(np.log10(Ein-GSE),np.log10(Efin-GSE),nE)
    for ie in range(nE):
        system = openHamiltonian(parameters)
        system.p.sca_temperature = system._temperature(GSE+Es[ie])
        system.computeRate()
        rates2[ie] = system.rates['2to2_1']
    #
    if save:
        np.savez(dataFn,
                 rates=rates,
                 evals=evals,
                 momenta=momenta,
                 rates2=rates2,
                 Es=Es,
                 GSE=GSE
                 )


fig = plt.figure(figsize=(4.65,1.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1], wspace=0.25)
s_norm = 10
s_small = 8
s_int = 8
s_verysmall = 7

### Full rate at different amplitudes
ax = fig.add_subplot(gs[0])
amplitudes = (0.5,1)#,10)

colors = ['navy', 'dodgerblue', 'lightblue']
firstOrder = [0,2,5]
secondOrder = [1,3,6]
thirdOrder = [4,]
for iA,A in enumerate(amplitudes):
    data = np.sum(rates[firstOrder],axis=0) + A**2/2 * np.sum(rates[secondOrder],axis=0) + A**4/4 * np.sum(rates[thirdOrder],axis=0)
    ax.scatter(
        evals[1:],
        data[1:],
        color=colors[iA],
        s=4,
        label=r'$A=%.2f$'%A,
    )
ax.legend(
    loc='upper left',
    fontsize=s_verysmall,
    handlelength=0.5,
    handletextpad=0.5
)

ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.tick_params(axis='both',
               which='both',      # apply to both major and minor ticks
               direction='in',    # ticks point inward
               top=True,          # show ticks on top
               bottom=True,       # show ticks on bottom
               left=True,       # show ticks on bottom
               labelsize=s_verysmall,
               pad=2,
               length=3,
               )          # tick length (optional)
ax.set_xlabel("Energy(g)",size=s_small)
ax.set_ylabel("Decay rate [MHz]",size=s_small)
#ax.set_yscale('log')

### Full rate as a function of amplitude for the first modes
ax = fig.add_subplot(gs[1])
#amplitudes = np.linspace(0.05,10,10)
amplitudes = np.logspace(-0.5,2,20)
ns = np.arange(1,15)

cmap = cm.cividis
norm = mcolors.Normalize(vmin=evals[ns[0]], vmax=evals[ns[-1]])

#colors = ['navy', 'dodgerblue', 'lightblue']
firstOrder = [0,2,5]
secondOrder = [1,3,6]
thirdOrder = [4,]
for i_n in ns:
    data = np.sum(rates[firstOrder,i_n],axis=0) + amplitudes**2/2 * np.sum(rates[secondOrder,i_n],axis=0) + amplitudes**4/4 * np.sum(rates[thirdOrder,i_n],axis=0)
    ax.plot(
        amplitudes,
        data,
        marker='o',
        #ls='--',
        color=cmap(norm(evals[i_n])),
        markersize=2,
        lw=0.5
    )
ax.set_yscale('log')
ax.set_xscale('log')

# Fit lines
ymin,ymax = ax.get_ylim()
pows = (2,3,4)
indy = (ns[-1],5,1)
colors = ['blue','r','aqua']
for i in range(len(pows)):
    data = np.sum(rates[firstOrder,indy[i]],axis=0) + amplitudes[-1]**2/2 * np.sum(rates[secondOrder,indy[i]],axis=0) + amplitudes[-1]**4/4 * np.sum(rates[thirdOrder,indy[i]],axis=0)
    ax.plot(
        amplitudes,
        amplitudes**pows[i] / amplitudes[-1]**pows[i] * data,
        ls='--',
        color=colors[i],
        zorder=10+i,
        lw=1,
        label=r"$\sim A^%d$"%pows[i]
    )

ax.set_ylim(ymin,ymax)
ax.legend(
    fontsize=s_verysmall,
    handlelength=1,
    #handletextpad=0.5
)

#ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
ax.tick_params(axis='both',
               which='both',      # apply to both major and minor ticks
               direction='in',    # ticks point inward
               top=True,          # show ticks on top
               bottom=True,       # show ticks on bottom
               left=True,       # show ticks on bottom
               labelsize=s_verysmall-2,
               pad=2,
               length=3
               )          # tick length (optional)
ax.set_xlabel("Amplitude",size=s_small)

### Decay rate as a function of state energy for first few modes
ax = fig.add_subplot(gs[2])

for i_n in ns:
    ax.plot(
        Es,
        rates2[:,i_n],
        marker='o',
        ls='--',
        color=cmap(norm(evals[i_n])),
        markersize=2,
        lw=0.5
    )
ax.set_yscale('log')
ax.set_xscale('log')
ax.tick_params(axis='both',
               which='both',      # apply to both major and minor ticks
               direction='in',    # ticks point inward
               top=True,          # show ticks on top
               bottom=True,       # show ticks on bottom
               left=True,       # show ticks on bottom
               labelsize=s_verysmall-2,
               pad=2,
               length=3,
               )          # tick length (optional)
ax.set_xlabel("State energy from GS",size=s_small)

# Fit line
ymin,ymax = ax.get_ylim()
po = 1.
ax.plot(
    Es,
    Es**po ,#/ Es[-1]**po * rates[-1,10],
    ls='--',
    color='r',
    zorder=10,
    lw=1,
    label=r"$\sim E$"
)
ax.legend(
    fontsize=s_verysmall,
    handlelength=1,
    #handletextpad=0.5
)

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required for colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Energy(g)',size=s_small)
cbar.ax.tick_params(
    axis='y',
    length=2,
    labelsize=s_verysmall
)


if 0:
    # Adjust figure
    plt.subplots_adjust(
        bottom = 0.092,
        top = 0.96,
        right = 0.956,
        left = 0.054,
    #    wspace=0.15,
    )

if final:
    fig.savefig(
        "Figures/lifetimeMagnons.pdf",
        bbox_inches="tight",
    )





































