""" Here I plot the decay rates at first order.
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
types = ('1to2_1','1to3_1','2to2_1')
Ts = (0,8)
broadening = 0.5
g1 = 10
stopRatios = (1,0.9)

### Data
dataFn = pf.getFilename(*('VerticesOrder1',Lx,Ly,Ns,g1,stopRatios,Ts,broadening),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    rates = np.load(dataFn)['rates']     #(Temp,Type,sr,Ns)
    evals = np.load(dataFn)['evals']
    momenta = np.load(dataFn)['momenta']
else:
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.sca_types = types
    parameters.sca_broadening = broadening
    rates = np.zeros((len(Ts),3,len(stopRatios),Ns))
    for it,T in enumerate(Ts):
        parameters.sca_temperature = T
        for ia,alpha in enumerate(stopRatios):
            parameters.dia_Hamiltonian = (g1*alpha,0,0,0,15*(1-alpha),0)
            system = openHamiltonian(parameters)
            system.computeRate()
            for ir in range(3):
                rates[it,ir,ia] = system.rates[parameters.sca_types[ir]]
            if alpha==1 and T==0:
                from wavespin.static.momentumTransformation import extractMomentum
                evals = system.evals
                momenta = np.zeros((Ns,2))
                for ik in range(1,Ns):
                    momenta[ik] = extractMomentum(system.Phi[:,ik].reshape(Lx,Ly))
    if save:
        np.savez(dataFn,rates=rates,evals=evals,momenta=momenta)

# Figure
fig = plt.figure(figsize=(4.65,2.5))
title = {
    '1to2_1':r"$\Gamma^{1\leftrightarrow2}_1$",
    '1to3_1':r"$\Gamma^{1\leftrightarrow3}_1$",
    '2to2_1':r"$\Gamma^{2\leftrightarrow2}_1$",
}
gs = fig.add_gridspec(
    2, 4,
    width_ratios=[1, 1, 0.3, 1.2],
    wspace=0.05,
    hspace=0.05,
)

s_norm = 10
s_small = 9
s_verysmall = 7

### Figure 1->2 and 1->3
#colors = ['khaki','y','olive']
colors = ['navy', 'dodgerblue', 'lightblue']
for it in range(2):
    ax0 = fig.add_subplot(gs[it,0])
    ax1 = fig.add_subplot(gs[it,1])
    ax1.sharey(ax0)
    axs = [ax0,ax1]
    for ir in range(2):
        ax = axs[ir]
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 2 decimals
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        for ia,alpha in enumerate(stopRatios):
            ax.scatter(np.arange(1,Ns),
                       rates[it,ir,ia,1:],
                       marker='o',
                       color=colors[ia],
                       label=r'$\alpha=%.1f$'%alpha,
                       s=6
                       )
        if ir==0:
            temp = it*8
            ax.set_ylabel(
                r"$\Gamma (T=%d)$ [MHz]"%temp,
                size=s_small,
                bbox=dict(facecolor='w',    # box fill color
                          linewidth=0.1,
                          edgecolor='black',   # border color
                          boxstyle='round,pad=0.2')
            )
        if it==1:
            ax.set_xlabel(
                "Mode number",
                size=s_small-1,
                labelpad=2
            )
        if it==0:
            ax.set_title(title[types[ir]],
                         size=s_norm,
                         x=0.2,
                         #y=1.03
                         y=1.1
                         )
            ax.tick_params(labelbottom=False)
        ax.tick_params(axis='x',
                       which='both',      # apply to both major and minor ticks
                       direction='in',    # ticks point inward
                       top=True,          # show ticks on top
                       bottom=True,       # show ticks on bottom
                       labelsize=s_verysmall,
                       pad=2,
                       length=3
                       )          # tick length (optional)
        ax.tick_params(axis='y',
                       which='both',      # apply to both major and minor ticks
                       direction='in',    # ticks point inward
                       left=True,       # show ticks on bottom
                       right=True,
                       labelsize=s_verysmall,
                       pad=1,
                       length=3
                       )          # tick length (optional)
        if ir==0 and it==0:
            ax.legend(
                fontsize=s_verysmall,
                handletextpad=0.4,
                handlelength=1,
                borderaxespad=0.5
            )
    ax1.tick_params(labelleft=False)

### Figure 2<->2
ax = fig.add_subplot(gs[0,3])

data = rates[1,2,0]
ax.scatter(
    np.arange(1,Ns),
    data[1:],
    marker='o',
    color=colors[0],
    s=6
)
bm = np.array([17,18,19,20,24,25,26,27,30,31,32,34])
ax.scatter(
    bm,
    data[bm],
    marker='o',
    facecolors='none',   # Empty inside
    edgecolors='red',
    color='r',
    lw=0.5,
    s=25,
    zorder=2,
    #label = 'saddle-modes' if ih==2 else '',
)
ax.tick_params(
    axis='x',
    which='both',      # apply to both major and minor ticks
    direction='in',    # ticks point inward
    top=True,          # show ticks on top
    bottom=True,       # show ticks on bottom
    labelsize=s_verysmall,
    pad=2,
    length=3
)          # tick length (optional)
ax.set_xlabel(
    "Mode number",
    size=s_small-1,
    labelpad=2
)
ax.tick_params(
    axis='y',
    which='both',      # apply to both major and minor ticks
    direction='in',    # ticks point inward
    left=True,       # show ticks on bottom
    right=True,
    labelsize=s_verysmall,
    pad=1,
    length=3
)          # tick length (optional)
ax.set_ylabel(
    r"$\Gamma (T=8)$ [MHz]",
    size=s_small,
    bbox=dict(facecolor='w',    # box fill color
              linewidth=0.1,
              edgecolor='black',   # border color
              boxstyle='round,pad=0.2')
)
ax.set_title(
    title[types[2]],
    size=s_norm,
    x=0.2,
    #y=1.03
    y=1.1
)

### Figure Bogoliubov saddle points
ax = fig.add_axes([0.55,0,0.4,0.4],projection='3d')

pane_col = (1.0, 0.973, 0.906)#,0.5)
pane_col2 = (1.0, 0.9, 0.85)

ax.plot_trisurf(momenta[:,0], momenta[:,1], evals, cmap='viridis', edgecolor='none',zorder=0)
ax.set_box_aspect([1, 1, 0.5])
ax.xaxis.set_pane_color(pane_col)
ax.yaxis.set_pane_color(pane_col)
ax.zaxis.set_pane_color(pane_col)
ax.grid(False)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(
    r"$k_x$",
    labelpad=-18,
    size=s_verysmall
)
ax.set_ylabel(
    r"$k_y$",
    labelpad=-18,
    size=s_verysmall
)

ax.set_zticklabels([])
ax.set_zlabel(r"$\epsilon({\bf k})$",size=s_small,labelpad=-15)
zmin,zmax = ax.get_zlim()
ax.set_zlim(0,zmax)
xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
#
x = [xmin,xmin + (xmax-xmin)/2,xmax]
y = [ymin,ymin + (ymax-ymin)/2,ymax]
for n in range(3):
    ax.plot([x[n],x[n]],[ymin,ymax],[0,0],color='gray',lw=0.5,zorder=-1)
    ax.plot([xmin,xmax],[y[n],y[n]],[0,0],color='gray',lw=0.5,zorder=-1)

ax.scatter(
    momenta[bm,0],
    momenta[bm,1],
    evals[bm],
    marker='o',
    facecolors='none',   # Empty inside
    edgecolors='red',
    color='r',
    lw=0.5,
    s=25,
    zorder=10
           )
#ax.set_title("Modes' energies",size=s_norm,y=1)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

if 0:
    # Adjust figure
    plt.subplots_adjust(
        bottom = 0.062,
        #top = 0.926,
        top = 0.88,
        right = 0.979,
        left = 0.107,
        wspace=0.15,
        hspace=0.
    )

if final:
    plt.savefig(
        "Figures/firstOrderDecay.pdf",
        bbox_inches="tight"
    )






























