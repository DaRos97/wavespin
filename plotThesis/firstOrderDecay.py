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
hvals = (0,0.5,1)

### Data
dataFn = pf.getFilename(*('VerticesOrder1',Lx,Ly,Ns,g1,hvals,Ts,broadening),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    rates = np.load(dataFn)['rates']     #(Temp,Type,h,Ns)
    evals = np.load(dataFn)['evals']
    momenta = np.load(dataFn)['momenta']
else:
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.sca_types = types
    parameters.sca_broadening = broadening
    rates = np.zeros((len(Ts),3,len(hvals),Ns))
    for it,T in enumerate(Ts):
        parameters.sca_temperature = T
        for ih,h in enumerate(hvals):
            parameters.dia_Hamiltonian = (g1,0,0,0,h,0)
            system = openHamiltonian(parameters)
            system.computeRate()
            for ir in range(3):
                rates[it,ir,ih] = system.rates[parameters.sca_types[ir]]
            if h==0 and T==0:
                from wavespin.static.momentumTransformation import extractMomentum
                evals = system.evals
                momenta = np.zeros((Ns,2))
                for ik in range(1,Ns):
                    momenta[ik] = extractMomentum(system.Phi[:,ik].reshape(Lx,Ly))
    if save:
        np.savez(dataFn,rates=rates,evals=evals,momenta=momenta)

# Figure
fig = plt.figure(figsize=(8.27*2,9))
title = {
    '1to2_1':r"$\Gamma^{1\leftrightarrow2}_1$",
    '1to3_1':r"$\Gamma^{1\leftrightarrow3}_1$",
    '2to2_1':r"$\Gamma^{2\leftrightarrow2}_1$",
}
s_label = 20
s_ticklabel = 15
s_legend = 15
s_title = 25
s_text = 25
#colors = ['khaki','y','olive']
colors = ['navy', 'dodgerblue', 'lightblue']
bm = np.array([17,18,19,20,24,25,26,27,30,31,32,34])
for it in range(2):
    for ir in range(3):
        ax = fig.add_subplot(2,3,1 + it*3 + ir)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 2 decimals
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        for ih,h in enumerate(hvals):
            if ir==0 or ih==0:
                ax.scatter(np.arange(1,Ns),
                           rates[it,ir,ih,1:],
                           marker='o',
                           color=colors[ih],
                           label='h=%.1f'%h,
                           s=70
                           )
            if (ir==0 and ih==2) or (ir>0 and ih==0):
                ax.scatter(bm,
                       rates[it,ir,ih,bm],
                       marker='o',
                       facecolors='none',   # Empty inside
                       edgecolors='red',
                       color='r',
                       lw=2,
                       s=200,
                       zorder=-1,
                       label = 'saddle-modes' if ih==2 else '',
                       )
        if ir==0:
            ax.set_ylabel("Decay rate [MHz]",size=s_label)
            if it==0:
                ax.legend(fontsize=s_legend)
                ymin,ymax = ax.get_ylim()
        if it==0 and ir==2:
            ax.set_ylim(ymin,ymax)
        if it==1:
            ax.set_xlabel("Mode number",size=s_label)
        if it==0:
            ax.set_title(title[types[ir]],
                         size=s_title,
                         x=0.2,
                         #y=1.03
                         y=1.1
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

# Temperature text
xd = 0.02
#yd = 0.63
#wd = 0.43
yd = 0.6
wd = 0.4
txt = [r"$T=0$ [MHz]",r"$T=8$ [MHz]"]
pane_col = (1.0, 0.973, 0.906)#,0.5)
pane_col2 = (1.0, 0.9, 0.85)
for i in range(2):
    plt.figtext(xd,yd-i*wd,
                txt[i],
                rotation=90,
                size=s_text,
                bbox=dict(facecolor=pane_col,    # box fill color
                          edgecolor='black',   # border color
                          boxstyle='round,pad=0.2')
                )
# Inset bogoliubov saddle points
ax = fig.add_axes([0.74,0.55,0.3,0.3],projection='3d')
ax.plot_trisurf(momenta[:,0], momenta[:,1], evals, cmap='viridis', edgecolor='none',zorder=0)
s_ticklabel2 = 12
ax.set_box_aspect([1, 1, 0.75])
ax.xaxis.set_pane_color(pane_col)
ax.yaxis.set_pane_color(pane_col)
ax.zaxis.set_pane_color(pane_col)
ax.grid(False)

xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ax.set_xticks([0,Lx/2,Lx-1],[r"$0$",r"$\pi/2$",r"$\pi$"],size=s_ticklabel2)
ax.set_yticks([0,Ly/2,Ly-1],[r"$0$",r"$\pi/2$",r"$\pi$"],size=s_ticklabel2)
ax.set_zticklabels([])
ax.set_zlabel(r"$\epsilon({\bf k})$",size=s_ticklabel2,labelpad=-15)
#
x = [0,Lx/2,Lx-1]
y = [0,Ly/2,Ly-1]
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
       lw=2,
       s=200,
    zorder=10
           )
ax.set_title("Modes' energies",size=15,y=1)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)


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
    fig.savefig("Figures/FirstOrderDecay.png")

plt.show()





























