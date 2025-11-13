""" Here I make the cool plot of the thesis with dispersions of J1-J2 model.
"""
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.static.open import openHamiltonian
from wavespin.tools.inputUtils import importParameters
import wavespin.tools.pathFinder as pf
from pathlib import Path
import matplotlib.pyplot as plt

parameters = importParameters()
parameters.lat_boundary = 'periodic'

final = True
if final:
    plt.rcParams.update({
        "text.usetex": True,              # Use LaTeX for all text
        "font.family": "serif",           # Set font family
        "font.serif": ["Computer Modern"], # Default LaTeX font
    })

save = True
Lx = 50 if final else 20
Ly = 50 if final else 20
hmin = 0
hmax = 3
nh = 49
hlist = np.linspace(hmin,hmax,nh)

# Load data
disps = []
gaps = []
gse = []
ths = []
Js = [(1,0),(1,1/2),(1,1)]
for J in Js:
    J1 = J[0]
    J2 = J[1]
    dataFn = pf.getFilename(*('dispersions',Lx,Ly,J1,J2,hmin,hmax,nh),dirname='Data/',extension='.npz')
    if Path(dataFn).is_file():
        disps.append(np.load(dataFn)['dispersions'])
        gaps.append(np.load(dataFn)['gaps'])
        gse.append(np.load(dataFn)['ens'])
        ths.append(np.load(dataFn)['thetas'])
    else:
        parameters.lat_Lx = Lx
        parameters.lat_Ly = Ly
        disp = np.zeros((nh,Lx,Ly))
        gap = np.zeros(nh)
        ens = np.zeros(nh)
        thetas = np.zeros(nh)
        for i in range(nh):
            parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,hlist[i],0)
            ham = openHamiltonian(parameters)
            disp[i] = ham.dispersion
            gap[i] = ham.gap
            ens[i] = ham.GSenergy
            thetas[i] = ham.theta
        if save:
            np.savez(dataFn,
                     dispersions=disp,
                     gaps=gap,
                     ens=ens,
                     thetas=thetas)
        disps.append(disp)
        gaps.append(gap)
        gse.append(ens)
        ths.append(thetas)

from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots
from matplotlib.ticker import MaxNLocator
fig = plt.figure(figsize=(4.65,2.5))
#fig = plt.figure(figsize=(24, 10))
gs = fig.add_gridspec(
    3, 6,
    width_ratios=[2, 0.2, 1, 1, 1, 1],
    wspace=0.1,
    hspace=0.05,
)
s_norm = 10
s_small = 9
s_verysmall = 8

### Line plots
parameters_text = [
    "Néel\n$J_2=0$",
    "Critical\n$J_2=1/2$",
    "Stripe\n$J_2=1$"]
c_th = 'dodgerblue'
c_gap = 'limegreen'
c_gse = 'darksalmon'
ax_col1 = [fig.add_subplot(gs[0, 0], sharex=None),]
ax_col1 += [fig.add_subplot(gs[i, 0], sharex=None if i == 0 else ax_col1[0]) for i in range(1,3)]
for i, ax in enumerate(ax_col1):
    # Theta
    l1 = ax.plot(hlist, ths[i],
                 color=c_th,
                 marker='',# if not final else '', 
                 label=r"$\theta$"
                 )      #Thetas
    # x ticks -> same for all
    ax.set_xticks([0,1,2,3],["$0$","$1$","$2$","$3$",])
    ax.tick_params(axis='x',
                   which='both',      # apply to both major and minor ticks
                   direction='in',    # ticks point inward
                   top=True,          # show ticks on top
                   bottom=True,       # show ticks on bottom
                   labelsize=7,
                   labeltop=True if i==0 else False,
                   labelbottom=False,
                   pad=1,
                   length=3,
                   )          # tick length (optional)
    if i == 0:
        #ax.xaxis.tick_top()
        ax.set_xlabel(r"Magnetic field $h$",
                      size=s_norm)
        ax.xaxis.set_label_position('top')
    # y -> theta
    ax.set_yticks([0,np.pi/4,np.pi/2],[r"$0$",r"$\frac{\pi}{4}$",r"$\frac{\pi}{2}$"])
    ax.tick_params(axis='y',
                   which='both',      # apply to both major and minor ticks
                   direction='in',    # ticks point inward
                   left=True,          # show ticks on top
                   right=False,       # show ticks on bottom
                   length=3,
                   labelsize=7,
                   colors=c_th
                   )          # tick length (optional)
    ax.set_ylabel(parameters_text[i],
                  rotation=0,
                  size=s_norm,
                  labelpad=25,
                  bbox=dict(facecolor='white',    # box fill color
                        edgecolor='black',   # border color
                        boxstyle='round,pad=0.3')
                  )
    # Gap
    ax_r = ax.twinx()
    l2 = ax_r.plot(hlist,
                   gaps[i]/4,
                   color=c_gap,
                   marker='',# if not final else '',
                   label="Gap"
                   )      #Gap
    # y -> gap
    ax_r.tick_params(axis='y',
                   which='both',      # apply to both major and minor ticks
                   direction='in',    # ticks point inward
                   left=False,          # show ticks on top
                   right=True,       # show ticks on bottom
                   length=3,
                   #width=2,
                   labelsize=5,
                   colors=c_gap
                   )          # tick length (optional)
    #ax_r.yaxis.set_major_locator(MaxNLocator(nbins=3))  # try 3–6 for a sparser axis
    if i in [0,1]:
        ax_r.set_yticks([0,0.1,0.2],[r"$0.0$",r"$0.1$",r"$0.2$"])
    else:
        ax_r.set_yticks([0,0.05,0.1],[r"$0.00$",r"$0.05$",r"$0.10$"])
    #ax_r.set_yticklabels([])
    # GSE
    ax_r = ax.twinx()
    l3 = ax_r.plot(hlist,
                   gse[i]/4,
                   color=c_gse,
                   marker='',# if not final else '',
                   label=r"$E_{GS}$"
                   )      #GS energy
    # y -> gap
    ax_r.set_yticks([-0.5,-0.3,-0.1],[r"$-0.5$",r"$-0.3$",r"$-0.1$"])
    ax_r.tick_params(axis='y',
                   which='both',      # apply to both major and minor ticks
                   direction='in',    # ticks point inward
                   left=False,          # show ticks on top
                   right=True,       # show ticks on bottom
                   length=3,
                   #width=2,
                   pad=-0.,
                   labelsize=5,#s_verysmall,
                   colors=c_gse
                   )          # tick length (optional)
    #ax_r.set_yticklabels([])
    # Legend
    if i==2:
        labels = [l.get_label() for l in l1+l2+l3]
        ax.legend(l1+l2+l3,
                  labels,
                  fontsize=s_verysmall,
                  handlelength=1,
                  handletextpad=0.3,
                  labelspacing=0.2,
                  loc=(0.05,0.1)
                  )


### Dispersion
pane_col = (1.0, 0.973, 0.906)#,0.5)

h_title = [r"$h=0$",r"$h=1$",r"$h=2$",r"$h=3$"]
axes_3d = [[fig.add_subplot(gs[i, j], projection='3d') for j in range(2, 6)] for i in range(3)]
X,Y = np.meshgrid(np.linspace(0,2*np.pi,Lx),np.linspace(0,2*np.pi,Ly),indexing='ij')
for i in range(3):      #row
    for j in range(4):      #columns
        ax = axes_3d[i][j]
        idx = np.where(hlist == j%4)[0][0]
        ax.plot_surface(X, Y, disps[i][idx].reshape(Lx,Ly), cmap='viridis')
        #### Make it pretty
        # Remove background
        ax.set_box_aspect([1, 1, 0.5])
        ax.xaxis.set_pane_color(pane_col)
        ax.yaxis.set_pane_color(pane_col)
        ax.zaxis.set_pane_color(pane_col)
        ax.grid(False)
        # x-y ticks
        ax.set_xticks([0,np.pi,2*np.pi],[r"$0$",r"$\pi$",r"$2\pi$"])
        ax.set_yticks([0,np.pi,2*np.pi],[r"$0$",r"$\pi$",r"$2\pi$"])
        ax.tick_params(
            axis='x',
            pad=-6,
            labelsize=5,
        )
        ax.tick_params(
            axis='y',
            pad=-6,
            labelsize=5,
        )
        # x-y grid
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        for n in range(3):
            ax.plot([n*np.pi,n*np.pi],[ymin,ymax],[0,0],color='gray',lw=0.5,zorder=-1)
            ax.plot([xmin,xmax],[n*np.pi,n*np.pi],[0,0],color='gray',lw=0.5,zorder=-1)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        # z-labels
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        if j!=3:
            ax.set_zticklabels([])
        ax.tick_params(
            axis='z',
            labelsize=5,
            pad=-4
        )
        # Title
        if i==0:
            ax.set_title(h_title[j],
                         size=s_norm,
                         bbox=dict(facecolor='white',    # box fill color
                                   edgecolor='black',   # border color
                                   boxstyle='round,pad=0.3')
                         )

# adjust z limits
for i in range(3):  # columns 2–5
    zmins, zmaxs = [], []
    for j in range(4):
        zlims = axes_3d[i][j].get_zlim()
        zmins.append(zlims[0])
        zmaxs.append(zlims[1])
    zmin, zmax = min(zmins), max(zmaxs)
    for j in range(4):
        axes_3d[i][j].set_zlim(zmin, zmax)

plt.subplots_adjust(
#    bottom = 0.014,
#    top = 0.944,
#    right = 0.987,
#    left = 0.104
)

if final:
    plt.savefig(
        "Figures/periodicDispersions.pdf",
        bbox_inches="tight"
    )
#plt.show()



















































