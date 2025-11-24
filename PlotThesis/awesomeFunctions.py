""" Here I plot the amazing functions in comparison with the cosine functions.
"""
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.static.open import openHamiltonian
from wavespin.tools.inputUtils import importParameters
import wavespin.tools.pathFinder as pf
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

final = True
if final:
    plt.rcParams.update({
        "text.usetex": True,              # Use LaTeX for all text
        "font.family": "serif",           # Set font family
        "font.serif": ["Computer Modern"], # Default LaTeX font
    })

save = True
Lx = 30
Ly = 31
J1 = 1
h = 0

# Load data
dataFn = pf.getFilename(*('PHI',Lx,Ly,J1,h),dirname='Data/',extension='.npz')
if Path(dataFn).is_file():
    phi = np.load(dataFn)['phi']
else:
    parameters = importParameters()
    parameters.lat_Lx = Lx
    parameters.lat_Ly = Ly
    parameters.dia_Hamiltonian = (J1/2,0,0,0,h,0)
    parameters.dia_plotWf = False
    system = openHamiltonian(parameters)
    system.diagonalize()
    phi = system.Phi
    if save:
        np.savez(dataFn,phi=phi)



X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly),indexing='ij')

data_a = []
inds_a = [2,3,4,6]
for ia in range(4):
    nn = phi[:,inds_a[ia]].reshape(Lx,Ly)
    nn[nn>0] /= np.max(nn[nn>0])
    nn[nn<0] /= abs(np.min(nn[nn<0]))
    if inds_a[ia] in [3,29]:
        nn *= -1
    data_a.append(nn)

data_cos = []
inds_k = [(1,0),(1,1),(0,2),(1,2)]
for ik in range(4):
    kx,ky = inds_k[ik]
    data_cos.append(np.cos(np.pi*kx*(2*X+1)/(2*Lx))*np.cos(np.pi*ky*(2*Y+1)/(2*Ly)))

data_diff = []
mins = np.zeros(4)
maxs = np.zeros(4)
for i in range(4):
    data_diff.append(np.absolute(data_cos[i]-data_a[i]))
    mins[i] = np.min(data_diff[-1])
    maxs[i] = np.max(data_diff[-1])
vmin = np.min(mins)
vmax = np.max(maxs)

#
datasets = [data_a,data_diff,data_cos]
mesh = []

fig = plt.figure(figsize=(4.65,3))
gs = fig.add_gridspec(3, 4, wspace=0.0, hspace=0.02)

s_norm = 10
s_small = 9
s_verysmall = 7

axes_3d = [[fig.add_subplot(gs[i, j], projection='3d') for j in range(4)] for i in [0,2]]
axes_2d = [fig.add_subplot(gs[1, j]) for j in range(4)]

pane_col = (1.0, 0.973, 0.906)#,0.5)
for i in range(3):      #row
    data = datasets[i]
    for j in range(4):      #columns
        ax = axes_3d[i//2][j] if i in [0,2] else axes_2d[j]
        if i in [0,2]:
            ax.plot_surface(X,Y,data[j],cmap='plasma')
            ax.set_box_aspect([1, 1, 0.8])
            ax.xaxis.set_pane_color(pane_col)
            ax.yaxis.set_pane_color(pane_col)
            ax.zaxis.set_pane_color(pane_col)
            #
            ax.set_zticks([-1,0,1],[r"$-1$",r"$0$",r"$1$"])
            ax.tick_params(
                axis='z',
                labelsize=s_verysmall,
                pad=-3
            )
            #
            ax.set_xlabel('x',size=s_verysmall,labelpad=-18)
            ax.set_ylabel('y',size=s_verysmall,labelpad=-18)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if i==0:
            ax.set_title(r"mode $n=%d$"%inds_a[j],
                         size=s_small,
                         bbox=dict(facecolor=pane_col,    # box fill color
                                   linewidth=0.2,
                                   edgecolor='black',   # border color
                                   boxstyle='round,pad=0.2')
                         )
        if i==2:
            ax.set_title(r"$k_x=%d$, $k_y=%d$"%inds_k[j],
                         size=s_small,
                         y=-0.4,
                         bbox=dict(facecolor=pane_col,    # box fill color
                                   linewidth=0.2,
                                   edgecolor='black',   # border color
                                   boxstyle='round,pad=0.2')
                         )
        if i==1:
            mesh.append(ax.pcolormesh(X,Y,
                          data[j],
                          cmap='bwr',
                          vmin=vmin,
                          vmax=vmax
                          ))
            ax.tick_params(axis='both',
                           which='both',      # apply to both major and minor ticks
                           direction='in',    # ticks point inward
                           left=True,          # show ticks on top
                           right=True,       # show ticks on bottom
                           top=True,
                           length=3,
                           )          # tick length (optional)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_aspect('equal')
            if 0:
                ax.set_ylabel("y",size=s_small,rotation=0)
            else:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            #ax.set_xlabel("x",size=s_label)

# Colorbar
cbar_ax = fig.add_axes([0.98, 0.37, 0.015, 0.25])  # [left, bottom, width, height]
cbar = fig.colorbar(mesh[2], cax=cbar_ax)
if 0:
    plt.figtext(
        0.96, 0.67,             # x (centered), y (slightly above top)
        r"$\Delta^2$",    # label text
        size=s_norm
    )
#cbar.ax.xaxis.set_label_position('top')  # move label to top
cbar.ax.tick_params(labelsize=s_verysmall)
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

# titles
xd = 0.005
hd = 0.81
wd = 0.305
txt = ["Eigenstate","Difference","Cosine"]
for i in range(3):
    plt.figtext(xd,hd-i*wd,
                txt[i],
                size=s_norm,
                bbox=dict(facecolor='white',    # box fill color
                          linewidth=0.1,
                          edgecolor='black',   # border color
                          boxstyle='round,pad=0.3')
                )


if 1:
    plt.subplots_adjust(
        #bottom = 0.078,
        #top = 0.956,
        right = 0.99,
        left = 0.15
    )

if final:
    plt.savefig(
        "Figures/awesomeFunctions.pdf",
        bbox_inches="tight"
    )
















































