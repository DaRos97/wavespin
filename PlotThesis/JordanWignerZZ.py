""" Here I plot the zz correlator in 1D with data computed from the other script.
5 stop ratios: [0.2,0.4,0.6,0.8,1]
500 ns time-evolved wf
N = 60
g1=10, h=15
measurement over 800ns, time step of 2ns
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
class SqrtNorm(mcolors.Normalize):
    def __call__(self, value, clip=None):
        return (super().__call__(value, clip))**(1)

final = True
if final:
    plt.rcParams.update({
        "text.usetex": True,              # Use LaTeX for all text
        "font.family": "serif",           # Set font family
        "font.serif": ["Computer Modern"], # Default LaTeX font
    })

dataFn = "Data/1DzzCorrelator.npy"
correlator = np.load(dataFn)

N = 60
full_time_measure = 0.8   #measure time in ms
Nt = 401        #time steps after ramp for the measurement
measure_time_list = np.linspace(0,full_time_measure,Nt)
omega_bound = Nt/full_time_measure/2
Nomega = int(8*omega_bound)
omega_list = np.linspace(-omega_bound,omega_bound,Nomega)
stop_ratio_list = np.linspace(0.2,1,5)     #ratios of ramp where we stop and measure

kx = np.fft.fftshift(np.fft.fftfreq(N,d=1))

fig = plt.figure(figsize=(4.65,1.5))
gs = fig.add_gridspec(1,5,wspace=0.1)
s_norm = 10
s_small = 9
s_verysmall = 7

vmax = np.max(np.abs(correlator))/10
for i_sr in range(len(stop_ratio_list)):
    stop_ratio = stop_ratio_list[i_sr]
    ax = fig.add_subplot(gs[i_sr])
    pm = ax.pcolormesh(
        kx,
        omega_list/10,
        (np.abs(correlator[i_sr]).T)/10,
        shading='auto',
        cmap='Blues',
        norm=SqrtNorm(vmin=0,vmax=vmax),
        rasterized=True
    )
    if i_sr==0:
        ax.set_ylabel(r"$\omega(g)$",size=s_norm)
        ax.set_yticks([-4,-2,0,2,4],['-4','-2','0','2','4'])
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-5,5)
    ax.set_xticks([-0.375,0,0.375],['','',''])
    ax.tick_params(
        axis='both',
        top=True,
        bottom=True,
        right=True,
        left=True,
        labelsize=s_verysmall,
        direction='in',
        length=4,
        width=0.8,
        pad=2
    )
    ax.set_xlabel(r"$k_x$",size=s_small)
#        if i_sr>4:
#            ax.set_xlabel('Momentum ($k_x$)')
#        if i_sr in [0,5]:
#            ax.set_ylabel(r'Frequency $\omega$ (MHz)')
    ax.set_title(
        r"$\alpha=$%.1f"%stop_ratio,
        size=s_norm
    )

#Blue cb
xb = 0.92
yb = 0.11
hb = 0.77
wb = 0.02
cbar_ax = fig.add_axes([xb, yb, wb, hb])  # [left, bottom, width, height]
cbar = fig.colorbar(pm, cax=cbar_ax)
cbar.set_label(r"$\vert\chi_{ZZ}(\omega)\vert$",fontsize=s_norm)
#cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
cbar.ax.tick_params(
    width=0.8,
    labelsize=s_verysmall
)


if final:
    plt.savefig(
        "Figures/JordanWignerZZ.pdf",
        bbox_inches="tight",
        dpi=600
    )
