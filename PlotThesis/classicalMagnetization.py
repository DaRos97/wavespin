""" Plot the classical J1-J2 phase diagram and the spin orientations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrow

final = True
plt.rcParams.update({
    "text.usetex": True,              # Use LaTeX for all text
    "font.family": "serif",           # Set font family
    "font.serif": ["Computer Modern"], # Default LaTeX font
    "text.latex.preamble": r"\usepackage{amsmath}",
})

### Figre Setup
fig = plt.figure(figsize=(4.65,2.5))
gs = GridSpec(2, 4,
              figure=fig,
              width_ratios=[1, 0.1, 1.6, 1],
              height_ratios=[1, 1],
              wspace=0.,
              hspace=0.)
s_norm = 10
s_small = 9

# --- Define colors (RGBA) ---
if 0:
    cNi = np.array([204, 0, 0, 1])/256   # reddish
    cSi = np.array([255, 128, 0, 1])/256   # greenish
    zN  = np.array([181, 253, 255, 1])/256   # uniform bluish
if 0:
    cNi = np.array([59, 130, 246, 1])/256   # reddish
    cSi = np.array([34, 197, 94, 1])/256   # greenish
    zN  = np.array([253, 224, 71, 1])/256   # uniform bluish
if 0:
    cNi = np.array([236, 72, 153, 1])/256   # reddish
    cSi = np.array([251, 146, 60, 1])/256   # greenish
    zN  = np.array([96, 165, 250, 1])/256   # uniform bluish
if 1:
    cNi = np.array([78, 147, 248, 1])/256   # reddish
    cSi = np.array([244, 109, 107, 1])/256   # greenish
    zN  = np.array([144, 210, 83, 1])/256   # uniform bluish
cNf = zN
cSf = zN

### Phase diagram plot
ax = fig.add_subplot(gs[:,2])
J1 = 1
D1 = D2 = 0
J2min,J2max = (0,1)
hmin,hmax = (0,3)
x = np.linspace(J2min, J2max, 500)
y = np.linspace(hmin, hmax, 500)
X, Y = np.meshgrid(x, y)

# --- Initialize an RGBA array ---
colors = np.zeros(X.shape + (4,))
for i in range(x.shape[0]):
    normy = y / y.max()
    if x[i] < J1/2:
        ycrit = 2*(J1*(1-D1) - x[i]*(1-D2))
        mask = y < ycrit
        thetas = np.arccos(y[mask]/ycrit) / np.pi*2
        colors[i,mask] = cNi[None,:] * thetas[:,None] + cNf[None,:] * (1-thetas[:,None])
    if x[i] > J1/2:
        ycrit = 2*x[i]*(1-D2)
        mask = y < ycrit
        thetas = np.arccos(y[mask]/ycrit) / np.pi*2
        colors[i,mask] = cSi[None,:] * thetas[:,None] + cSf[None,:] * (1-thetas[:,None])
    colors[i,~mask] = zN
colors[..., 3] = 1.0
colors = colors.transpose(1,0,2)

# --- Plot ---
ax.imshow(colors, origin='lower', extent=(0,1,0,3), aspect='auto')

# Draw boundaries
ax.plot(x[x<J1/2], 2*(1 - x[x<J1/2]), 'k', lw=1)
ax.plot(x[x>J1/2], 2*x[x>J1/2], 'k', lw=1)
ax.plot([J1/2,J1/2],[0,1],'k--')

ax.set_xlabel(r"$J_2$",size=s_norm)
ax.set_ylabel("h",size=s_norm,rotation=0)#,labelpad=15)
ax.set_xlim(J2min,J2max)
ax.set_ylim(hmin,hmax)
ax.tick_params(axis='both',
               direction='in',
               #length=4,
               #width=1,
               #pad=3,
               labelsize=s_small
               )
if 0:
    # Colorbars
    x_ = 0.7
    w_ = 0.02
    x_2 = x_+w_
    h_ = 0.75
    y_ = 0.1
    cax = fig.add_axes([x_,y_,w_,h_])
    cmap = mcolors.ListedColormap(colors[y<2*(J1*(1-D1)),0])
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=mcolors.Normalize(vmin=0,vmax=1))
    cbar = fig.colorbar(sm,
                        cax=cax,
                        orientation='vertical')
    #cbar.set_ticks([0,0.5,1])
    cbar.set_ticks([])
    #cbar.set_ticklabels([r"$\pi/2$",r"$\pi/4$",r"$0$"],size=s_small)
    #cbar.ax.yaxis.set_ticks_position('left')
    #cbar.ax.set_title(r"$\theta^\text{canted-Néel}$",
    #                  fontsize=s_norm,
    #                  pad=10
    #                  )

    cax = fig.add_axes([x_2,y_,w_,h_])
    cmap = mcolors.ListedColormap(colors[y<2*(x[-1]*(1-D1)),-1])
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=mcolors.Normalize(vmin=0,vmax=1))
    cbar = fig.colorbar(sm,
                        cax=cax,
                        orientation='vertical')
    cbar.set_ticks([0,0.5,1])
    cbar.set_ticklabels([r"$\pi/2$",r"$\pi/4$",r"$0$"],size=s_small)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.set_title(
        #r"$\theta^\text{canted-stripe}$",
        r"$\theta$",
        fontsize=s_norm,
        pad=10
    )
# Text phases
ax.text(
    0.04,
    0.6,
    "Canted-Néel",
    size=s_norm
)
ax.text(
    0.54,
    0.6,
    "Canted-stripe",
    size=s_norm
)
ax.text(
    0.35,
    2.3,
    "Staggered",
    size=s_norm
)
# Text transitions
ax.text(
    0.1,
    1.25,
    "second-order",
    rotation=-32.5,
    size=s_small
)
ax.text(
    0.6,
    1.3,
    "second-order",
    rotation=32.5,
    size=s_small
)
ax.text(
    0.439,
    0.1,
    "first-order",
    rotation=90,
    size=s_small
)

### Plot the spin configurations
Lx = Ly = 4
J1 = 1
D1 = D2 = 0
pars = [(0,0),(0,1.5),(1,0),(1,1.5)]
arrLen = 0.8     # Arrow length
def makeArrow(x,dx,y,dy,arrowColor,zorder=1):
    return FancyArrow(
        x, y, dx, dy,              # start and end points
        width=0.07,
        length_includes_head=True,
        head_width=0.3,
        head_length=0.3,
        color=arrowColor,
        path_effects=[pe.withSimplePatchShadow(offset=(1,-1),alpha=0.9)],
        zorder=zorder
    )
axsa = [(1,0),(0,0),(1,3),(0,3)]
colAr = []
ths = []
for i in range(4):
    axx,axy = axsa[i]
    ax = fig.add_subplot(gs[axx,axy])
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5,3.5)
    ax.set_ylim(-0.5,3.5)
    J2,h = pars[i]
    if J2 < J1/2:
        hcrit = 2*(J1*(1-D1) - J2*(1-D2))
        order = 'c-Neel'
        c_dot = ['k','k']
    else:
        hcrit = 2*J2*(1-D2)
        order = 'c-stripe'
        c_dot = ['aqua','b']
    th = np.arccos(h/hcrit)
    ths.append(th)
    ci,cf = (cNi,cNf) if order=='c-Neel' else (cSi,cSf)
    color = ci * th/np.pi*2 + cf * (1-th/np.pi*2)
    colAr.append(color)
    color[-1] = 1
    # Lattice
    for ix in range(Lx):
        for iy in range(Ly):
            ax.scatter(ix,iy,c=c_dot[(ix+iy)%2],marker='o',s=3,zorder=2)
            if ix+1<Lx:
                ax.plot([ix,ix+1],[iy,iy],c='gray',lw=1,zorder=-1)
            if iy+1<Ly:
                ax.plot([ix,ix],[iy,iy+1],c='gray',lw=1,zorder=-1)
            #
            if order == 'c-Neel':
                th2 = th + np.pi*((ix+iy)%2)
            else:
                if ix%2==1 and iy%2==0:
                    th2 = th + np.pi
                elif ix%2==0 and iy%2==1:
                    th2 = -th + np.pi
                elif ix%2==1 and iy%2==1:
                    th2 = -th
                else:
                    th2 = th
            dx = arrLen * np.sin(th2)   # total x displacement
            dy = arrLen * np.cos(th2)
            x_start = ix - dx / 2
            y_start = iy - dy / 2
            ax.add_patch(makeArrow(x_start,dx,y_start,dy,color))

### Plot axis
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Arc
for axDir in range(2):
    x_ = 0.1 if axDir==0 else 0.83
    y_ = -0.0
    w_ = 0.1
    h_ = w_
    ax = fig.add_axes([x_,y_,w_,h_])
    # Move spines to the center
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Hide the top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Add arrows at the end of axes (optional)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=2)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=2)

    # Labels
    ax.set_xlabel('x', loc='right',labelpad=2)
    ax.set_ylabel('z',rotation=0,labelpad=5)
    # Arrows
    ax.add_patch(makeArrow(0,1,0,0,colAr[0+2*axDir],zorder=10))
    dx = np.sin(ths[1+2*axDir])
    dy = np.cos(ths[1+2*axDir])
    ax.add_patch(makeArrow(0,dx,0,dy,colAr[1+2*axDir],zorder=10))
    # Arc
    arc = Arc(
        (0, 0),
        0.8, 0.8,
        angle=0,
        theta1=0,
        theta2=90,#90-ths[1+2*axDir]/np.pi*180,
        color=colAr[0+2*axDir],
        lw=1,
        ls=(0,(0.8,0.5)),
        zorder=0
    )
    ax.add_patch(arc)
    arc = Arc(
        (0, 0),
        1.1, 1.1,
        angle=0,
        theta1=45,
        theta2=90,#90-ths[1+2*axDir]/np.pi*180,
        color=colAr[1+2*axDir],
        lw=1,
        ls=(0,(0.8,0.5)),
        zorder=0
    )
    ax.add_patch(arc)
    # Axis
    ax.set_xlim(-0.2,1.5)
    ax.set_ylim(-0.2,1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# Adjust figure
plt.subplots_adjust(
#    bottom = 0.062,
    #top = 0.926,
#    top = 0.847,
    right = 1,
    left = 0,
#    wspace=0.15,
)

if final:
    plt.savefig(
        "Figures/classicalMagnetization.pdf",
        bbox_inches="tight"
    )

#plt.show()











