"""Plotting functions related to classical spin solutions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow
import matplotlib.patheffects as pe
from wavespin.tools import pathFinder as pf
from wavespin.plots import latticePlots


def addAngleArrows(ax, lattice, thetas, *, arrowColor='royalblue'):
    """Add spin quantization angle arrows to an existing lattice axis.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis to draw arrows on.
    lattice : latticeClass
        Lattice geometry (provides site coordinates via ``_xy``).
    thetas : np.ndarray
        Quantization angle for each site.
    arrowColor : any matplotlib colour
        Colour of the arrows.
    """
    a = 0.8
    for i in range(lattice.Ns):
        x, y = lattice._xy(i)
        th = thetas[i]
        dx = a * np.sin(th)
        dy = a * np.cos(th)
        x_start = x - dx / 2
        y_start = y - dy / 2
        arrow = FancyArrow(
            x_start, y_start, dx, dy,
            width=0.1,
            length_includes_head=True,
            head_width=0.3,
            head_length=0.3,
            color=arrowColor,
            path_effects=[pe.withSimplePatchShadow(offset=(2, -2), alpha=0.9)],
        )
        ax.add_patch(arrow)


def plotLatticeWithAngles(lattice, thetas, *,
                          arrowColor='royalblue',
                          indices=False,
                          sublatticeColors=False,
                          perturbationSite=None,
                          boundary='auto',
                          figsize=(12, 12),
                          ax=None,
                          filename=None,
                          show=True):
    """Plot lattice with spin quantization angle arrows overlaid.

    Parameters
    ----------
    lattice : latticeClass
        The lattice object to plot.
    thetas : np.ndarray
        Quantization angles for each site.
    arrowColor : any matplotlib colour
        Colour of the arrows.
    indices : bool
        Show site indices.
    sublatticeColors : bool
        Colour sites by sublattice.
    perturbationSite : (int, int) or None
        Highlight this coordinate.
    boundary : bool or 'auto'
        Passed through to :func:`latticePlots.plotLattice`.
    figsize : (float, float)
        Figure size.
    ax : matplotlib Axes or None
        Plot on an existing axis.
    filename : str or None
        Save figure to this path.
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    fig, ax
    """
    fig, ax = latticePlots.plotLattice(
        lattice,
        indices=indices,
        sublatticeColors=sublatticeColors,
        perturbationSite=perturbationSite,
        boundary=boundary,
        figsize=figsize,
        ax=ax,
        show=False,
    )
    addAngleArrows(ax, lattice, thetas, arrowColor=arrowColor)

    if filename is not None:
        fig.tight_layout()
        fig.savefig(filename)

    if show:
        plt.show()

    return fig, ax


def plotQuantizationAngles(sim, thetas, phis, **kwargs):
    """Plot theta and phi of each site of the lattice."""
    verbose = kwargs.get('verbose', False)
    Lx, Ly = (sim.Lx, sim.Ly)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(121, projection='3d')
    func = thetas.reshape(Lx, Ly)
    X, Y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij')
    ax.plot_surface(X, Y, func, cmap='plasma')

    ax = fig.add_subplot(122, projection='3d')
    for i in range(sim.Ns):
        ix, iy = sim._xy(i)
        if (ix + iy) % 2 == 1:
            thetas[i] += np.pi
    func2 = thetas.reshape(Lx, Ly)
    ax.plot_surface(X, Y,
                    func2,
                    cmap='plasma',
                    alpha=0.8
                    )
    ax.plot_surface(X, Y,
                    np.ones((Lx, Ly)) * sim.periodicTheta,
                    color='g',
                    alpha=0.4
                    )
    if sim.p.savePlotSolution:
        argsFn = (sim.txtSim + '_solution', sim.Lx, sim.Ly, sim.Ns,
                  sim.g1, sim.g2, sim.d1, sim.d2, sim.h, sim.boundary)
        figureFn = pf.getFilename(*argsFn, dirname=sim.figureDn, extension='.png')
        Path(figureFn).parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print("Saving picture to file: " + figureFn)
        fig.savefig(figureFn)

    showFig = kwargs.get('showFigure', False)
    if showFig:
        plt.show()
