"""Lattice geometry plotting.
"""

import matplotlib.pyplot as plt
import numpy as np


def plotLattice(lattice, *,
                indices=False,
                sublatticeColors=False,
                perturbationSite=None,
                figsize=(12, 12),
                ax=None,
                filename=None,
                show=True):
    """Plot the square lattice grid with sites, bonds, and optional markers.

    Parameters
    ----------
    lattice : latticeClass
        The lattice object to plot.
    indices : bool
        If True, display site indices on each active site.
    sublatticeColors : bool
        If True, colour sites by sublattice (A/B checkerboard).
    perturbationSite : (int, int) or None
        If given, highlight this coordinate as a perturbation site.
    figsize : (float, float)
        Figure size in inches (only used when *ax* is None).
    ax : matplotlib Axes or None
        If provided, plot on this axis instead of creating a new figure.
    filename : str or None
        If provided, save the figure to this path.
    show : bool
        If True, call ``plt.show()``.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    offSiteSet = lattice.offSiteSet
    siteToIndex = lattice.siteToIndex
    Lx, Ly = lattice.Lx, lattice.Ly
    cols = ['b', 'c'] if sublatticeColors else ['k', 'k']

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    for ix in range(Lx):
        for iy in range(Ly):
            is_off = (ix, iy) in offSiteSet
            if is_off:
                ax.scatter(ix, iy, c='orange', marker='x', s=30, zorder=2)
            else:
                ax.scatter(ix, iy, c=cols[(ix + iy) % 2], marker='o', s=150, zorder=2)
                if indices:
                    ax.text(ix + 0.05, iy + 0.15,
                            str(siteToIndex[(ix, iy)]), size=20)
            if ix + 1 < Lx:
                if is_off or (ix + 1, iy) in offSiteSet:
                    ax.plot([ix, ix + 1], [iy, iy],
                            c='y', ls='--', lw=0.5, zorder=-1)
                else:
                    ax.plot([ix, ix + 1], [iy, iy],
                            c='k', ls='-', lw=2, zorder=-1)
            if iy + 1 < Ly:
                if is_off or (ix, iy + 1) in offSiteSet:
                    ax.plot([ix, ix], [iy, iy + 1],
                            c='y', ls='--', lw=0.5, zorder=-1)
                else:
                    ax.plot([ix, ix], [iy, iy + 1],
                            c='k', lw=2, zorder=-1)

    if perturbationSite is not None:
        ax.scatter(perturbationSite[0], perturbationSite[1],
                   c='w', edgecolor='m', lw=2, marker='o', s=300, zorder=1)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)

    if show:
        plt.show()

    return fig, ax
