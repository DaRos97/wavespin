"""Demonstrate lattice plotting with ``plotLattice``, ``addAngleArrows``,
and ``plotLatticeWithAngles``.
"""

import numpy as np
from wavespin.tools.inputUtils import LatticeParams
from wavespin.lattice.lattice import latticeClass
from wavespin.plots import latticePlots
from wavespin.plots import classicPlots


def demo1_basic():
    """Plain lattice with defaults."""
    params = LatticeParams(Lx=6, Ly=4)
    lattice = latticeClass(params)
    fig, ax = latticePlots.plotLattice(lattice, show=False)
    fig.savefig(lattice.figureDn + "demo1_basic.png")
    print("Saved", lattice.figureDn + "demo1_basic.png")


def demo2_off_sites_and_indices():
    """Lattice with missing sites and index labels."""
    params = LatticeParams(Lx=6, Ly=5, offSiteList=((2, 2), (3, 1), (4, 3)))
    lattice = latticeClass(params)
    fig, ax = latticePlots.plotLattice(lattice, indices=True, show=False)
    fig.savefig(lattice.figureDn + "demo2_offSites_indices.png")
    print("Saved", lattice.figureDn + "demo2_offSites_indices.png")


def demo3_sublattice_colors():
    """Lattice with checkerboard (sublattice) colours and perturbation site."""
    params = LatticeParams(Lx=6, Ly=5, offSiteList=((2, 2), (3, 1)))
    lattice = latticeClass(params)
    fig, ax = latticePlots.plotLattice(
        lattice,
        sublatticeColors=True,
        perturbationSite=(4, 2),
        show=False,
    )
    fig.savefig(lattice.figureDn + "demo3_sublattice.png")
    print("Saved", lattice.figureDn + "demo3_sublattice.png")


def demo4_angle_arrows():
    """Lattice with spin angle arrows overlaid."""
    params = LatticeParams(Lx=6, Ly=5, offSiteList=((2, 2), (3, 1)))
    lattice = latticeClass(params)
    thetas = np.linspace(0, 2 * np.pi, lattice.Ns)
    fig, ax = classicPlots.plotLatticeWithAngles(
        lattice, thetas,
        sublatticeColors=True,
        arrowColor='crimson',
        show=False,
    )
    fig.savefig(lattice.figureDn + "demo4_angles.png")
    print("Saved", lattice.figureDn + "demo4_angles.png")


def demo5_composable():
    """Composable usage: ``plotLattice`` as base, layer arrows manually."""
    params = LatticeParams(Lx=6, Ly=4)
    lattice = latticeClass(params)
    thetas = np.linspace(0, np.pi, lattice.Ns)

    fig, ax = latticePlots.plotLattice(lattice, indices=True, show=False)
    classicPlots.addAngleArrows(ax, lattice, thetas, arrowColor='forestgreen')
    fig.savefig(lattice.figureDn + "demo5_composable.png")
    print("Saved", lattice.figureDn + "demo5_composable.png")


if __name__ == "__main__":
    demo1_basic()
    demo2_off_sites_and_indices()
    demo3_sublattice_colors()
    demo4_angle_arrows()
    demo5_composable()
