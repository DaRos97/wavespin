"""Demonstrate lattice plotting for open and periodic boundary conditions,
plus diamond-shaped preset geometries.
"""

from wavespin.lattice.presets import PRESETS, get_preset
from wavespin.tools.inputUtils import LatticeParams
from wavespin.lattice.lattice import latticeClass
from wavespin.plots import latticePlots


def demo_obc():
    """Open boundary condition lattice."""
    params = LatticeParams(Lx=6, Ly=4)
    lattice = latticeClass(params)
    fig, ax = latticePlots.plotLattice(lattice, show=False)
    fig.savefig(lattice.figureDn + "demo_obc.png")
    print("Saved", lattice.figureDn + "demo_obc.png")


def demo_pbc():
    """Periodic boundary condition lattice with ghost sites and wrap bonds."""
    params = LatticeParams(Lx=6, Ly=4, boundary='periodic')
    lattice = latticeClass(params)
    fig, ax = latticePlots.plotLattice(lattice, show=False)
    fig.savefig(lattice.figureDn + "demo_pbc.png")
    print("Saved", lattice.figureDn + "demo_pbc.png")


def demo_diamonds():
    """Diamond-shaped OBC preset geometries."""
    for name in PRESETS:
        lattice = get_preset(name)
        fig, ax = latticePlots.plotLattice(lattice, show=False)
        fig.savefig(lattice.figureDn + f"demo_{name}.png")
        print("Saved", lattice.figureDn + f"demo_{name}.png")


if __name__ == "__main__":
    demo_obc()
    demo_pbc()
    demo_diamonds()
