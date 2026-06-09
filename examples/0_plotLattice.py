""" Plot and save example lattice geometries for each available boundary condition
and diamond-shaped preset geometry.

This script generates PNG figures saved to the ``Figures/`` directory showing:
  - Open boundary condition (OBC) rectangular lattice
  - Periodic boundary condition (PBC) rectangular lattice with ghost sites and wrap bonds
  - All available diamond-shaped OBC preset geometries

The plots display site labels, coordinate axes, nearest-neighbor and next-nearest-neighbor
bonds, and the off-site (removed) regions for shaped geometries.

Usage
-----
    python 0_plotLatticeDemo.py

No input file or command-line arguments are required --- lattice parameters are
hard-coded for demonstration purposes.
"""

from wavespin.lattice.presets import PRESETS, get_preset
from wavespin.tools.inputUtils import LatticeParams
from wavespin.lattice.lattice import latticeClass
from wavespin.plots import latticePlots


def demo_obc():
    """Plot and save an open boundary condition rectangular lattice."""
    params = LatticeParams(Lx=6, Ly=4)
    lattice = latticeClass(params)
    fig, ax = latticePlots.plotLattice(lattice, show=False)
    fig.savefig(lattice.figureDn + "demo_obc.png")
    print("Saved", lattice.figureDn + "demo_obc.png")


def demo_pbc():
    """Plot and save a periodic boundary condition lattice with ghost sites and wrap bonds."""
    params = LatticeParams(Lx=6, Ly=4, boundary='periodic')
    lattice = latticeClass(params)
    fig, ax = latticePlots.plotLattice(lattice, show=False)
    fig.savefig(lattice.figureDn + "demo_pbc.png")
    print("Saved", lattice.figureDn + "demo_pbc.png")


def demo_diamonds():
    """Plot and save all available diamond-shaped OBC preset geometries."""
    for name in PRESETS:
        lattice = get_preset(name)
        fig, ax = latticePlots.plotLattice(lattice, show=False)
        fig.savefig(lattice.figureDn + f"demo_{name}.png")
        print("Saved", lattice.figureDn + f"demo_{name}.png")


if __name__ == "__main__":
    demo_obc()
    demo_pbc()
    demo_diamonds()
