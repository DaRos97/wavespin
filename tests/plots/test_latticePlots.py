import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

from wavespin.tools.inputUtils import LatticeParams
from wavespin.lattice.lattice import latticeClass
from wavespin.lattice.presets import PRESETS, get_preset
from wavespin.plots import latticePlots, classicPlots


def _obc_lattice(Lx=6, Ly=4, offSiteList=()):
    return latticeClass(LatticeParams(Lx=Lx, Ly=Ly, offSiteList=offSiteList))


def _pbc_lattice(Lx=6, Ly=4):
    return latticeClass(LatticeParams(Lx=Lx, Ly=Ly, boundary='periodic'))


class TestPlotLattice:

    def test_returns_fig_ax(self):
        lattice = _obc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_show_false_does_not_crash(self):
        lattice = _obc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, show=False)
        plt.close(fig)

    def test_indices_option(self):
        lattice = _obc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, indices=True, show=False)
        plt.close(fig)

    def test_sublattice_colors_option(self):
        lattice = _obc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, sublatticeColors=True,
                                            show=False)
        plt.close(fig)

    def test_perturbation_site_option(self):
        lattice = _obc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, perturbationSite=(2, 1),
                                            show=False)
        plt.close(fig)

    def test_boundary_auto_obc(self):
        lattice = _obc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, boundary='auto', show=False)
        plt.close(fig)

    def test_boundary_auto_pbc(self):
        lattice = _pbc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, boundary='auto', show=False)
        plt.close(fig)

    def test_boundary_false_override(self):
        lattice = _pbc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, boundary=False, show=False)
        plt.close(fig)

    def test_boundary_true_on_obc(self):
        lattice = _obc_lattice()
        fig, ax = latticePlots.plotLattice(lattice, boundary=True, show=False)
        plt.close(fig)

    def test_existing_ax(self):
        lattice = _obc_lattice()
        fig, ax = plt.subplots()
        result = latticePlots.plotLattice(lattice, ax=ax, show=False)
        assert result[1] is ax
        plt.close(fig)

    def test_filename_saves_file(self, tmp_path):
        lattice = _obc_lattice()
        fname = str(tmp_path / "test.png")
        latticePlots.plotLattice(lattice, filename=fname, show=False)
        assert (tmp_path / "test.png").exists()

    def test_all_diamond_presets(self):
        for name in PRESETS:
            lattice = get_preset(name)
            fig, ax = latticePlots.plotLattice(lattice, show=False)
            plt.close(fig)


class TestAddAngleArrows:

    def test_adds_patches_to_axes(self):
        lattice = _obc_lattice(Lx=4, Ly=4)
        thetas = np.linspace(0, np.pi, lattice.Ns)
        fig, ax = plt.subplots()
        n_patches_before = len(ax.patches)
        classicPlots.addAngleArrows(ax, lattice, thetas)
        assert len(ax.patches) > n_patches_before
        plt.close(fig)

    def test_arrow_color_respected(self):
        lattice = _obc_lattice(Lx=4, Ly=4)
        thetas = np.linspace(0, np.pi, lattice.Ns)
        fig, ax = plt.subplots()
        classicPlots.addAngleArrows(ax, lattice, thetas, arrowColor='red')
        for p in ax.patches:
            if isinstance(p, FancyArrow):
                from matplotlib.colors import to_rgba
                assert to_rgba(p.get_facecolor()) == to_rgba('red')


class TestPlotLatticeWithAngles:

    def test_returns_fig_ax(self):
        lattice = _obc_lattice(Lx=4, Ly=4)
        thetas = np.linspace(0, np.pi, lattice.Ns)
        fig, ax = classicPlots.plotLatticeWithAngles(lattice, thetas, show=False)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_passes_boundary_through(self):
        lattice = _pbc_lattice(Lx=4, Ly=4)
        thetas = np.linspace(0, np.pi, lattice.Ns)
        fig, ax = classicPlots.plotLatticeWithAngles(lattice, thetas,
                                                      boundary='auto', show=False)
        plt.close(fig)

    def test_off_site_lattice(self):
        lattice = _obc_lattice(Lx=5, Ly=5, offSiteList=((2, 2), (1, 3)))
        thetas = np.linspace(0, np.pi, lattice.Ns)
        fig, ax = classicPlots.plotLatticeWithAngles(lattice, thetas, show=False)
        plt.close(fig)
