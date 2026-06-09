"""Square lattice with nearest-neighbor and next-nearest-neighbor adjacency.

Defines ``latticeClass``, the base class for all simulations.  It builds an
:math:`L_x \\times L_y` grid of sites with optional missing (off-site)
positions, and precomputes nearest-neighbor (NN) and next-nearest-neighbor
(NNN) index lists for open or periodic boundary conditions.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wavespin.plots import fancyLattice
from wavespin.tools import pathFinder as pf

if TYPE_CHECKING:
    from wavespin.tools.inputUtils import LatticeParams


class latticeClass:
    """Square-lattice geometry with site indexing and precomputed neighbor lists.

    Parameters
    ----------
    p : LatticeParams
        Parameter dataclass containing ``Lx``, ``Ly``, ``offSiteList``,
        ``boundary``, and ``plotLattice``.

    Attributes
    ----------
    Lx, Ly : int
        Number of sites along x and y directions.
    Ns : int
        Total number of *active* sites (``Lx * Ly - len(offSiteList)``).
    offSiteList : tuple of (int, int)
        Coordinates of sites excluded from the lattice.
    offSiteSet : set of (int, int)
        Set version of ``offSiteList`` for O(1) membership checks.
    indexToSite : list of (int, int)
        Maps site index ``0..Ns-1`` → ``(x, y)`` coordinate.
    siteToIndex : dict mapping (int, int) → int
        Inverse of ``indexToSite``; O(1) coordinate → index lookup.
    boundary : str
        ``'open'`` or ``'periodic'``.
    NN : list of list of int
        ``NN[i]`` — nearest-neighbour indices of site *i*.
    NNN : list of list of int
        ``NNN[i]`` — next-nearest-neighbour indices of site *i*.
    dataDn, figureDn : str
        Absolute paths to ``Data/`` and ``Figures/`` output directories.

    Raises
    ------
    ValueError
        If ``Lx`` or ``Ly`` is smaller than 1.
    ValueError
        If all sites are excluded via ``offSiteList`` (``Ns == 0``).
    ValueError
        If ``boundary == 'periodic'`` and either ``offSiteList`` is non-empty
        or ``Lx``/``Ly`` is odd.
    """

    def __init__(self, p: LatticeParams) -> None:
        self.p = copy.copy(p)
        self.Lx: int = self.p.Lx
        self.Ly: int = self.p.Ly
        if self.Lx < 1 or self.Ly < 1:
            raise ValueError(
                f"Lx and Ly must be >= 1, got Lx={self.Lx}, Ly={self.Ly}"
            )
        self.offSiteList: tuple = self.p.offSiteList
        self.offSiteSet: set = set(self.offSiteList)
        self.indexToSite: list[tuple[int, int]] = self._mapIndexSite()
        self.siteToIndex: dict[tuple[int, int], int] = {
            site: idx for idx, site in enumerate(self.indexToSite)
        }
        self.Ns: int = self.Lx * self.Ly - len(self.offSiteList)
        if self.Ns < 1:
            raise ValueError(
                f"Lattice must contain at least one active site, "
                f"got Ns={self.Ns} (Lx={self.Lx}, Ly={self.Ly}, "
                f"offSiteList={len(self.offSiteList)})"
            )
        self.boundary: str = self.p.boundary
        if self.boundary == 'periodic':
            if len(self.offSiteList) != 0:
                raise ValueError(
                    "Periodic and non-square lattice not implemented"
                )
            if self.Lx % 2 or self.Ly % 2:
                raise ValueError(
                    "For a periodic boundary you need even Lx and Ly!"
                )

        self.NN: list[list[int]] = self._build_nn()
        self.NNN: list[list[int]] = self._build_nnn()

        self.dataDn: str = pf.getHomeDirname(str(Path.cwd()), '/Data/')
        Path(self.dataDn).mkdir(parents=True, exist_ok=True)
        self.figureDn: str = pf.getHomeDirname(str(Path.cwd()), '/Figures/')
        Path(self.figureDn).mkdir(parents=True, exist_ok=True)

        if p.plotLattice:
            fancyLattice.plotSitesGrid(self)

    def _xy(self, i: int) -> tuple[int, int]:
        """Return the ``(x, y)`` coordinate of site *i*."""
        return self.indexToSite[i]

    def _idx(self, x: int, y: int) -> int:
        """Return the site index for coordinate ``(x, y)``."""
        return self.siteToIndex[(x, y)]

    # ------------------------------------------------------------------
    # Nearest neighbours
    # ------------------------------------------------------------------

    def _build_nn(self) -> list[list[int]]:
        """Build the nearest-neighbour adjacency list for every site."""
        NN: list[list[int]] = [[] for _ in range(self.Ns)]
        for ind in range(self.Ns):
            if self.boundary == 'periodic':
                x, y = self._xy(ind)
                NN[ind] = [
                    self._idx((x + 1) % self.Lx, y),
                    self._idx((x - 1) % self.Lx, y),
                    self._idx(x, (y + 1) % self.Ly),
                    self._idx(x, (y - 1) % self.Ly),
                ]
            else:
                NN[ind] = self._open_nn(ind)
        return NN

    def _open_nn(self, ind: int) -> list[int]:
        """Nearest neighbours of site *ind* with open boundary conditions.

        Boundary and off-site checks exclude neighbours that would lie
        outside the lattice or on a removed site.
        """
        Lx, Ly = self.Lx, self.Ly
        ix, iy = self._xy(ind)
        result: list[int] = []
        off = self.offSiteSet
        sti = self.siteToIndex

        if ix != Lx - 1 and (ix + 1, iy) not in off:          # right
            result.append(sti[(ix + 1, iy)])
        if ix != 0 and (ix - 1, iy) not in off:               # left
            result.append(sti[(ix - 1, iy)])
        if iy != Ly - 1 and (ix, iy + 1) not in off:          # up
            result.append(sti[(ix, iy + 1)])
        if iy != 0 and (ix, iy - 1) not in off:               # down
            result.append(sti[(ix, iy - 1)])
        return result

    # ------------------------------------------------------------------
    # Next-nearest neighbours
    # ------------------------------------------------------------------

    def _build_nnn(self) -> list[list[int]]:
        """Build the next-nearest-neighbour adjacency list for every site."""
        NNN: list[list[int]] = [[] for _ in range(self.Ns)]
        for ind in range(self.Ns):
            if self.boundary == 'periodic':
                x, y = self._xy(ind)
                NNN[ind] = [
                    self._idx((x + 1) % self.Lx, (y + 1) % self.Ly),
                    self._idx((x - 1) % self.Lx, (y + 1) % self.Ly),
                    self._idx((x + 1) % self.Lx, (y - 1) % self.Ly),
                    self._idx((x - 1) % self.Lx, (y - 1) % self.Ly),
                ]
            else:
                NNN[ind] = self._open_nnn(ind)
        return NNN

    def _open_nnn(self, ind: int) -> list[int]:
        """Next-nearest neighbours of site *ind* with open boundary conditions.

        Boundary and off-site checks exclude NNNs that would lie outside
        the lattice or on a removed site.
        """
        Lx, Ly = self.Lx, self.Ly
        ix, iy = self._xy(ind)
        result: list[int] = []
        off = self.offSiteSet
        sti = self.siteToIndex

        if ix != Lx - 1 and iy != Ly - 1 and (ix + 1, iy + 1) not in off:
            result.append(sti[(ix + 1, iy + 1)])               # right-up
        if ix != 0 and iy != Ly - 1 and (ix - 1, iy + 1) not in off:
            result.append(sti[(ix - 1, iy + 1)])               # left-up
        if ix != Lx - 1 and iy != 0 and (ix + 1, iy - 1) not in off:
            result.append(sti[(ix + 1, iy - 1)])               # right-down
        if ix != 0 and iy != 0 and (ix - 1, iy - 1) not in off:
            result.append(sti[(ix - 1, iy - 1)])               # left-down
        return result

    # ------------------------------------------------------------------
    # Index ↔ coordinate mapping
    # ------------------------------------------------------------------

    def _mapIndexSite(self) -> list[tuple[int, int]]:
        """Build the ordered index → coordinate mapping.

        Iterates over the full :math:`L_x \\times L_y` grid, skipping
        coordinates that appear in ``offSiteSet``, and returns a list
        whose *i*-th entry is the ``(x, y)`` coordinate of site *i*.
        """
        return [
            (ix, iy)
            for ix in range(self.Lx)
            for iy in range(self.Ly)
            if (ix, iy) not in self.offSiteSet
        ]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def patchFunction(self, func: np.ndarray) -> np.ndarray:
        """Reshape a site-indexed array to the :math:`L_x \\times L_y` grid.

        Pads missing sites with ``NaN`` when ``offSiteList`` is non-empty.

        Parameters
        ----------
        func : np.ndarray of shape ``(Ns,)``
            Function defined on the active sites of the lattice.

        Returns
        -------
        np.ndarray
            ``(Lx, Ly)``-shaped array with ``NaN`` at off-site positions.
        """
        if not self.offSiteList:
            return func.reshape(self.Lx, self.Ly)

        formatted = np.full((self.Lx, self.Ly), np.nan)
        for i, (x, y) in enumerate(self.indexToSite):
            formatted[x, y] = func[i]
        return formatted
