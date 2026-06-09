"""Predefined lattice geometries — diamond-shaped open-boundary lattices.

Each entry is a ``(Lx, Ly, offSiteList)`` tuple suitable for
:class:`~wavespin.tools.inputUtils.LatticeParams`.
"""

from wavespin.tools.inputUtils import LatticeParams
from wavespin.lattice.lattice import latticeClass


def _diamond_offSites(L):
    """Return the ``offSiteList`` for an L×L diamond (L even).

    Crops the four corners: sites whose Manhattan distance from a
    corner is strictly less than ``L/2 - 1``.
    """
    if L % 2 != 0:
        raise ValueError(f"Diamond requires even L, got {L}")
    h = L // 2 - 1
    if h <= 0:
        return ()
    off = []
    for x in range(L):
        for y in range(L):
            if (x + y < h or
                x + (L - 1 - y) < h or
                (L - 1 - x) + y < h or
                (L - 1 - x) + (L - 1 - y) < h):
                off.append((x, y))
    return tuple(off)


PRESETS: dict[str, tuple[int, int, tuple]] = {
    '12-diamond':  (4,  4,  _diamond_offSites(4)),
    '24-diamond':  (6,  6,  _diamond_offSites(6)),
    '40-diamond':  (8,  8,  _diamond_offSites(8)),
    '60-diamond':  (10, 10, _diamond_offSites(10)),
    '84-diamond':  (12, 12, _diamond_offSites(12)),
    '97-tilted-diamond': (
        13, 13,
        (
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (1, 8), (1, 9), (1, 10), (1, 11), (1, 12),
            (2, 0), (2, 1), (2, 2),
            (2, 9), (2, 10), (2, 11), (2, 12),
            (3, 0), (3, 1), (3, 10), (3, 11), (3, 12),
            (4, 0), (4, 11), (4, 12),
            (5, 12),
            (7, 0),
            (8, 0), (8, 1), (8, 12),
            (9, 0), (9, 1), (9, 2), (9, 11), (9, 12),
            (10, 0), (10, 1), (10, 2), (10, 3),
            (10, 10), (10, 11), (10, 12),
            (11, 0), (11, 1), (11, 2), (11, 3), (11, 4),
            (11, 9), (11, 10), (11, 11), (11, 12),
            (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5),
            (12, 8), (12, 9), (12, 10), (12, 11), (12, 12),
        ),
    ),
    '112-diamond': (14, 14, _diamond_offSites(14)),
    '144-diamond': (16, 16, _diamond_offSites(16)),
}


def get_preset(name, *, boundary='open', plotLattice=False):
    """Return a :class:`latticeClass` instance for a named preset geometry.

    Parameters
    ----------
    name : str
        Key in :data:`PRESETS` (e.g. ``'60-diamond'``).
    boundary : str
        ``'open'`` or ``'periodic'``.
    plotLattice : bool
        Passed to :class:`LatticeParams`.

    Returns
    -------
    latticeClass
    """
    Lx, Ly, offSiteList = PRESETS[name]
    params = LatticeParams(
        Lx=Lx, Ly=Ly, offSiteList=offSiteList,
        boundary=boundary, plotLattice=plotLattice,
    )
    return latticeClass(params)
