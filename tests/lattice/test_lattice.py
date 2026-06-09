import numpy as np
import pytest
from wavespin.tools.inputUtils import myParameters
from wavespin.lattice.lattice import latticeClass


def _make_params(Lx=3, Ly=4, boundary="open", offSiteList=()):
    p = myParameters()
    p.lat_Lx = Lx
    p.lat_Ly = Ly
    p.lat_boundary = boundary
    p.lat_offSiteList = offSiteList
    return p


class TestSquareOpen:
    """Square lattice with open boundary conditions."""

    def test_ns(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=4))
        assert lattice.Ns == 12

    def test_index_to_site_length(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=4))
        assert len(lattice.indexToSite) == 12

    def test_xy_idx_inverses(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=4))
        for i in range(lattice.Ns):
            x, y = lattice._xy(i)
            assert lattice._idx(x, y) == i

    def test_nn_counts(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=5))
        for i in range(lattice.Ns):
            x, y = lattice._xy(i)
            is_corner = (x == 0 or x == 3) and (y == 0 or y == 4)
            is_edge = (x == 0 or x == 3 or y == 0 or y == 4)
            if is_corner:
                assert len(lattice.NN[i]) == 2, f"corner site {i}=({x},{y}) has {len(lattice.NN[i])} NN, expected 2"
            elif is_edge:
                assert len(lattice.NN[i]) == 3, f"edge site {i}=({x},{y}) has {len(lattice.NN[i])} NN, expected 3"
            else:
                assert len(lattice.NN[i]) == 4, f"interior site {i}=({x},{y}) has {len(lattice.NN[i])} NN, expected 4"

    def test_nnn_counts(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=5))
        for i in range(lattice.Ns):
            x, y = lattice._xy(i)
            is_corner = (x == 0 or x == 3) and (y == 0 or y == 4)
            is_edge = (x == 0 or x == 3 or y == 0 or y == 4)
            if is_corner:
                assert len(lattice.NNN[i]) == 1, f"corner site {i}=({x},{y}) has {len(lattice.NNN[i])} NNN, expected 1"
            elif is_edge:
                assert len(lattice.NNN[i]) == 2, f"edge site {i}=({x},{y}) has {len(lattice.NNN[i])} NNN, expected 2"
            else:
                assert len(lattice.NNN[i]) == 4, f"interior site {i}=({x},{y}) has {len(lattice.NNN[i])} NNN, expected 4"

    def test_nn_symmetry(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=5))
        for i in range(lattice.Ns):
            for j in lattice.NN[i]:
                assert i in lattice.NN[j], f"NN asymmetry: {i}->{j} but {j} not in NN[{i}]"

    def test_nnn_symmetry(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=5))
        for i in range(lattice.Ns):
            for j in lattice.NNN[i]:
                assert i in lattice.NNN[j], f"NNN asymmetry: {i}->{j} but {j} not in NNN[{i}]"


class TestSquarePeriodic:
    """Square lattice with periodic boundary conditions."""

    def test_ns(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=6, boundary="periodic"))
        assert lattice.Ns == 24

    def test_every_site_has_4_nn(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=4, boundary="periodic"))
        for i in range(lattice.Ns):
            assert len(lattice.NN[i]) == 4, f"site {i} has {len(lattice.NN[i])} NN, expected 4"

    def test_every_site_has_4_nnn(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=4, boundary="periodic"))
        for i in range(lattice.Ns):
            assert len(lattice.NNN[i]) == 4, f"site {i} has {len(lattice.NNN[i])} NNN, expected 4"

    def test_nn_wraparound(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=4, boundary="periodic"))
        idx0 = lattice._idx(0, 0)
        neighbors = set(lattice._xy(j) for j in lattice.NN[idx0])
        assert (1, 0) in neighbors   # right
        assert (3, 0) in neighbors   # left wrap-around
        assert (0, 1) in neighbors   # up
        assert (0, 3) in neighbors   # down wrap-around

    def test_nnn_wraparound(self):
        lattice = latticeClass(_make_params(Lx=4, Ly=4, boundary="periodic"))
        idx0 = lattice._idx(0, 0)
        neighbors = set(lattice._xy(j) for j in lattice.NNN[idx0])
        assert (1, 1) in neighbors   # right-up
        assert (3, 1) in neighbors   # left-up wrap
        assert (1, 3) in neighbors   # right-down wrap
        assert (3, 3) in neighbors   # left-down wrap

    def test_odd_lx_raises(self):
        with pytest.raises(ValueError, match="even"):
            latticeClass(_make_params(Lx=3, Ly=4, boundary="periodic"))

    def test_odd_ly_raises(self):
        with pytest.raises(ValueError, match="even"):
            latticeClass(_make_params(Lx=4, Ly=3, boundary="periodic"))

    def test_off_site_with_periodic_raises(self):
        with pytest.raises(ValueError, match="square"):
            latticeClass(_make_params(Lx=4, Ly=4, boundary="periodic", offSiteList=((0, 0),)))


class TestOffSiteOpen:
    """Open-boundary lattice with removed (off-site) positions."""

    def off_sites(self):
        return ((1, 1), (2, 2))

    def test_ns(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=3, offSiteList=self.off_sites()))
        assert lattice.Ns == 7   # 9 - 2

    def test_off_sites_not_in_index_to_site(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=3, offSiteList=self.off_sites()))
        for off in self.off_sites():
            assert off not in lattice.indexToSite

    def test_off_sites_not_in_site_to_index(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=3, offSiteList=self.off_sites()))
        for off in self.off_sites():
            assert off not in lattice.siteToIndex

    def test_idx_raises_keyerror_for_off_site(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=3, offSiteList=self.off_sites()))
        for off in self.off_sites():
            with pytest.raises(KeyError):
                lattice._idx(*off)

    def test_nn_skips_off_sites(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=3, offSiteList=self.off_sites()))
        # Site (1, 0) sits on the bottom edge.
        # Its upper neighbour (1, 1) is off-site → excluded.
        # So it only keeps left (0,0) and right (2,0): 2 neighbours.
        idx_10 = lattice._idx(1, 0)
        nn_coords = {lattice._xy(j) for j in lattice.NN[idx_10]}
        assert (0, 0) in nn_coords
        assert (2, 0) in nn_coords
        assert (1, 1) not in nn_coords
        assert len(lattice.NN[idx_10]) == 2

    def test_nn_symmetry(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=3, offSiteList=self.off_sites()))
        for i in range(lattice.Ns):
            for j in lattice.NN[i]:
                assert i in lattice.NN[j], f"NN asymmetry: {i}->{j} but {j} not in NN[{i}]"

    def test_nnn_symmetry(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=3, offSiteList=self.off_sites()))
        for i in range(lattice.Ns):
            for j in lattice.NNN[i]:
                assert i in lattice.NNN[j], f"NNN asymmetry: {i}->{j} but {j} not in NNN[{i}]"


class TestPatchFunction:
    """Reshaping of site-indexed arrays to the Lx×Ly grid."""

    def test_rectangular(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=4))
        data = np.arange(lattice.Ns, dtype=float)
        result = lattice.patchFunction(data)
        assert result.shape == (3, 4)
        assert np.array_equal(result, data.reshape(3, 4))

    def test_with_off_sites(self):
        lattice = latticeClass(_make_params(Lx=3, Ly=3, offSiteList=((1, 1),)))
        data = np.arange(lattice.Ns, dtype=float)  # 0..7
        result = lattice.patchFunction(data)
        assert result.shape == (3, 3)
        assert np.isnan(result[1, 1])
        # All off-site positions should be NaN
        assert np.isnan(result[1, 1])
        # Active sites must contain the right values
        assert result[0, 0] == data[lattice._idx(0, 0)]
        assert result[2, 2] == data[lattice._idx(2, 2)]
        # No NaN on any active site
        for x, y in lattice.indexToSite:
            assert not np.isnan(result[x, y])


class TestDegenerateDimensions:
    """1×N and N×1 lattices (thin lines)."""

    def test_1xn_nn_counts(self):
        lattice = latticeClass(_make_params(Lx=1, Ly=5))
        for i in range(lattice.Ns):
            x, y = lattice._xy(i)
            is_edge = (y == 0 or y == 4)
            if is_edge:
                assert len(lattice.NN[i]) == 1, f"edge site {i}=({x},{y}) has {len(lattice.NN[i])} NN"
            else:
                assert len(lattice.NN[i]) == 2, f"interior site {i}=({x},{y}) has {len(lattice.NN[i])} NN"

    def test_1xn_nnn_counts(self):
        lattice = latticeClass(_make_params(Lx=1, Ly=5))
        for i in range(lattice.Ns):
            assert len(lattice.NNN[i]) == 0, f"site {i} has {len(lattice.NNN[i])} NNN, expected 0"

    def test_nx1_nn_counts(self):
        lattice = latticeClass(_make_params(Lx=5, Ly=1))
        for i in range(lattice.Ns):
            x, y = lattice._xy(i)
            is_edge = (x == 0 or x == 4)
            if is_edge:
                assert len(lattice.NN[i]) == 1, f"edge site {i}=({x},{y}) has {len(lattice.NN[i])} NN"
            else:
                assert len(lattice.NN[i]) == 2, f"interior site {i}=({x},{y}) has {len(lattice.NN[i])} NN"

    def test_nx1_nnn_counts(self):
        lattice = latticeClass(_make_params(Lx=5, Ly=1))
        for i in range(lattice.Ns):
            assert len(lattice.NNN[i]) == 0, f"site {i} has {len(lattice.NNN[i])} NNN, expected 0"


class TestValidation:
    """Invalid lattice configurations."""

    def test_lx_zero_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            latticeClass(_make_params(Lx=0, Ly=4))

    def test_ly_zero_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            latticeClass(_make_params(Lx=4, Ly=0))

    def test_all_sites_off_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            latticeClass(_make_params(Lx=2, Ly=2, offSiteList=((0, 0), (0, 1), (1, 0), (1, 1))))

    def test_lx_ly_one_no_off_sites(self):
        lattice = latticeClass(_make_params(Lx=1, Ly=1))
        assert lattice.Ns == 1
        assert lattice.NN[0] == []
        assert lattice.NNN[0] == []

    def test_p_is_independent(self):
        p = _make_params(Lx=3, Ly=4)
        lattice = latticeClass(p)
        p.lat_Lx = 100
        assert lattice.Lx == 3
        assert lattice.Ns == 12
