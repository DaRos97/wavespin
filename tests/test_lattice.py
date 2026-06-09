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
    """Test 1: square lattice with open boundary conditions."""

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
    """Test 2: square lattice with periodic boundary conditions."""

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
