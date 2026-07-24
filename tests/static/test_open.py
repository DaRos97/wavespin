import numpy as np
import pytest

from wavespin.tools.inputUtils import SimParams, LatticeParams, DiagParams
from wavespin.static.open import openHamiltonian


def _make_params(Lx=4, Ly=4, g1=5, g2=0, d1=0, d2=0, h=0, h_disorder=0, boundary='open'):
    return SimParams(
        lattice=LatticeParams(Lx=Lx, Ly=Ly, boundary=boundary),
        diag=DiagParams(Hamiltonian=(g1, g2, d1, d2, h, h_disorder)),
    )


def _make_system(**kwargs):
    return openHamiltonian(_make_params(**kwargs))


# ---------------------------------------------------------------------------
# P0 — Critical invariants
# ---------------------------------------------------------------------------

class TestConstruction:
    """Smoke tests for openHamiltonian instantiation."""

    def test_shapes(self):
        s = _make_system(Lx=4, Ly=4)
        assert s.Ns == 16
        assert s.evals.shape == (16,)
        assert s.U_.shape == (16, 16)
        assert s.V_.shape == (16, 16)
        assert s.Phi.shape == (16, 16)
        assert s.thetas.shape == (16,)
        assert s.Ps.shape == (2, 16, 16, 3, 3)

    def test_g1_g2_detects_order(self):
        s_neel = _make_system(g1=5, g2=0)
        assert s_neel.order == 'canted-Neel'

    def test_stripe_order_on_small_lattice_is_unstable(self):
        """Canted-stripe on a small open lattice produces a non-positive-
        definite Hamiltonian — this is a known physics limitation."""
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            _make_system(Lx=4, Ly=4, g1=5, g2=3, h=0)

    def test_gse_not_set_when_g1_zero(self):
        s = _make_system(g1=0)
        assert not hasattr(s, 'GSE')

    def test_gse_set_when_g1_nonzero(self):
        s = _make_system(g1=5)
        assert hasattr(s, 'GSE')
        assert np.isfinite(s.GSE)


class TestParaUnitarity:
    """Bogoliubov transformation should satisfy U†U − V†V = I."""

    def test_unitarity(self):
        s = _make_system(Lx=3, Ly=3, g1=5)
        U = s.U_
        V = s.V_
        diff = U.T.conj() @ U - V.T.conj() @ V
        assert np.allclose(np.eye(s.Ns)[1:, 1:], diff[1:, 1:], atol=1e-10)


class TestHermitianHamiltonian:
    """The real-space BdG Hamiltonian must be Hermitian."""

    def test_hermitian(self):
        s = _make_system(Lx=3, Ly=3, g1=5)
        H = s._realSpaceHamiltonian()
        assert np.allclose(H, H.T.conj(), atol=1e-10)


class TestZeroMode:
    """Goldstone theorem: zero mode when staggered field is absent."""

    def test_zero_mode_without_field(self):
        s = _make_system(Lx=4, Ly=4, g1=5, h=0)
        assert s.evals[0] < 1e-8

    def test_gap_opens_with_field(self):
        s0 = _make_system(Lx=4, Ly=4, g1=5, h=0)
        s1 = _make_system(Lx=4, Ly=4, g1=5, h=1.0)
        assert s1.evals[0] > 1e-8
        assert s1.evals[0] > s0.evals[0]

    def test_gap_grows_with_h(self):
        prev = -1
        for h in [0, 0.5, 1.0]:
            s = _make_system(Lx=4, Ly=4, g1=5, h=h)
            assert s.evals[0] > prev - 1e-10
            prev = s.evals[0]


class TestEvalsNonNegative:
    """All quasiparticle energies must be non-negative."""

    def test_all_nonnegative(self):
        s = _make_system(Lx=4, Ly=4, g1=5, h=1.0)
        assert np.all(s.evals >= -1e-10)


# ---------------------------------------------------------------------------
# P1 — Important behaviour
# ---------------------------------------------------------------------------

class TestNNTerms:
    """_NNterms should populate only nearest-neighbour bonds."""

    def test_structure(self):
        s = _make_system(Lx=3, Ly=3)
        mat = s._NNterms(1.0)
        assert mat.shape == (9, 9)
        assert np.allclose(np.diag(mat), 0)  # no on-site
        assert np.allclose(mat, mat.T)       # undirected
        # Each interior site (index 4) has exactly 4 neighbours
        assert mat[4].sum() == 4.0

    def test_value(self):
        s = _make_system(Lx=3, Ly=3)
        mat = s._NNterms(3.0)
        assert np.all((mat == 0) | (mat == 3.0))


class TestNNNTerms:
    """_NNNterms should populate only next-nearest-neighbour bonds."""

    def test_structure(self):
        s = _make_system(Lx=4, Ly=4)
        mat = s._NNNterms(1.0)
        assert mat.shape == (16, 16)
        assert np.allclose(np.diag(mat), 0)
        assert np.allclose(mat, mat.T)
        assert mat[5].sum() == 4.0  # interior site has 4 NNN


class TestHterms:
    """_Hterms should produce a staggered diagonal pattern."""

    def test_no_disorder(self):
        s = _make_system(Lx=2, Ly=2)
        mat = s._Hterms(1.0, 0.0)
        assert mat.shape == (4, 4)
        diag = np.diag(mat)
        # (0,0): -1, (1,0): +1, (0,1): +1, (1,1): -1
        assert diag[0] == -1.0
        assert diag[1] == 1.0
        assert diag[2] == 1.0
        assert diag[3] == -1.0

    def test_disorder_is_nonzero(self):
        s = _make_system(Lx=4, Ly=4)
        mat = s._Hterms(1.0, 0.5)
        diag = np.diag(mat)
        expected = np.array(
            [(-1) ** (x + y + 1) * 1.0 for y in range(4) for x in range(4)]
        )
        assert not np.allclose(diag, expected)  # disorder shifts values


class TestPhiIsReal:
    """Phi should be real-valued."""

    def test_real(self):
        s = _make_system(Lx=4, Ly=4, g1=5, h=1.0)
        assert np.allclose(s.Phi.imag, 0, atol=1e-12)


class TestDiskCaching:
    """Second construction with same params should reload from cache."""

    def test_cache_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            'wavespin.tools.pathFinder.getHomeDirname',
            lambda cwd, sub: str(tmp_path) + '/' + sub.lstrip('/'),
        )
        import os
        os.makedirs(str(tmp_path / 'data'), exist_ok=True)

        s1 = _make_system(Lx=4, Ly=4, g1=5, h=1.0)
        evals1 = s1.evals.copy()

        s2 = _make_system(Lx=4, Ly=4, g1=5, h=1.0)
        assert np.allclose(evals1, s2.evals, atol=1e-12)


# ---------------------------------------------------------------------------
# P2 — Edge cases & physics
# ---------------------------------------------------------------------------

class TestThetaCheckerboard:
    """Canted-Néel order should produce a π-alternating theta pattern."""

    def test_neel_alternation(self):
        s = _make_system(Lx=4, Ly=4, g1=5, g2=0, h=0)
        for i in range(s.Ns):
            x, y = s._xy(i)
            expected = s.theta + np.pi * ((x + y) % 2)
            assert np.allclose(s.thetas[i] % (2 * np.pi), expected % (2 * np.pi), atol=1e-10)

    def test_stripe_order_on_small_lattice_is_unstable(self):
        """Canted-stripe on a small open lattice produces a non-positive-
        definite Hamiltonian — a known physics limitation of the stripe
        phase on open boundaries."""
        with pytest.raises(np.linalg.LinAlgError):
            _make_system(Lx=4, Ly=4, g1=5, g2=3, h=0)


class TestGSE:
    """Ground-state energy should be finite and repeatable."""

    def test_finite(self):
        s = _make_system(Lx=4, Ly=4, g1=5, h=1.0)
        assert np.isfinite(s.GSE)

    def test_stable(self):
        s = _make_system(Lx=4, Ly=4, g1=5, h=1.0)
        gse1 = s.get_GSE()
        gse2 = s.get_GSE()
        assert gse1 == gse2


class Test1DChain:
    """Works on a 1×N lattice (1D chain)."""

    def test_construction(self):
        s = _make_system(Lx=1, Ly=5, g1=5)
        assert s.Ns == 5
        assert s.evals.shape == (5,)
        assert s.NNN[0] == []  # no NNN in 1×N


class TestOffSiteGeometry:
    """Works on a diamond-shaped preset."""

    def test_diamond(self):
        from wavespin.lattice.presets import get_preset
        lat = get_preset('24-diamond')
        params = SimParams(
            lattice=lat.p,
            diag=DiagParams(Hamiltonian=(5, 0, 0, 0, 0, 0)),
        )
        s = openHamiltonian(params)
        assert s.Ns == lat.Ns
        assert s.evals.shape == (lat.Ns,)
        assert s.Phi.shape == (lat.Ns, lat.Ns)


class TestExcludeZeroMode:
    """When excludeZeroMode=True, the zero-mode column of U_/V_ is zeroed out."""

    def test_zero_mode_excluded(self):
        s = _make_system(Lx=4, Ly=4, g1=5, h=0)
        assert np.allclose(s.U_[:, 0], 0)
        assert np.allclose(s.V_[:, 0], 0)

    def test_zero_mode_included(self):
        params = _make_params(Lx=4, Ly=4, g1=5, h=0)
        params.diag.excludeZeroMode = False
        s = openHamiltonian(params)
        assert not np.allclose(s.U_[:, 0], 0)
        assert not np.allclose(s.V_[:, 0], 0)
