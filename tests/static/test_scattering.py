import numpy as np
import pytest

from wavespin.tools.inputUtils import SimParams, LatticeParams, DiagParams, ScatteringParams
from wavespin.static.open import openHamiltonian


def _make_system(Lx=3, Ly=3, g1=5, h=0, T=0, types=('1to2_1',)):
    params = SimParams(
        lattice=LatticeParams(Lx=Lx, Ly=Ly),
        diag=DiagParams(Hamiltonian=(g1, 0, 0, 0, h, 0)),
        scattering=ScatteringParams(types=types, temperature=T),
    )
    return openHamiltonian(params)


# ---------------------------------------------------------------------------
# Vertex tests
# ---------------------------------------------------------------------------

class TestVertexShapes:
    """Vertex tensors have the correct rank and size."""

    def test_1to2_shape(self):
        s = _make_system()
        s.computeVertex('1to2')
        assert s.vertex1to2.shape == (s.Ns, s.Ns, s.Ns)

    def test_2to2_shape(self):
        s = _make_system()
        s.computeVertex('2to2')
        assert s.vertex2to2.shape == (s.Ns, s.Ns, s.Ns, s.Ns)

    def test_1to3_shape(self):
        s = _make_system()
        s.computeVertex('1to3')
        assert s.vertex1to3.shape == (s.Ns, s.Ns, s.Ns, s.Ns)


class TestVertexSymmetry:
    """Vertex tensors are properly symmetrized."""

    def test_1to2_lm_symmetry(self):
        s = _make_system()
        s.computeVertex('1to2')
        V = s.vertex1to2
        for n in range(min(s.Ns, 3)):
            diff = V[n] - V[n].T
            assert np.allclose(diff, 0, atol=1e-12)

    def test_2to2_nl_symmetry(self):
        s = _make_system()
        s.computeVertex('2to2')
        V = s.vertex2to2
        diff_nl = V - np.transpose(V, (1, 0, 2, 3))
        assert np.allclose(diff_nl, 0, atol=1e-12)

    def test_2to2_mp_symmetry(self):
        s = _make_system()
        s.computeVertex('2to2')
        V = s.vertex2to2
        diff_mp = V - np.transpose(V, (0, 1, 3, 2))
        assert np.allclose(diff_mp, 0, atol=1e-12)

    def test_1to3_lmp_symmetry(self):
        s = _make_system()
        s.computeVertex('1to3')
        V = s.vertex1to3
        diff = V - np.transpose(V, (0, 2, 1, 3))
        assert np.allclose(diff, 0, atol=1e-12)


class TestVertexReal:
    """Vertex tensors are real-valued."""

    def test_real_1to2(self):
        s = _make_system()
        s.computeVertex('1to2')
        assert np.allclose(s.vertex1to2.imag, 0, atol=1e-12)

    def test_real_2to2(self):
        s = _make_system()
        s.computeVertex('2to2')
        assert np.allclose(s.vertex2to2.imag, 0, atol=1e-12)

    def test_real_1to3(self):
        s = _make_system()
        s.computeVertex('1to3')
        assert np.allclose(s.vertex1to3.imag, 0, atol=1e-12)


class TestVertexCaching:
    """Second computation returns identical result (disk cache)."""

    def test_cached_identical(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            'wavespin.tools.pathFinder.getHomeDirname',
            lambda cwd, sub: str(tmp_path) + '/' + sub.lstrip('/'),
        )
        import os
        os.makedirs(str(tmp_path / 'data'), exist_ok=True)

        s1 = _make_system()
        s1.computeVertex('1to2')
        v1 = s1.vertex1to2.copy()

        s2 = _make_system()
        s2.computeVertex('1to2')
        v2 = s2.vertex1to2

        assert np.allclose(v1, v2, atol=1e-12)


# ---------------------------------------------------------------------------
# Rate tests
# ---------------------------------------------------------------------------

class TestRateShape:
    """Rates have shape (Ns-1,) (zero mode excluded)."""

    def test_1to2_1(self):
        s = _make_system(types=('1to2_1',))
        s.computeRate()
        assert s.rates['1to2_1'].shape == (s.Ns - 1,)

    def test_1to2_2(self):
        s = _make_system(types=('1to2_2',))
        s.computeRate()
        assert s.rates['1to2_2'].shape == (s.Ns - 1,)

    def test_2to2_1(self):
        s = _make_system(types=('2to2_1',), T=1)
        s.computeRate()
        assert s.rates['2to2_1'].shape == (s.Ns - 1,)

    def test_1to3_1(self):
        s = _make_system(types=('1to3_1',))
        s.computeRate()
        assert s.rates['1to3_1'].shape == (s.Ns - 1,)


class TestRateNonNegative:
    """All rates are non-negative."""

    def test_1to2_1(self):
        s = _make_system(types=('1to2_1',))
        s.computeRate()
        assert np.all(s.rates['1to2_1'] >= -1e-12)

    def test_2to2_1(self):
        s = _make_system(types=('2to2_1',), T=1)
        s.computeRate()
        assert np.all(s.rates['2to2_1'] >= -1e-12)

    def test_1to3_1(self):
        s = _make_system(types=('1to3_1',))
        s.computeRate()
        assert np.all(s.rates['1to3_1'] >= -1e-12)


class TestRateReal:
    """Rates are real-valued."""

    def test_real(self):
        s = _make_system(types=('1to2_1',))
        s.computeRate()
        assert np.allclose(s.rates['1to2_1'].imag, 0, atol=1e-12)


class TestRateTemperatureDependence:
    """Temperature effects are correct."""

    def test_T0_gives_zero_2to2(self):
        s = _make_system(types=('2to2_1',), T=0)
        s.computeRate()
        assert np.allclose(s.rates['2to2_1'], 0)

    def test_T_positive_gives_nonzero_2to2(self):
        s = _make_system(types=('2to2_1',), T=5)
        s.computeRate()
        assert not np.allclose(s.rates['2to2_1'], 0)

    def test_1to2_has_finite_T0_rate(self):
        s = _make_system(types=('1to2_1',), T=0)
        s.computeRate()
        assert s.rates['1to2_1'][0] >= 0

    def test_1to2_grows_with_temperature(self):
        s0 = _make_system(types=('1to2_1',), T=0)
        s0.computeRate()
        s5 = _make_system(types=('1to2_1',), T=5)
        s5.computeRate()
        assert np.any(s5.rates['1to2_1'] >= s0.rates['1to2_1'])


class TestRateAllFinite:
    """No NaN or Inf in any rate."""

    def test_finite(self):
        s = _make_system(types=('1to2_1', '2to2_1', '1to3_1'), T=1)
        s.computeRate()
        for r in s.rates.values():
            assert np.all(np.isfinite(r))


class TestRateDictPopulated:
    """All requested processes appear in rates dict."""

    def test_all_entries(self):
        requested = ('1to2_1', '2to2_1', '1to3_1')
        s = _make_system(types=requested, T=1)
        s.computeRate()
        for proc in requested:
            assert proc in s.rates
