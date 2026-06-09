import pytest
from wavespin.lattice.presets import PRESETS, get_preset, _diamond_offSites
from wavespin.lattice.lattice import latticeClass


class TestPresetIntegrity:

    def test_all_names_are_strings(self):
        for name in PRESETS:
            assert isinstance(name, str)
            assert isinstance(PRESETS[name], tuple)
            assert len(PRESETS[name]) == 3

    def test_dimensions_match(self):
        for name, (Lx, Ly, _) in PRESETS.items():
            assert Lx == Ly, f"{name}: expected square lattice, got {Lx}×{Ly}"

    def test_ns_matches_expected(self):
        for name, (Lx, Ly, off) in PRESETS.items():
            lattice = get_preset(name)
            expected = Lx * Ly - len(off)
            assert lattice.Ns == expected, f"{name}: Ns mismatch"

    def test_off_sites_in_bounds(self):
        for name, (Lx, Ly, off) in PRESETS.items():
            for x, y in off:
                assert 0 <= x < Lx, f"{name}: off-site x={x} out of bounds"
                assert 0 <= y < Ly, f"{name}: off-site y={y} out of bounds"

    def test_no_duplicate_off_sites(self):
        for name, (Lx, Ly, off) in PRESETS.items():
            assert len(off) == len(set(off)), f"{name}: duplicate off-sites"

    def test_every_preset_creates_valid_lattice(self):
        for name in PRESETS:
            lattice = get_preset(name)
            assert isinstance(lattice, latticeClass)
            assert lattice.Ns > 0

    def test_even_l_symmetric_diamonds_match_generator(self):
        for name, (Lx, Ly, off) in PRESETS.items():
            if Lx % 2 == 0 and 'tilted' not in name:
                generated = _diamond_offSites(Lx)
                assert set(off) == set(generated), \
                    f"{name}: offSiteList does not match generator"

    def test_tilted_diamond_is_asymmetric(self):
        name = '97-tilted-diamond'
        assert name in PRESETS
        Lx, Ly, off = PRESETS[name]
        assert Lx % 2 == 1, f"{name}: expected odd L, got {Lx}"
        generated = _diamond_offSites(Lx) if Lx % 2 == 0 else None
        if generated is not None:
            assert set(off) != set(generated), \
                f"{name}: unexpectedly matches symmetric generator"


class TestGetPreset:

    def test_returns_lattice_instance(self):
        lattice = get_preset('60-diamond')
        assert isinstance(lattice, latticeClass)

    def test_respects_boundary_parameter(self):
        lattice = get_preset('24-diamond', boundary='open')
        assert lattice.boundary == 'open'

    def test_respects_plotLattice_parameter(self):
        lattice = get_preset('24-diamond', plotLattice=False)
        assert not lattice.p.plotLattice

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            get_preset('nonexistent')

    def test_periodic_with_off_sites_raises(self):
        with pytest.raises(ValueError, match="square"):
            get_preset('24-diamond', boundary='periodic')
