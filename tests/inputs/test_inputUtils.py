import pytest
from io import StringIO
import sys

from wavespin.tools.inputUtils import (
    parseValue,
    LatticeParams,
    DiagParams,
    CorrelatorParams,
    ScatteringParams,
    SimParams,
    importParameters,
    checkParameters,
    _build_field_map,
)


# ---------------------------------------------------------------------------
# parseValue
# ---------------------------------------------------------------------------

class TestParseValue:
    def test_int(self):
        assert parseValue("42") == 42
        assert parseValue("-7") == -7
        assert parseValue(" 10 ") == 10

    def test_float(self):
        assert parseValue("3.14") == 3.14
        assert parseValue("-2.5") == -2.5

    def test_bool(self):
        assert parseValue("True") is True
        assert parseValue("False") is False

    def test_tuple(self):
        assert parseValue("(1, 2, 3)") == (1, 2, 3)
        assert parseValue("((0, 0), (1, 1))") == ((0, 0), (1, 1))

    def test_string_fallback(self):
        assert parseValue("some_arbitrary_text") == "some_arbitrary_text"
        assert parseValue("not_a_literal") == "not_a_literal"


# ---------------------------------------------------------------------------
# LatticeParams
# ---------------------------------------------------------------------------

class TestLatticeParams:
    def test_defaults(self):
        lp = LatticeParams()
        assert lp.Lx == 7
        assert lp.Ly == 9
        assert lp.offSiteList == ()
        assert lp.boundary == 'open'
        assert lp.plotLattice is False

    def test_custom_values(self):
        lp = LatticeParams(Lx=3, Ly=4, offSiteList=((1, 1),), boundary='periodic', plotLattice=True)
        assert lp.Lx == 3
        assert lp.Ly == 4
        assert lp.offSiteList == ((1, 1),)
        assert lp.boundary == 'periodic'
        assert lp.plotLattice is True

    def test_lx_zero_raises(self):
        with pytest.raises(ValueError, match="Lx value not acceptable"):
            LatticeParams(Lx=0)

    def test_lx_negative_raises(self):
        with pytest.raises(ValueError, match="Lx value not acceptable"):
            LatticeParams(Lx=-5)

    def test_ly_zero_raises(self):
        with pytest.raises(ValueError, match="Ly value not acceptable"):
            LatticeParams(Ly=0)

    def test_invalid_boundary_raises(self):
        with pytest.raises(ValueError, match="Invalid boundary type"):
            LatticeParams(boundary='closed')

    def test_offsite_x_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Coordinate x is not between"):
            LatticeParams(Lx=4, Ly=4, offSiteList=((5, 0),))

    def test_offsite_x_negative_raises(self):
        with pytest.raises(ValueError, match="Coordinate x is not between"):
            LatticeParams(Lx=4, Ly=4, offSiteList=((-1, 0),))

    def test_offsite_y_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Coordinate y is not between"):
            LatticeParams(Lx=4, Ly=4, offSiteList=((0, 5),))

    def test_offsite_y_negative_raises(self):
        with pytest.raises(ValueError, match="Coordinate y is not between"):
            LatticeParams(Lx=4, Ly=4, offSiteList=((0, -1),))

    def test_offsite_at_boundary_is_valid(self):
        lp = LatticeParams(Lx=4, Ly=4, offSiteList=((3, 3),))
        assert lp.offSiteList == ((3, 3),)


# ---------------------------------------------------------------------------
# DiagParams
# ---------------------------------------------------------------------------

class TestDiagParams:
    def test_defaults(self):
        dp = DiagParams()
        assert dp.Hamiltonian == (0, 0, 0, 0, 0, 0)
        assert dp.excludeZeroMode is True
        assert dp.uniformQA is True
        assert dp.saveWf is False
        assert dp.plotWf is False
        assert dp.plotDiffusionSolutions is False
        assert dp.plotMomenta is False

    def test_wrong_hamiltonian_length_raises(self):
        with pytest.raises(ValueError, match="Hamiltonian parameters not of right length"):
            DiagParams(Hamiltonian=(1, 2, 3))

    def test_hamiltonian_length_6_passes(self):
        dp = DiagParams(Hamiltonian=(10, 0, 0, 0, 5, 0))
        assert len(dp.Hamiltonian) == 6


# ---------------------------------------------------------------------------
# CorrelatorParams
# ---------------------------------------------------------------------------

class TestCorrelatorParams:
    def test_defaults(self):
        cp = CorrelatorParams()
        assert cp.correlatorType == 'zz'
        assert cp.transformType == 'dct'
        assert cp.perturbationSite == (0, 0)
        assert cp.magnonOrder == (1, 2, 3, 4)
        assert cp.energy == -100
        assert cp.saveXT is False
        assert cp.saveXTbonds is False
        assert cp.saveKW is False
        assert cp.plotKW is False
        assert cp.savePlotKW is False

    def test_invalid_correlator_type_raises(self):
        with pytest.raises(ValueError, match="Invalid correlator type"):
            CorrelatorParams(correlatorType='invalid')

    def test_all_valid_correlator_types(self):
        for t in ['zz', 'ee', 'jj', 'ze', 'ez', 'xx']:
            cp = CorrelatorParams(correlatorType=t)
            assert cp.correlatorType == t

    def test_invalid_transform_type_raises(self):
        with pytest.raises(ValueError, match="Invalid momentum transform type"):
            CorrelatorParams(transformType='invalid')

    def test_all_valid_transform_types(self):
        for t in ['fft', 'dst', 'dct', 'dat', 'dat2']:
            cp = CorrelatorParams(transformType=t)
            assert cp.transformType == t

    def test_invalid_magnon_mode_raises(self):
        with pytest.raises(ValueError, match="not an acceptable"):
            CorrelatorParams(magnonOrder=(1, 5))

    def test_all_valid_magnon_modes(self):
        cp = CorrelatorParams(magnonOrder=(1, 2, 3, 4))
        assert cp.magnonOrder == (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# ScatteringParams
# ---------------------------------------------------------------------------

class TestScatteringParams:
    def test_defaults(self):
        sp = ScatteringParams()
        assert sp.types == ('2to2_1',)
        assert sp.temperature == 0.0
        assert sp.broadening == 0.5
        assert sp.saveVertex is False
        assert sp.saveRate is False
        assert sp.plotRate is False

    def test_invalid_scattering_type_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            ScatteringParams(types=('invalid_type',))

    def test_all_valid_scattering_types(self):
        valid = ['1to2_1', '1to2_2', '1to3_1', '1to3_2', '1to3_3',
                 '2to2_1', '2to2_2']
        for t in valid:
            sp = ScatteringParams(types=(t,))
            assert sp.types == (t,)

    def test_multiple_scattering_types(self):
        sp = ScatteringParams(types=('1to2_1', '2to2_1'))
        assert sp.types == ('1to2_1', '2to2_1')

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="Temperature has to be positive"):
            ScatteringParams(temperature=-1.0)

    def test_zero_temperature_passes(self):
        sp = ScatteringParams(temperature=0.0)
        assert sp.temperature == 0.0

    def test_negative_broadening_raises(self):
        with pytest.raises(ValueError, match="Broadening of energy delta"):
            ScatteringParams(broadening=-0.1)

    def test_zero_broadening_passes(self):
        sp = ScatteringParams(broadening=0.0)
        assert sp.broadening == 0.0


# ---------------------------------------------------------------------------
# SimParams
# ---------------------------------------------------------------------------

class TestSimParams:
    def test_defaults(self):
        sp = SimParams()
        assert isinstance(sp.lattice, LatticeParams)
        assert isinstance(sp.diag, DiagParams)
        assert isinstance(sp.correlator, CorrelatorParams)
        assert isinstance(sp.scattering, ScatteringParams)
        assert sp.lattice.Lx == 7
        assert sp.diag.Hamiltonian == (0, 0, 0, 0, 0, 0)
        assert sp.correlator.correlatorType == 'zz'
        assert sp.scattering.types == ('2to2_1',)

    def test_override_sub_params(self):
        sp = SimParams(
            lattice=LatticeParams(Lx=10, Ly=12),
            diag=DiagParams(Hamiltonian=(1, 2, 3, 4, 5, 6)),
        )
        assert sp.lattice.Lx == 10
        assert sp.lattice.Ly == 12
        assert sp.diag.Hamiltonian == (1, 2, 3, 4, 5, 6)
        assert sp.correlator.correlatorType == 'zz'

    def test_sub_objects_are_independent(self):
        sp1 = SimParams()
        sp2 = SimParams()
        sp1.lattice.Lx = 99
        assert sp2.lattice.Lx == 7

    def test_field_access_via_nested_paths(self):
        sp = SimParams(
            lattice=LatticeParams(Lx=5, boundary='periodic'),
            scattering=ScatteringParams(temperature=2.5),
        )
        assert sp.lattice.Lx == 5
        assert sp.lattice.boundary == 'periodic'
        assert sp.scattering.temperature == 2.5


# ---------------------------------------------------------------------------
# _build_field_map
# ---------------------------------------------------------------------------

class TestBuildFieldMap:
    def test_all_expected_keys_present(self):
        fm = _build_field_map()
        expected = {
            'Lx', 'Ly', 'offSiteList', 'boundary', 'plotLattice',
            'Hamiltonian', 'excludeZeroMode', 'uniformQA', 'saveWf',
            'plotWf', 'plotDiffusionSolutions', 'plotMomenta',
            'correlatorType', 'transformType', 'perturbationSite',
            'magnonOrder', 'energy', 'fullTimeMeasure', 'nTimes', 'nOmega',
            'saveXT', 'saveXTbonds',
            'saveKW', 'plotKW', 'savePlotKW',
            'types', 'temperature', 'broadening', 'saveVertex',
            'saveRate', 'plotRate',
        }
        assert set(fm.keys()) == expected

    def test_no_duplicate_keys(self):
        fm = _build_field_map()
        assert len(fm) == len(set(fm.keys()))

    def test_values_point_to_correct_sub_object(self):
        fm = _build_field_map()
        assert fm['Lx'] == ('lattice', 'Lx')
        assert fm['Hamiltonian'] == ('diag', 'Hamiltonian')
        assert fm['correlatorType'] == ('correlator', 'correlatorType')
        assert fm['temperature'] == ('scattering', 'temperature')


# ---------------------------------------------------------------------------
# importParameters
# ---------------------------------------------------------------------------

class TestImportParameters:
    def test_empty_filename_returns_defaults(self):
        p = importParameters()
        assert isinstance(p, SimParams)
        assert p.lattice.Lx == 7
        assert p.lattice.Ly == 9

    def test_parses_valid_file(self, tmp_path):
        content = """# comment line
Lx: 4
Ly: 5
boundary: periodic
Hamiltonian: (10, 0, 0, 0, 5, 0)
temperature: 1.5
"""
        f = tmp_path / "input.txt"
        f.write_text(content)
        p = importParameters(str(f))
        assert p.lattice.Lx == 4
        assert p.lattice.Ly == 5
        assert p.lattice.boundary == 'periodic'
        assert p.diag.Hamiltonian == (10, 0, 0, 0, 5, 0)
        assert p.scattering.temperature == 1.5

    def test_parses_boolean_values(self, tmp_path):
        content = "plotLattice: True\nsaveWf: False\n"
        f = tmp_path / "input.txt"
        f.write_text(content)
        p = importParameters(str(f))
        assert p.lattice.plotLattice is True
        assert p.diag.saveWf is False

    def test_parses_tuple_value(self, tmp_path):
        content = "offSiteList: ((1, 1), (2, 2))\n"
        f = tmp_path / "input.txt"
        f.write_text(content)
        p = importParameters(str(f))
        assert p.lattice.offSiteList == ((1, 1), (2, 2))

    def test_unknown_parameter_prints_warning(self, tmp_path):
        content = "bogusKey: 42\n"
        f = tmp_path / "input.txt"
        f.write_text(content)
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            importParameters(str(f))
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "Unknown parameter 'bogusKey' ignored" in output

    def test_missing_colon_raises(self, tmp_path):
        content = "bad line without colon\n"
        f = tmp_path / "input.txt"
        f.write_text(content)
        with pytest.raises(ValueError, match="Invalid line"):
            importParameters(str(f))

    def test_comment_and_blank_lines_skipped(self, tmp_path):
        content = """
# this is a comment

Lx: 8

# another comment
Ly: 9
"""
        f = tmp_path / "input.txt"
        f.write_text(content)
        p = importParameters(str(f))
        assert p.lattice.Lx == 8
        assert p.lattice.Ly == 9

    def test_unknown_keys_dont_break_known_keys(self, tmp_path):
        content = "Lx: 5\nbogus: ignored\nLy: 6\n"
        f = tmp_path / "input.txt"
        f.write_text(content)
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            p = importParameters(str(f))
        finally:
            sys.stdout = old_stdout
        assert p.lattice.Lx == 5
        assert p.lattice.Ly == 6
        assert "Unknown parameter 'bogus' ignored" in captured.getvalue()


# ---------------------------------------------------------------------------
# checkParameters
# ---------------------------------------------------------------------------

class TestCheckParameters:
    def test_valid_params_pass(self):
        sp = SimParams(lattice=LatticeParams(Lx=8, Ly=8))
        checkParameters(sp)

    def test_perturbation_x_out_of_range_raises(self):
        sp = SimParams(
            lattice=LatticeParams(Lx=4, Ly=4),
            correlator=CorrelatorParams(perturbationSite=(5, 0)),
        )
        with pytest.raises(ValueError, match="Perturbation site coordinate x"):
            checkParameters(sp)

    def test_perturbation_y_out_of_range_raises(self):
        sp = SimParams(
            lattice=LatticeParams(Lx=4, Ly=4),
            correlator=CorrelatorParams(perturbationSite=(0, 5)),
        )
        with pytest.raises(ValueError, match="Perturbation site coordinate y"):
            checkParameters(sp)

    def test_perturbation_x_negative_raises(self):
        sp = SimParams(
            lattice=LatticeParams(Lx=4, Ly=4),
            correlator=CorrelatorParams(perturbationSite=(-1, 0)),
        )
        with pytest.raises(ValueError, match="Perturbation site coordinate x"):
            checkParameters(sp)

    def test_perturbation_y_negative_raises(self):
        sp = SimParams(
            lattice=LatticeParams(Lx=4, Ly=4),
            correlator=CorrelatorParams(perturbationSite=(0, -1)),
        )
        with pytest.raises(ValueError, match="Perturbation site coordinate y"):
            checkParameters(sp)
