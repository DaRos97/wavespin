""" Functions to import parameters from input files.
"""

from dataclasses import dataclass, field, fields
from typeguard import typechecked
import ast


def parseValue(value_str):
    """
    Try to evaluate as Python literal (works for bool, int, float, list, tuple, etc.)
    If it fails, treat it as a raw string.
    """
    value_str = value_str.strip()
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str


@typechecked
@dataclass
class LatticeParams:
    Lx: int = 7
    Ly: int = 9
    offSiteList: tuple = ()
    boundary: str = 'open'
    plotLattice: bool = False

    def __post_init__(self):
        if self.Lx <= 0:
            raise ValueError("Lx value not acceptable (>= 1): " + str(self.Lx))
        if self.Ly <= 0:
            raise ValueError("Ly value not acceptable (>= 1): " + str(self.Ly))
        if self.boundary not in ['open', 'periodic']:
            raise ValueError(f"Invalid boundary type: {self.boundary}")
        for coord in self.offSiteList:
            if coord[0] >= self.Lx or coord[0] < 0:
                raise ValueError(
                    "Coordinate x is not between 0 and Lx-1="
                    + str(self.Lx - 1) + ': ' + str(coord[0])
                )
            if coord[1] >= self.Ly or coord[1] < 0:
                raise ValueError(
                    "Coordinate y is not between 0 and Ly-1="
                    + str(self.Ly - 1) + ': ' + str(coord[1])
                )


@typechecked
@dataclass
class DiagParams:
    Hamiltonian: tuple = (0, 0, 0, 0, 0, 0)
    excludeZeroMode: bool = True
    uniformQA: bool = True
    saveWf: bool = False
    plotWf: bool = False
    plotDiffusionSolutions: bool = False
    plotMomenta: bool = False

    def __post_init__(self):
        if len(self.Hamiltonian) != 6:
            raise ValueError(
                "Hamiltonian parameters not of right length: "
                + str(self.Hamiltonian)
            )


@typechecked
@dataclass
class CorrelatorParams:
    correlatorType: str = 'zz'
    transformType: str = 'dct'
    perturbationSite: tuple = (0, 0)
    magnonModes: tuple = (1, 2, 3, 4)
    energy: float = -100
    saveXT: bool = False
    saveXTbonds: bool = False
    saveKW: bool = False
    plotKW: bool = False
    savePlotKW: bool = False

    def __post_init__(self):
        if self.correlatorType not in ['zz', 'ee', 'jj', 'ze', 'ez', 'xx']:
            raise ValueError(f"Invalid correlator type: {self.correlatorType}")
        if self.transformType not in ['fft', 'dst', 'dct', 'dat', 'dat2']:
            raise ValueError(
                f"Invalid momentum transform type: {self.transformType}"
            )
        for t in self.magnonModes:
            if t not in [1, 2, 3, 4]:
                raise ValueError(
                    "Term " + str(t) + " not an acceptable 'magnonModes' "
                    "term: [1,2,3,4]."
                )


@typechecked
@dataclass
class ScatteringParams:
    types: tuple = ('2to2_1',)
    temperature: float = 0.0
    broadening: float = 0.5
    saveVertex: bool = False
    saveRate: bool = False
    plotRate: bool = False

    def __post_init__(self):
        valid = [
            '1to2_1', '1to2_2', '1to3_1', '1to3_2', '1to3_3',
            '2to2_1', '2to2_2',
        ]
        for tt in self.types:
            if tt not in valid:
                raise ValueError("Scattering type %s not recognised" % tt)
        if self.temperature < 0:
            raise ValueError(
                "Temperature has to be positive or 0, not "
                + str(self.temperature)
            )
        if self.broadening < 0:
            raise ValueError(
                "Broadening of energy delta has to be positive, not "
                + str(self.broadening)
            )


@typechecked
@dataclass
class SimParams:
    lattice: LatticeParams = field(default_factory=LatticeParams)
    diag: DiagParams = field(default_factory=DiagParams)
    correlator: CorrelatorParams = field(default_factory=CorrelatorParams)
    scattering: ScatteringParams = field(default_factory=ScatteringParams)


def _build_field_map():
    """Build a mapping from input file key to (sub_attr, field_name)."""
    mapping = {}
    for sim_field in fields(SimParams):
        sub_cls = sim_field.type  # e.g. LatticeParams
        sub_name = sim_field.name  # e.g. 'lattice'
        for sub_field in fields(sub_cls):
            mapping[sub_field.name] = (sub_name, sub_field.name)
    return mapping


def importParameters(inputFn='', **kwargs):
    """ Function to import all the parameters for the calculation from the
    input file and store them in a SimParams instance.

    Lines starting with '#' are skipped.

    Parameters
    ----------
    inputFn : str, input filename.
    **kwargs : 'verbose':bool.

    Returns
    -------
    SimParams : class of parameters for the calculation.
    """
    verbose = kwargs.get('verbose', False)
    parameters = SimParams()
    if inputFn == '':
        return parameters
    field_map = _build_field_map()
    with open(inputFn, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                raise ValueError(f"Invalid line (missing ':'): {line}")
            key, value = line.split(':', 1)
            key = key.strip()
            if key in field_map:
                sub_name, field_name = field_map[key]
                sub_obj = getattr(parameters, sub_name)
                setattr(sub_obj, field_name, parseValue(value))
            else:
                print(f"Warning: Unknown parameter '{key}' ignored.")
    if verbose:
        print("------------------ Chosen input parameters -----------------")
        print(parameters)
    checkParameters(parameters)
    return parameters


def checkParameters(parameters):
    """ Cross-parameter validation requiring multiple sub-parameter groups.
    """
    lat = parameters.lattice
    cor = parameters.correlator
    coord = cor.perturbationSite
    if coord[0] >= lat.Lx or coord[0] < 0:
        raise ValueError(
            "Perturbation site coordinate x is not between 0 and Lx-1="
            + str(lat.Lx - 1) + ': ' + str(coord[0])
        )
    if coord[1] >= lat.Ly or coord[1] < 0:
        raise ValueError(
            "Perturbation site coordinate y is not between 0 and Ly-1="
            + str(lat.Ly - 1) + ': ' + str(coord[1])
        )
