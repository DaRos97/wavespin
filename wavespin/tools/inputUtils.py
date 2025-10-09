""" Functions to import parameters from input files.
"""

from dataclasses import dataclass, fields
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
class myParameters:
    # Lattice
    lat_Lx: int = 6                             # 
    lat_Ly: int = 7                             #
    lat_offSiteList: tuple = ()                 #
    lat_boundary: str = 'open'                  #
    lat_plotLattice: bool = False               #
    # Diagonalization
    dia_Hamiltonian: tuple = (0,0,0,0,0,0)      #
    dia_excludeZeroMode: bool = False           #
    dia_uniformQA: bool = True                  #
    dia_saveWf: bool = False                    #
    dia_plotWf: bool = False                    #
    dia_plotMomenta: bool = False               #
    # Correlators
    cor_correlatorType: str = 'zz'              #
    cor_transformType: str = 'dct'              #
    cor_perturbationSite: tuple = (0,0)         #
    cor_magnonModes: tuple = (1,2,3,4)          #
    cor_saveXT: bool = False                    #
    cor_saveXTbonds: bool = False               #
    cor_saveKW: bool = False                    #
    cor_plotKW: bool = False                    #
    cor_savePlotKW: bool = False                #
    # Decay and scattering
    sca_types: tuple = ('2to2_1',)              #
    sca_temperature: float = 0.0                #
    sca_broadening: float = 0.5                 #
    sca_saveVertex: bool = True                 #
    sca_plotVertex: bool = False                #

def importParameters(inputFn,**kwargs):
    """ Function to import all the parameters for the calculation from the input file and store the in the class myParameters.
    Lines starting with '#' are skipped.

    Parameters
    ----------
    inputFn : str, input filename.
    **kwargs : 'verbose':bool.

    Returns
    -------
    myParameters : class of parameters for the calculation.
    """
    verbose = kwargs.get('verbose',False)
    parameters = myParameters()
    with open(inputFn, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                raise ValueError(f"Invalid line (missing ':'): {line}")
            key, value = line.split(':', 1)
            key = key.strip()
            if key in [f.name[4:] for f in fields(myParameters)]:
                ik = [f.name[4:] for f in fields(myParameters)].index(key)
                attr = fields(myParameters)[ik].name
                setattr(parameters,attr,parseValue(value))
            else:
                print(f"Warning: Unknown parameter '{key}' ignored.")
    checkParameters(parameters)
    if verbose:
        print("------------------ Chosen input parameters -----------------")
        print(parameters)
    return parameters

def checkParameters(parameters):
    """ Check that each input parameters is of right type and of reasonable value.
    """
    # Lattice parameters
    for coord in parameters.lat_offSiteList:
        if coord[0]>=parameters.lat_Lx or coord[0]<0:
            raise ValueError("Coordinate x is not between 0 and Lx-1="+str(parameters.lat_Lx-1)+': '+str(coord[0]))
        if coord[1]>=parameters.lat_Ly or coord[1]<0:
            raise ValueError("Coordinate y is not between 0 and Ly-1="+str(parameters.lat_Ly-1)+': '+str(coord[1]))
    if parameters.lat_Lx <= 0:
        raise ValueError("Lx value not acceptable (>=0): "+str(parameters.lat_Lx))
    if parameters.lat_Ly <= 0:
        raise ValueError("Ly value not acceptable (>=0): "+str(parameters.lat_Ly))
    if parameters.lat_boundary not in ['open','periodic']:
        raise ValueError(f"Invalid boundary type: {parameters.lat_boundary}")
    # Diagonalization parameters
    if len(parameters.dia_Hamiltonian) != 6:
        raise ValueError("Hamiltonian parameters not of right legth: ",parameters.dia_Hamiltonian)
    # Correlator parameters
    if parameters.cor_correlatorType not in ['zz','ee','jj','ze','ez','xx']:
        raise ValueError(f"Invalid correlator type: {parameters.cor_type}")
    if parameters.cor_transformType not in ['fft','dst','dct','dat']:
        raise ValueError(f"Invalid momentum transform type: {parameters.cor_transformType}")
    coord = parameters.cor_perturbationSite
    if coord[0]>=parameters.lat_Lx or coord[0]<0:
        raise ValueError("Perturbation site coordinate x is not between 0 and Lx-1="+str(parameters.lat_Lx-1)+': '+str(coord[0]))
    if coord[1]>=parameters.lat_Ly or coord[1]<0:
        raise ValueError("Perturbation site coordinate y is not between 0 and Ly-1="+str(parameters.lat_Ly-1)+': '+str(coord[1]))
    for t in parameters.cor_magnonModes:
        if t not in [1,2,3,4]:
            raise ValueError("Term "+str(t)+" not an acceptable 'magnonModes' term: [1,2,3,4].")
    # Decay and scattering parameters
    for tt in parameters.sca_types:
        if tt not in ['1to2_1','1to2_2','1to3_1','1to3_2','1to3_3','2to2_1','2to2_2']:
            raise ValueError("Scattering type %s not recognised"%tt)
    if parameters.sca_temperature < 0:
        raise ValueError("Temperature has to be positive or 0, not ",parameters.sca_temperature)
    if parameters.sca_broadening < 0:
        raise ValueError("Broadening of energy delta has to be positive, not ",parameters.sca_broadening)



