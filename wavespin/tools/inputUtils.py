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
class openParameters:
    correlatorType: str = 'zz'          #
    transformType: str = 'dct'          #
    Lx: int = 6                         #
    Ly: int = 7                         #
    offSiteList: tuple = ()             #
    perturbationSite: tuple = (0,0)     #
    magnonModes: tuple = (1,2,3,4)      #
    excludeZeroMode: bool = False       #
    plotSites: bool = False             #
    saveWf: bool = False                #
    plotWf: bool = False                #
    saveCorrelatorXT: bool = False      #
    saveCorrelatorXTbonds: bool = False #
    saveCorrelatorKW: bool = False      #
    plotCorrelatorKW: bool = False      #
    savePlotCorrelatorKW: bool = False  #
    uniformQA: bool = True              #
    scatteringType: str = '1to2'        #

def importOpenParameters(inputFn,**kwargs):
    """ Function to import all the parameters for the calculation from the input file and store the in the class openParameters.
    Lines starting with '#' are skipped.

    Parameters
    ----------
    inputFn : str, input filename.
    **kwargs : 'verbose':bool.

    Returns
    -------
    openParameters : class of parameters for the calculation.
    """
    verbose = kwargs.get('verbose',False)
    parameters = openParameters()
    with open(inputFn, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                raise ValueError(f"Invalid line (missing ':'): {line}")
            key, value = line.split(':', 1)
            key = key.strip()
            if key in [f.name for f in fields(openParameters)]:
                setattr(parameters,key,parseValue(value))
            else:
                print(f"Warning: Unknown parameter '{key}' ignored.")
    checkOpenParameters(parameters)
    if verbose:
        print("------------------ Chosen input parameters -----------------")
        print(parameters)
    return parameters

def checkOpenParameters(parameters):
    """ Check that each input parameters is of right type and of reasonable value.
    """
    if parameters.correlatorType not in ['zz','ee','jj','ze','ez','xx']:
        raise ValueError(f"Invalid correlator type: {parameters.correlatorType}")
    if parameters.transformType not in ['fft','dst','dct','dat']:
        raise ValueError(f"Invalid momentum transform type: {parameters.transformType}")
    if parameters.Lx <= 0:
        raise ValueError("Lx value not acceptable (>=0): "+str(parameters.Lx))
    if parameters.Ly <= 0:
        raise ValueError("Ly value not acceptable (>=0): "+str(parameters.Ly))
    for coord in parameters.offSiteList:
        if coord[0]>=parameters.Lx or coord[0]<0:
            raise ValueError("Coordinate x is not between 0 and Lx-1="+str(parameters.Lx-1)+': '+str(coord[0]))
        if coord[1]>=parameters.Ly or coord[1]<0:
            raise ValueError("Coordinate y is not between 0 and Ly-1="+str(parameters.Ly-1)+': '+str(coord[1]))
    coord = parameters.perturbationSite
    if coord[0]>=parameters.Lx or coord[0]<0:
        raise ValueError("Perturbation site coordinate x is not between 0 and Lx-1="+str(parameters.Lx-1)+': '+str(coord[0]))
    if coord[1]>=parameters.Ly or coord[1]<0:
        raise ValueError("Perturbation site coordinate y is not between 0 and Ly-1="+str(parameters.Ly-1)+': '+str(coord[1]))
    for t in parameters.magnonModes:
        if t not in [1,2,3,4]:
            raise ValueError("Term "+str(t)+" not an acceptable 'magnonModes' term: [1,2,3,4].")
    if parameters.scatteringType not in ['1to2','1to3','2to2']:
        raise ValueError(f"Invalid scattering type: {parameters.scatteringType}")

@typechecked
@dataclass
class periodicParameters:
    correlatorType: str = 'zz'          #
    transformType: str = 'fft'          #
    Lx: int = 10                        #
    Ly: int = 10                        #
    plotSites: bool = False             #
    plotValues: bool = False            #
    plotDispersions: bool = False       #
    saveCorrelatorXT: bool = False      #
    saveCorrelatorKW: bool = False      #
    plotCorrelatorKW: bool = False      #
    saveFigureCorrelatorKW: bool = False#

def importPeriodicParameters(inputFn,**kwargs):
    """ Function to import all the parameters for the calculation from the input file and store the in the class openParameters.
    Lines starting with '#' are skipped.

    Parameters
    ----------
    inputFn : str, input filename.
    **kwargs : 'verbose':bool.

    Returns
    -------
    periodicParameters : class of parameters for the calculation.
    """
    verbose = kwargs.get('verbose',False)
    parameters = periodicParameters()
    with open(inputFn, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                raise ValueError(f"Invalid line (missing ':'): {line}")
            key, value = line.split(':', 1)
            key = key.strip()
            if key in [f.name for f in fields(periodicParameters)]:
                setattr(parameters,key,parseValue(value))
            else:
                print(f"Warning: Unknown parameter '{key}' ignored.")
    checkPeriodicParameters(parameters)
    if verbose:
        print("------------------ Chosen input parameters -----------------")
        print(parameters)
    return parameters

def checkPeriodicParameters(parameters):
    """ Check that each input parameters is of right type and of reasonable value.
    """
    if parameters.correlatorType not in ['zz',]:
        raise ValueError(f"Invalid correlator type: {parameters.correlatorType}")
    if parameters.Lx <= 0:
        raise ValueError("Lx value not acceptable (>=0): "+str(parameters.Lx))
    if parameters.Ly <= 0:
        raise ValueError("Ly value not acceptable (>=0): "+str(parameters.Ly))

@typechecked
@dataclass
class classicParameters:
    Lx: int = 6                         #
    Ly: int = 7                         #
    offSiteList: tuple = ()             #
    plotSites: bool = False             #
    T_max: float = 2.5                  # start temperature
    T_min: float = 1e-3                 # end temperature
    alpha: float = 0.95                 # geometric cooling factor
    sweeps_per_T_high: int = 200        # sweeps at high T
    sweeps_per_T_low: int = 1000        # sweeps at low T
    overrelax_every: int = 1            # do 1 overrelax sweep per Metropolis sweep (0 disables)
    proposal_step: float = 0.35         # small rotation angle scale (~0.2–0.5)
    seed: int = 42                      #
    boundary: str = 'open'              #
    saveSolution: bool = False          #
    savePlotSolution: bool = False      #

def importClassicParameters(inputFn,**kwargs):
    """ Function to import all the parameters for the calculation from the input file and store the in the class openParameters.
    Lines starting with '#' are skipped.

    Parameters
    ----------
    inputFn : str, input filename.
    **kwargs : 'verbose':bool.

    Returns
    -------
    openParameters : class of parameters for the calculation.
    """
    verbose = kwargs.get('verbose',False)
    parameters = classicParameters()
    with open(inputFn, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                raise ValueError(f"Invalid line (missing ':'): {line}")
            key, value = line.split(':', 1)
            key = key.strip()
            if key in [f.name for f in fields(classicParameters)]:
                setattr(parameters,key,parseValue(value))
            else:
                print(f"Warning: Unknown parameter '{key}' ignored.")
    checkClassicParameters(parameters)
    if verbose:
        print("------------------ Chosen input parameters -----------------")
        print(parameters)
    return parameters

def checkClassicParameters(parameters):
    """ Check that each input parameters is of right type and of reasonable value.
    """
    for coord in parameters.offSiteList:
        if coord[0]>=parameters.Lx or coord[0]<0:
            raise ValueError("Coordinate x is not between 0 and Lx-1="+str(parameters.Lx-1)+': '+str(coord[0]))
        if coord[1]>=parameters.Ly or coord[1]<0:
            raise ValueError("Coordinate y is not between 0 and Ly-1="+str(parameters.Ly-1)+': '+str(coord[1]))
    if parameters.Lx <= 0:
        raise ValueError("Lx value not acceptable (>=0): "+str(parameters.Lx))
    if parameters.Ly <= 0:
        raise ValueError("Ly value not acceptable (>=0): "+str(parameters.Ly))
    if parameters.T_min <= 0 or parameters.T_max <= 0 or parameters.T_min>parameters.T_max:
        raise ValueError("Not acceptable values of T_min and/or T_max : %.4f, %.4f"%(parameters.T_min,parameters.T_max))
    if parameters.alpha <= 0 or parameters.alpha>1:
        raise ValueError("alpha value not acceptable (>0 and <1): %.3f"%parameters.alpha)
    if parameters.sweeps_per_T_low <= 0 or parameters.sweeps_per_T_high <= 0:
        raise ValueError("Not acceptable values of sweeps : %, %"%(parameters.sweeps_per_T_low,parameters.sweeps_per_T_high))
    if parameters.overrelax_every < 0:
        raise ValueError("overrelax_every value not acceptable (>0): "+str(parameters.overrelax_every))
    if parameters.boundary not in ['open','periodic']:
        raise ValueError(f"Invalid boundary type: {parameters.boundary}")


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
    cor_correlatorType: str = 'zz'                        #
    cor_transformType: str = 'dct'              #
    cor_perturbationSite: tuple = (0,0)         #
    cor_magnonModes: tuple = (1,2,3,4)          #
    cor_saveXT: bool = False                    #
    cor_saveXTbonds: bool = False               #
    cor_saveKW: bool = False                    #
    cor_plotKW: bool = False                    #
    cor_savePlotKW: bool = False                #
    # Decay and scattering
    sca_types: tuple = ('1to2','2to2a','1to3')  #
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
        if tt not in ['1to2','1to3','2to2a','2to2b']:
            raise ValueError("Scattering type %s not recognised"%tt)
    if parameters.sca_temperature < 0:
        raise ValueError("Temperature has to be positive or 0, not ",parameters.sca_temperature)
    if parameters.sca_broadening < 0:
        raise ValueError("Broadening of energy delta has to be positive, not ",parameters.sca_broadening)



