""" Functions to import parameters from input files.
"""

from dataclasses import dataclass, fields
from typeguard import typechecked
import ast

@typechecked
@dataclass
class openParameters:
    correlatorType: str = 'zz'          #
    transformType: str = 'dct'          #
    Lx: int = 6                         #
    Ly: int = 7                         #
    offSiteList: tuple = ()             #
    perturbationSite: tuple = (2,3)     #
    magnonModes: tuple = (1,2,3,4)      #
    excludeZeroMode: bool = False       #
    plotSites: bool = False             #
    saveWf: bool = False                #
    plotWf: bool = False                #
    saveCorrelatorXT: bool = False      #
    saveCorrelatorKW: bool = False      #
    plotCorrelatorKW: bool = False      #

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

@typechecked
@dataclass
class periodicParameters:
    correlatorType: str = 'zz'          #
    transformType: str = 'fft'          #
    Lx: int = 6                         #
    Ly: int = 7                         #
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


