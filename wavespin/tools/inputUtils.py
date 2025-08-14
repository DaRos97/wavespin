""" Functions to import parameters from input files.
"""

def importOpenParameters(filename,**kwargs):
    """ Function to import all the parameters for the calculation from the input file.
    Lines starting with '#' are skipped.

    Parameters
    ----------
    filename : str, input filename.
    **kwargs : 'verbose':bool.

    Returns
    -------
    tuple of input parameters, each has a default value.
    """
    verbose = kwargs.get('verbose',False)
    default_params = {
        'correlator_type': 'zz',
        'fourier_type': 'dat',
        'exclude_zero_mode': True,
        'use_experimental_parameters': True,
        'Lx': 6,
        'Ly': 7,
        'offSiteList': [],
        'perturbationSite': (2,3),
        'include_list': [2,4,6,8],
        'plotSites': False,
        'save_wf': True,
        'plot_wf': True,
        'save_correlator': True,
        'save_bonds': False,
        'plot_correlator': True,
        'save_fig': True,
    }
    params = dict(default_params)
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                raise ValueError(f"Invalid line (missing ':'): {line}")
            key, value = line.split(':', 1)
            key = key.strip()
            if key in default_params:
                params[key] = parse_value(value)
            else:
                print(f"Warning: Unknown parameter '{key}' ignored.")
    checkParameters(params)
    if verbose:
        print("------------------ Chosen input parameters -----------------")
        for key in params.keys():
            print(key,': ',params[key])
    return params

def checkParameters(params):
    """ Check that each input parameters is of right type and of reasonable value.
    """
    if params['correlator_type'] not in ['zz','ee','jj','ze','ez']:
        raise ValueError(f"Invalid correlator type: {params['correlator_type']}")
    if params['fourier_type'] not in ['fft','dst','dct','dat']:
        raise ValueError(f"Invalid fourier type: {params['fourier_type']}")
    boolList = ['exclude_zero_mode','use_experimental_parameters','plotSites','save_wf','plot_wf','save_correlator','save_bonds','plot_correlator','save_fig']
    for t in boolList:
        if type(params[t])!=bool:
            raise ValueError(f"{t} is not a bool: {params[t]}")
    if type(params['Lx'])!=int:
        raise ValueError(f"Lx is not a int: {params['Lx']}")
    if type(params['Ly'])!=int:
        raise ValueError(f"Ly is not a int: {params['Ly']}")
    if type(params['offSiteList'])!=list:
        raise ValueError(f"offSiteList is not a list: {params['offSiteList']}")
    for coord in params['offSiteList']:
        if type(coord[0])!=int or type(coord[1])!=int:
            raise ValueError("Coordinate is not a int tuple: "+str(coord[0])+','+str(coord[1]))
        if coord[0]>=params['Lx'] or coord[0]<0:
            raise ValueError("Coordinate x is not between 0 and Lx="+str(params['Lx'])+': '+str(coord[0]))
        if coord[1]>=params['Ly'] or coord[1]<0:
            raise ValueError("Coordinate y is not between 0 and Ly="+str(params['Ly'])+': '+str(coord[1]))
    if type(params['perturbationSite'])!=tuple:
        raise ValueError(f"perturbationSite is not a tuple: {params['perturbationSite']}")
    coord = params['perturbationSite']
    if type(coord[0])!=int or type(coord[1])!=int:
        raise ValueError("Perturbation site coordinate is not a int tuple: "+str(coord[0])+','+str(coord[1]))
    if coord[0]>=params['Lx'] or coord[0]<0:
        raise ValueError("Perturbation site coordinate x is not between 0 and Lx="+str(params['Lx'])+': '+str(coord[0]))
    if coord[1]>=params['Ly'] or coord[1]<0:
        raise ValueError("Perturbation site coordinate y is not between 0 and Ly="+str(params['Ly'])+': '+str(coord[1]))
    if type(params['include_list'])!=list:
        raise ValueError(f"include_list is not a list: {params['include_list']}")
    for t in params['include_list']:
        if t not in [2,4,6,8]:
            raise ValueError("Term "+str(t)+" not an acceptable 'include_list' term: [2,4,6,8]->1-2-3-4 magnon terms.")


