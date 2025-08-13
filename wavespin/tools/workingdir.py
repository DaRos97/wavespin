import numpy as np

def getHomeDirname(cwd,subfolder=''):
    """ Find the home folder.

    Parameters
    ----------
    cwd : str, current woring directory.
    subfolder : str, should end in '/'.

    Returns
    -------
    dirname : str, path of desired directory.
    """
    if len(subfolder)>0 and subfolder[-1]!='/':
        raise ValueError("sub-directory name  %s must end with '/'"%subfolder)
    pos = cwd.find('WaveSpin')
    if pos==-1:
        raise ValueError("Script is running outside the WaveSpin folder: "+cwd)
    dirname = cwd[:pos]+'WaveSpin/'+subfolder
    return dirname


