import os
from pathlib import Path
import numpy as np

def getFilename(*args,dirname='',extension='',floatPrecision=4):
    """ Get filename for set of parameters.

    Parameters
    ----------
    *args: list of arguments to put in the filename.
    dirname: str, should end with '/'
    extension: string, should start with '.'
    floatPrecision: int, detail of floating point arguments.

    Returns
    -------
    filename : string.
    """
    if len(dirname)>0 and dirname[-1]!='/':
        raise ValueError("directory name  %s must end with '/'"%dirname)
    if len(extension)>0 and extension[0]!='.':
        raise ValueError("extension name  %s must begin with '.'"%extension)

    filename = ''
    filename += dirname
    for i,a in enumerate(args):
        t = type(a)
        if t in [str,np.str_]:
            filename += a
        elif t in [int, np.int64, np.int32]:
            filename += str(a)
        elif t in [float, np.float32, np.float64]:
            filename += f"{a:.{floatPrecision}f}"
        elif t==tuple:
            filename += getFilename(*a)
        else:
            raise TypeError("Parameter %s has unsupported type: %s"%(a,t))
        if not i==len(args)-1:
            filename += '_'
    filename += extension
    return filename

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
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    if len(subfolder)>0 and subfolder[-1]!='/':
        raise ValueError("sub-directory name  %s must end with '/'"%subfolder)
    if Path(scriptPath+'/homePath.txt').is_file():
        with open(scriptPath+'/homePath.txt','r') as f:
            homeDn = f.readline()[:-1]
    else:
        homeDn = cwd
    return homeDn + subfolder


