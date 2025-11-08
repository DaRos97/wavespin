""" Here we keep some usefull geometric functions.
"""

import numpy as np

def vector_to_polar_angles(v):
    """
    Compute polar angles (theta, phi) from a 3D unit vector.

    Parameters:
        v (array-like): unit vector [x, y, z]

    Returns:
        theta (float): polar angle (0 to pi)
        phi (float): azimuthal angle (-pi to pi)
    """
    x, y, z = v
    # theta = angle from z-axis
    theta = np.arccos(z)
    # phi = azimuth in xy-plane
    phi = np.arctan2(y, x)
    return theta, phi

def lorentz(arg,gamma):
    """ Lorentz distribution used for scattering and decay rates.
    """
    return (1.0/np.pi) * (gamma / (arg**2 + gamma**2))

def Ry(theta):
    """ Compute 3D rotation matrix around y-axis of angle theta """
    R = np.zeros((3,3))
    R[0,0] = np.cos(theta)
    R[0,2] = np.sin(theta)
    R[1,1] = 1
    R[2,0] = -np.sin(theta)
    R[2,2] = np.cos(theta)
    return R
