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

