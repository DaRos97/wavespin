""" Here we keep some usefull geometric functions.
"""

import numpy as np
import scipy

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


def solve_diffusion_eigenmodes_xy(sites, dx=1., dy=1., direction='xy'):
    """
    Solves for the eigenmodes of the diffusion equation on a specified geometry.

    Args:
        sites (list of tuples): A list of (x, y) coordinates for each site.
    Returns:
        tuple: (eigenvalues, eigenvectors)
               The eigenvectors are the columns of the returned matrix.
    """
    num_sites = len(sites)
    # --- Build the Laplacian Matrix (L) ---
    L = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        x, y = sites[i]
        degree = 0
        # Check all four cardinal neighbors
        list_nn_dic = {'x': [(x+1,y),(x-1,y)], 'y':[(x,y+1),(x,y-1)], 'xy':[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]}
        for nx, ny in list_nn_dic[direction]:
            neighbor_coord = (nx, ny)
            if neighbor_coord in sites:
                # This is a valid neighbor within our defined geometry
                degree += abs(x - nx)*dx + abs(y-ny)*dy
                # Set the off-diagonal element L_ij = -1
                j = sites.index(neighbor_coord)
                L[i, j] = - abs(x - nx)*dx - abs(y-ny)*dy
        # Set the diagonal element L_ii = degree
        L[i, i] = degree
    # --- Solve the Eigenproblem ---
    eigenvalues, eigenvectors = scipy.linalg.eigh(L)
    return eigenvalues, eigenvectors

def rotation_pi_apply(v, Lx, Ly):
    V = v.reshape((Lx, Ly))
    R = V[::-1, ::-1]
    return R.ravel()

def detect_degenerate_groups(eigvals, tol=1e-10):
    groups = []
    M = len(eigvals)
    i = 0
    while i < M:
        j = i + 1
        while j < M and abs(eigvals[j] - eigvals[i]) <= tol * max(1, abs(eigvals[i])):
            j += 1
        groups.append((i, j))
        i = j
    return groups

def fix_degeneracy_with_rotation(eigvals, eigvecs, Lx, Ly, tol=1e-10):
    n, M = eigvecs.shape
    groups = detect_degenerate_groups(eigvals, tol)
    eigvecs_fixed = eigvecs.copy()
    block_info = []

    for i0, i1 in groups:
        k = i1 - i0
        if k <= 1:
            # fix sign consistently
            v = eigvecs_fixed[:, i0]
            imax = np.argmax(np.abs(v))
            if v[imax] < 0:
                eigvecs_fixed[:, i0] = -v
            continue

        V = eigvecs_fixed[:, i0:i1]     #subspace of degenerate eigenvectors
        RV = np.zeros_like(V)
        for c in range(k):
            RV[:, c] = rotation_pi_apply(V[:, c], Lx, Ly)

        Bsub = V.T @ RV
        Bsub = 0.5 * (Bsub + Bsub.T) # symmetrize

        lamB, U = scipy.linalg.eigh(Bsub)
        order = np.argsort(lamB)
        lamB = lamB[order]
        U = U[:, order]

        Vnew = V @ U

        # fix sign
        for r in range(k):
            v = Vnew[:, r]
            imax = np.argmax(np.abs(v))
            if v[imax] < 0:
                Vnew[:, r] = -v
        eigvecs_fixed[:, i0:i1] = Vnew
        block_info.append({'range': (i0, i1), 'multiplicity': k, 'rotation_eigs': lamB})
    return eigvecs_fixed, groups, block_info


