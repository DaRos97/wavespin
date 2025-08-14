""" Functions used in the periodic setting.
"""

import numpy as np

def momentumGrid(Lx,Ly):
    """ Compute momenta in the Brillouin zone for a (periodic) rectangular shape.

    Parameters
    ----------
    Lx,Ly : int, linear size.

    Returns
    -------
    gridk : (Lx,Ly,2)-array.
    """
    dx = 2*np.pi/Lx
    dy = 2*np.pi/Ly
    gridk = np.zeros((Lx,Ly,2))
    for i1 in range(Lx):
        for i2 in range(Ly):
            gridk[i1,i2,0] = dx*(1+i1) #- np.pi
            gridk[i1,i2,1] = dy*(1+i2) #- np.pi
    return gridk

def neighborDispersion(gridk):
    r""" Compute the '$\Gamma$' dispersion at first and second nearest neighbor.

    Parameters
    ----------
    gridk : (Lx,Ly,2)-array
        Momenta of BZ.

    Returns
    -------
    Gamma : 2-tuple
        Dispersions at 1st and 2nd nearest neighbor.
    """
    Gamma1 = np.cos(gridk[:,:,0])+np.cos(gridk[:,:,1])  #cos(kx) + cos(ky)
    Gamma2 = np.cos(gridk[:,:,0]+gridk[:,:,1])+np.cos(gridk[:,:,0]-gridk[:,:,1])  #cos(kx+ky) + cos(kx-ky)
    Gamma = (Gamma1,Gamma2)
    return Gamma

def quantizationAxis(S,J_i,D_i,h_i):
    r""" Compute angles theta and phi of quantization axis depending on Hamiltonian parameters.
    Works for both uniform and site-dependent Hamiltonian parameters, where we take the average.
    In PBC each site has the same Q-axis, in OBC there could be border effects.

    Parameters
    ----------
    S : float, spin size.
    J,D,h : (Lx,Ly)-arrays of Hamiltonian parameters.

    Returns
    -------
    angles : 2-tuple.
        $\theta$,$\phi$ : polar and azimuthal angle.
    """
    if type(J_i[0]) in [float,int,np.float64]:   #if we give a single number for J1,J2,H etc.. -> static_dispersion.py
        J = J_i
        D = D_i
        h = h_i
    else:   #f we give a site dependent value of J1, J2 etc.., we need an average -> static_ZZ_*.py
        J = []
        D = []
        for i in range(2):
            if not (J_i[i] == np.zeros(J_i[i].shape)).all():
                J.append(abs(float(np.sum(J_i[i])/(J_i[i][np.nonzero(J_i[i])]).shape)))
            else:
                J.append(0)
            if not (D_i[i] == np.zeros(D_i[i].shape)).all():
                D.append(float(np.sum(D_i[i])/(D_i[i][np.nonzero(D_i[i])]).shape))
            else:
                D.append(0)
        if J[0]!=0:
            D[0] = D[0]/J[0]        #As we defined in notes
        if J[1]!=0:
            D[1] = D[1]/J[1]        #As we defined in notes
        if not (h_i == np.zeros(h_i.shape)).all():
            h_av = float(np.sum(h_i)/(h_i[np.nonzero(h_i)]).shape)
            h_stag = np.absolute(h_i[np.nonzero(h_i)]-h_av)
            h = float(np.sum(h_stag)/(h_stag[np.nonzero(h_stag)]).shape)
        else:
            h = 0
    if J[1]<J[0]/2 and h<4*S*(J[0]*(1-D[0])-J[1]*(1-D[1])):
        theta = np.arccos(h/(4*S*(J[0]*(1-D[0])-J[1]*(1-D[1]))))
    else:
        theta = 0
    #
    phi = 0
    angles = (theta, phi)
    return angles

def computeN11(*pars):
    """ Compute N_11 as in notes. """
    S,Gamma,h,ts,theta,phi,J,D = pars
    p_zz = computePs(0,0,ts,J,D)
    p_xx = computePs(1,1,ts,J,D)
    p_yy = computePs(2,2,ts,J,D)
    result = h/2*np.cos(theta)
    for i in range(2):
        result += S*(Gamma[i]*(p_xx[i]+p_yy[i])/2-2*p_zz[i])
    return result

def computeN12(*pars):
    """ Compute N_12 as in notes. """
    S,Gamma,h,ts,theta,phi,J,D = pars
    p_xx = computePs(1,1,ts,J,D)
    p_yy = computePs(2,2,ts,J,D)
    p_xy = computePs(1,2,ts,J,D)
    result = 0
    for i in range(2):
        result += S/2*Gamma[i]*(p_xx[i]-p_yy[i]-2*1j*p_xy[i])
    return result

def computePs(alpha,beta,ts,J,D,offSiteList=[],Lx=0,Ly=0,order='c-Neel'):
    """ Compute coefficient p_gamma^{alpha,beta} for a given classical order.
    alpha,beta=0,1,2 -> z,x,y like for ts.
    J and D are tuple with 1st and 2nd nn.
    Each can be either a number or a Ns*Ns matrix of values for site dependent case.

    Parameters
    ----------

    Returns
    -------
    """
    if order=='c-Neel': #nn: A<->B, nnn: A<->A
        #Nearest neighor
        nn =  J[0]*ts[0][alpha][0]*ts[1][beta][0] + J[0]*ts[0][alpha][1]*ts[1][beta][1] + J[0]*D[0]*ts[0][alpha][2]*ts[1][beta][2]
        nnn = J[1]*ts[0][alpha][0]*ts[0][beta][0] + J[1]*ts[0][alpha][1]*ts[0][beta][1] + J[1]*D[1]*ts[0][alpha][2]*ts[0][beta][2]
    for offTerm in offSiteList:
        ind = offTerm[0]*Ly + offTerm[1]
        nn[:,ind] *= 0
        nn[ind,:] *= 0
    return (nn,nnn)

def computeTs(theta,phi):
    """ Compute the parameters t_z, t_x and t_y as in notes for sublattice A and B.
    Sublattice A has negative magnetic feld.
    """
    result = [
        [   #sublattice A
            (np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)),  #t_zx,t_zy,t_zz
            (np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)), #t_xx,t_xy,t_xz
            (-np.sin(phi),np.cos(phi),0)  ],                                      #t_yx,t_yy,t_yz
        [   #sublattice B
            (-np.sin(theta)*np.cos(phi),-np.sin(theta)*np.sin(phi),-np.cos(theta)),  #t_zx,t_zy,t_zz
            (-np.cos(theta)*np.cos(phi),-np.cos(theta)*np.sin(phi),np.sin(theta)),   #t_xx,t_xy,t_xz
            (-np.sin(phi),np.cos(phi),0)  ],                                         #t_yx,t_yy,t_yz
    ]
    return result

def computeEpsilon(*pars):
    """
    Compute dispersion epsilon as in notes.
    Controls are neded for ZZ in k.
    """
    N_11 = computeN11(*pars)
    N_12 = computeN12(*pars)
    result = np.sqrt(N_11**2-np.absolute(N_12)**2,where=(N_11**2>=np.absolute(N_12)**2))
#    result[N_11**2<np.absolute(N_12)**2] = 0
    return result

def computeGsE(epsilon,*pars):
    """Compute ground state energy as in notes."""
    E_0 = computeE0(*pars)
    Ns = epsilon.shape[0]*epsilon.shape[1]
#    return E_0 + np.sum(epsilon[~np.isnan(epsilon)])/Ns
    return E_0 + np.sum(epsilon)/Ns

def computeE0(*pars):
    r""" Compute $E_0$ as in notes.
    """
    S,Gamma,h,ts,theta,phi,J,D = pars
    p_zz = computePs(0,0,ts,J,D)
    result = -h*(S+1/2)*np.cos(theta)
    for i in range(2):
        result += 2*S*(S+1)*p_zz[i]
    return result

def computeSolution(Lx,Ly,S,HamiltonianParameters):
    """ Compute dispersion with a given set of Hamiltonian parameters

    Parameters
    ----------

    Returns
    -------
    epsilon : (Lx,Ly)-array, dispersion.
    (theta,phi) : floats, quantizatio axis.
    gsE : float, ground state energy.
    gap : float, gap.
    """
    J,D,h = HamiltonianParameters
    gridk = momentumGrid(Lx,Ly)
    Gamma = neighborDispersion(gridk)
    theta,phi = quantizationAxis(S,J,D,h)
    ts = computeTs(theta,phi)
    parameters = (S,Gamma,h,ts,theta,phi,J,D)
    epsilon = computeEpsilon(*parameters)
    gsE = computeGsE(epsilon,*parameters)
    gap = np.min(epsilon)
    return epsilon, (theta,phi), gsE, gap







