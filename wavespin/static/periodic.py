""" Functions used in the periodic setting.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from wavespin.tools import pathFinder as pf
from wavespin.tools import inputUtils as iu
from wavespin.static import momentumTransformation

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


class periodicSystem:
    def __init__(self, p: iu.periodicParameters, termsHamiltonian):
        self.fullParameters = p
        #Lattice parameters
        self.Lx = p.Lx
        self.Ly = p.Ly
        self.gridRealSpace = np.stack(np.meshgrid(np.arange(self.Lx), np.arange(self.Ly), indexing="ij"), axis=-1)
        self.gridk = momentumGrid(self.Lx,self.Ly)#BZ grid
        self.Gamma = (
            np.cos(self.gridk[:,:,0])+np.cos(self.gridk[:,:,1]),  #cos(kx) + cos(ky)
            np.cos(self.gridk[:,:,0]+self.gridk[:,:,1])+np.cos(self.gridk[:,:,0]-self.gridk[:,:,1])  #cos(kx+ky) + cos(kx-ky)
        )
        #Hamiltonian parameters
        self.g1,self.g2,self.d1,self.d2,self.h = termsHamiltonian
        self.J = (self.g1,self.g2)
        self.D = (self.d1,self.d2)
        self.S = 0.5     #spin value
        self.theta,self.phi = quantizationAxis(self.S,self.J,self.D,self.h)
        self.ts = computeTs(self.theta,self.phi)       #All t-parameters for A and B sublattice
        self.dispersion = self._dispersion()
        self.rk = self._rk()
        self.phik = self._phik()
        #XT correlator parameters
        self.correlatorType = p.correlatorType
        self.fullTimeMeasure = 0.8     #measure time in ms
        self.nTimes = 401        #time steps after ramp for the measurement
        self.measureTimeList = np.linspace(0,self.fullTimeMeasure,self.nTimes)
        self.saveCorrelatorXT = p.saveCorrelatorXT
        #KW correlator parameters
        self.transformType = p.transformType
        self.nOmega = 2000
        self.saveCorrelatorKW = p.saveCorrelatorKW
        self.plotCorrelatorKW = p.plotCorrelatorKW
        self.saveFigureCorrelatorKW = p.saveFigureCorrelatorKW

    def _xy(self, i):
        return i // self.Ly, i % self.Ly

    def _idx(self, x, y):
        return x*self.Ly + y

    def _dispersion(self):
        """
        Compute dispersion epsilon as in notes.
        Controls are neded for ZZ in k.
        """
        N_11 = self._N11()
        N_12 = self._N12()
        result = np.sqrt(N_11**2-np.absolute(N_12)**2,where=(N_11**2>=np.absolute(N_12)**2))
    #    result[N_11**2<np.absolute(N_12)**2] = 0
        return result

    def _N11(self):
        """ Compute N_11 as in notes. """
        p_zz = computePs(0,0,self.ts,self.J,self.D)
        p_xx = computePs(1,1,self.ts,self.J,self.D)
        p_yy = computePs(2,2,self.ts,self.J,self.D)
        result = self.h/2*np.cos(self.theta)
        for i in range(2):
            result += self.S*(self.Gamma[i]*(p_xx[i]+p_yy[i])/2-2*p_zz[i])
        return result

    def _N12(self):
        """ Compute N_12 as in notes. """
        p_xx = computePs(1,1,self.ts,self.J,self.D)
        p_yy = computePs(2,2,self.ts,self.J,self.D)
        p_xy = computePs(1,2,self.ts,self.J,self.D)
        result = 0
        for i in range(2):
            result += self.S/2*self.Gamma[i]*(p_xx[i]-p_yy[i]-2*1j*p_xy[i])
        return result

    def _rk(self):
        """
        Compute rk as in notes.
        Controls are neded for ZZ in k.
        """
        N_11 = self._N11()
        N_12 = self._N12()
        frac = np.divide(np.absolute(N_12),N_11,where=(N_11!=0))
        result = -1/2*np.arctanh(frac,where=(frac<1))
        result[frac>=1] = np.nan
        return result

    def _phik(self):
        """Compute e^i*phik as in notes."""
        N_12 = self._N12()
        result = np.exp(1j*np.angle(N_12))
        return result

    def realSpaceCorrelator(self,verbose=False):
        """ Here we compute the real space correlator: see notes for details.
        """
        Lx = self.Lx
        Ly = self.Ly
        argsFn = ('correlatorXT',self.correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly)
        dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
        correlatorFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npy')
        if not Path(correlatorFn).is_file():
            self.correlatorXT = np.zeros((Lx,Ly,self.nTimes),dtype=complex)
            exp_e = np.exp(-1j*2*np.pi*self.measureTimeList[None,None,:]*self.dispersion[:,:,None])
            exp_e = np.reshape(exp_e, shape=(Lx*Ly,self.nTimes))
            rk = self.rk.reshape(Lx*Ly)
            phik = self.phik.reshape(Lx*Ly)
            cosh_rk = np.cosh(rk)
            cosh_rk[np.isnan(cosh_rk)] = 0
            cosh_rk[np.absolute(cosh_rk)>1e3] = 0
            sinh_rk = np.sinh(rk)
            sinh_rk[np.isnan(sinh_rk)] = 0
            sinh_rk[np.absolute(sinh_rk)>1e3] = 0
            tj = self.ts[0]
            for ii in range(Lx*Ly): #take r_j=0 -> loop only over i. Good because we have PBC
                ix,iy = self._xy(ii)
                ti = self.ts[(ix+iy)%2]
                exp_k = (np.exp(-1j*np.dot(self.gridk,self.gridRealSpace[ix,iy]))).reshape(Lx*Ly)
                corr1 = self.S/2/Lx/Ly*np.einsum('kt,k,k->t',
                                         exp_e,exp_k,
        #                                 ((ti[1][2]-1j*ti[2][2])*cosh_rk + (ti[1][2]+1j*ti[2][2])*phik.conj()*sinh_rk) * ((tj[1][2]+1j*tj[2][2])*cosh_rk + (tj[1][2]-1j*tj[2][2])*phik*sinh_rk),
                                         ti[1][2]*tj[1][2]*(np.absolute(cosh_rk+phik.conj()*sinh_rk)**2),
                                         optimize=True)
                corr2 = np.sum( ti[0][2]*tj[0][2]*(self.S-1/Lx/Ly*sinh_rk**2) )*0
                corr3 = ti[0][2]*tj[0][2]/((Lx*Ly)**2)*(np.einsum('kt,qt,k,q,k,q->t',
                                                           exp_e,exp_e,exp_k,exp_k,
                                                           sinh_rk**2,cosh_rk**2,
                                                           optimize=True) +
                                                 np.einsum('kt,qt,k,q,k,q->t',
                                                           exp_e,exp_e,exp_k,exp_k,
                                                           phik.conj()*cosh_rk*sinh_rk,phik*cosh_rk*sinh_rk,
                                                           optimize=True)*2 )
                self.correlatorXT[ix,iy] = 2*1j*np.imag(corr1 + corr2 + corr3)
            if self.saveCorrelatorXT:
                if not Path(dataDn).is_dir():
                    print("Creating 'Data/' folder in home directory.")
                    os.system('mkdir '+dataDn)
                np.save(correlatorFn,self.correlatorXT)
        else:
            if verbose:
                print("Loading real-space correlator from file: "+correlatorFn)
            self.correlatorXT = np.load(correlatorFn)

    def momentumSpaceCorrelator(self,verbose=False):
        """ Here we simply Fourier transform the correlator.
        """
        Lx = self.Lx
        Ly = self.Ly
        argsFn = ('correlatorKW',self.correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly)
        dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
        correlatorFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npy')
        if not Path(correlatorFn).is_file():
            self.correlatorKW = momentumTransformation.dicTransformType['fft'](self)
            if self.saveCorrelatorKW:
                if not Path(dataDn).is_dir():
                    print("Creating 'Data/' folder in home directory.")
                    os.system('mkdir '+dataDn)
                np.save(correlatorFn,self.correlatorKW)
        else:
            if verbose:
                print("Loading moemntum-space correlator from file: "+correlatorFn)
            self.correlatorKW = np.load(correlatorFn)

#######################################################################
#######################################################################

class periodicRamp():
    def __init__(self, systems : list[periodicSystem] = None):
        """ systems should be a list of openSystem objects. """
        self.rampElements = systems or []
        self.nP = len(self.rampElements)

    def addSystem(self,system):
        """ Custom function to add an element to the ramp """
        self.rampElements.append(system)
        self.nP = len(self.rampElements)

    def correlatorsXT(self,verbose=False):
        """ Compute correlators in real space for each system in the ramp.
        """
        iterCorr = tqdm(range(self.nP),desc="Computing correlator") if verbose else range(self.nP)
        for i in iterCorr:
            # Compute Correlators
            self.rampElements[i].realSpaceCorrelator()

    def correlatorsKW(self,verbose=False):
        """ Here we Fourier transform the XT correlators and plot them nicely.
        """
        iterKW = tqdm(range(self.nP),desc="Computing Fourier transformation of correlator") if verbose else range(self.nP)
        for i in iterKW:
            self.rampElements[i].momentumSpaceCorrelator()




