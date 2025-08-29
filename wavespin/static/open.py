""" Functions used for the open boundary conditions' calculations.
"""

import os
import scipy
import numpy as np
from pathlib import Path
from tqdm import tqdm

from wavespin.lattice.lattice import latticeClass, hamiltonianClass
from wavespin.tools import pathFinder as pf
from wavespin.tools import inputUtils as iu
from wavespin.static import periodic as pe
from wavespin.static import openCorrelators
from wavespin.static import momentumTransformation

class openSystem(hamiltonianClass):
    def __init__(self, p: iu.openParameters, termsHamiltonian, **kwargs):
        self.p = p
        # Construct lattice and Hamiltonian
        super().__init__(Lx=p.Lx,Ly=p.Ly,offSiteList=p.offSiteList,boundary='open',termsHamiltonian=termsHamiltonian,**kwargs)
        #
#        self.thetas,self.phis = self._quantizationAngles()
        self.theta, self.phi = pe.quantizationAxis(self.S,self.J_i,self.D_i,self.h_i)
        self.ts = pe.computeTs(self.theta,self.phi)       #All t-parameters for A and B sublattice
        #XT correlator parameters
        self.correlatorType = p.correlatorType
        self.perturbationSite = p.perturbationSite
        self.perturbationIndex = self.indexesMap.index(self.perturbationSite)
        self.magnonModes = p.magnonModes
        self.site0 = 0 #if h_t_i[0,0,0]<0 else 1     #decide sublattice A and B of reference lattice site
        self.fullTimeMeasure = 0.8     #measure time in ms
        self.nTimes = 401        #time steps after ramp for the measurement
        self.measureTimeList = np.linspace(0,self.fullTimeMeasure,self.nTimes)
        self.saveCorrelatorXT = p.saveCorrelatorXT
        #KW correlator parameters
        self.transformType = p.transformType
        self.nOmega = 2000
        self.saveCorrelatorKW = p.saveCorrelatorKW
        self.saveCorrelatorKW = p.saveCorrelatorKW
        self.plotCorrelatorKW = p.plotCorrelatorKW

    def _quantizationAngles(self):
        """ Import the angles from the MC simulation.
        """
        argsFn = ('MC',self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,self.Ns, 'open')
        dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
        solutionFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npz')
        if Path(solutionFn).is_file():
            Spins = np.load(solutionFn)['Spins']
            thetas = np.zeros(self.Ns)
            phis = np.zeros(self.Ns)
            for i in range(self.Ns):
                thetas[i], phis[i] = fs.vector_to_polar_angles(self.Spins[i])
        else:
            print("Quantization angles for this system (lattice and/or Hamiltonian parameters) has not been computed yet")
            print("First do the MC simulation, then come back")
            exit()
        return thetas, phis

    def bogoliubovTransformation(self,verbose=False):
        """ Compute the Bogoliubov transformation for the real-space Hamiltonian.
        Initialize U_, V_ and evals : bogoliubov transformation matrices U and V and eigenvalues.
        """
        Ns = self.Ns
        #
        argsFn = ('bogoliubov_rs',self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns)
        dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
        transformationFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npz')
        if not Path(transformationFn).is_file():
            hamiltonian = self.realSpaceHamiltonian()
            if np.max(np.absolute(hamiltonian-hamiltonian.conj()))>1e-5:
                raise ValueError("Hamiltonian is not real! Procedure might be wrong")
            # Para-diagonalization, see notes (appendix) for details
            A = hamiltonian[:Ns,:Ns]
            B = hamiltonian[:Ns,Ns:]
            try:
                K = scipy.linalg.cholesky(A-B)
            except:
                K = scipy.linalg.cholesky(A-B+np.identity(Ns)*1e-5)
            lam2,chi_ = scipy.linalg.eigh(K@(A+B)@K.T.conj())
            if self.p.excludeZeroMode:
                lam2[0] = 1
            self.evals = np.sqrt(lam2)         #dispersion -> positive
            #
            chi = chi_ / self.evals**(1/2)     #normalized eigenvectors: divide each column of chi_ by the corresponding eigenvalue -> of course for the gapless mode there is a problem here
            phi_ = K.T.conj()@chi
            psi_ = (A+B)@phi_/self.evals       # Problem also here
            self.U_ = 1/2*(phi_+psi_)
            self.V_ = 1/2*(phi_-psi_)
            if self.p.saveWf:
                if not Path(dataDn).is_dir():
                    print("Creating 'Data/' folder in home directory.")
                    os.system('mkdir '+dataDn)
                np.savez(transformationFn,awesomeU=self.U_,awesomeV=self.V_,evals=self.evals)
        else:
            if verbose:
                print("Loading Bogoliubov transformation from file: "+transformationFn)
            self.U_ = np.load(transformationFn)['awesomeU']
            self.V_ = np.load(transformationFn)['awesomeV']
            self.evals = np.load(transformationFn)['evals']

        if self.p.excludeZeroMode:       #Put to 0 the eigenstate corresponding to the zero energy mode -> a bit far fetched
            self.U_[:,0] *= 0
            self.V_[:,0] *= 0

    def realSpaceHamiltonian(self):
        """
        Compute the real space Hamiltonian -> (2Ns x 2Ns).
        Conventions for the real space wavefunction and parameters are in the notes.
        SECOND NEAREST-NEIGHBOR NOT IMPLEMENTED.
        Built for the rectangular shape and AFTER removed the rows/columns corresponding to off sites.

        Returns
        -------
        ham : 2Ns,2Ns matrix of real space Hamiltonian.
        """
        S = self.S
        Lx = self.Lx
        Ly = self.Ly
        ts = self.ts
        J_i = self.J_i
        D_i = self.D_i
        h_i = self.h_i
        offSiteList = self.offSiteList
        #
        p_zz = pe.computePs(0,0,ts,J_i,D_i,offSiteList,Lx,Ly)
        p_xx = pe.computePs(1,1,ts,J_i,D_i,offSiteList,Lx,Ly)
        p_yy = pe.computePs(2,2,ts,J_i,D_i,offSiteList,Lx,Ly)
        p_xy = pe.computePs(1,2,ts,J_i,D_i,offSiteList,Lx,Ly)
        fac0 = 1      #Need to change this in notes -> counting of sites from 2 to 1 sites per UC
        fac1 = 1
        fac2 = 2
        ham = np.zeros((2*Lx*Ly,2*Lx*Ly),dtype=complex)
        #p_zz sums over nn but is on-site -> problem when geometry is not square
        #diagonal
        ham[:Lx*Ly,:Lx*Ly] = abs(h_i)/fac0*np.cos(self.theta) - S/fac1*np.diag(np.sum(p_zz[0],axis=0))
        ham[Lx*Ly:,Lx*Ly:] = abs(h_i)/fac0*np.cos(self.theta) - S/fac1*np.diag(np.sum(p_zz[0],axis=0))
        #off_diag 1 - nn
        off_diag_1_nn = S/fac2*(p_xx[0]+p_yy[0])
        ham[:Lx*Ly,:Lx*Ly] += off_diag_1_nn
        ham[Lx*Ly:,Lx*Ly:] += off_diag_1_nn
        #off_diag 2 - nn
        off_diag_2_nn = S/fac2*(p_xx[0]-p_yy[0]+2*1j*p_xy[0])
        ham[:Lx*Ly,Lx*Ly:] += off_diag_2_nn
        ham[Lx*Ly:,:Lx*Ly] += off_diag_2_nn.T.conj().T
        #Remove offSiteList
        indexesToRemove = []
        for offTerm in offSiteList:
            ind = _idx(offTerm[0],offTerm[1])
            indexesToRemove.append(ind)
            indexesToRemove.append(ind + Lx*Ly)
        ham = np.delete(ham,indexesToRemove,axis=0)
        ham = np.delete(ham,indexesToRemove,axis=1)
        return ham

    def realSpaceCorrelator(self,verbose=False):
        """ Here we compute the correlator in real space.
        """
        Lx = self.Lx
        Ly = self.Ly
        Ns = self.Ns
        txtZeroEnergy = 'without0energy' if self.p.excludeZeroMode else 'with0energy'
        argsFn = ('correlatorXT_rs',self.correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy)
        dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
        correlatorFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npy')
        if not Path(correlatorFn).is_file():
            self.correlatorXT = np.zeros((Lx,Ly,self.nTimes),dtype=complex)
            #
            U = np.zeros((2*Ns,2*Ns),dtype=complex)
            U[:Ns,:Ns] = self.U_
            U[:Ns,Ns:] = self.V_
            U[Ns:,:Ns] = self.V_
            U[Ns:,Ns:] = self.U_
            #Correlator -> can make this faster, we actually only need U_ and V_
            exp_e = np.exp(-1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
            A = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[Ns:,Ns:],optimize=True)
            B = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[:Ns,Ns:],optimize=True)
            G = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[:Ns,Ns:],optimize=True)
            H = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[Ns:,Ns:],optimize=True)
            #
            for ind_i in range(Ns):
                ix, iy = self._xy(ind_i)
                if (ix,iy) in self.offSiteList:
                    continue
                self.correlatorXT[ix,iy] = openCorrelators.dicCorrelators[self.correlatorType](self,ind_i,A,B,G,H)
            if self.saveCorrelatorXT:
                if not Path(dataDn).is_dir():
                    print("Creating 'Data/' folder in home directory.")
                    os.system('mkdir '+dataDn)
                np.save(correlatorFn,self.correlatorXT)
        else:
            if verbose:
                print("Loading real-space correlator from file: "+correlatorFn)
            self.correlatorXT = np.load(correlatorFn)

    def realSpaceCorrelatorBond(self,verbose=False):
        """ Here we compute the correlator in real space for each bond, like for the jj.
        """
        Lx = self.Lx
        Ly = self.Ly
        Ns = self.Ns
        txtZeroEnergy = 'without0energy' if self.p.excludeZeroMode else 'with0energy'
        argsFn_h = ('correlator_horizontal_bonds',self.correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy)
        argsFn_v = ('correlator_vertical_bonds',self.correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy)
        dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
        correlatorFn_h = pf.getFilename(*argsFn_h,dirname=dataDn,extension='.npy')
        correlatorFn_v = pf.getFilename(*argsFn_v,dirname=dataDn,extension='.npy')
        if not Path(correlatorFn_h).is_file() or Path(correlatorFn_v).is_file():
            self.correlatorXT_h = np.zeros((Lx-1,Ly,self.nTimes),dtype=complex)
            self.correlatorXT_v = np.zeros((Lx,Ly-1,self.nTimes),dtype=complex)
            #
            U = np.zeros((2*Ns,2*Ns),dtype=complex)
            U[:Ns,:Ns] = self.U_
            U[:Ns,Ns:] = self.V_
            U[Ns:,:Ns] = self.V_
            U[Ns:,Ns:] = self.U_
            #Correlator -> can make this faster, we actually only need U_ and V_
            exp_e = np.exp(-1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
            A = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[Ns:,Ns:],optimize=True)
            B = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[:Ns,Ns:],optimize=True)
            G = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[:Ns,Ns:],optimize=True)
            H = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[Ns:,Ns:],optimize=True)
            #
            for ihx in range(Lx-1):
                for ihy in range(Ly):
                    ind_i = self._idx(ihx,ihy)
                    self.correlatorXT_h[ihx,ihy] = openCorrelators.jjCorrelatorBond(self,ind_i,A,B,G,H,'h')
            for ivx in range(Lx):
                for ivy in range(Ly-1):
                    ind_i = self._idx(ivx,ivy)
                    self.correlatorXT_v[ivx,ivy] = openCorrelators.jjCorrelatorBond(self,ind_i,A,B,G,H,'v')
            if self.saveCorrelatorXT:
                if not Path(dataDn).is_dir():
                    print("Creating 'Data/' folder in home directory.")
                    os.system('mkdir '+dataDn)
                np.save(correlatorFn_h,self.correlatorXT_h)
                np.save(correlatorFn_v,self.correlatorXT_v)
        else:
            if verbose:
                print("Loading real-space bond correlator from file: "+correlatorFn_h)
            self.correlatorXT_h = np.load(correlatorFn_h)
            self.correlatorXT_v = np.load(correlatorFn_v)

    def momentumSpaceCorrelator(self,verbose=False):
        """ Here we simply Fourier transform the correlator.
        """
        Lx = self.Lx
        Ly = self.Ly
        Ns = self.Ns
        txtZeroEnergy = 'without0energy' if self.p.excludeZeroMode else 'with0energy'
        argsFn = ('correlatorKW_rs',self.correlatorType,self.transformType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy)
        dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
        correlatorFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npy')
        if not Path(correlatorFn).is_file():
            self.correlatorKW = momentumTransformation.dicTransformType[self.transformType](self)
            if self.saveCorrelatorKW:
                if not Path(dataDn).is_dir():
                    print("Creating 'Data/' folder in home directory.")
                    os.system('mkdir '+dataDn)
                np.save(correlatorFn,self.correlatorKW)
        else:
            if verbose:
                print("Loading moemntum-space correlator from file: "+correlatorFn)
            self.correlatorKW = np.load(correlatorFn)

##########################################################
##########################################################

class openRamp():
    def __init__(self, systems : list[openSystem] = None):
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
        iterBog = tqdm(range(self.nP),desc="Computing Bogoliubov transformation and correlator") if verbose else range(self.nP)
        for i in iterBog:
            # Compute Bogoliubov transformation matrices and eigenvalues
            self.rampElements[i].bogoliubovTransformation()
            # Compute Correlators
            self.rampElements[i].realSpaceCorrelator()
            # Bond correlators
            self.rampElements[i].realSpaceCorrelatorBond()

    def correlatorsKW(self,verbose=False):
        """ Here we Fourier transform the XT correlators and plot them nicely.
        """
        iterKW = tqdm(range(self.nP),desc="Computing Fourier transformation of correlator") if verbose else range(self.nP)
        for i in iterKW:
            self.rampElements[i].momentumSpaceCorrelator()



















