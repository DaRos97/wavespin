""" Functions used for the open boundary conditions' calculations.
"""

import os
import scipy
import numpy as np
from pathlib import Path
from tqdm import tqdm
from time import time

from wavespin.lattice.lattice import latticeClass
from wavespin.tools import pathFinder as pf
from wavespin.tools import inputUtils as iu
from wavespin.tools.functions import lorentz
from wavespin.static import openCorrelators
from wavespin.static import momentumTransformation
from wavespin.static.periodic import quantizationAxis
from wavespin.plots.rampPlots import *

class openHamiltonian(latticeClass):
    def __init__(self, p: iu.myParameters):
        super().__init__(p)
        self.p = p
        #Hamiltonian parameters
        self.g1,self.g2,self.d1,self.d2,self.h,self.h_disorder = p.dia_Hamiltonian
        self.S = 0.5     #spin value
        self.g_i = (self._NNterms(self.g1), self._NNNterms(self.g2))
        self.D_i = (self._NNterms(self.d1), self._NNNterms(self.d2))
        self.h_i = self._Hterms(self.h,self.h_disorder)
#        self.realSpaceHamiltonian = self._realSpaceHamiltonian()

    def _NNterms(self,val):
        """ Construct Ns,Ns matrix for real space Hamiltonian using the lattice nn.
        """
        vals = np.zeros((self.Ns,self.Ns))
        for i in range(self.Ns):
            vals[i,self.NN[i]] = val
        return vals

    def _NNNterms(self,val):
        """ Construct Ns,Ns matrix for real space Hamiltonian using the lattice nnn.
        """
        vals = np.zeros((self.Lx*self.Ly,self.Lx*self.Ly))
        for i in range(self.Ns):
            vals[i,self.NNN[i]] = val
        return vals

    def _Hterms(self,val,disorder_val):
        vals = np.zeros((self.Lx*self.Ly,self.Lx*self.Ly))
        disorder = (np.random.rand(self.Ns)-0.5)*2 * disorder_val
        for ix in range(self.Lx):
            for iy in range(self.Ly):
                ind = self._idx(ix,iy)
                vals[ind,ind] = -(-1)**(ix+iy) * val + disorder[ind]
        return vals

    def quantizationAxisAngles(self,verbose=False):
        """ Here we get the quantization axis angles to use for the diagonalization.
        """
        self.theta, self.phi = quantizationAxis(self.S,self.g_i,self.D_i,self.h_i)
        if self.p.dia_uniformQA:
            self.thetas = np.ones(self.Ns)*self.theta
            self.phis = np.ones(self.Ns)*self.phi
        else:
            argsFn = ('minimization_solution',self.g1,self.g2,self.d1,self.d2,self.h,self.h_disorder,self.Lx,self.Ly,self.Ns,'open')
            anglesFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npy')
            if not Path(anglesFn).is_file():
                print("File of quantization axis angles not found: "+anglesFn)
                print("computing it now..")
                from wavespin.classicSpins.minimization import minHam
                from wavespin.tools.inputUtils import classicParameters
                parameters = classicParameters()
                setattr(parameters,'saveSolution',True)
                setattr(parameters,'Lx',self.Lx)
                setattr(parameters,'Ly',self.Ly)
                setattr(parameters,'offSiteList',self.offSiteList)
                simulation = minHam(parameters,(self.g1,self.g2,self.d1,self.d2,self.h,self.h_disorder))
                simulation.minimization(verbose=verbose)

            self.thetas = np.load(anglesFn)
            for i in range(self.Ns):
                ix,iy = self._xy(i)
                if (ix+iy)%2==1:
                    self.thetas[i] += np.pi
            self.phis = np.zeros(self.Ns)

    def computeTs(self):
        """ Compute the parameters t_z, t_x and t_y as in notes for sublattice A and B.
        Sublattice A has negative magnetic feld.
        """
        self.ts = np.zeros((self.Ns,3,3))
        for i in range(self.Ns):
            ix,iy = self._xy(i)
            sign = (-1)**(ix+iy)
            th = self.thetas[i]
            ph = self.phis[i]
            self.ts[i,0] = sign*np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th) ]) #t_zx,t_zy,t_zz
            self.ts[i,1] = sign*np.array([np.cos(th)*np.cos(ph),np.cos(th)*np.sin(ph),-np.sin(th)]) #t_xx,t_xy,t_xz
            self.ts[i,2] =      np.array([-np.sin(ph)          ,np.cos(ph)           ,0             ]) #t_yx,t_yy,t_yz

    def computePs(self,order='c-Neel'):
        """ Compute coefficient p_gamma^{alpha,beta}_ij for a given classical order.
        alpha,beta=0,1,2 -> z,x,y like for ts.

        Parameters
        ----------

        Returns
        -------
        """
        self.Ps = np.zeros((2,self.Ns,self.Ns,3,3))     # number of nearest-neighbor(2), Ns, Ns, zxy, xyz 
        vecGnn = np.array([self.g_i[0],self.g_i[0],self.g_i[0]*self.D_i[0]])        #3,Ns,Ns
        vecGnnn = np.array([self.g_i[1],self.g_i[1],self.g_i[1]*self.D_i[1]])
        if order=='c-Neel': #nn: A<->B, nnn: A<->A
            for i in range(self.Ns):
                #nn
                for j in self.NN[i]:
                    self.Ps[0,i,j] = np.einsum('d,ad,bd->ab',vecGnn[:,i,j],self.ts[i],self.ts[j],optimize=True)
                #nnn
                for j in self.NNN[i]:
                    self.Ps[1,i,j] = np.einsum('d,ad,bd->ab',vecGnnn[:,i,j],self.ts[i],self.ts[j],optimize=True)
        for offTerm in self.offSiteList:
            ind = offTerm[0]*Ly + offTerm[1]
            nn[:,ind] *= 0
            nn[ind,:] *= 0

    def _realSpaceHamiltonian(self,verbose=False):
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
        g_i = self.g_i
        D_i = self.D_i
        h_i = self.h_i
        offSiteList = self.offSiteList
        self.quantizationAxisAngles(verbose)
        self.computeTs()
        self.computePs()
        p_zz = self.Ps[0,:,:,0,0]
        p_xx = self.Ps[0,:,:,1,1]
        p_yy = self.Ps[0,:,:,2,2]
        #
        ham = np.zeros((2*Lx*Ly,2*Lx*Ly),dtype=complex)
        #p_zz sums over nn but is on-site -> problem when geometry is not rectangular
        ham[:Lx*Ly,:Lx*Ly] = abs(h_i)*np.cos(np.diag(self.thetas)) / S - np.diag(np.sum(p_zz,axis=1)) / 2 / S
        ham[Lx*Ly:,Lx*Ly:] = abs(h_i)*np.cos(np.diag(self.thetas)) / S - np.diag(np.sum(p_zz,axis=1)) / 2 / S
        #off_diag 1 - nn
        off_diag_1_nn = (p_xx+p_yy) / 4 / S
        ham[:Lx*Ly,:Lx*Ly] += off_diag_1_nn
        ham[Lx*Ly:,Lx*Ly:] += off_diag_1_nn
        #off_diag 2 - nn
        off_diag_2_nn = (p_xx-p_yy) / 4 / S
        ham[:Lx*Ly,Lx*Ly:] += off_diag_2_nn
        ham[Lx*Ly:,:Lx*Ly] += off_diag_2_nn.T.conj()
        #Remove offSiteList
        indexesToRemove = []
        for offTerm in offSiteList:
            ind = _idx(offTerm[0],offTerm[1])
            indexesToRemove.append(ind)
            indexesToRemove.append(ind + Lx*Ly)
        ham = np.delete(ham,indexesToRemove,axis=0)
        ham = np.delete(ham,indexesToRemove,axis=1)
        return ham

    def diagonalize(self,verbose=False,**kwargs):
        """ Compute the Bogoliubov transformation for the real-space Hamiltonian.
        Initialize U_, V_ and evals : bogoliubov transformation matrices U and V and eigenvalues.
        """
        Ns = self.Ns
        #
        argsFn = ('bogoliubov_rs',self.g1,self.g2,self.d1,self.d2,self.h,self.h_disorder,self.Lx,self.Ly,self.Ns)
        transformationFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npz')
        hamiltonian = self._realSpaceHamiltonian(verbose)
        if not Path(transformationFn).is_file():
            if np.max(np.absolute(hamiltonian-hamiltonian.conj()))>1e-5:
                raise ValueError("Hamiltonian is not real! Procedure might be wrong")
            # Para-diagonalization, see notes (appendix) for details
            A = hamiltonian[:Ns,:Ns]
            B = hamiltonian[:Ns,Ns:]
            try:
                K = scipy.linalg.cholesky(A-B)
            except:
                K = scipy.linalg.cholesky(A-B+np.identity(Ns)*1e-4)
            lam2,chi_ = scipy.linalg.eigh(K@(A+B)@K.T.conj())
            if self.p.dia_excludeZeroMode:
                lam2[0] = 1
            self.evals = np.sqrt(lam2)         #dispersion -> positive
            #
            chi = chi_ / self.evals**(1/2)     #normalized eigenvectors: divide each column of chi_ by the corresponding eigenvalue -> of course for the gapless mode there is a problem here
            phi_ = K.T.conj()@chi
            psi_ = (A+B)@phi_/self.evals       # Problem also here
            self.U_ = 1/2*(phi_+psi_)
            self.V_ = 1/2*(phi_-psi_)
            if self.p.dia_saveWf:
                if not Path(self.dataDn).is_dir():
                    print("Creating 'Data/' folder in directory: "+self.dataDn)
                    os.system('mkdir '+self.dataDn)
                np.savez(transformationFn,awesomeU=self.U_,awesomeV=self.V_,evals=self.evals)
        else:
            if verbose:
                print("Loading Bogoliubov transformation from file: "+transformationFn)
            self.U_ = np.load(transformationFn)['awesomeU']
            self.V_ = np.load(transformationFn)['awesomeV']
            self.evals = np.load(transformationFn)['evals']

        if self.p.dia_excludeZeroMode:       #Put to 0 the eigenstate corresponding to the zero energy mode -> a bit far fetched
            self.U_[:,0] *= 0
            self.V_[:,0] *= 0
        if self.p.dia_plotWf:
            plotWf(self)
            plotWfCos(self)
        if self.p.dia_plotMomenta:
            plotBogoliubovMomenta(self,**kwargs)

    def decayRates(self,temperature,verbose=False):
        """ Compute decay rates.
        Use sca_broadening * mean energy difference for the broadening of the energy delta function.
        """
        if (self.J_i[1] != 0).any() or (self.D_i[1] != 0).any():
            raise ValueError("Decay rates are so far only implemented for first nearest neighbor couplings.")
        edif = self.evals[2:] - self.evals[1:-1]
        gamma = self.p.sca_broadening * np.mean(edif)
        #
        evals = self.evals[1:]
        U = np.real(self.U_)[:,1:]
        V = np.real(self.V_)[:,1:]
        # Parameters
        p_xz = self.J_i[0] * np.sqrt(self.S/8) * (1-self.D_i[0]) * np.sin(2 * self.thetas)
        p_x = -self.J_i[0] * self.S**2 * (np.cos(self.thetas)**2 + self.D_i[0]*np.sin(self.thetas)**2)
        p_y = self.J_i[0] * self.S**2
        p_z = -self.J_i[0] * self.S**2 * (np.sin(self.thetas)**2 + self.D_i[0]*np.cos(self.thetas)**2)
        f1 = -(p_x+p_y)/(8*self.S**2)
        f2 = (p_y-p_x)/(8*self.S**2)
        f3 = p_z/self.S**2
        # Filename and looping types
        self.dataScattering = {}
        argsFn = ['scatteringVertex',temperature,self.g1,self.g2,self.d1,self.d2,self.h,self.h_disorder,self.Lx,self.Ly,self.Ns,self.p.sca_broadening]
        for scatteringType in self.p.sca_types:
            argsFnS = argsFn.insert(1,scatteringType)
            if verbose:
                print("\nScattering %s"%scatteringType)
            vertexFn = pf.getFilename(*tuple(argsFn),dirname=self.dataDn,extension='.npy')
            if Path(vertexFn).is_file():
                if verbose:
                    print("Loading from file")
                self.dataScattering[scatteringType] = np.load(vertexFn)
                continue
            ti = time()
            if scatteringType == '1to2':
                Omega = 1/2 * (U[:,None,:,None] * V[None,:,None,:] + U[:,None,None,:]*V[None,:,:,None])
                Kappa = 1/2 * (U[:,None,:,None] * U[None,:,None,:] + V[:,None,:,None]*V[None,:,None,:])
                Vn_lm =  np.einsum('ij,iilm,jn->nlm',p_xz,Omega,U+V,optimize=True)
                Vn_lm += np.einsum('ij,iiln,jm->nlm',p_xz,Kappa,U+V,optimize=True)
                Vn_lm += np.einsum('ij,iimn,jl->nlm',p_xz,Kappa,U+V,optimize=True)
                # i <-> j
                Vn_lm *= 2
                en = evals[:,None,None]
                el = evals[None,:,None]
                em = evals[None,None,:]
                ## 1 -> 2
                # Delta
                arg = en - el -  em
                delta_vals = lorentz(arg, gamma)
                # Bose Factor
                if temperature != 0:
                    beta = 1/temperature
                    bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*(em+el)) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1)
                else:
                    bose_factor = 1
                # Decay rate
                Gamma_1to2 = 2 * np.pi * np.einsum('nlm,nlm,nlm->n',Vn_lm**2,delta_vals,bose_factor)
                ## 1 <- 2
                # Delta
                arg = en + em - el
                delta_vals = lorentz(arg, gamma)
                # Bose Factor
                if temperature != 0:
                    beta = 1/temperature
                    bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*el) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1)
                else:
                    bose_factor = 1
                # Decay rate
                Gamma_2to1 = 4 * np.pi * np.einsum('lnm,nlm,nlm->n',Vn_lm**2,delta_vals,bose_factor)
                # Final
                Gamma_n = Gamma_1to2 + Gamma_2to1
            if scatteringType == '1to3':
                # Terms I need
                Omega = 1/2 * (U[:,None,:,None] * V[None,:,None,:] + U[:,None,None,:]*V[None,:,:,None])
                Kappa = 1/2 * (U[:,None,:,None] * U[None,:,None,:] + V[:,None,:,None]*V[None,:,None,:])
                OmS = Omega + np.transpose(Omega,(1,0,2,3))
                KaS = Kappa + np.transpose(Kappa,(0,1,3,2))
                # f1
                Vn_lmr =  np.einsum('ij,jjlm,ijrn->nlmr',f1,Omega,KaS,optimize=True)
                Vn_lmr += np.einsum('ij,jjlr,ijmn->nlmr',f1,Omega,KaS,optimize=True)
                Vn_lmr += np.einsum('ij,jjmr,ijln->nlmr',f1,Omega,KaS,optimize=True)
                Vn_lmr += np.einsum('ij,jjln,ijmr->nlmr',f1,Kappa,OmS,optimize=True)
                Vn_lmr += np.einsum('ij,jjmn,ijlr->nlmr',f1,Kappa,OmS,optimize=True)
                Vn_lmr += np.einsum('ij,jjrn,ijml->nlmr',f1,Kappa,OmS,optimize=True)
                # f2                                   
                Vn_lmr += np.einsum('ij,jjlm,ijnr->nlmr',f2,Omega,OmS,optimize=True)
                Vn_lmr += np.einsum('ij,jjlr,ijnm->nlmr',f2,Omega,OmS,optimize=True)
                Vn_lmr += np.einsum('ij,jjmr,ijnl->nlmr',f2,Omega,OmS,optimize=True)
                Vn_lmr += np.einsum('ij,jjln,ijmr->nlmr',f2,Kappa,KaS,optimize=True)
                Vn_lmr += np.einsum('ij,jjmn,ijlr->nlmr',f2,Kappa,KaS,optimize=True)
                Vn_lmr += np.einsum('ij,jjrn,ijml->nlmr',f2,Kappa,KaS,optimize=True)
                # f3                                   
                Vn_lmr += np.einsum('ij,jjlm,iinr->nlmr',f3/2,Omega,Kappa,optimize=True)
                Vn_lmr += np.einsum('ij,jjlr,iimn->nlmr',f3/2,Omega,Kappa,optimize=True)
                Vn_lmr += np.einsum('ij,jjmr,iiln->nlmr',f3/2,Omega,Kappa,optimize=True)
                Vn_lmr += np.einsum('ij,jjln,iimr->nlmr',f3/2,Kappa,Omega,optimize=True)
                Vn_lmr += np.einsum('ij,jjmn,iilr->nlmr',f3/2,Kappa,Omega,optimize=True)
                Vn_lmr += np.einsum('ij,jjrn,iiml->nlmr',f3/2,Kappa,Omega,optimize=True)
                # i <-> j and symmetrization
                Vn_lmr /= 3

                en = evals[:,None,None,None]
                el = evals[None,:,None,None]
                em = evals[None,None,:,None]
                er = evals[None,None,None,:]
                ## 1->3
                # Delta
                arg = en - el - em - er
                delta_vals = lorentz(arg,gamma)
                # Bose factor
                if temperature != 0:
                    beta = 1/temperature
                    bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*(el+em+er)) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1) / (np.exp(beta*er)-1)
                else:
                    bose_factor = 1
                # Decay rate
                Gamma_1to3 = 6 * np.pi * np.einsum('nlmr,nlmr,nlmr->n',Vn_lmr**2,delta_vals,bose_factor)
                ## 3->1
                # Delta
                arg = el - en - em - er
                delta_vals = lorentz(arg,gamma)
                # Bose factor
                if temperature != 0:
                    beta = 1/temperature
                    bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*el) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1) / (np.exp(beta*er)-1)
                else:
                    bose_factor = 1
                # Decay rate
                Gamma_3to1 = 18 * np.pi * np.einsum('lnmr,nlmr,nlmr->n',Vn_lmr**2,delta_vals,bose_factor)
                # Total
                Gamma_n = Gamma_1to3 + Gamma_3to1
            if scatteringType in ['2to2a','2to2b']:
                # Terms I need
                Omega = 1/2 * (U[:,None,:,None] * V[None,:,None,:] + U[:,None,None,:]*V[None,:,:,None])
                Kappa = 1/2 * (U[:,None,:,None] * U[None,:,None,:] + V[:,None,:,None]*V[None,:,None,:])
                OmS = Omega + np.transpose(Omega,(1,0,2,3))
                KaS = Kappa + np.transpose(Kappa,(0,1,3,2))
                # f1
                Vnr_lm =  np.einsum('ij,jjlm,ijrn->nrlm',f1,Omega,OmS,optimize=True)
                Vnr_lm += np.einsum('ij,jjnr,ijlm->nrlm',f1,Omega,OmS,optimize=True)
                Vnr_lm += np.einsum('ij,jjlr,ijmn->nrlm',f1,Kappa,KaS,optimize=True)
                Vnr_lm += np.einsum('ij,jjmr,ijln->nrlm',f1,Kappa,KaS,optimize=True)
                Vnr_lm += np.einsum('ij,jjln,ijmr->nrlm',f1,Kappa,KaS,optimize=True)
                Vnr_lm += np.einsum('ij,jjmn,ijlr->nrlm',f1,Kappa,KaS,optimize=True)
                # f2
                Vnr_lm += np.einsum('ij,jjlm,ijrn->nrlm',f2,Omega,KaS,optimize=True)
                Vnr_lm += np.einsum('ij,jjnr,ijlm->nrlm',f2,Omega,KaS,optimize=True)
                Vnr_lm += np.einsum('ij,jjlr,ijnm->nrlm',f2,Kappa,OmS,optimize=True)
                Vnr_lm += np.einsum('ij,jjmr,ijnl->nrlm',f2,Kappa,OmS,optimize=True)
                Vnr_lm += np.einsum('ij,jjln,ijrm->nrlm',f2,Kappa,OmS,optimize=True)
                Vnr_lm += np.einsum('ij,jjmn,ijrl->nrlm',f2,Kappa,OmS,optimize=True)
                # f3
                Vnr_lm += np.einsum('ij,jjlm,iinr->nrlm',f3/2,Omega,Omega,optimize=True)
                Vnr_lm += np.einsum('ij,jjnr,iilm->nrlm',f3/2,Omega,Omega,optimize=True)
                Vnr_lm += np.einsum('ij,jjlr,iimn->nrlm',f3/2,Kappa,Kappa,optimize=True)
                Vnr_lm += np.einsum('ij,jjmr,iiln->nrlm',f3/2,Kappa,Kappa,optimize=True)
                Vnr_lm += np.einsum('ij,jjln,iimr->nrlm',f3/2,Kappa,Kappa,optimize=True)
                Vnr_lm += np.einsum('ij,jjmn,iilr->nrlm',f3/2,Kappa,Kappa,optimize=True)
                # i <-> j and symmetrization
                Vnr_lm *= 2

                if scatteringType == '2to2a':
                    en = evals[:,None,None,None]
                    er = evals[None,:,None,None]
                    el = evals[None,None,:,None]
                    em = evals[None,None,None,:]
                    # Delta
                    arg = en + er - el - em
                    delta_vals = lorentz(arg,gamma)
                    # Bose Factor
                    if temperature != 0:
                        beta = 1/temperature
                        bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*(em+el)) / (np.exp(beta*er)-1) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1)
                    else:
                        bose_factor = 1
                    # Decay rate
                    Gamma_n = 4 * np.pi * np.sum(Vnr_lm**2 * delta_vals * bose_factor,axis=(1,2,3))
                if scatteringType == '2to2b':
                    en = evals[:,None,None]
                    el = evals[None,:,None]
                    em = evals[None,None,:]
                    # Delta
                    arg = 2*en - el - em
                    delta_vals = lorentz(arg,gamma)
                    # Bose Factor
                    if temperature != 0:
                        beta = 1/temperature
                        bose_factor = (1-np.exp(-2*beta*en)) * np.exp(beta*(em+el)) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1)
                    else:
                        bose_factor = 1
                    # Decay rate
                    Gamma_n = np.einsum('lmii,ilm,ilm->i',Vnr_lm**2,delta_vals,bose_factor)
                    Gamma_n *= 2*np.pi
            if verbose:
                print("Computation took: %.3f seconds"%(time()-ti))
            self.dataScattering[scatteringType] = Gamma_n
            if self.p.sca_saveVertex:
                if verbose:
                    print("Saving result to file")
                np.save(vertexFn,Gamma_n)
        if self.p.sca_plotVertex:
            plotVertex(self)

##########################################################
##########################################################

class openSystem(openHamiltonian):
    def __init__(self, p: iu.openParameters):
        # Construct lattice and Hamiltonian
        super().__init__(p)
        #XT correlator parameters
        self.perturbationSite = p.cor_perturbationSite
        self.perturbationIndex = self.indexesMap.index(self.perturbationSite)
        #
        self.site0 = 0 #if h_t_i[0,0,0]<0 else 1     #decide sublattice A and B of reference lattice site
        self.fullTimeMeasure = 0.8     #measure time in ms
        self.nTimes = 401        #time steps after ramp for the measurement
        self.measureTimeList = np.linspace(0,self.fullTimeMeasure,self.nTimes)
        #KW correlator parameters
        self.nOmega = 2000

    def realSpaceCorrelator(self,verbose=False):
        """ Here we compute the correlator in real space.
        """
        Lx = self.Lx
        Ly = self.Ly
        Ns = self.Ns
        txtZeroEnergy = 'without0energy' if self.p.dia_excludeZeroMode else 'with0energy'
        argsFn = ('correlatorXT_rs',self.p.cor_correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,self.Ns,txtZeroEnergy,'magnonModes',self.p.cor_magnonModes)
        correlatorFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npy')
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
                self.correlatorXT[ix,iy] = openCorrelators.dicCorrelators[self.p.cor_correlatorType](self,ind_i,A,B,G,H)
            if self.p.cor_saveXT:
                if not Path(self.dataDn).is_dir():
                    print("Creating 'Data/' folder in directory: "+self.dataDn)
                    os.system('mkdir '+self.dataDn)
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
        txtZeroEnergy = 'without0energy' if self.p.dia_excludeZeroMode else 'with0energy'
        argsFn_h = ('correlator_horizontal_bonds',self.p.cor_correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy,'magnonModes',self.p.cor_magnonModes,self.perturbationSite)
        argsFn_v = ('correlator_vertical_bonds',self.p.cor_correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy,'magnonModes',self.p.cor_magnonModes,self.perturbationSite)
        correlatorFn_h = pf.getFilename(*argsFn_h,dirname=self.dataDn,extension='.npy')
        correlatorFn_v = pf.getFilename(*argsFn_v,dirname=self.dataDn,extension='.npy')
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
            if self.p.cor_saveXTbonds:
                if not Path(self.dataDn).is_dir():
                    print("Creating 'Data/' folder in directory: "+self.dataDn)
                    os.system('mkdir '+self.dataDn)
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
        txtZeroEnergy = 'without0energy' if self.p.dia_excludeZeroMode else 'with0energy'
        argsFn = ('correlatorKW_rs',self.p.cor_correlatorType,self.p.cor_transformType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,self.Ns,txtZeroEnergy,'magnonModes',self.p.cor_magnonModes)
        correlatorFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npy')
        if not Path(correlatorFn).is_file():
            self.correlatorKW = momentumTransformation.dicTransformType[self.p.cor_transformType](self)
            if self.p.cor_saveKW:
                if not Path(self.dataDn).is_dir():
                    print("Creating 'Data/' folder in directory: "+self.dataDn)
                    os.system('mkdir '+self.dataDn)
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
            self.rampElements[i].diagonalize(verbose=verbose)
            # Compute Correlators
            self.rampElements[i].realSpaceCorrelator(verbose=verbose)
            # Bond correlators
            if self.rampElements[i].p.cor_saveXTbonds:
                self.rampElements[i].realSpaceCorrelatorBond(verbose=verbose)

    def correlatorsKW(self,verbose=False):
        """ Here we Fourier transform the XT correlators and plot them nicely.
        """
        iterKW = tqdm(range(self.nP),desc="Computing Fourier transformation of correlator") if verbose else range(self.nP)
        for i in iterKW:
            self.rampElements[i].momentumSpaceCorrelator()

        if self.rampElements[0].p.cor_plotKW:
            """ Plot the Fourier-transformed correlators of the ramp """
            plotRampKW(self,
                       kwargs={
                           'numKbins' : 50,
                           'ylim' : 70,            #MHz
                           'saveFigure' : self.rampElements[0].p.cor_savePlotKW,
                           'showFigure' : True,
                       }
                       )



















