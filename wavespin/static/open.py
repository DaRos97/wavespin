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
from wavespin.tools.functions import lorentz, Ry
from wavespin.static import correlators
from wavespin.static import momentumTransformation
from wavespin.static.periodic import quantizationAxis
from wavespin.plots.rampPlots import *
from wavespin.plots import fancyLattice
from wavespin.static.decayProcesses import dic_processes
import itertools

class openHamiltonian(latticeClass):
    def __init__(self, p: iu.myParameters):
        super().__init__(p)
        # Hamiltonian parameters
        self.g1,self.g2,self.d1,self.d2,self.h,self.h_disorder = self.p.dia_Hamiltonian
        self.order = 'canted-Néel' if self.g2<=self.g1/2 else 'canted-stripe'
        self.S = 0.5     #spin value
        self.g_i = (self._NNterms(self.g1), self._NNNterms(self.g2))
        self.D_i = (self._NNterms(self.d1), self._NNNterms(self.d2))
        self.h_i = self._Hterms(self.h,self.h_disorder)
        # Diagonalize Hamiltonian to get eigenvalues and Bogoliubov fuunctions
        self.diagonalize()
        # self.realSpaceHamiltonian = self._realSpaceHamiltonian()
        if self.boundary == 'periodic':
            self.gridRealSpace = np.stack(np.meshgrid(np.arange(self.Lx), np.arange(self.Ly), indexing="ij"), axis=-1)
            self._momentumGrid()
            self.gamma = self._gamma()
            self.dispersion = self._dispersion()
            self.gap = np.min(self.dispersion)
            self.GSenergy = np.sum(self.dispersion)/self.Ns/2 + self._E0()

# Periodic functions
    def _momentumGrid(self):
        """ Compute momenta in the Brillouin zone for a (periodic) rectangular shape.

        Parameters
        ----------
        Lx,Ly : int, linear size.

        """
        dx = 2*np.pi/self.Lx
        dy = 2*np.pi/self.Ly
        self.gridk = np.zeros((self.Lx,self.Ly,2))
        for i1 in range(self.Lx):
            for i2 in range(self.Ly):
                self.gridk[i1,i2,0] = dx*(1+i1) #- np.pi
                self.gridk[i1,i2,1] = dy*(1+i2) #- np.pi

    def _gamma(self):
        r""" Compute the '$\Gamma$' dispersion at first and second nearest neighbor.

        Returns
        -------
        Gamma : 2-tuple
            Dispersions at 1st and 2nd nearest neighbor.
        """
        gridk = self.gridk
        Gamma1 = 1/2*( np.cos(gridk[:,:,0])+np.cos(gridk[:,:,1]) ) #cos(kx) + cos(ky)
        Gamma2 = 1/2*( np.cos(gridk[:,:,0]+gridk[:,:,1])+np.cos(gridk[:,:,0]-gridk[:,:,1]))  #cos(kx+ky) + cos(kx-ky)
        return (Gamma1,Gamma2)

    def _dispersion(self):
        """
        Compute dispersion epsilon as in notes.
        Controls are neded for ZZ in k.
        """
        self.quantizationAxisAngles()
        self.computeTs()
        self.computePs()
        N_11 = self._N11()
        N_12 = self._N12()
        result = np.zeros(N_11.shape)
        mask = (N_11**2>=np.absolute(N_12)**2)
        result[mask] = np.sqrt(N_11[mask]**2-np.absolute(N_12[mask])**2)
        return result

    def _E0(self):
        r""" Compute $E_0$ as in notes.
        """
        pzz_nn = np.sum(self.Ps[0,0,:,0,0]) / 4
        pzz_nnn = np.sum(self.Ps[1,0,:,0,0]) / 4
        result = 2*self.S*(self.S+1) * (pzz_nn + pzz_nnn)
        result += - self.h * np.cos(self.theta) * (self.S + 1/2)
        return result

    def _N11(self):
        """ Compute N_11 as in notes. """
        #nn
        pxx_nn = np.sum(self.Ps[0,0,:,1,1]) / 4
        pyy_nn = np.sum(self.Ps[0,0,:,2,2]) / 4
        pzz_nn = np.sum(self.Ps[0,0,:,0,0]) / 4
        #pyy_nn = self.Ps[0,0,1,2,2]
        #pzz_nn = self.Ps[0,0,1,0,0]
        result = 1/2/self.S * ( (pxx_nn+pyy_nn)*self.gamma[0] - 2*pzz_nn)
        #nnn
        if 0:
            for i in range(100,150):
                x,y = self._xy(i)
                if x+1!=self.Lx and y!=0:
                    innn = self._idx(x+1,y-1)
                    print(self.Ps[1,i,innn,1,1],self.Ps[1,i,innn,2,2],self.Ps[1,i,innn,0,0])
            input()
        pxx_nnn = np.sum(self.Ps[1,0,:,1,1]) / 4
        pyy_nnn = np.sum(self.Ps[1,0,:,2,2]) / 4
        pzz_nnn = np.sum(self.Ps[1,0,:,0,0]) / 4
        result += 1/2/self.S * ( (pxx_nnn+pyy_nnn)*self.gamma[1] - 2*pzz_nnn)
        #z
        result += self.h/2*np.cos(self.theta)
        return result

    def _N12(self):
        """ Compute N_12 as in notes. """
        #nn
        pxx_nn = np.sum(self.Ps[0,0,:,1,1]) / 4
        pyy_nn = np.sum(self.Ps[0,0,:,2,2]) / 4
        result = 1/2/self.S * self.gamma[0] * (pxx_nn - pyy_nn)
        #nnn
        pxx_nnn = np.sum(self.Ps[1,0,:,1,1]) / 4
        pyy_nnn = np.sum(self.Ps[1,0,:,2,2]) / 4
        result += 1/2/self.S * self.gamma[1] * (pxx_nnn - pyy_nnn)
        return result

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
        vals = np.zeros((self.Ns,self.Ns))
        for i in range(self.Ns):
            vals[i,self.NNN[i]] = val
        return vals

    def _Hterms(self,val,disorder_val):
        vals = np.zeros((self.Ns,self.Ns))
        disorder = (np.random.rand(self.Ns)-0.5)*2 * disorder_val
        for i in range(self.Ns):
            ix,iy = self._xy(i)
            vals[i,i] = (-1)**(ix+iy+1) * val + disorder[i]
        return vals

    def _temperature(self,Eref):
        """ Compute temperature given the energy.
        Since the E(T) function is not invertible we have to compute E for a bunch of Ts and extract graphically the T.
        """
        if Eref == -100:
            return 0
        self.diagonalize()
        Nbonds = np.sum(self._NNterms(1)) // 2
        GS_energy = -3/2 + np.sum(self.evals) / Nbonds / self.g1 / 2
        if Eref < GS_energy:
            raise ValueError("Input state energy smaller than GS energy: %.3f"%GS_energy)
        tempE = np.zeros(self.Ns)
        tempE[0] = GS_energy
        for i in range(1,self.Ns):
            B = 1/(np.exp(self.evals[1:]/self.evals[i])-1)
            tempE[i] = GS_energy + np.sum(self.evals[1:]*B) / Nbonds / self.g1
        if Eref > tempE[-1]:
            raise ValueError("Input state energy larger then max magnon one %.3f"%tempE[-1])
        Tmin = self.evals[ np.argmin( (Eref-tempE)[Eref>tempE] ) ]
        Tmax = self.evals[ np.argmin( (Eref-tempE)[Eref>tempE] ) +1 ]
        Tlist = np.linspace(Tmin,Tmax,100)
        e_list = np.zeros(len(Tlist))
        for i in range(len(Tlist)):
            B = 1/(np.exp(self.evals[1:]/Tlist[i])-1)
            e_list[i] = GS_energy + np.sum(self.evals[1:]*B) / Nbonds / self.g1
        indT = np.argmin(abs(e_list-Eref))
        if 0:   # Plot temperature
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot()
            #ax.plot(self.evals,tempE)
            ax.plot(Tlist,e_list)
            ax.axhline(Eref)
            ax.axvline(Tlist[indT])
            plt.show()
            exit()
        return Tlist[indT]

    def quantizationAxisAngles(self,verbose=False):
        """ Here we get the quantization axis angles to use for the diagonalization.
        Phi is 0, we would need it just when the dynamics is implemented.
        """
        self.theta, self.phi = quantizationAxis(self.S,self.g_i,self.D_i,self.h_i)
        self.phis = np.zeros(self.Ns)
        if self.p.dia_uniformQA:
            self.thetas = np.ones(self.Ns)*self.theta
            if self.order == 'canted-Néel':
                for i in range(self.Ns):
                    x,y = self._xy(i)
                    self.thetas[i] += np.pi*((x+y)%2)
            else:       # canted-stripe
                for i in range(self.Ns):
                    x,y = self._xy(i)
                    if x%2==1 and y%2==0:
                        self.thetas[i] += np.pi
                    if x%2==0 and y%2==1:
                        self.thetas[i] *= -1
                        self.thetas[i] += np.pi
                    if x%2==1 and y%2==1:
                        self.thetas[i] *= -1
            if 0:   # Plot solution
                kwargs = {'indices':False, 'angles':True}
                fancyLattice.plotSitesGrid(self,**kwargs)
        else:
            argsFn = ('quantAngle',self.Lx,self.Ly,self.Ns,self.p.dia_Hamiltonian)
            anglesFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npy')
            if not Path(anglesFn).is_file():
                print("File of quantization axis angles not found: "+anglesFn)
                print("computing it now..")
                from wavespin.classicSpins.anglesOBC import classicMagnetization
                result = classicMagnetization(self,verbose)
                self.thetas = result.bestAngles
                if 1:   # Plot solution
                    kwargs = {'indices':False, 'angles':True}
                    fancyLattice.plotSitesGrid(self,**kwargs)
                    exit()
                if input("Save result?[y/N]")=='y':
                    argsFn = ('anglesOBC',self.Lx,self.Ly,self.Ns,self.dia_Hamiltonian)
                    solutionFn = pf.getFilename(*argsFn,dirname=obj.dataDn,extension='.npy')
                    np.save(solutionFn,self.thetas)
            else:
                self.thetas = np.load(anglesFn)

    def computeTs(self):
        """ Compute the vector parameters t_z, t_x and t_y as in notes for sublattice A and B.
        Sublattice A has negative magnetic feld.
        """
        self.ts = np.zeros((self.Ns,3,3))
        for i in range(self.Ns):
            #print(i,self._xy(i),self.thetas[i]/np.pi*180)
            #input()
            rot= Ry(self.thetas[i])
            self.ts[i,0] = rot @ np.array([0,0,1])#t_zx,t_zy,t_zz
            self.ts[i,1] = rot @ np.array([1,0,0])#t_xx,t_xy,t_xz
            self.ts[i,2] = rot @ np.array([0,1,0])#t_yx,t_yy,t_yz

    def computePs(self):
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
        for i in range(self.Ns):
            #nn
            for j in self.NN[i]:
                self.Ps[0,i,j] = np.einsum('d,ad,bd->ab',vecGnn[:,i,j],self.ts[i],self.ts[j],optimize=True)
            #nnn
            for j in self.NNN[i]:
                self.Ps[1,i,j] = np.einsum('d,ad,bd->ab',vecGnnn[:,i,j],self.ts[i],self.ts[j],optimize=True)

    def _realSpaceHamiltonian(self,verbose=False):
        """
        Compute the real space Hamiltonian -> (2Ns x 2Ns).
        Conventions for the real space wavefunction and parameters are in the notes.
        SECOND NEAREST-NEIGHBOR: implemented Ps but not in the Hamiltonian.

        Returns
        -------
        ham : 2Ns,2Ns matrix of real space Hamiltonian.
        """
        S = self.S
        Ns = self.Ns
        g_i = self.g_i
        D_i = self.D_i
        h_i = self.h_i
        self.quantizationAxisAngles(verbose)
        self.computeTs()
        self.computePs()
        p_zz = self.Ps[0,:,:,0,0]
        p_xx = self.Ps[0,:,:,1,1]
        p_yy = self.Ps[0,:,:,2,2]
        #
        ham = np.zeros((2*Ns,2*Ns),dtype=complex)
        #p_zz sums over nn but is on-site -> problem when geometry is not rectangular
        ham[:Ns,:Ns] = -h_i*np.cos(np.diag(self.thetas)) / 2 / S - np.diag(np.sum(p_zz,axis=1)) / 2 / S
        ham[Ns:,Ns:] = -h_i*np.cos(np.diag(self.thetas)) / 2 / S - np.diag(np.sum(p_zz,axis=1)) / 2 / S
        #off_diag 1 - nn
        off_diag_1_nn = (p_xx+p_yy) / 4 / S
        ham[:Ns,:Ns] += off_diag_1_nn
        ham[Ns:,Ns:] += off_diag_1_nn
        #off_diag 2 - nn
        off_diag_2_nn = (p_xx-p_yy) / 4 / S
        ham[:Ns,Ns:] += off_diag_2_nn
        ham[Ns:,:Ns] += off_diag_2_nn.T.conj()
        return ham

    def diagonalize(self,verbose=False,**kwargs):
        """ Compute the Bogoliubov transformation for the real-space Hamiltonian.
        Initialize U_, V_ and evals : bogoliubov transformation matrices U and V and eigenvalues.
        """
        argsFn = ('bogWf',self.Lx,self.Ly,self.Ns,self.p.dia_Hamiltonian)
        transformationFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npz')
        hamiltonian = self._realSpaceHamiltonian(verbose)
        if not Path(transformationFn).is_file():
            if np.max(np.absolute(hamiltonian-hamiltonian.conj()))>1e-5:
                raise ValueError("Hamiltonian is not real! Procedure might be wrong")
            # Para-diagonalization
            Ns = self.Ns
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
            self.Phi = np.real(self.U_-self.V_)
            for n in range(self.Ns):
                x,y = self._xy(n)
                self.Phi[n,:] *= 2/np.pi*(-1)**(x+y+1)
            if self.p.dia_saveWf:
                if not Path(self.dataDn).is_dir():
                    print("Creating 'Data/' folder in directory: "+self.dataDn)
                    os.system('mkdir '+self.dataDn)
                np.savez(transformationFn,awesomeU=self.U_,awesomeV=self.V_,evals=self.evals,Phi=self.Phi)
        else:
            if verbose:
                print("Loading Bogoliubov transformation from file: "+transformationFn)
            self.U_ = np.load(transformationFn)['awesomeU']
            self.V_ = np.load(transformationFn)['awesomeV']
            self.Phi = np.load(transformationFn)['Phi']
            self.evals = np.load(transformationFn)['evals']

        if self.p.dia_excludeZeroMode:       #Put to 0 the eigenstate corresponding to the zero energy mode -> a bit far fetched
            self.U_[:,0] *= 0
            self.V_[:,0] *= 0
        if self.p.dia_plotWf:
            #plotWf3D(self)
            plotWf2D(self,nModes=30)#self.Ns)
            #plotWfCos(self)
        if self.p.dia_plotMomenta:
            plotBogoliubovMomenta(self,**kwargs)

    def computeRate(self,verbose=False):
        """ Compute the required decay/scattering rate for each mode.
        """
        self.rates = {}
        for process in self.p.sca_types:
            argsDecayFn = ['decay',process,self.p.sca_temperature,self.p.dia_Hamiltonian,self.Lx,self.Ly,self.Ns,self.p.sca_broadening]
            decayFn = pf.getFilename(*tuple(argsDecayFn),dirname=self.dataDn,extension='.npy')
            if Path(decayFn).is_file():
                self.rates[process] = np.load(decayFn)
                continue
            if not hasattr(self,process[:4]):
                argsVertexFn = ['vertex',process[:4],self.p.dia_Hamiltonian,self.Lx,self.Ly,self.Ns]
                vertexFn = pf.getFilename(*tuple(argsVertexFn),dirname=self.dataDn,extension='.npy')
                if Path(vertexFn).is_file():
                    setattr(self,'vertex'+process[:4],np.load(vertexFn))
                else:
                    if verbose:
                        print("Computing ",process[:4]," vertex")
                    self.computeVertex(process[:4])
                if self.p.sca_saveVertex:
                    np.save(vertexFn,getattr(self,'vertex'+process[:4]))
            self.rates[process] = dic_processes[process](self)
            if self.p.sca_saveRate:
                np.save(decayFn,self.rates[process])
        if self.p.sca_plotRate:
            plotRate(self)

    def computeVertex(self,vertex):
        """ Compute the vertex of the interaction, may be 1->2, 1->3 or 2->2.
        """
        U = np.real(self.U_)
        V = np.real(self.V_)
        if vertex=='1to2':
            # f_ij
            f = self.g_i[0] / np.sqrt(2*self.S)/4/self.S * (1-self.D_i[0]) * np.sin(2 * self.thetas)
            f /= 2 # Since we sum over all bonds, we count each twice
            ### Vn(l,m)
            Vn_lm = np.zeros((self.Ns,self.Ns,self.Ns))
            # f1
            Vn_lm += np.einsum('ij,in,jl,jm->nlm',f,V,V,U,optimize=True)
            Vn_lm += np.einsum('ij,jn,il,jm->nlm',f,U,U,U,optimize=True)
            Vn_lm += np.einsum('ij,jn,il,jm->nlm',f,V,U,V,optimize=True)
            Vn_lm += np.einsum('ij,in,jl,jm->nlm',f,U,V,U,optimize=True)
            Vn_lm += np.einsum('ij,jn,il,jm->nlm',f,U,V,U,optimize=True)
            Vn_lm += np.einsum('ij,jn,il,jm->nlm',f,V,V,V,optimize=True)
            # Symmetrize l,m
            result = (
                Vn_lm
                + np.transpose(Vn_lm, (0,2,1))  # l↔m
            ) / 2.0
            # i <-> j counterpart
            result *= 2
        elif vertex=='2to2':
            # f_ij
            f1 = self.g_i[0] / 16 / self.S**2 * (np.cos(self.thetas)**2 + self.D_i[0]*np.sin(self.thetas)**2 + 1)
            f2 = self.g_i[0] / 16 / self.S**2 * (np.cos(self.thetas)**2 + self.D_i[0]*np.sin(self.thetas)**2 - 1)
            f3 = -self.g_i[0] / 4 / self.S**2 * (np.sin(self.thetas)**2 + self.D_i[0]*np.cos(self.thetas)**2)
            f1 /= 2 # Since we sum over all bonds, we count each twice
            f2 /= 2
            f3 /= 2
            ### V_nl(m,p)
            Vnl_mp = np.zeros((self.Ns,self.Ns,self.Ns,self.Ns))
            # f1
            Vnl_mp += np.einsum('ij,jn,il,jm,jp->nlmp',f1,U,V,U,U,optimize=True)
            Vnl_mp += np.einsum('ij,in,jl,jm,jp->nlmp',f1,V,V,U,V,optimize=True) * 2
            Vnl_mp += np.einsum('ij,jn,jl,im,jp->nlmp',f1,U,V,U,U,optimize=True) * 2
            Vnl_mp += np.einsum('ij,jn,jl,im,jp->nlmp',f1,V,V,U,V,optimize=True)
            Vnl_mp += np.einsum('ij,in,jl,jm,jp->nlmp',f1,U,U,U,V,optimize=True) * 2
            Vnl_mp += np.einsum('ij,jn,il,jm,jp->nlmp',f1,U,V,V,V,optimize=True)
            Vnl_mp += np.einsum('ij,jn,jl,jm,ip->nlmp',f1,U,U,U,V,optimize=True)
            Vnl_mp += np.einsum('ij,jn,jl,im,jp->nlmp',f1,U,V,V,V,optimize=True) * 2
            # f2
            Vnl_mp += np.einsum('ij,in,jl,jm,jp->nlmp',f2,U,U,U,U,optimize=True)
            Vnl_mp += np.einsum('ij,in,jl,jm,jp->nlmp',f2,U,V,U,V,optimize=True) * 2
            Vnl_mp += np.einsum('ij,jn,jl,jm,ip->nlmp',f2,U,V,U,V,optimize=True) * 2
            Vnl_mp += np.einsum('ij,jn,jl,im,jp->nlmp',f2,V,V,V,V,optimize=True)
            Vnl_mp += np.einsum('ij,jn,il,jm,jp->nlmp',f2,U,V,U,V,optimize=True) * 2
            Vnl_mp += np.einsum('ij,in,jl,jm,jp->nlmp',f2,V,V,V,V,optimize=True)
            Vnl_mp += np.einsum('ij,jn,jl,im,jp->nlmp',f2,U,U,U,U,optimize=True)
            Vnl_mp += np.einsum('ij,jn,jl,im,jp->nlmp',f2,U,V,U,V,optimize=True) * 2
            # f3
            Vnl_mp += np.einsum('ij,in,il,jm,jp->nlmp',f3,U,V,U,V,optimize=True)
            Vnl_mp += np.einsum('ij,in,jl,im,jp->nlmp',f3,U,U,U,U,optimize=True)
            Vnl_mp += np.einsum('ij,in,jl,im,jp->nlmp',f3,U,V,U,V,optimize=True)
            Vnl_mp += np.einsum('ij,jn,il,jm,ip->nlmp',f3,U,V,U,V,optimize=True)
            Vnl_mp += np.einsum('ij,in,jl,im,jp->nlmp',f3,V,V,V,V,optimize=True)
            Vnl_mp += np.einsum('ij,jn,jl,im,ip->nlmp',f3,U,V,U,V,optimize=True)
            # Symmetrize n,l and m,p
            result = (
                Vnl_mp
                + np.transpose(Vnl_mp, (1,0,2,3))  # n↔l
                + np.transpose(Vnl_mp, (0,1,3,2))  # m↔p
                + np.transpose(Vnl_mp, (1,0,3,2))  # n↔l, m↔p
            ) / 4.0
            # i <-> j counterpart
            result *= 2
        elif vertex=='1to3':
            # f_ij
            f1 = self.g_i[0] / 16 / self.S**2 * (np.cos(self.thetas)**2 + self.D_i[0]*np.sin(self.thetas)**2 + 1)
            f2 = self.g_i[0] / 16 / self.S**2 * (np.cos(self.thetas)**2 + self.D_i[0]*np.sin(self.thetas)**2 - 1)
            f3 = -self.g_i[0] / 4 / self.S**2 * (np.sin(self.thetas)**2 + self.D_i[0]*np.cos(self.thetas)**2)
            f1 /= 2 # Since we sum over all bonds, we count each twice
            f2 /= 2
            f3 /= 2
            ### V_n(l,m,p)
            Vn_lmp = np.zeros((self.Ns,self.Ns,self.Ns,self.Ns))
            # f1
            Vn_lmp += np.einsum('ij,in,jl,jm,jp->nlmp',f1,V,V,U,U,optimize=True)
            Vn_lmp += np.einsum('ij,jn,il,jm,jp->nlmp',f1,U,U,U,U,optimize=True)
            Vn_lmp += np.einsum('ij,jn,il,jm,jp->nlmp',f1,V,U,V,U,optimize=True) * 2
            Vn_lmp += np.einsum('ij,in,jl,jm,jp->nlmp',f1,U,V,V,U,optimize=True)
            Vn_lmp += np.einsum('ij,jn,il,jm,jp->nlmp',f1,U,V,V,U,optimize=True) * 2
            Vn_lmp += np.einsum('ij,jn,il,jm,jp->nlmp',f1,V,V,V,V,optimize=True)
            # f2
            Vn_lmp += np.einsum('ij,in,jl,jm,jp->nlmp',f2,U,V,U,U,optimize=True)
            Vn_lmp += np.einsum('ij,jn,il,jm,jp->nlmp',f2,U,V,U,U,optimize=True)
            Vn_lmp += np.einsum('ij,jn,il,jm,jp->nlmp',f2,V,V,V,U,optimize=True) * 2
            Vn_lmp += np.einsum('ij,in,jl,jm,jp->nlmp',f2,V,V,V,U,optimize=True)
            Vn_lmp += np.einsum('ij,jn,il,jm,jp->nlmp',f2,U,U,V,U,optimize=True) * 2
            Vn_lmp += np.einsum('ij,jn,il,jm,jp->nlmp',f2,V,U,V,V,optimize=True)
            # f3
            Vn_lmp += np.einsum('ij,in,il,jm,jp->nlmp',f3,U,U,V,U,optimize=True)
            Vn_lmp += np.einsum('ij,in,il,jm,jp->nlmp',f3,V,V,V,U,optimize=True)
            Vn_lmp += np.einsum('ij,jn,jl,im,ip->nlmp',f3,U,U,V,U,optimize=True)
            Vn_lmp += np.einsum('ij,jn,jl,im,ip->nlmp',f3,V,V,V,U,optimize=True)
            # Symmetrize l,m,p
            perms = list(itertools.permutations(range(3)))
            result = sum( np.transpose(Vn_lmp, (0,) + tuple(1 + np.array(perm))) for perm in perms ) / len(perms)
            # i <-> j counterpart
            result *= 2
        setattr(self,'vertex'+vertex,result)

##########################################################
##########################################################

class openSystem(openHamiltonian):
    def __init__(self, p: iu.myParameters):
        # Construct lattice and Hamiltonian
        super().__init__(p)
        #XT correlator parameters
        self.perturbationSite = p.cor_perturbationSite
        self.perturbationIndex = self._idx(*self.perturbationSite)
        #
        self.site0 = 0 #if h_t_i[0,0,0]<0 else 1     #decide sublattice A and B of reference lattice site
        self.fullTimeMeasure = 0.8     #measure time in ms
        self.nTimes = 401        #time steps after ramp for the measurement
        self.measureTimeList = np.linspace(0,self.fullTimeMeasure,self.nTimes)
        #KW correlator parameters
        self.nOmega = 2000
        # Diagonalize
        #self.diagonalize()

    def realSpaceCorrelator(self,verbose=False):
        """ Here we compute the correlator in real space.
        """
        temperature = self._temperature(self.p.cor_energy)
        txtZeroEnergy = 'without0energy' if self.p.dia_excludeZeroMode else 'with0energy'
        argsFn = ('correlatorXT',self.p.cor_correlatorType,self.Lx,self.Ly,self.Ns,self.p.dia_Hamiltonian,
                  txtZeroEnergy,'magnonModes',self.p.cor_magnonModes,self.p.cor_energy)
        correlatorFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npy')
        if not Path(correlatorFn).is_file():
            self.correlatorXT = np.zeros((self.Ns,self.nTimes),dtype=complex)
            #
            #U = np.zeros((2*Ns,2*Ns),dtype=complex)
            #U[:Ns,:Ns] = self.U_
            #U[:Ns,Ns:] = self.V_
            #U[Ns:,:Ns] = self.V_
            #U[Ns:,Ns:] = self.U_
            #Correlator -> can make this faster, we actually only need U_ and V_
            exp_e = np.exp(-1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
            #A_GS = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[Ns:,Ns:],optimize=True)
            #B_GS = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[:Ns,Ns:],optimize=True)
            #G_GS = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[:Ns,Ns:],optimize=True)
            #H_GS = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[Ns:,Ns:],optimize=True)
            A_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.V_,self.U_,optimize=True)
            B_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.U_,self.V_,optimize=True)
            G_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.V_,self.V_,optimize=True)
            H_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.U_,self.U_,optimize=True)
            if temperature != 0:
                exp_e_c = np.exp(1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
                BF = 1/(np.exp(self.evals/temperature)-1)
                A2  = np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.U_,exp_e, optimize=True)
                A2 += np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.V_,exp_e_c,optimize=True)
                #A2 += np.einsum('l,in,jn,tn->ijt',BF,self.V_,self.U_,exp_e, optimize=True)
                #A2 += np.einsum('l,in,jn,tn->ijt',BF,self.U_,self.V_,exp_e_c,optimize=True)
                #
                B2  = np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.V_,exp_e, optimize=True)
                B2 += np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.U_,exp_e_c,optimize=True)
                #B2 += np.einsum('l,in,jn,tn->ijt',BF,self.U_,self.V_,exp_e, optimize=True)
                #B2 += np.einsum('l,in,jn,tn->ijt',BF,self.V_,self.U_,exp_e_c,optimize=True)
                #
                G2  = np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.V_,exp_e, optimize=True)
                G2 += np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.U_,exp_e_c,optimize=True)
                #G2 += np.einsum('l,in,jn,tn->ijt',BF,self.V_,self.V_,exp_e, optimize=True)
                #G2 += np.einsum('l,in,jn,tn->ijt',BF,self.U_,self.U_,exp_e_c,optimize=True)
                #
                H2  = np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.U_,exp_e, optimize=True)
                H2 += np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.V_,exp_e_c,optimize=True)
                #H2 += np.einsum('l,in,jn,tn->ijt',BF,self.U_,self.U_,exp_e, optimize=True)
                #H2 += np.einsum('l,in,jn,tn->ijt',BF,self.V_,self.V_,exp_e_c,optimize=True)
            else:
                A2 = B2 = G2 = H2 = 0
            Af = A_GS + A2
            Bf = B_GS + B2
            Gf = G_GS + G2
            Hf = H_GS + H2
            if temperature != 0 and 0:
                fig = plt.figure(figsize=(12,12))
                funcs = [A_GS,B_GS,G_GS,H_GS]
                funcs2 = [A2,B2,G2,H2]
                ss = [-1,+1,-self.Ly,self.Ly]
                for iA in range(4):
                    f = np.real(funcs[iA][:,self.perturbationIndex,:])
                    fi = np.imag(funcs[iA][:,self.perturbationIndex,:])
                    f2 = np.real(funcs2[iA][:,self.perturbationIndex,:])
                    f2i = np.imag(funcs2[iA][:,self.perturbationIndex,:])
                    for ix in range(4):
                        ax = fig.add_subplot(4,4,iA*4+ix+1)
                        ax.plot(self.measureTimeList,f[self.perturbationIndex+ss[ix],:],color='r')
                        ax.plot(self.measureTimeList,fi[self.perturbationIndex+ss[ix],:],color='orange')
                        ax.plot(self.measureTimeList,f2[self.perturbationIndex+ss[ix],:],color='b')
                        ax.plot(self.measureTimeList,f2i[self.perturbationIndex+ss[ix],:],color='aqua')
                        ax.set_xlim(0,0.1)
                plt.show()
            #
            for ind_i in range(self.Ns):
                self.correlatorXT[ind_i] = correlators.dicCorrelators[self.p.cor_correlatorType](self,ind_i,Af,Bf,Gf,Hf)
            if self.p.cor_saveXT:
                np.save(correlatorFn,self.correlatorXT)
        else:
            if verbose:
                print("Loading real-space correlator from file: "+correlatorFn)
            self.correlatorXT = np.load(correlatorFn)

    def realSpaceCorrelatorBond(self,verbose=False):
        """ Here we compute the correlator in real space for each bond, like for the jj.
        """
        #temperature = self._temperature(self.p.cor_energy)
        temperature = 0
        Lx = self.Lx
        Ly = self.Ly
        Ns = self.Ns
        txtZeroEnergy = 'without0energy' if self.p.dia_excludeZeroMode else 'with0energy'
        argsFn_h = ('correlator_horizontal_bonds',self.p.cor_correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy,'magnonModes',self.p.cor_magnonModes,self.perturbationSite,self.p.cor_energy)
        argsFn_v = ('correlator_vertical_bonds',self.p.cor_correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy,'magnonModes',self.p.cor_magnonModes,self.perturbationSite,self.p.cor_energy)
        correlatorFn_h = pf.getFilename(*argsFn_h,dirname=self.dataDn,extension='.npy')
        correlatorFn_v = pf.getFilename(*argsFn_v,dirname=self.dataDn,extension='.npy')
        if not Path(correlatorFn_h).is_file() or not Path(correlatorFn_v).is_file():
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
            if temperature != 0:
                exp_e_c = np.exp(1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
                BF = 1/(np.exp(self.evals/temperature)-1)
                A += np.einsum('l,p,il,jp,tl->ijt',BF,BF,self.U_,self.V_,exp_e_c,optimize=True)
                A += np.einsum('l,p,ip,jl,tp->ijt',BF,BF,self.V_,self.U_,exp_e,  optimize=True)
                A += np.einsum('l,ip,jp,tp->ijt',  BF**2,self.V_,self.U_,exp_e, optimize=True)
                #
                B += np.einsum('l,p,il,jp,tl->ijt',BF,BF,self.V_,self.U_,exp_e_c,optimize=True)
                B += np.einsum('l,p,ip,jl,tp->ijt',BF,BF,self.U_,self.V_,exp_e,  optimize=True)
                B += np.einsum('l,ip,jp,tp->ijt',  BF**2,self.U_,self.V_,exp_e, optimize=True)
                #
                G += np.einsum('l,p,il,jp,tl->ijt',BF,BF,self.U_,self.U_,exp_e_c,optimize=True)
                G += np.einsum('l,p,ip,jl,tp->ijt',BF,BF,self.V_,self.V_,exp_e,  optimize=True)
                G += np.einsum('l,ip,jp,tp->ijt',  BF**2,self.V_,self.V_,exp_e, optimize=True)
                #
                H += np.einsum('l,p,il,jp,tl->ijt',BF,BF,self.V_,self.V_,exp_e_c,optimize=True)
                H += np.einsum('l,p,ip,jl,tp->ijt',BF,BF,self.U_,self.U_,exp_e,  optimize=True)
                H += np.einsum('l,ip,jp,tp->ijt',  BF**2,self.U_,self.U_,exp_e, optimize=True)
            #
            for ihx in range(Lx-1):
                for ihy in range(Ly):
                    ind_i = self._idx(ihx,ihy)
                    self.correlatorXT_h[ihx,ihy] = correlators.jjCorrelatorBond(self,ind_i,A,B,G,H,'h')
            for ivx in range(Lx):
                for ivy in range(Ly-1):
                    ind_i = self._idx(ivx,ivy)
                    self.correlatorXT_v[ivx,ivy] = correlators.jjCorrelatorBond(self,ind_i,A,B,G,H,'v')
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
        temperature = self._temperature(self.p.cor_energy)
        txtZeroEnergy = 'without0energy' if self.p.dia_excludeZeroMode else 'with0energy'
        argsFn = ('correlatorKW',self.p.cor_correlatorType,self.p.cor_transformType,self.Lx,self.Ly,self.Ns,self.p.dia_Hamiltonian,
                  txtZeroEnergy,'magnonModes',self.p.cor_magnonModes,self.p.cor_energy)
        correlatorFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npz')
        if not Path(correlatorFn).is_file():
            self.correlatorKW, self.momentum = momentumTransformation.dicTransformType[self.p.cor_transformType](self)
            if self.p.cor_saveKW:
                np.savez(correlatorFn,correlator=self.correlatorKW,momentum=self.momentum)
        else:
            if verbose:
                print("Loading momentum-space correlator from file: "+correlatorFn)
            self.correlatorKW = np.load(correlatorFn)['correlator']
            self.momentum = np.load(correlatorFn)['momentum']

##########################################################
##########################################################

class openRamp():
    def __init__(self, systems = None):
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



















