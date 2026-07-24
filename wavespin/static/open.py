""" Functions used for the open boundary conditions' calculations.
"""

import copy
import scipy
import numpy as np
from pathlib import Path
from tqdm import tqdm
from time import time

from wavespin.lattice.lattice import latticeClass
from wavespin.tools import pathFinder as pf
from wavespin.tools.inputUtils import SimParams
from wavespin.tools.functions import lorentz, Ry
from wavespin.static import correlators
from wavespin.static import momentumTransformation
from wavespin.static.periodic import quantizationAxis
from wavespin.plots.rampPlots import *
from wavespin.plots import latticePlots
from wavespin.plots import classicPlots
from wavespin.static.decayProcesses import dic_processes
import itertools

class openHamiltonian(latticeClass):
    """Real-space Bogoliubov diagonalization for open boundary conditions.

    Builds the quadratic bosonic Hamiltonian from Holstein-Primakoff
    expansion coefficients, then para-diagonalizes it via Cholesky
    decomposition to obtain quasiparticle energies, wavefunctions, and
    Bogoliubov transformation matrices.

    Parameters
    ----------
    p : SimParams
        Simulation parameters.  Only ``p.lattice`` and ``p.diag`` are
        consumed; ``p.correlator`` and ``p.scattering`` are ignored.

    Attributes
    ----------
    g1, g2, d1, d2, h, h_disorder : float
        Unpacked from ``p.diag.Hamiltonian``.
    order : str
        ``'canted-Neel'`` if ``g2 <= g1/2`` else ``'canted-stripe'``.
    S : float
        Spin magnitude (fixed at 0.5).
    g_i, D_i : tuple of (Ns,Ns) ndarray
        Nearest-neighbour and next-nearest-neighbour coupling matrices.
    h_i : (Ns,Ns) ndarray
        Staggered-field matrix (diagonal).
    theta : float
        Base canting angle (radians).
    phis : (Ns,) ndarray
        Azimuthal angles (all zero for the planar model).
    thetas : (Ns,) ndarray
        Per-site canting angles, patterned after the magnetic order.
    ts : (Ns,3,3) ndarray
        Rotation vectors :math:`t_z, t_x, t_y` at each site.
    Ps : (2,Ns,Ns,3,3) ndarray
        Holstein-Primakoff :math:`P^{\\alpha\\beta}_{ij}` coefficients
        for nearest-neighbour (index 0) and next-nearest-neighbour
        (index 1).
    evals : (Ns,) ndarray
        Quasiparticle eigen-energies :math:`E_n`.
    U_, V_ : (Ns,Ns) ndarray
        Bogoliubov transformation matrices.
    Phi : (Ns,Ns) ndarray
        Real-space wavefunctions (:math:`\\Phi = U - V`).
    GSE : float
        Ground-state energy per bond (set only if ``g1 != 0``).
    rates : dict
        Decay rates keyed by process type (populated after
        :meth:`computeRate`).
    """

    def __init__(self, p: SimParams):
        super().__init__(p.lattice)
        self.p = copy.copy(p)
        # Hamiltonian parameters
        self.g1,self.g2,self.d1,self.d2,self.h,self.h_disorder = self.p.diag.Hamiltonian
        self.order = 'canted-Neel' if self.g2<=self.g1/2 else 'canted-stripe'
        self.S = 0.5     #spin value
        self.g_i = (self._NNterms(self.g1), self._NNNterms(self.g2))
        self.D_i = (self._NNterms(self.d1), self._NNNterms(self.d2))
        self.h_i = self._Hterms(self.h,self.h_disorder)
        # Diagonalize Hamiltonian to get eigenvalues and Bogoliubov fuunctions
        self.diagonalize()
        if self.g1!=0:
            self.GSE = self.get_GSE()


    def _NNterms(self,val):
        """Build an Ns×Ns matrix with ``val`` on every nearest-neighbour bond.

        Parameters
        ----------
        val : float
            Coupling value to place on NN bonds.

        Returns
        -------
        (Ns,Ns) ndarray
            Matrix where ``result[i, j] = val`` for each NN pair.
        """
        vals = np.zeros((self.Ns,self.Ns))
        for i in range(self.Ns):
            vals[i,self.NN[i]] = val
        return vals

    def _NNNterms(self,val):
        """Build an Ns×Ns matrix with ``val`` on every next-nearest-neighbour bond.

        Parameters
        ----------
        val : float
            Coupling value to place on NNN bonds.

        Returns
        -------
        (Ns,Ns) ndarray
            Matrix where ``result[i, j] = val`` for each NNN pair.
        """
        vals = np.zeros((self.Ns,self.Ns))
        for i in range(self.Ns):
            vals[i,self.NNN[i]] = val
        return vals

    def _Hterms(self,val,disorder_val):
        """Build the diagonal staggered-field matrix.

        Parameters
        ----------
        val : float
            Staggered field strength ``h``.
        disorder_val : float
            Disorder strength (uniform random ±``disorder_val``).

        Returns
        -------
        (Ns,Ns) ndarray
            Diagonal matrix with :math:`(-1)^{x+y+1} h + \\delta_i`
            at site *i*.
        """
        vals = np.zeros((self.Ns,self.Ns))
        disorder = (np.random.rand(self.Ns)-0.5)*2 * disorder_val
        for i in range(self.Ns):
            ix,iy = self._xy(i)
            vals[i,i] = (-1)**(ix+iy+1) * val + disorder[i]
        return vals

    def get_GSE(self):
        """Ground-state energy per bond.

        Returns
        -------
        float
            :math:`E_\\text{GS} = -3/2 + \\sum_n E_n / N_\\text{bonds} / (2g_1)`.
        """
        Nbonds = np.sum(self._NNterms(1)) // 2
        GS_energy = -3/2 + np.sum(self.evals) / Nbonds / self.g1 / 2
        return GS_energy

    def _temperature(self,Eref):
        """Invert the energy-temperature relation to find the temperature
        corresponding to a given mean energy ``Eref``.

        Parameters
        ----------
        Eref : float
            Target mean energy (must be ≥ ``GSE``).

        Returns
        -------
        float
            Temperature in the same units as ``evals``.
        """
        if Eref == self.GSE or Eref==-100:
            return 0
        Nbonds = np.sum(self._NNterms(1)) // 2
        GS_energy = self.get_GSE()
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
        if Tmin==0:
            Tmin += 0.5
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
        """Determine the per-site quantization-axis angles.

        Computes the base canting angle :math:`\\theta` from the averaged
        Hamiltonian parameters, then spatially patterns it across the
        lattice according to the magnetic order (canted-Néel or
        canted-stripe).

        Sets
        ----
        theta : float
            Base canting angle (radians).
        phi : float
            Azimuthal angle (0 for the planar XY model).
        phis : (Ns,) ndarray
            Per-site azimuthal angles (all zero).
        thetas : (Ns,) ndarray
            Per-site polar angles :math:`\\theta_i` with the appropriate
            sublattice pattern.
        """
        self.theta, self.phi = quantizationAxis(self.S,self.g_i,self.D_i,self.h_i)
        self.phis = np.zeros(self.Ns)
        if self.p.diag.uniformQA:
            self.thetas = np.ones(self.Ns)*self.theta
            if self.order == 'canted-Neel':
                for i in range(self.Ns):
                    x,y = self._xy(i)
                    self.thetas[i] += np.pi*((x+y)%2)
            elif self.order == 'canted-stripe':
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
                classicPlots.plotLatticeWithAngles(self, self.thetas)
        else:
            raise NotImplementedError(
                "Non-uniform quantization angles (uniformQA=False) are not "
                "yet implemented. The classicMagnetization minimization via "
                "scipy is available in classicSpins/anglesOBC.py for future "
                "integration."
            )

    def computeTs(self):
        """Build the per-site rotation vectors.

        For each site *i*, rotates the reference axes (z, x, y) by the
        canting angle :math:`\\theta_i` around the y-axis:

        .. math::
            t_z = R_y(\\theta_i) \\cdot \\hat{z}, \\quad
            t_x = R_y(\\theta_i) \\cdot \\hat{x}, \\quad
            t_y = R_y(\\theta_i) \\cdot \\hat{y}.

        Sets
        ----
        ts : (Ns,3,3) ndarray
            ``ts[i, 0]`` = rotated z, ``ts[i, 1]`` = rotated x,
            ``ts[i, 2]`` = rotated y.
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
        """Compute the Holstein-Primakoff expansion coefficients
        :math:`P^{\\alpha\\beta}_{ij}` for each bond.

        .. math::
            P^{\\alpha\\beta}_{ij} =
            \\sum_\\gamma g^\\gamma_{ij} \\, t^\\alpha_{i\\gamma} \\,
            t^\\beta_{j\\gamma}

        where :math:`\\alpha,\\beta,\\gamma \\in \\{z, x, y\\}` and
        :math:`g^\\gamma_{ij}` are the bond-dependent coupling components.

        Sets
        ----
        Ps : (2,Ns,Ns,3,3) ndarray
            ``Ps[0, i, j]`` for NN bonds, ``Ps[1, i, j]`` for NNN bonds.
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
        ## Nearest neighbor
        p_zz = self.Ps[0,:,:,0,0]
        p_xx = self.Ps[0,:,:,1,1]
        p_yy = self.Ps[0,:,:,2,2]
        ## Second eearest neighbor
        p2_zz = self.Ps[1,:,:,0,0]
        p2_xx = self.Ps[1,:,:,1,1]
        p2_yy = self.Ps[1,:,:,2,2]
        #
        ham = np.zeros((2*Ns,2*Ns),dtype=complex)
        #p_zz sums over neighbor but is on-site -> problem when geometry is not rectangular ?
        ham[:Ns,:Ns] = -h_i*np.cos(np.diag(self.thetas)) / 2 / S - np.diag(np.sum(p_zz,axis=1)) / 2 / S - np.diag(np.sum(p2_zz,axis=1)) / 2 / S
        ham[Ns:,Ns:] = -h_i*np.cos(np.diag(self.thetas)) / 2 / S - np.diag(np.sum(p_zz,axis=1)) / 2 / S - np.diag(np.sum(p2_zz,axis=1)) / 2 / S
        #off_diag 1
        off_diag_1 = (p_xx+p_yy + p2_xx+p2_yy) / 4 / S
        ham[:Ns,:Ns] += off_diag_1
        ham[Ns:,Ns:] += off_diag_1.T.conj()
        #off_diag 2
        off_diag_2 = (p_xx-p_yy + p2_xx-p2_yy) / 4 / S
        ham[:Ns,Ns:] += off_diag_2
        ham[Ns:,:Ns] += off_diag_2.T.conj()
        return ham

    def diagonalize(self,verbose=False,**kwargs):
        """Bogoliubov para-diagonalization of the real-space Hamiltonian.

        Builds the :math:`2N_s \\times 2N_s` Bogoliubov-de Gennes
        matrix, then solves the generalized eigenvalue problem via
        Cholesky decomposition:

        1. ``A, B`` = upper-left / upper-right blocks of H
        2. ``K = chol(A - B)``
        3. Solve ``K (A+B) K^T`` for :math:`\\omega_n^2` and
           eigenvectors :math:`\\chi_n`
        4. :math:`E_n = \\sqrt{\\omega_n^2}`,
           :math:`\\phi_n = K^T \\chi_n / \\sqrt{E_n}`,
           :math:`\\psi_n = (A+B) \\phi_n / E_n`
        5. ``U = (φ + ψ) / 2``, ``V = (φ - ψ) / 2``,
           ``Φ = U - V``

        Results are cached to disk via ``Data/*.npz``.

        Sets
        ----
        evals : (Ns,) ndarray
            Quasiparticle eigen-energies :math:`E_n`.
        U_, V_ : (Ns,Ns) ndarray
            Bogoliubov transformation matrices.
        Phi : (Ns,Ns) ndarray
            Real-space wavefunctions.
        """
        argsFn = ('bogWf',self.Lx,self.Ly,self.Ns,self.p.diag.Hamiltonian,self.boundary)
        transformationFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npz')
        hamiltonian = self._realSpaceHamiltonian(verbose)
        if not Path(transformationFn).is_file():
            if np.max(np.absolute(hamiltonian-hamiltonian.conj()))>1e-5:
                raise ValueError("Hamiltonian is not real! Procedure might be wrong")
            # Para-diagonalization
            Ns = self.Ns
            A = hamiltonian[:Ns,:Ns]
            B_off = hamiltonian[:Ns,Ns:]
            try:
                K = scipy.linalg.cholesky(A - B_off)
            except np.linalg.LinAlgError:
                if verbose:
                    print("Cholesky failed, regularizing ...")
                reg = 1e-4
                while reg < 1.0:
                    try:
                        K = scipy.linalg.cholesky(A - B_off + np.identity(Ns) * reg)
                        if verbose:
                            print(f"    converged with reg = {reg:.1e}")
                        break
                    except np.linalg.LinAlgError:
                        reg *= 10
                else:
                    raise np.linalg.LinAlgError(
                        "A - B_off is not positive definite even with "
                        f"regularisation up to {reg}"
                    )
            lam2,chi_ = scipy.linalg.eigh(K@(A+B_off)@K.T.conj())
            if self.p.diag.excludeZeroMode:
                mask0modes = lam2<1e-8
                lam2[mask0modes] = 1
            self.evals = np.sqrt(lam2)         #dispersion -> positive
            #
            chi = chi_ / self.evals**(1/2)     #normalized eigenvectors: divide each column of chi_ by the corresponding eigenvalue -> of course for the gapless mode there is a problem here
            phi_ = K.T.conj()@chi
            psi_ = (A+B_off)@phi_/self.evals       # Problem also here
            self.U_ = 1/2*(phi_+psi_)
            self.V_ = 1/2*(phi_-psi_)
            if self.p.diag.excludeZeroMode:
                self.evals[mask0modes] = 0               # Set again to 0 the gapless mode
            self.Phi = np.real(self.U_-self.V_)
            # Re-rotate back to normal basis
            for ind in range(self.Ns):
                x,y = self._xy(ind)
                if self.order=='canted-Neel':
                    self.Phi[ind,:] *= 2/np.pi*(-1)**(x+y)
                    continue
                elif self.order=='canted-stripe':
                    self.Phi[ind,:] *= (-1)**(x%2)
                    continue
                    if x%2==1 and y%2==0:
                        self.Phi[ind,:] *= 2/np.pi*(-1)**(x+y)
                        continue
                    if x%2==0 and y%2==1:
                        continue
                        #self.thetas[i] *= -1
                        #self.thetas[i] += np.pi
                    if x%2==1 and y%2==1:
                        self.Phi[ind,:] *= 2/np.pi*(-1)**(x+y)
                    #raise ValueError("Not implemented")
            if self.p.diag.saveWf:
                Path(self.dataDn).mkdir(parents=True, exist_ok=True)
                np.savez(transformationFn, awesomeU=self.U_, awesomeV=self.V_, evals=self.evals, Phi=self.Phi)
        else:
            if verbose:
                print("Loading Bogoliubov transformation from file: "+transformationFn)
            self.U_ = np.load(transformationFn)['awesomeU']
            self.V_ = np.load(transformationFn)['awesomeV']
            self.Phi = np.load(transformationFn)['Phi']
            self.evals = np.load(transformationFn)['evals']

        if self.p.diag.excludeZeroMode:       #Put to 0 the eigenstate corresponding to the zero energy mode -> a bit far fetched
            self.U_[:,0] *= 0
            self.V_[:,0] *= 0
        if self.p.diag.plotWf:
            #plotWf3D(self)
            plotWf2D(self,nModes=self.Ns if self.Ns<=30 else 30)
            #plotWfCos(self)
        if self.p.diag.plotMomenta:
            plotBogoliubovMomenta(self,**kwargs)

    def computeRate(self,verbose=False):
        """Compute magnon decay/scattering rates via Fermi's Golden Rule.

        For each process type listed in ``p.scattering.types``, the
        corresponding vertex is computed (or loaded from disk) and the
        rate is evaluated via the dispatch function in
        :mod:`wavespin.static.decayProcesses`.

        Rates are cached to disk using deterministic filenames.

        Sets
        ----
        rates : dict
            ``{process_type: array_of_shape_(Ns,)}``.
        """
        self.rates = {}
        for process in self.p.scattering.types:
            argsDecayFn = ['decay',process,self.p.scattering.temperature,self.p.scattering.broadening,self.p.diag.Hamiltonian,self.Lx,self.Ly,self.Ns,self.boundary]
            decayFn = pf.getFilename(*tuple(argsDecayFn),dirname=self.dataDn,extension='.npy',floatPrecision=8)
            if Path(decayFn).is_file():
                self.rates[process] = np.load(decayFn)
                if verbose:
                    print("Loading rates of process %s"%process)
                continue
            if not hasattr(self,'vertex'+process[:4]):
                self.computeVertex(process[:4],verbose=verbose)
            self.rates[process] = dic_processes[process](self)
            if self.p.scattering.saveRate:
                np.save(decayFn,self.rates[process])
        if self.p.scattering.plotRate:
            plotRate(self)

    def computeVertex(self,vertex,verbose=False):
        """Compute the interaction vertex tensor for a given process.

        Parameters
        ----------
        vertex : str
            One of ``'1to2'``, ``'2to2'``, ``'1to3'``.
        verbose : bool
            If True, prints progress messages.

        Returns
        -------
        ndarray
            Vertex tensor whose shape depends on the process:
            ``(Ns,Ns,Ns)`` for 1→2, ``(Ns,Ns,Ns,Ns)`` for 2→2 and 1→3.

        Notes
        -----
        The tensor is computed via :func:`numpy.einsum` contractions
        over the Bogoliubov matrices ``U_``, ``V_`` and the
        site-dependent coupling factors.  The result is attached as
        ``self.vertex1to2`` (or ``self.vertex2to2``, ``self.vertex1to3``)
        and cached to disk.
        """
        argsVertexFn = ['vertex',vertex,self.p.diag.Hamiltonian,self.Lx,self.Ly,self.Ns,self.boundary]
        vertexFn = pf.getFilename(*tuple(argsVertexFn),dirname=self.dataDn,extension='.npy')
        if Path(vertexFn).is_file():
            result = np.load(vertexFn)
            if 0:   # Checks on sparsity
                sparsity = np.count_nonzero(result==0) / result.size
                sparsity = result[np.absolute(result)<np.max(np.absolute(result))/100].size / result.size
                print(vertexFn)
                print("Vertex: %s has sparcity %.5f"%(vertex,sparsity))
                print(np.max(result**2))
                input()
        else:
            if verbose:
                print("Computing ",vertex," vertex")
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
            if self.p.scattering.saveVertex:
                    print("Saving vertex %s"%vertex)
                    np.save(vertexFn,result)
            setattr(self,'vertex'+vertex,result)

##########################################################
##########################################################

class openSystem(openHamiltonian):
    def __init__(self, p: SimParams):
        # Construct lattice and Hamiltonian
        super().__init__(p)
        #XT correlator parameters
        self.perturbationSite = p.correlator.perturbationSite
        self.perturbationIndex = self._idx(*self.perturbationSite)
        #
        self.site0 = 0 #if h_t_i[0,0,0]<0 else 1     #decide sublattice A and B of reference lattice site
        #self.fullTimeMeasure, self.nTimes, self.nOmega = (0.8,401,2000)
        self.fullTimeMeasure, self.nTimes, self.nOmega = (16,401,2000)
        print("Using fullTimeMeasure=%d"%self.fullTimeMeasure)
        #self.fullTimeMeasure = 0.8     #measure time in ms
        #self.nTimes = 401        #time steps after ramp for the measurement
        self.measureTimeList = np.linspace(0,self.fullTimeMeasure,self.nTimes)
        #KW correlator parameters
        #self.nOmega = 2000
        # Diagonalize
        #self.diagonalize()

    def realSpaceCorrelator(self,verbose=False):
        """ Here we compute the correlator in real space.
        """
        temperature = self._temperature(self.p.correlator.energy)
        print("Temperature: %.3f MHz"%temperature)
        txtZeroEnergy = 'without0energy' if self.p.diag.excludeZeroMode else 'with0energy'
        argsFn = ('correlatorXT',self.p.correlator.correlatorType,self.Lx,self.Ly,self.Ns,self.p.diag.Hamiltonian,
                  txtZeroEnergy,'magnonModes',self.p.correlator.magnonModes,self.p.correlator.energy)
        correlatorFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npy')
        if not Path(correlatorFn).is_file():
            self.correlatorXT = np.zeros((self.Ns,self.nTimes),dtype=complex)
            #Correlator -> can make this faster, we actually only need U_ and V_
            exp_e = np.exp(-1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
            A_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.V_,self.U_,optimize=True)
            B_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.U_,self.V_,optimize=True)
            G_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.V_,self.V_,optimize=True)
            H_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.U_,self.U_,optimize=True)
            if temperature != 0:
                exp_e_c = np.exp(1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
                #exp_e_c = exp_e.conj()
                BF = np.zeros(self.Ns)
                BF[1:] = 1/(np.exp(self.evals[1:]/temperature)-1)
                #
                A2  = np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.U_,exp_e, optimize=True)
                A2 += np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.V_,exp_e_c,optimize=True)
                #
                B2  = np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.V_,exp_e, optimize=True)
                B2 += np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.U_,exp_e_c,optimize=True)
                #
                G2  = np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.V_,exp_e, optimize=True)
                G2 += np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.U_,exp_e_c,optimize=True)
                #
                H2  = np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.U_,exp_e, optimize=True)
                H2 += np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.V_,exp_e_c,optimize=True)
            else:
                A2 = B2 = G2 = H2 = 0
            Af = A_GS + A2
            Bf = B_GS + B2
            Gf = G_GS + 2*G2
            Hf = H_GS + 2*H2
            if 0:
                print("A_ij")
                print("GS : %.3f,%.3f"%(np.max(np.absolute(np.real(A_GS))),np.max(np.absolute(np.imag(A_GS)))))
                print("T  : %.3f,%.3f"%(np.max(np.absolute(np.real(A2  ))),np.max(np.absolute(np.imag(A2  )))))
                print("fin: %.3f,%.3f"%(np.max(np.absolute(np.real(Af  ))),np.max(np.absolute(np.imag(Af  )))))
                print("B_ij")
                print("GS : %.3f,%.3f"%(np.max(np.absolute(np.real(B_GS))),np.max(np.absolute(np.imag(B_GS)))))
                print("T  : %.3f,%.3f"%(np.max(np.absolute(np.real(B2  ))),np.max(np.absolute(np.imag(B2  )))))
                print("fin: %.3f,%.3f"%(np.max(np.absolute(np.real(Bf  ))),np.max(np.absolute(np.imag(Bf  )))))
                print("G_ij")
                print("GS : %.3f,%.3f"%(np.max(np.absolute(np.real(G_GS))),np.max(np.absolute(np.imag(G_GS)))))
                print("T  : %.3f,%.3f"%(np.max(np.absolute(np.real(G2  ))),np.max(np.absolute(np.imag(G2  )))))
                print("fin: %.3f,%.3f"%(np.max(np.absolute(np.real(Gf  ))),np.max(np.absolute(np.imag(Gf  )))))
                print("H_ij")
                print("GS : %.3f,%.3f"%(np.max(np.absolute(np.real(H_GS))),np.max(np.absolute(np.imag(H_GS)))))
                print("T  : %.3f,%.3f"%(np.max(np.absolute(np.real(H2  ))),np.max(np.absolute(np.imag(H2  )))))
                print("fin: %.3f,%.3f"%(np.max(np.absolute(np.real(Hf  ))),np.max(np.absolute(np.imag(Hf  )))))
                input()
            if temperature != 0 and 0:
                fig = plt.figure(figsize=(12,12))
                funcs = [Af,Bf,Gf,Hf,Gf+Hf]
                funcs1 = [A_GS,B_GS,G_GS,H_GS,G_GS+H_GS]
                funcs2 = [A2,B2,G2,H2,G2+H2]
                pI = self.perturbationIndex
                ss = pI + np.array([-1,+1,-self.Ly,self.Ly],dtype=int)
                for iA in range(5):
                    f = np.real(funcs[iA][:,pI,:])
                    fi = np.imag(funcs[iA][:,pI,:])
                    f1 = np.real(funcs1[iA][:,pI,:])
                    f1i = np.imag(funcs1[iA][:,pI,:])
                    f2 = np.real(funcs2[iA][:,pI,:])
                    f2i = np.imag(funcs2[iA][:,pI,:])
                    for ix in range(4):
                        ax = fig.add_subplot(5,4,iA*4+ix+1)
                        #ax.plot(self.measureTimeList,f[ss[ix],:],color='r')
                        ax.plot(self.measureTimeList,fi[ss[ix],:],color='orange')
                        #ax.plot(self.measureTimeList,f1[ss[ix],:],color='g')
                        ax.plot(self.measureTimeList,f1i[ss[ix],:],color='forestgreen')
                        #ax.plot(self.measureTimeList,f2[ss[ix],:],color='b')
                        ax.plot(self.measureTimeList,f2i[ss[ix],:],color='aqua')
                        ax.set_xlim(0,0.1)
                plt.show()
            #
            for ind_i in range(self.Ns):
                self.correlatorXT[ind_i] = correlators.dicCorrelators[self.p.correlator.correlatorType](self,ind_i,Af,Bf,Gf,Hf)
            if self.p.correlator.saveXT:
                np.save(correlatorFn,self.correlatorXT)
        else:
            if verbose:
                print("Loading real-space correlator from file: "+correlatorFn)
            self.correlatorXT = np.load(correlatorFn)

    def realSpaceCorrelatorBond(self,verbose=False):
        """ Here we compute the correlator in real space for each bond, like for the jj.
        """
        temperature = self._temperature(self.p.correlator.energy)
        Lx = self.Lx
        Ly = self.Ly
        Ns = self.Ns
        txtZeroEnergy = 'without0energy' if self.p.diag.excludeZeroMode else 'with0energy'
        argsFn_h = ('correlator_horizontal_bonds',self.p.correlator.correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy,'magnonModes',self.p.correlator.magnonModes,self.perturbationSite,self.p.correlator.energy)
        argsFn_v = ('correlator_vertical_bonds',self.p.correlator.correlatorType,self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,Ns,txtZeroEnergy,'magnonModes',self.p.correlator.magnonModes,self.perturbationSite,self.p.correlator.energy)
        correlatorFn_h = pf.getFilename(*argsFn_h,dirname=self.dataDn,extension='.npy')
        correlatorFn_v = pf.getFilename(*argsFn_v,dirname=self.dataDn,extension='.npy')
        if not Path(correlatorFn_h).is_file() or not Path(correlatorFn_v).is_file():
            self.correlatorXT_h = np.zeros((Lx-1,Ly,self.nTimes),dtype=complex)
            self.correlatorXT_v = np.zeros((Lx,Ly-1,self.nTimes),dtype=complex)
            #
            exp_e = np.exp(-1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
            A_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.V_,self.U_,optimize=True)
            B_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.U_,self.V_,optimize=True)
            G_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.V_,self.V_,optimize=True)
            H_GS = np.einsum('tk,ik,jk->ijt',exp_e,self.U_,self.U_,optimize=True)
            if temperature != 0:
                exp_e_c = np.exp(1j*2*np.pi*self.measureTimeList[:,None]*self.evals[None,:])
                #exp_e_c = exp_e.conj()
                BF = np.zeros(self.Ns)
                BF[1:] = 1/(np.exp(self.evals[1:]/temperature)-1)
                #
                A2  = np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.U_,exp_e, optimize=True)
                A2 += np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.V_,exp_e_c,optimize=True)
                #
                B2  = np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.V_,exp_e, optimize=True)
                B2 += np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.U_,exp_e_c,optimize=True)
                #
                G2  = np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.V_,exp_e, optimize=True)
                G2 += np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.U_,exp_e_c,optimize=True)
                #
                H2  = np.einsum('l,il,jl,tl->ijt',BF,self.U_,self.U_,exp_e, optimize=True)
                H2 += np.einsum('l,il,jl,tl->ijt',BF,self.V_,self.V_,exp_e_c,optimize=True)
            else:
                A2 = B2 = G2 = H2 = 0
            Af = A_GS + A2
            Bf = B_GS + B2
            Gf = G_GS + 2*G2
            Hf = H_GS + 2*H2
            #
            for ihx in range(Lx-1):
                for ihy in range(Ly):
                    ind_i = self._idx(ihx,ihy)
                    self.correlatorXT_h[ihx,ihy] = correlators.jjCorrelatorBond(self,ind_i,Af,Bf,Gf,Hf,'h')
            for ivx in range(Lx):
                for ivy in range(Ly-1):
                    ind_i = self._idx(ivx,ivy)
                    self.correlatorXT_v[ivx,ivy] = correlators.jjCorrelatorBond(self,ind_i,Af,Bf,Gf,Hf,'v')
            if self.p.correlator.saveXTbonds:
                Path(self.dataDn).mkdir(parents=True, exist_ok=True)
                np.save(correlatorFn_h, self.correlatorXT_h)
                np.save(correlatorFn_v, self.correlatorXT_v)
        else:
            if verbose:
                print("Loading real-space bond correlator from file: "+correlatorFn_h)
            self.correlatorXT_h = np.load(correlatorFn_h)
            self.correlatorXT_v = np.load(correlatorFn_v)

    def momentumSpaceCorrelator(self,verbose=False):
        """ Here we simply Fourier transform the correlator.
        """
        temperature = self._temperature(self.p.correlator.energy)
        txtZeroEnergy = 'without0energy' if self.p.diag.excludeZeroMode else 'with0energy'
        argsFn = ('correlatorKW',self.p.correlator.correlatorType,self.p.correlator.transformType,self.Lx,self.Ly,self.Ns,self.p.diag.Hamiltonian,
                  txtZeroEnergy,'magnonModes',self.p.correlator.magnonModes,self.p.correlator.energy)
        correlatorFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npz')
        if not Path(correlatorFn).is_file():
            self.correlatorKW, self.momentum = momentumTransformation.dicTransformType[self.p.correlator.transformType](self)
            if self.p.correlator.saveKW:
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
            #self.rampElements[i].diagonalize(verbose=verbose)
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
                       **{
                           'numKbins' : 50,
                           'ylim' : 7 if self.rampElements[0].g1==10 else 3.5,
                           'saveFigure' : self.rampElements[0].p.cor_savePlotKW,
                           'showFigure' : True,
                       }
                       )



















