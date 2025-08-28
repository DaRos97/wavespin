""" Functions for montecarlo simulation of the classical ground state.

"""

import numpy as np
from numpy.random import default_rng
from pathlib import Path

from wavespin.lattice.lattice import latticeClass, hamiltonianClass
from wavespin.tools.inputUtils import classicParameters
from wavespin.static.openCorrelators import get_nn, get_nnn
import wavespin.tools.pathFinder as pf
from wavespin.static.periodic import quantizationAxis

rng = default_rng()

class XXZJ1J2MC(hamiltonianClass):
    def __init__(self, p: classicParameters, termsHamiltonian, **kwargs):
        self.p = p
        # Construct lattice and Hamiltonian
        super().__init__(Lx=p.Lx,Ly=p.Ly,offSiteList=p.offSiteList,boundary=p.boundary,termsHamiltonian=termsHamiltonian,**kwargs)
        # Random number generator
        self.rng = default_rng(p.seed)
        # Spins as unit vectors (N,3)
        self.Spins = self._random_unit_vectors(self.Ns)
        # Staggering factor eta_i = (-1)^(x+y)
        self.eta = self._staggering()
        # Hamiltonian parameters
        self.periodicTheta, self.periodicPhi = quantizationAxis(0.5,(self.g1,self.g2),(self.d1,self.d2),self.h)
        # Anisotropy matrix A = diag(1,1,g) -> x,y,z
        self.A1 = np.diag([1.0, 1.0, self.d1])
        self.A2 = np.diag([1.0, 1.0, self.d2])

    def _staggering(self):
        eta = np.empty(self.Ns, dtype=np.int8)
        for i in range(self.Ns):
            x, y = self._xy(i)
            eta[i] = 1 if ((x + y) % 2 == 0) else -1
        return eta

    # ---------- spin helpers ----------
    def _random_unit_vectors(self, n):
        v = self.rng.normal(size=(n,3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    def _random_small_rotation(self, s):
        # Propose a random small rotation of vector s with angle ~ proposal_step
        # Generate a random axis perpendicular to s
        axis = self.rng.normal(size=3)
        axis -= axis.dot(s) * s
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return s  # rare
        axis /= n
        # Small angle around this axis
        angle = self.p.proposal_step * (1.0 + 0.2*self.rng.standard_normal())
        ca, sa = np.cos(angle), np.sin(angle)
        # Rodrigues' formula: rotate s around 'axis' by 'angle'
        return ca * s + sa * np.cross(axis, s) + (1 - ca) * (axis * (axis @ s))

    # ---------- energy pieces ----------
    def _pair_energy_nn(self, si, sj):
        # si^T A sj = SxSx + SySy + g SzSz
        return si @ (self.A1 @ sj)

    def _pair_energy_nnn(self, si, sj):
        # si^T A sj = SxSx + SySy + g SzSz
        return si @ (self.A2 @ sj)

    def local_energy(self, i, s=None):
        # Energy contributions involving site i only (avoid double counting by halving bonds elsewhere if you sum over all sites)
        if s is None:
            s = self.Spins[i]
        g1, g2, h = self.g1, self.g2, self.h
        e = 0.0
        for j in self.NN[i]:
            e += g1 * self._pair_energy_nn(s, self.Spins[j])
        for k in self.NNN[i]:
            e += g2 * self._pair_energy_nnn(s, self.Spins[k])
        e += -h * self.eta[i] * s[2]
        return e

    def total_energy(self):
        g1, g2, h = self.g1, self.g2, self.h
        E = 0.0
        # Bonds each counted twice -> multiply by 0.5
        for i in range(self.Ns):
            si = self.Spins[i]
            E += 0.5 * g1 * sum(self._pair_energy_nn(si, self.Spins[j]) for j in self.NN[i])
            E += 0.5 * g2 * sum(self._pair_energy_nnn(si, self.Spins[k]) for k in self.NNN[i])
        E += -h * np.sum(self.eta * self.Spins[:,2])
        return E

    def local_field(self, i):
        # Effective field h_eff so that E_i = S_i · h_eff
        g1, g2, h = self.g1, self.g2, self.h
        h_eff = np.zeros(3)
        # Anisotropic neighbor contribution
        h_eff += g1 * (self.A1 @ self.Spins[self.NN[i]].T).sum(axis=1)
        h_eff += g2 * (self.A2 @ self.Spins[self.NNN[i]].T).sum(axis=1)
        # Staggered field term contributes (0,0,-h*eta)
        h_eff += np.array([0.0, 0.0, -h * self.eta[i]])
        return h_eff

    # ---------- MC sweeps ----------
    def metropolis_sweep(self, T):
        accepts = 0
        for _ in range(self.Ns):
            i = int(self.rng.integers(self.Ns))
            s_old = self.Spins[i]
            s_new = self._random_small_rotation(s_old)
            dE = self.local_energy(i, s_new) - self.local_energy(i, s_old)
            if dE <= 0 or self.rng.random() < np.exp(-dE / T):
                self.Spins[i] = s_new / np.linalg.norm(s_new)
                accepts += 1
        return accepts / self.Ns

    def overrelaxation_sweep(self):
        # Reflect spin i about local field h_eff: s' = 2 (s·h) h/||h||^2 - s
        for i in range(self.Ns):
            h = self.local_field(i)
            hn2 = h @ h
            if hn2 < 1e-14:
                continue
            s = self.Spins[i]
            s_new = 2.0 * (s @ h) * h / hn2 - s
            # normalize to be safe
            self.Spins[i] = s_new / np.linalg.norm(s_new)

    # ---------- measurements ----------
    def magnetization(self):
        return self.Spins.mean(axis=0)

    def staggered_mz(self):
        return float((self.eta * self.Spins[:,2]).mean())

    def structure_factor(self, component='z'):
        # Compute S(q) for chosen component ('x','y','z' or 'tot')
        Lx = self.Lx
        Ly = self.Ly
        coords = np.indices((Lx, Ly))
        kx = 2*np.pi*np.fft.fftfreq(Lx)
        ky = 2*np.pi*np.fft.fftfreq(Ly)
        if component == 'tot':
            field = np.linalg.norm(self.Spins.reshape(Lx, Ly, 3), axis=2)
        else:
            comp = {'x':0,'y':1,'z':2}[component]
            field = self.Spins[:, comp].reshape(Lx, Ly)
        F = np.fft.fft2(field)
        S_q = (F * np.conj(F)).real / self.Ns
        return np.fft.fftshift(S_q), np.fft.fftshift(kx), np.fft.fftshift(ky)

    # ---------- annealing runner ----------
    def sweeps_at_T(self, T):
        # Increase sweeps as we cool
        hi, lo = self.p.sweeps_per_T_high, self.p.sweeps_per_T_low
        # linear interpolation in log T
        t = (np.log(T) - np.log(self.p.T_min)) / (np.log(self.p.T_max) - np.log(self.p.T_min) + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        return int(lo + (hi - lo) * t)

    def anneal(self, verbose=False):
        """ Perform annealing to get the best configuration S.)
        """
        argsFn = ('MC',self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,self.Ns, self.p.boundary)
        dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
        solutionFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npz')
        if not Path(solutionFn).is_file():
            T = self.p.T_max
            best_E = np.inf
            best_S = None
            history = []
            while T > self.p.T_min:
                nsw = self.sweeps_at_T(T)
                acc = 0.0
                for _ in range(nsw):
                    acc += self.metropolis_sweep(T)
                    if self.p.overrelax_every > 0:
                        for _ in range(self.p.overrelax_every):
                            self.overrelaxation_sweep()
                acc /= max(nsw,1)
                E = self.total_energy()
                m = self.magnetization()
                ms = self.staggered_mz()
                history.append((T, E/self.Ns, acc, m[0], m[1], m[2], ms))
                if verbose:
                    print(f"T={T:.4f}  E/N={E/self.Ns:.6f}  acc={acc:.2f}  m=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f})  m_stag_z={ms:.3f}")
                if E < best_E:
                    best_E = E
                    best_S = self.Spins.copy()
                T *= self.p.alpha
            # set best configuration
            if best_S is not None:
                self.Spins = best_S
            history = np.array(history)
            if self.p.saveSolution:
                if not Path(dataDn).is_dir():
                    print("Creating 'Data/' folder in home directory.")
                    os.system('mkdir '+dataDn)
                np.savez(solutionFn,Spins=self.Spins,history=history)
        else:
            self.Spins = np.load(solutionFn)['Spins']
            history = np.load(solutionFn)['history']
        return history












