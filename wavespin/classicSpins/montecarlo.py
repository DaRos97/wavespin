""" Functions for montecarlo simulation of the classical ground state.

"""

# Strategy: simulated annealing with local Metropolis updates + optional overrelaxation

import numpy as np

from numpy.random import default_rng
rng = default_rng()

from dataclasses import dataclass

@dataclass
class Params:
    L: int = 24                    # linear system size; N = L*L
    J1: float = -1.0               # NN coupling (FM<0, AFM>0)
    J2: float = 0.5                # NNN coupling
    D1: float = 1.2                # anisotropy on 1st nn ZZ (D1=1 is isotropic)
    D2: float = 1.2                # anisotropy on 2nd nn ZZ (D1=1 is isotropic)
    h: float = 0.2                 # staggered field strength along z
    T_max: float = 2.5             # start temperature
    T_min: float = 1e-3            # end temperature
    alpha: float = 0.95            # geometric cooling factor
    sweeps_per_T_high: int = 200   # sweeps at high T
    sweeps_per_T_low: int = 1000   # sweeps at low T
    overrelax_every: int = 1       # do 1 overrelax sweep per Metropolis sweep (0 disables)
    proposal_step: float = 0.35    # small rotation angle scale (~0.2–0.5)
    seed: int = 0

class XXZJ1J2MC:
    def __init__(self, p: Params):
        self.p = p
        self.rng = default_rng(p.seed)
        self.L = p.L
        self.N = p.L * p.L
        # Spins as unit vectors (N,3)
        self.S = self._random_unit_vectors(self.N)
        # Precompute neighbors
        self.NN = self._build_nn()
        self.NNN = self._build_nnn()
        # Staggering factor eta_i = (-1)^(x+y)
        self.eta = self._staggering()
        # Anisotropy matrix A = diag(1,1,g) -> x,y,z
        self.A1 = np.diag([1.0, 1.0, self.p.D1])
        self.A2 = np.diag([1.0, 1.0, self.p.D2])

    # ---------- lattice helpers ----------
    def _xy(self, i):
        return i % self.L, i // self.L

    def _idx(self, x, y):
        return (x % self.L) + self.L * (y % self.L)

    def _build_nn(self):
        NN = [[] for _ in range(self.N)]
        for y in range(self.L):
            for x in range(self.L):
                i = self._idx(x, y)
                NN[i] = [
                    self._idx(x+1, y),
                    self._idx(x-1, y),
                    self._idx(x, y+1),
                    self._idx(x, y-1),
                ]
        return np.array(NN, dtype=np.int32)

    def _build_nnn(self):
        NNN = [[] for _ in range(self.N)]
        for y in range(self.L):
            for x in range(self.L):
                i = self._idx(x, y)
                NNN[i] = [
                    self._idx(x+1, y+1),
                    self._idx(x-1, y+1),
                    self._idx(x+1, y-1),
                    self._idx(x-1, y-1),
                ]
        return np.array(NNN, dtype=np.int32)

    def _staggering(self):
        eta = np.empty(self.N, dtype=np.int8)
        for i in range(self.N):
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
            s = self.S[i]
        J1, J2, h = self.p.J1, self.p.J2, self.p.h
        e = 0.0
        for j in self.NN[i]:
            e += J1 * self._pair_energy_nn(s, self.S[j])
        for k in self.NNN[i]:
            e += J2 * self._pair_energy_nnn(s, self.S[k])
        e += -h * self.eta[i] * s[2]
        return e

    def total_energy(self):
        J1, J2, h = self.p.J1, self.p.J2, self.p.h
        E = 0.0
        # Bonds each counted twice -> multiply by 0.5
        for i in range(self.N):
            si = self.S[i]
            E += 0.5 * J1 * sum(self._pair_energy_nn(si, self.S[j]) for j in self.NN[i])
            E += 0.5 * J2 * sum(self._pair_energy_nnn(si, self.S[k]) for k in self.NNN[i])
        E += -h * np.sum(self.eta * self.S[:,2])
        return E

    def local_field(self, i):
        # Effective field h_eff so that E_i = S_i · h_eff
        J1, J2, h = self.p.J1, self.p.J2, self.p.h
        h_eff = np.zeros(3)
        # Anisotropic neighbor contribution
        h_eff += J1 * (self.A1 @ self.S[self.NN[i]].T).sum(axis=1)
        h_eff += J2 * (self.A2 @ self.S[self.NNN[i]].T).sum(axis=1)
        # Staggered field term contributes (0,0,-h*eta)
        h_eff += np.array([0.0, 0.0, -h * self.eta[i]])
        return h_eff

    # ---------- MC sweeps ----------
    def metropolis_sweep(self, T):
        accepts = 0
        for _ in range(self.N):
            i = int(self.rng.integers(self.N))
            s_old = self.S[i]
            s_new = self._random_small_rotation(s_old)
            dE = self.local_energy(i, s_new) - self.local_energy(i, s_old)
            if dE <= 0 or self.rng.random() < np.exp(-dE / T):
                self.S[i] = s_new / np.linalg.norm(s_new)
                accepts += 1
        return accepts / self.N

    def overrelaxation_sweep(self):
        # Reflect spin i about local field h_eff: s' = 2 (s·h) h/||h||^2 - s
        for i in range(self.N):
            h = self.local_field(i)
            hn2 = h @ h
            if hn2 < 1e-14:
                continue
            s = self.S[i]
            s_new = 2.0 * (s @ h) * h / hn2 - s
            # normalize to be safe
            self.S[i] = s_new / np.linalg.norm(s_new)

    # ---------- measurements ----------
    def magnetization(self):
        return self.S.mean(axis=0)

    def staggered_mz(self):
        return float((self.eta * self.S[:,2]).mean())

    def structure_factor(self, component='z'):
        # Compute S(q) for chosen component ('x','y','z' or 'tot')
        L = self.L
        coords = np.indices((L, L))
        kx = 2*np.pi*np.fft.fftfreq(L)
        ky = 2*np.pi*np.fft.fftfreq(L)
        if component == 'tot':
            field = np.linalg.norm(self.S.reshape(L, L, 3), axis=2)
        else:
            comp = {'x':0,'y':1,'z':2}[component]
            field = self.S[:, comp].reshape(L, L)
        F = np.fft.fft2(field)
        S_q = (F * np.conj(F)).real / self.N
        return np.fft.fftshift(S_q), np.fft.fftshift(kx), np.fft.fftshift(ky)

    # ---------- annealing runner ----------
    def sweeps_at_T(self, T):
        # Increase sweeps as we cool
        hi, lo = self.p.sweeps_per_T_high, self.p.sweeps_per_T_low
        # linear interpolation in log T
        t = (np.log(T) - np.log(self.p.T_min)) / (np.log(self.p.T_max) - np.log(self.p.T_min) + 1e-12)
        t = np.clip(t, 0.0, 1.0)
        return int(lo + (hi - lo) * t)

    def anneal(self, verbose=True):
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
            history.append((T, E/self.N, acc, m[0], m[1], m[2], ms))
            if verbose:
                print(f"T={T:.4f}  E/N={E/self.N:.6f}  acc={acc:.2f}  m=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f})  m_stag_z={ms:.3f}")
            if E < best_E:
                best_E = E
                best_S = self.S.copy()
            T *= self.p.alpha
        # set best configuration
        if best_S is not None:
            self.S = best_S
        return np.array(history)












