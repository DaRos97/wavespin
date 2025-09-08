""" Functions for minimization of quantization axis angles.
"""

import numpy as np
from numpy.random import default_rng
from pathlib import Path
from wavespin.lattice.lattice import latticeClass
from wavespin.tools.inputUtils import classicParameters
import wavespin.tools.pathFinder as pf
from wavespin.static.periodic import quantizationAxis
from scipy.optimize import minimize


class minHam(latticeClass):
    def __init__(self, p: classicParameters, termsHamiltonian):
        self.p = p
        self.txtSim = 'minimization'
        # Construct lattice and Hamiltonian
        super().__init__(p,boundary=p.boundary)
        # Hamiltonian parameters
        self.g1,self.g2,self.d1,self.d2,self.h = termsHamiltonian
        self.planar = True if (self.d1==0 and self.d2==0) else False
        self.S = 1/2
        self.periodicTheta, self.periodicPhi = quantizationAxis(self.S,(self.g1,self.g2),(self.d1,self.d2),self.h)
        # Staggering factor eta_i = (-1)^(x+y)
        self.eta = self._staggering()
        # Anisotropy matrix A = diag(1,1,g) -> x,y,z
        self.A1 = np.diag([1.0, 1.0, self.d1])
        self.A2 = np.diag([1.0, 1.0, self.d2])

    def _staggering(self):
        eta = np.empty(self.Ns, dtype=np.int8)
        for i in range(self.Ns):
            x, y = self._xy(i)
            eta[i] = (-1)**((x+y+1) % 2)
        return eta

    def getSpin(self,th,ph):
        """ Build spin from angles.
        """
        return np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])

    def _pair_energy_nn(self, si, sj):
        """ si^T A sj = SxSx + SySy + g SzSz.
        """
        return si @ (self.A1 @ sj) * self.S**2

    def _pair_energy_nnn(self, si, sj):
        """ si^T A sj = SxSx + SySy + g SzSz.
        """
        return si @ (self.A2 @ sj) * self.S**2

    def total_energy(self,angles):
        """ Compute total energy.
        Bonds each counted twice -> multiply by 1/2.
        """
        if self.planar:
            thetas = angles
            phis = np.zeros(self.Ns)
        else:
            thetas = angles[:self.Ns]
            phis = angles[self.Ns:]
        E = 0.0
        for i in range(self.Ns):
            si = self.getSpin(thetas[i],phis[i])
            E += 1/2 * self.g1 * sum(self._pair_energy_nn(si, self.getSpin(thetas[j],phis[j])) for j in self.NN[i])
#            E += 1/2 * self.g2 * sum(self._pair_energy_nnn(si, self.getSpin(thetas[k],phis[k])) for k in self.NNN[i])
        E += self.h * np.sum(self.eta * np.cos(thetas)) * self.S
        E /= self.Ns
        return E

    def minimization(self,verbose=False):
        """ Perform the energy minimization.
        """
        argsFn = (self.txtSim+'_solution',self.g1,self.g2,self.d1,self.d2,self.h,self.Lx,self.Ly,self.Ns,self.p.boundary)
        solutionFn = pf.getFilename(*argsFn,dirname=self.dataDn,extension='.npy')
        if not Path(solutionFn).is_file():
            if self.planar:
                x_initial = np.ones(self.Ns)*self.periodicTheta + 1e-2*np.random.rand(self.Ns)
                for i in range(self.Ns):
                    ix,iy = self._xy(i)
                    if (ix+iy)%2==1:
                        x_initial[i] -= np.pi
                bounds = [(-np.pi,np.pi) for _ in range(self.Ns)]
            else:
                x_initial = np.ones(self.Ns*2)
                x_initial[:self.Ns] *= self.periodicTheta
                x_initial[self.Ns:] *= self.periodicPhi
                bounds = [(0,np.pi) for _ in range(self.Ns)]
                bounds += [(0,2*np.pi) for _ in range(self.Ns)]

            res = minimize(
                self.total_energy,
                x0 = x_initial,
                bounds = bounds,
                method = 'Nelder-Mead',
                tol = 1e-4,
                options= {
                    'disp':verbose,
                    'maxiter':1e10}
            )
            bestAngles = res.x
            if self.p.saveSolution:
                if not Path(self.dataDn).is_dir():
                    print("Creating 'Data/' folder in home directory.")
                    os.system('mkdir '+self.dataDn)
                if verbose:
                    print("Saving solution to file: "+solutionFn)
                np.save(solutionFn,bestAngles)
        else:
            if verbose:
                print("Loading angles from file: "+solutionFn)
            bestAngles = np.load(solutionFn)
        return bestAngles







