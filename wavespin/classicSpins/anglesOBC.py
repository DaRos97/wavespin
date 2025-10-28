""" Functions for minimization of quantization axis angles.
"""

import numpy as np
from numpy.random import default_rng
from pathlib import Path
from wavespin.lattice.lattice import latticeClass
import wavespin.tools.inputUtils as iu
from wavespin.static.periodic import quantizationAxis
import wavespin.tools.pathFinder as pf
from scipy.optimize import minimize

class classicMagnetization():
    def __init__(self,obj,verbose=False):
        """ Obj is an openHamiltonian object.
        """
        self.S = obj.S
        self.planar = True
        self.A1 = np.diag([1.0, 1.0, obj.d1])
        self.A2 = np.diag([1.0, 1.0, obj.d2])
        #
        self.findSolution(obj,verbose)

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

    def total_energy(self,angles,*args):
        """ Compute total energy.
        Bonds each counted twice -> multiply by 1/2.
        """
        obj = args[0]
        if self.planar:
            thetas = angles
            phis = np.zeros(obj.Ns)
        else:
            thetas = angles[:obj.Ns]
            phis = angles[obj.Ns:]
        E = 0.0
        for i in range(obj.Ns):
            si = self.getSpin(thetas[i],phis[i])
            E += 1/2 * obj.g1 * sum(self._pair_energy_nn(si, self.getSpin(thetas[j],phis[j])) for j in obj.NN[i])
            E += 1/2 * obj.g2 * sum(self._pair_energy_nnn(si, self.getSpin(thetas[k],phis[k])) for k in obj.NNN[i])
        E += np.sum(obj.h_i * np.cos(thetas)) * obj.S
        E /= obj.Ns
        return E

    def findSolution(self,obj,verbose):
        """ Minimize energy to get the angles.
        Start with periodic solution as initial guess
        """
        # Initial condition
        if self.planar:
            x_initial = np.ones(obj.Ns)*obj.theta + 1e-2*np.random.rand(obj.Ns)
            for i in range(obj.Ns):
                ix,iy = obj._xy(i)
                x_initial[i] -= np.pi*((ix+iy)%2)
            bounds = [(-np.pi,np.pi) for _ in range(obj.Ns)]
        else:
            x_initial = np.ones(2*obj.Ns)
            x_initial[:obj.Ns] *= obj.theta
            x_initial[obj.Ns:] = np.zeros(obj.Ns)
            bounds = [(0,np.pi) for _ in range(obj.Ns)]
            bounds += [(0,2*np.pi) for _ in range(obj.Ns)]
        # Minimization
        res = minimize(
            self.total_energy,
            x0 = x_initial,
            args = (obj,),
            bounds = bounds,
            method = 'Nelder-Mead',
            tol = 1e-4,
            options= {
                'disp':verbose,
                'maxiter':1e10}
        )
        self.bestAngles = res.x








