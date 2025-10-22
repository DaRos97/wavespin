""" Functions defining the lattice.
"""

import numpy as np
from wavespin.tools import pathFinder as pf
from pathlib import Path
from wavespin.plots import fancyLattice
import copy

class latticeClass():
    def __init__(self,p):
        self.p = copy.deepcopy(p)
        self.Lx = p.lat_Lx
        self.Ly = p.lat_Ly
        self.offSiteList = p.lat_offSiteList
        self.indexesMap = self._mapSiteIndex()
        self.Ns = self.Lx*self.Ly - len(self.offSiteList)
        self.boundary = p.lat_boundary
        # Precompute neighbors
        self.NN = self._build_nn()
        self.NNN = self._build_nnn()
        # Directory names
        self.dataDn = pf.getHomeDirname(str(Path.cwd()),'/Data/')
        self.figureDn = pf.getHomeDirname(str(Path.cwd()),'/Figures/')
        # Plotting
        if p.lat_plotLattice:
            fancyLattice.plotSitesGrid(self)

    def _xy(self, i):
        return i // self.Ly, i % self.Ly

    def _idx(self, x, y):
        return self.Ly * (x % self.Lx) + (y % self.Ly) if self.boundary=='periodic' else self.Ly * x + y

    def _build_nn(self):
        """ Construct a list of nn indexes for each site index.
        """
        NN = [[] for _ in range(self.Ns)]
        for x in range(self.Lx):
            for y in range(self.Ly):
                i = self._idx(x, y)
                if self.boundary=='periodic':
                    NN[i] = [
                        self._idx(x+1, y),
                        self._idx(x-1, y),
                        self._idx(x, y+1),
                        self._idx(x, y-1),
                    ]
                else:
                    NN[i] = self._open_nn(i)
        return NN

    def _build_nnn(self):
        """ Construct a list of nnn indexes for each site index.
        """
        NNN = [[] for _ in range(self.Ns)]
        for x in range(self.Lx):
            for y in range(self.Ly):
                i = self._idx(x, y)
                if self.boundary=='periodic':
                    NNN[i] = [
                        self._idx(x+1, y+1),
                        self._idx(x-1, y+1),
                        self._idx(x+1, y-1),
                        self._idx(x-1, y-1),
                    ]
                else:
                    NNN[i] = self._open_nnn(i)
        return NNN

    def _open_nn(self,ind):
        """ Compute indices of nearest neighbors of site ind.
        """
        Lx = self.Lx
        Ly = self.Ly
        result= []
        if ind//Ly!=Lx-1:        #right neighbor
            result.append(ind+Ly)
        if ind//Ly!=0:           #left neighbor
            result.append(ind-Ly)
        if ind%Ly!=Ly-1:         #up neighbor
            result.append(ind+1)
        if ind%Ly!=0:            #bottom neighbor
            result.append(ind-1)
        return result

    def _open_nnn(self,ind):
        """ Compute indices of next-nearest neighbors of site ind.
        """
        Lx = self.Lx
        Ly = self.Ly
        result= []
        if ind//Ly!=Lx-1 and ind%Ly!=Ly-1:        #right-up neighbor
            result.append(ind+Ly+1)
        if ind//Ly!=Lx-1 and ind%Ly!=0:        #right-down neighbor
            result.append(ind+Ly-1)
        if ind//Ly!=0 and ind%Ly!=Ly-1:        #left-up neighbor
            result.append(ind-Ly+1)
        if ind//Ly!=0 and ind%Ly!=0:        #left-down neighbor
            result.append(ind-Ly-1)
        return result

    def _mapSiteIndex(self):
        """ Here we define a map: from an index between 0 and Ns-1 to (ix,iy) between 0 and Lx/y-1.
        To each index in the actual used qubits assign the corresponding ix,iy.

        Returns
        -------
        indexesMap : list of 2-tuple.
            Coordinates of sites which are been considered, in order of their index.
        """
        indexesMap = []
        for ix in range(self.Lx):
            for iy in range(self.Ly):
                if (ix,iy) not in self.offSiteList:
                    indexesMap.append((ix,iy))
        return indexesMap



