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
        self.indexToSite = self._mapIndexSite()
        self.Ns = self.Lx*self.Ly - len(self.offSiteList)
        self.boundary = p.lat_boundary
        if self.boundary == 'periodic' and len(self.offSiteList) != 0:
            raise ValueError("Periodic and non-rectangular lattice not implemented")
        # Precompute neighbors
        self.NN = self._build_nn()
        self.NNN = self._build_nnn()
        # Directory names
        self.dataDn = pf.getHomeDirname(str(Path.cwd()),'/Data/')
        self.figureDn = pf.getHomeDirname(str(Path.cwd()),'/Figures/')
        # Plotting
        if p.lat_plotLattice:
            fancyLattice.plotSitesGrid(self)
            exit()

    def _xy(self, i):
        return self.indexToSite[i]

    def _idx(self, x, y):
        return self.Ly * (x % self.Lx) + (y % self.Ly) if self.boundary=='periodic' else self.Ly * x + y

    def _build_nn(self):
        """ Construct a list of nn indexes for each site index.
        """
        NN = [[] for _ in range(self.Ns)]
        for ind in range(self.Ns):
            if self.boundary=='periodic':
                x,y = self._xy(ind)
                NN[ind] = [
                    self._idx(x+1, y),
                    self._idx(x-1, y),
                    self._idx(x, y+1),
                    self._idx(x, y-1),
                ]
            else:
                NN[ind] = self._open_nn(ind)
        return NN

    def _open_nn(self,ind):
        """ Compute indices of nearest neighbors of site ind.
        """
        Lx = self.Lx
        Ly = self.Ly
        ix, iy = self._xy(ind)
        result= []
        if ix != Lx-1 and not (ix+1,iy) in self.offSiteList:        #right neighbor
            result.append( self.indexToSite.index((ix+1,iy)) )
        if ix != 0 and not (ix-1,iy) in self.offSiteList:        #left neighbor
            result.append( self.indexToSite.index((ix-1,iy)) )
        if iy != Ly-1 and not (ix,iy+1) in self.offSiteList:        #upper neighbor
            result.append( self.indexToSite.index((ix,iy+1)) )
        if iy != 0 and not (ix,iy-1) in self.offSiteList:        #lower neighbor
            result.append( self.indexToSite.index((ix,iy-1)) )
        return result

    def _build_nnn(self):
        """ Construct a list of nnn indexes for each site index.
        """
        NNN = [[] for _ in range(self.Ns)]
        for ind in range(self.Ns):
            if self.boundary=='periodic':
                x,y = self._xy(ind)
                NNN[ind] = [
                    self._idx(x+1, y+1),
                    self._idx(x-1, y+1),
                    self._idx(x+1, y-1),
                    self._idx(x-1, y-1),
                ]
            else:
                NNN[ind] = self._open_nnn(ind)
        return NNN

    def _open_nnn(self,ind):
        """ Compute indices of next-nearest neighbors of site ind.
        """
        Lx = self.Lx
        Ly = self.Ly
        ix, iy = self._xy(ind)
        result= []
        if ix != Lx-1 and iy != Ly-1 and not (ix+1,iy+1) in self.offSiteList:        #right-up neighbor
            result.append( self.indexToSite.index((ix+1,iy+1)) )
        if ix != 0 and iy != Ly-1 and not (ix-1,iy+1) in self.offSiteList:        #left-up neighbor
            result.append( self.indexToSite.index((ix-1,iy+1)) )
        if ix != Lx-1 and iy != 0 and not (ix+1,iy-1) in self.offSiteList:        #right-down neighbor
            result.append( self.indexToSite.index((ix+1,iy-1)) )
        if ix != 0 and iy != 0 and not (ix-1,iy-1) in self.offSiteList:        #left-down neighbor
            result.append( self.indexToSite.index((ix-1,iy-1)) )
        return result

    def _mapIndexSite(self):
        """ Here we define a map: from an index between 0 and Ns-1 to (ix,iy) between 0 and Lx/y-1.
        To each index in the actual used qubits assign the corresponding site ix,iy.

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



