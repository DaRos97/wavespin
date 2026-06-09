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
        self.Lx = self.p.lat_Lx
        self.Ly = self.p.lat_Ly
        self.offSiteList = self.p.lat_offSiteList
        self.offSiteSet = set(self.offSiteList)
        self.indexToSite = self._mapIndexSite()
        self.siteToIndex = {site: idx for idx, site in enumerate(self.indexToSite)}
        self.Ns = self.Lx*self.Ly - len(self.offSiteList)
        self.boundary = self.p.lat_boundary
        if self.boundary == 'periodic':
            if len(self.offSiteList) != 0:
                raise ValueError("Periodic and non-square lattice not implemented")
            if self.Lx%2 or self.Ly%2:
                raise ValueError("For a periodic boundary you need even Lx and Ly!")
        # Precompute neighbors
        self.NN = self._build_nn()
        self.NNN = self._build_nnn()
        # Directory names
        self.dataDn = pf.getHomeDirname(str(Path.cwd()),'/Data/')
        Path(self.dataDn).mkdir(parents=True, exist_ok=True)
        self.figureDn = pf.getHomeDirname(str(Path.cwd()),'/Figures/')
        Path(self.figureDn).mkdir(parents=True, exist_ok=True)
        # Plotting
        if p.lat_plotLattice:
            fancyLattice.plotSitesGrid(self)

    def _xy(self, i):
        return self.indexToSite[i]

    def _idx(self, x, y):
        return self.siteToIndex[(x, y)]

    def _build_nn(self):
        """ Construct a list of nn indexes for each site index.
        """
        NN = [[] for _ in range(self.Ns)]
        for ind in range(self.Ns):
            if self.boundary=='periodic':
                x,y = self._xy(ind)
                NN[ind] = [
                    self._idx((x+1)%self.Lx, y),
                    self._idx((x-1)%self.Lx, y),
                    self._idx(x, (y+1)%self.Ly),
                    self._idx(x, (y-1)%self.Ly),
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
        if ix != Lx-1 and (ix+1, iy) not in self.offSiteSet:
            result.append(self.siteToIndex[(ix+1, iy)])
        if ix != 0 and (ix-1, iy) not in self.offSiteSet:
            result.append(self.siteToIndex[(ix-1, iy)])
        if iy != Ly-1 and (ix, iy+1) not in self.offSiteSet:
            result.append(self.siteToIndex[(ix, iy+1)])
        if iy != 0 and (ix, iy-1) not in self.offSiteSet:
            result.append(self.siteToIndex[(ix, iy-1)])
        return result

    def _build_nnn(self):
        """ Construct a list of nnn indexes for each site index.
        """
        NNN = [[] for _ in range(self.Ns)]
        for ind in range(self.Ns):
            if self.boundary=='periodic':
                x,y = self._xy(ind)
                NNN[ind] = [
                    self._idx((x+1)%self.Lx, (y+1)%self.Ly),
                    self._idx((x-1)%self.Lx, (y+1)%self.Ly),
                    self._idx((x+1)%self.Lx, (y-1)%self.Ly),
                    self._idx((x-1)%self.Lx, (y-1)%self.Ly),
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
        if ix != Lx-1 and iy != Ly-1 and (ix+1, iy+1) not in self.offSiteSet:
            result.append(self.siteToIndex[(ix+1, iy+1)])
        if ix != 0 and iy != Ly-1 and (ix-1, iy+1) not in self.offSiteSet:
            result.append(self.siteToIndex[(ix-1, iy+1)])
        if ix != Lx-1 and iy != 0 and (ix+1, iy-1) not in self.offSiteSet:
            result.append(self.siteToIndex[(ix+1, iy-1)])
        if ix != 0 and iy != 0 and (ix-1, iy-1) not in self.offSiteSet:
            result.append(self.siteToIndex[(ix-1, iy-1)])
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
                if (ix,iy) not in self.offSiteSet:
                    indexesMap.append((ix,iy))
        return indexesMap

    def patchFunction(self,func):
        """ Tool for masking a function defined over a non-square geometry.
        func has to have size Ns
        """
        if len(self.offSiteList)==0:
            formattedFunc = func.reshape(self.Lx,self.Ly)
        else:
            formattedFunc = np.zeros((self.Lx,self.Ly))
            for ix in range(self.Lx):
                for iy in range(self.Ly):
                    if (ix,iy) in self.offSiteSet:
                        formattedFunc[ix,iy] = np.nan
                    else:
                        formattedFunc[ix,iy] = func[self._idx(ix,iy)]
        return formattedFunc


