""" Here we try the sites for a Corbino disk.
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.lattice.lattice import latticeClass
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian

LxP = 10
LyP = 12
parameters = importParameters()
parameters.lat_Lx = LxP
parameters.lat_Ly = LyP
parameters.lat_boundary = 'periodic'
parameters.lat_plotLattice = False
parameters.dia_Hamiltonian = (10,0,0,0,0,0)
parameters.dia_plotWf = False
sca_types = ('2to2_1',)
parameters.sca_types = sca_types
parameters.sca_broadening = 0.5
parameters.sca_saveVertex = True
parameters.sca_temperature = 8
system = openHamiltonian(parameters)
evalsPeriod = system.evals[1:]
system.computeRate()
gammaPeriod = system.rates[sca_types[0]]
print(system.Ns)

# Corbino geometry
r1 = 1.5
r2 = 6.5

offSiteList = []
Lx = int(2*r2)
Ly = Lx
for ix in range(Lx):
    for iy in range(Ly):
        # Center in 0
        x = ix-Lx//2
        y = iy-Ly//2
        d = x**2+y**2
        if d<r1**2 or d>r2**2:
            offSiteList.append((ix,iy))

parameters = importParameters()

parameters.lat_Lx = Lx
parameters.lat_Ly = Ly
parameters.lat_offSiteList = tuple(offSiteList)
parameters.lat_boundary = 'open'
parameters.lat_plotLattice = False

parameters.dia_Hamiltonian = (10,0,0,0,0,0)
parameters.dia_plotWf = False

sca_types = ('2to2_1',)
parameters.sca_types = sca_types
parameters.sca_broadening = 0.5
parameters.sca_saveVertex = True
parameters.sca_temperature = 8

#lattice = latticeClass(parameters)

system = openHamiltonian(parameters)
print(system.Ns)
evals = system.evals[1:]
system.computeRate()
gamma = system.rates[sca_types[0]]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
ax.scatter(
    evals,
    gamma,
    lw=0,
    alpha=0.7,
    color='orange',
    label="Disk $(r_1,r_2)=(%.1f,%.1f)$, %d sites"%(r1,r2,Lx*Ly-len(offSiteList)),
)
ax.scatter(
    evalsPeriod,
    gammaPeriod,
    lw=0,
    alpha=0.7,
    color='blue',
    label="Periodic $(L_x,L_y)=(%d,%d)$, %d sites"%(LxP,LyP,LxP*LyP),
)
ax.legend(loc='upper left',fontsize=20)
#ax.set_title("Disc of internal radius %.1f and external %.1f"%(r1,r2))
plt.show()


