""" 
Here we check the normalization of the awesome functions.
"""

import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.tools.inputUtils import importParameters
from wavespin.static.open import openHamiltonian
import matplotlib.pyplot as plt

parsLattices = {
    '24-diamond': [
        6,
        6,
        ( (0,0), (0,1), (0,4), (0,5), (1,0), (1,5), (4,0), (4,5), (5,0), (5,1), (5,4), (5,5) )
    ],
    '60-diamond': [
        10,
        10,
        ( (0,0), (0,1), (0,2), (0,3), (0,6), (0,7), (0,8), (0,9), (1,0), (1,1), (1,2), (1,7), (1,8), (1,9), (2,0), (2,1), (2,8), (2,9), (3,0), (3,9), (6,0), (6,9), (7,0), (7,1), (7,8), (7,9), (8,0), (8,1), (8,2), (8,7), (8,8), (8,9), (9,0), (9,1), (9,2), (9,3), (9,6), (9,7), (9,8), (9,9) )
    ],
    '60-diamond2': [
        10,
        10,
        (
            (0,0), (0,1), (0,2), (0,7), (0,8), (0,9),
            (1,0), (1,1), (1,8), (1,9),
            (2,0), (2,9),
            (7,0), (7,9),
            (8,0), (8,1), (8,8), (8,9),
            (9,0), (9,1), (9,2), (9,7), (9,8), (9,9)
        )
    ],
    '7x8-rectangle': [
        7,
        8,
        (),
    ],
    '8x6-rectangle': [
        8,
        6,
        (),
    ],
    '20x20-rectangle': [
        20,
        20,
        (),
    ],
    '10x10-rectangle': [
        10,
        10,
        (),
    ],
    '80-diamond': [
        7,      #not finished
        8,
        (),
    ],
    '144-diamond': [
        16,      #not finished
        16,
        (
            (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (0,15),
            (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,10), (1,11), (1,12), (1,13), (1,14), (1,15),
            (2,0), (2,1), (2,2), (2,3), (2,4), (2,11), (2,12), (2,13), (2,14), (2,15),
            (3,0), (3,1), (3,2), (3,3), (3,12), (3,13), (3,14), (3,15),
            (4,0), (4,1), (4,2), (4,13), (4,14), (4,15),
            (5,0), (5,1), (5,14), (5,15),
            (6,0), (6,15),
            (9,0), (9,15),
            (10,0), (10,1), (10,14), (10,15),
            (11,0), (11,1), (11,2), (11,13), (11,14), (11,15),
            (12,0), (12,1), (12,2), (12,3), (12,12), (12,13), (12,14), (12,15),
            (13,0), (13,1), (13,2), (13,3), (13,4), (13,11), (13,12), (13,13), (13,14), (13,15),
            (14,0), (14,1), (14,2), (14,3), (14,4), (14,5), (14,10), (14,11), (14,12), (14,13), (14,14), (14,15),
            (15,0), (15,1), (15,2), (15,3), (15,4), (15,5), (15,6), (15,9), (15,10), (15,11), (15,12), (15,13), (15,14), (15,15),
        )
    ],
}

lattices = (
    '10x10-rectangle',
    '20x20-rectangle'
)
parameters = importParameters()

J1 = 1
J2 = 0
h = 0
parameters.dia_Hamiltonian = (J1/2,J2/2,0,0,h,0)

results = {}
for lattice in lattices:
    parameters.lat_Lx = parsLattices[lattice][0]
    parameters.lat_Ly = parsLattices[lattice][1]
    parameters.lat_offSiteList = parsLattices[lattice][2]
    parameters.lat_boundary = 'open'
    #
    parameters.lat_plotLattice = False
    parameters.dia_plotWf = False
    mySystem = openHamiltonian(parameters)
    #
    phi = mySystem.Phi
    res = np.sum(phi**2,axis=0)
    results[lattice] = res

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
for lattice in lattices:
    res = results[lattice]
    N = res.shape[0]
    ax.scatter(np.arange(N),res,label=lattice)
ax.legend()
plt.show()

