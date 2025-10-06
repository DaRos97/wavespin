""" Code for computing numerically using RMOs the classical phase diagram of magnetic orders in the J1-J2 XY model plus staggered H.
Since there are no other magnetic orders then the canted-Neel and canted-Stripe, an analytical plot is actually better.
"""

import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.classicSpins.RMO import *


J1 = 1  # Positive

J2min = 0
J2max = 1
nJ2 = 21

Hmin = 0
Hmax = 3
nH = 21

phaseDiagramParameters = (J1,J2min,J2max,nJ2,Hmin,Hmax,nH)

en = computeClassicalGroundState(phaseDiagramParameters,verbose=True,save=False)

plotClassicalPhaseDiagram(en,phaseDiagramParameters,show=True)

plotClassicalPhaseDiagramParameters(en,phaseDiagramParameters,show=True)
