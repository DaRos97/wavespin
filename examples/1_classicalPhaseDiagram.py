""" Code for computing numerically using RMOs the classical phase diagram of magnetic orders in the J1-J2 XY model plus staggered H.
Since there are no other magnetic orders then the canted-Neel and canted-Stripe, an analytical plot is actually better.
"""

import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.classicSpins.RMO import *
from wavespin.tools import pathFinder


saveData = True

g1 = 1  # Positive

g2min = 0
g2max = 1
ng2 = 21

hmin = 0
hmax = 3
nh = 21

dataDn = '  '
argsFn = ('classicalPhaseDiagram',g1,g2min,g2max,ng2,hmin,hmax,nh)
transformationFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npz')

phaseDiagramParameters = (J1,J2min,J2max,nJ2,Hmin,Hmax,nH)

en = computeClassicalGroundState(phaseDiagramParameters,verbose=True,save=saveData)

plotClassicalPhaseDiagram(en,phaseDiagramParameters,show=True)

plotClassicalPhaseDiagramParameters(en,phaseDiagramParameters,show=True)
