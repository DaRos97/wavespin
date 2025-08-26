import numpy as np
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.classicSpins.RMO import *


J1 = 1
J2min = 0
J2max = 1
nJ2 = 5
Hmin = 0
Hmax = 3
nH = 5

phaseDiagramParameters = (J1,J2min,J2max,nJ2,Hmin,Hmax,nH)

en = computeClassicalGroundState(phaseDiagramParameters,verbose=True,save=False)

plotClassicalPhaseDiagram(en,phaseDiagramParameters,show=True)

plotClassicalPhaseDiagramParameters(en,phaseDiagramParameters,show=True)
