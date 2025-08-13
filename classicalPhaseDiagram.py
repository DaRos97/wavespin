import numpy as np
import os,sys
from wavespin.classicalGroundState import classical_j2_energy as ce


J1 = 1
J2min = 0
J2max = 1
nJ2 = 5
Hmin = 0
Hmax = 3
nH = 5

phaseDiagramParameters = (J1,J2min,J2max,nJ2,Hmin,Hmax,nH)

en = ce.computeClassicalGroundState(phaseDiagramParameters,verbose=True,save=False)

ce.plotClassicalPhaseDiagram(en,phaseDiagramParameters,show=True)

ce.plotClassicalPhaseDiagramParameters(en,phaseDiagramParameters,show=True)
