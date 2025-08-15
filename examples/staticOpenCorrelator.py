import numpy as np
import sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from wavespin.static.open import *
from wavespin.tools.inputUtils import importOpenParametrs as importParameters
from wavespin.tools.pathFinder import *

""" Parameters and options """
parser = argparse.ArgumentParser(description="Static correlator calculation using Holstein-Primakoff formalism")
parser.add_argument("inputFile", help="Name of the file where computation parameters and options are stored")
parser.add_argument("-v","--verbose", help="Enable verbose output", action="store_true")
inputArguments = parser.parse_args()
verbose = inputArguments.verbose
parameters = importParameters(inputArguments.inputFile,{'verbose':verbose})
(correlatorType,
 fourierType,
 excludeZeromode,
 useExperimentalParameters,
 Lx,Ly,offSiteList,perturbationSite,
 includeList,plotSites,
 saveWf,plotWf,saveCorrelator,saveBonds,plotCorrelator,saveFig,
 ) = parameters.values()

""" Derived parameters """
indexesMap = mapSiteIndex(Lx,Ly,offSiteList)
Ns = len(indexesMap)
perturbationIndex = indexesMap.index(perturbationSite) #site_j[1] + site_j[0]*Ly
excludeList,magnonText = getExcludeList(includeList)

""" Plot circuit """
if plotSites:
    plotSitesGrid(Lx,Ly,offSiteList,perturbationSite,indexesMap)

""" Parameters of the Hamiltonian """
S = 0.5     #spin value
nP = 11
g_val = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
h_val = 15
g1_t_i,g2_t_i,d1_t_i,h_t_i = getHamiltonianParameters(Lx,Ly,nP,g_val,h_val)   #parameters of Hamiltonian which depend on time

""" Correlator parameters """
site0 = 0 if h_t_i[0,0,0]<0 else 1     #decide sublattice A and B of reference lattice site
fullTimeMeasure = 0.8     #measure time in ms
nTimes = 401        #time steps after ramp for the measurement
measureTimeList = np.linspace(0,fullTimeMeasure,nTimes)
nOmega = 2000

""" Bogoliubov transformation"""
U_, V_, evals = bogoliubovTransformation()

""" Computation of correlator"""
correlator = computeCorrelator()

""" Fourier transform """
args = (N_omega,U_,V_,perturbationIndex,offSiteList,indexesMap)
correlator_kw = fourierTransform[fourier_type](correlator,*args)

""" Figure """
title = 'Commutator '+correlator_type+', fourier: '+fourier_type+', '+txt_magnon
args_fn += (fourier_type,)
figname = 'Figures/' + fs.get_fn(*args_fn) + '.png'

plotCorrelator(
    correlator_kw,
    n_bins=100,
    fourier_type=fourier_type,
    title=title,
    figname=figname,
    showfig=True,
)






