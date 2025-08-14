"""
Here we have the functions to compute the lowest energy configuration for the classical spin model varying J2 and H.
This is given by minimizing a function of 5 angles: orientation of two spins in the unit cell (3 angles),
rotation angle of translation in the 2 directions. This is equivalent to a Regular Magnetic Order(RMO)
construction considering only translations and the U(1) symmetry of the Hamiltonian.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution as D_E
from tqdm import tqdm
from wavespin.tools.pathFinder import getFilename, getHomeDirname

def classicalEnergyRMO(angles,*args):
    r""" Computes the classical energy of the considered RMO construction.

    Parameters
    ----------
    angles : 3-float tuple.
        Three angles of the RMO: $\theta$ sublattice A, $\phi$ sublattice B, $\phi$ of translation along $a_2$.
    *args: Hamiltonian parameters.
        j1,j2,h -> 1st nn, 2nd nn, magnetic field.

    Returns
    -------
    energy : float, classical energy.
    """
    thA,phB,ph2 = angles
    thB = np.pi-thA
    j1,j2,h = args
    ph1 = 0
    energy = (j1/2*np.sin(thA)*np.sin(thB)*(np.cos(phB)+np.cos(phB+ph2)+np.cos(phB-ph1)+np.cos(phB-ph1-ph2))
             + j2/2*(np.cos(ph1+ph2)+np.cos(ph2))*(np.sin(thA)**2+np.sin(thB)**2)
             + h*(np.cos(thB)-np.cos(thA))
             )
    return energy

def computeClassicalGroundState(phaseDiagramParameters,**kwargs):
    """ Run the minimization algorithm for the classical ground state phase diagram.

    Parameters
    ----------
    phaseDiagramParameters: tuple.
        J1, J2min, J2max, nJ2, Hmin, Hmax, nH
    **kwargs: keyword arguments.
        'verbose': bool, 'save': bool

    Returns
    -------
    en : (nJ2,nH,4)-array -> energy + 3 angles for each point of the phase diagram.
    """
    J1,J2min,J2max,nJ2,Hmin,Hmax,nH = phaseDiagramParameters
    listJ2 = np.linspace(J2min,J2max,nJ2)
    listH = np.linspace(Hmin,Hmax,nH)

    verbose = kwargs.get('verbose',False)
    save = kwargs.get('save',False)

    filenameArgs = ('energies_',) + phaseDiagramParameters
    dataFn = getFilename(*filenameArgs,dirname=getHomeDirname(str(Path.cwd()),'Data/classicalEnergies/'),extension='.npy')

    if not Path(dataFn).is_file():
        en = np.zeros((nJ2,nH,4))   #energy and 3 angles
        iterJ2 = tqdm(range(nJ2)) if verbose else range(nJ2)
        for ij2 in iterJ2:
            for ih in range(nH):
                args = (J1,listJ2[ij2],listH[ih])
                res = D_E(     classicalEnergyRMO,
                               bounds = [(0,np.pi),(-np.pi,np.pi),(-np.pi,np.pi)],
                               args=args,
                               #method='Nelder-Mead',
                               strategy='rand1exp',
                               tol=1e-8,
    #                           options={'disp':False}
                              )
                en[ij2,ih,0] = res.fun
                en[ij2,ih,1:] = res.x
        #Process data
        iterJ2 = tqdm(range(nJ2)) if verbose else range(nJ2)
        for ij2 in iterJ2:
            for ih in range(nH):
                if abs(en[ij2,ih,1])<1e-4:
                    en[ij2,ih,3] = np.nan
                    en[ij2,ih,2] = np.nan
                if abs(abs(en[ij2,ih,3])-np.pi)<1e-4:
                    en[ij2,ih,3] = abs(en[ij2,ih,3])
        if save:
            dataDn = Path(getHomeDirname(str(Path.cwd())),'Data/')
            if not dataDn.is_dir():
                print("Creating 'Data/' folder in home directory.")
                os.system('mkdir '+dataDn)
            dataDn = Path(getHomeDirname(str(Path.cwd())),'Data/classicalEnergies/')
            if not dataDn.is_dir():
                print("Creating 'classicalEnergies/' folder in 'Data/' directory.")
                os.system('mkdir '+dataDn)
            np.save(dataFn,en)
    else:
        en = np.load(dataFn)
    return en

def plotClassicalPhaseDiagram(en,phaseDiagramParameters,**kwargs):
    """ Plot the classical phase diagram.

    Parameters
    ----------
    en : (nJ2,nH,4)-array.
        energy + 3 angles for each point of the phase diagram.
    phaseDiagramParameters: tuple.
        J1, J2min, J2max, nJ2, Hmin, Hmax, nH. Has to be consistent with en.
    **kwargs: keyword arguments.
        'show': bool, 'save': bool
    """
    J1,J2min,J2max,nJ2,Hmin,Hmax,nH = phaseDiagramParameters
    listJ2 = np.linspace(J2min,J2max,nJ2)
    listH = np.linspace(Hmin,Hmax,nH)

    show = kwargs.get('show',False)
    save = kwargs.get('save',False)

    filenameArgs = ('phaseDiagram_',) + phaseDiagramParameters
    figureFn = getFilename(*filenameArgs,dirname=getHomeDirname(str(Path.cwd()),'Figures/classicalPhaseDiagram/'),extension='.png')

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    for ij2 in range(nJ2):
        for ih in range(nH):
            e,th,phb,ph2 = en[ij2,ih]
            if abs(th)<1e-3:
                color='k'
            elif abs(abs(phb)-np.pi)<1e-3 and abs(ph2)<1e-3:  #neel
                color='r'
            elif abs(abs(phb)-np.pi)<1e-3:#
                color='y'
            elif abs(phb)<1e-3:#
                color='orange'
            else:   #unknown
                color = 'b'
                print(phb)
            ax.scatter(listJ2[ij2],listH[ih],color=color,marker='o')

    # Missing the legend

    if save:
        figureDn = Path(getHomeDirname(str(Path.cwd())),'Figures/')
        if not figureDn.is_dir():
            print("Creating 'Figures/' folder in home directory.")
            os.system('mkdir '+figureDn)
        figureDn = Path(getHomeDirname(str(Path.cwd())),'Figures/classicalPhaseDiagram/')
        if not figureDn.is_dir():
            print("Creating 'classicalPhaseDiagram/' folder in 'Figures/' directory.")
            os.system('mkdir '+figureDn)
        fig.savefig(figureFn)
    if show:
        plt.show()
    plt.close()

def plotClassicalPhaseDiagramParameters(en,phaseDiagramParameters,**kwargs):
    """ Plot the classical phase diagram parameters.

    Parameters
    ----------
    en : (nJ2,nH,4)-array.
        energy + 3 angles for each point of the phase diagram.
    phaseDiagramParameters: tuple.
        J1, J2min, J2max, nJ2, Hmin, Hmax, nH. Has to be consistent with en.
    **kwargs: keyword arguments.
        'show': bool, 'save': bool
    """
    J1,J2min,J2max,nJ2,Hmin,Hmax,nH = phaseDiagramParameters
    listJ2 = np.linspace(J2min,J2max,nJ2)
    listH = np.linspace(Hmin,Hmax,nH)

    show = kwargs.get('show',False)
    save = kwargs.get('save',False)

    filenameArgs = ('parametersPhaseDiagram_',) + phaseDiagramParameters
    figureFn = getFilename(*filenameArgs,dirname=getHomeDirname(str(Path.cwd()),'Figures/classicalPhaseDiagram/'),extension='.png')

    fig = plt.figure(figsize=(12,12))
    X,Y = np.meshgrid(listJ2,listH)
    titles = ['energy',r'$\theta_A$',r'$\theta_B$',r'$\phi_B$',r'$\phi_2$']
    for i in range(4):
        ax = fig.add_subplot(2,2,i+1,projection='3d')
        ax.plot_surface(X,Y,en[:,:,i].T,cmap='plasma_r')
        ax.set_title(titles[i])

    # Missing legend

    if save:
        figureDn = Path(getHomeDirname(str(Path.cwd())),'Figures/')
        if not figureDn.is_dir():
            print("Creating 'Figures/' folder in home directory.")
            os.system('mkdir '+figureDn)
        figureDn = Path(getHomeDirname(str(Path.cwd())),'Figures/classicalPhaseDiagram/')
        if not figureDn.is_dir():
            print("Creating 'classicalPhaseDiagram/' folder in 'Figures/' directory.")
            os.system('mkdir '+figureDn)
        fig.savefig(figureFn)
    if show:
        plt.show()
    plt.close()




