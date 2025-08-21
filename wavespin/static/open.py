""" Functions used for the open boundary conditions.
"""

import os
import scipy
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wavespin.tools import pathFinder as pf
from wavespin.static import periodic as pe
from scipy.fft import fftfreq, fftshift, fft, fft2, dstn, dctn

def mapSiteIndex(Lx,Ly,offSiteList):
    """ Here we define a map: from an index between 0 and Ns-1 to (ix,iy) between 0 and Lx/y-1.
    To each index in the actual used qubits assign the corresponding ix,iy.

    Parameters
    ----------
    Lx,Ly : int, rectangular linear sizes.
    offSiteList : list of 2-tuple.
        Coordinates of sites which are turned off.

    Returns
    -------
    indexesMap : list of 2-tuple.
        Coordinates of sites which are been considered, in order of their index.
    """
    indexesMap = []
    for ix in range(Lx):
        for iy in range(Ly):
            if (ix,iy) not in offSiteList:
                indexesMap.append((ix,iy))
    return indexesMap

def getMagnonText(includeList):     #not used yet
    """ Get magnon text from list of included terms.

    Parameters
    ----------

    Returns
    -------
    """
    if includeList ==  [2,4,6,8]:
        magnonText = 'allMagnon'
    else:
        magnonText = ''
        for i,term in enumerate(includeList):
            magnonText += str(int(term/2))
            if i!=len(includeList)-1:
                magnonText += ','
        magnonText += 'Magnon'
    return magnonText

def plotSitesGrid(Lx,Ly,offSiteList,perturbationSite,indexesMap):
    """ Here we plot the grid structure to see which sites are considered in the calculation.
    """
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    for ix in range(Lx):
        for iy in range(Ly):
            if (ix,iy) in offSiteList:
                ax.scatter(ix,iy,c='r',marker='x',s=80,zorder=2)
            else:
                ax.scatter(ix,iy,c='b',marker='o',s=80,zorder=2)
                ax.text(ix+0.05,iy+0.15,str(indexesMap.index((ix,iy))),size=20)
            if ix+1<Lx:
                if (ix,iy) in offSiteList or (ix+1,iy) in offSiteList:
                    ax.plot([ix,ix+1],[iy,iy],c='y',ls='--',lw=0.5,zorder=-1)
                else:
                    ax.plot([ix,ix+1],[iy,iy],c='darkgreen',ls='-',lw=2,zorder=-1)
            if iy+1<Ly:
                if (ix,iy) in offSiteList or (ix,iy+1) in offSiteList:
                    ax.plot([ix,ix],[iy,iy+1],c='y',ls='--',lw=0.5,zorder=-1)
                else:
                    ax.plot([ix,ix],[iy,iy+1],c='darkgreen',lw=2,zorder=-1)
    ax.scatter(perturbationSite[0],perturbationSite[1],c='w',edgecolor='m',lw=2,marker='o',s=200,zorder=1)
    ax.set_aspect('equal')
    ax.set_xlabel('x',size=30)
    ax.set_ylabel('y',size=30)
    fig.tight_layout()
    plt.show()

def getHamiltonianParameters(Lx,Ly,nP,gFinal,hInitial):
    """ Compute Hamiltonian parameters for real space.
    Each terms is a Lx*Ly,Lx*Ly matrix.
    In this way on-site terms like the magnetic field are diagonal,
    while first or second nearest neighbor ones are not.

    Parameters
    ----------
    Lx,Ly : int, rectangular linear sizes.
    nP : int, number of steps from initial to final state.
    gFinal, hInitial : floats

    Returns
    -------
    List of nP,Lx*Ly,Lx*Ly matrices for each Hamiltonian term.
    """
    Ns = Lx*Ly
    g1In = np.zeros((Ns,Ns))
    g1Fin = np.zeros((Ns,Ns))
    for ix in range(Lx):
        for iy in range(Ly):
            ind = ix*Ly + iy
            ind_plus_y = ind+1
            if ind_plus_y//Ly==ind//Ly:
                g1Fin[ind,ind_plus_y] = g1Fin[ind_plus_y,ind] = gFinal
            ind_plus_x = ind+Ly
            if ind_plus_x<Lx*Ly:
                g1Fin[ind,ind_plus_x] = g1Fin[ind_plus_x,ind] = gFinal
    g2In = np.zeros((Ns,Ns))
    g2Fin = np.zeros((Ns,Ns))
    d1In = np.zeros((Ns,Ns))
    d1Fin = np.zeros((Ns,Ns))
    hIn = np.zeros((Ns,Ns))
    for ix in range(Lx):
        for iy in range(Ly):
            hIn[iy+ix*Ly,iy+ix*Ly] = -(-1)**(ix+iy) * hInitial
    hFin = np.zeros((Ns,Ns))
    t_values = np.linspace(0,1,nP).reshape(nP,1,1)
    g1_t_i = (1-t_values)*g1In + t_values*g1Fin
    g2_t_i = (1-t_values)*g2In + t_values*g2Fin
    d1_t_i = (1-t_values)*d1In + t_values*d1Fin
    h_t_i = (1-t_values)*hIn + t_values*hFin
    return g1_t_i,g2_t_i,d1_t_i,h_t_i

def computeHamiltonianRs(*parameters):
    """
    Compute the real space Hamiltonian -> (2Ns x 2Ns).
    Conventions for the real space wavefunction and parameters are in the notes.
    SECOND NEAREST-NEIGHBOR NOT IMPLEMENTED.

    Parameters
    ----------
    *args

    Returns
    -------
    ham : 2Ns,2Ns matrix of real space Hamiltonian.
    """
    S,Lx,Ly,h_i,ts,theta,phi,J_i,D_i,offSiteList = parameters
    Ns = Lx*Ly
    #
    p_zz = pe.computePs(0,0,ts,J_i,D_i,offSiteList,Lx,Ly)
    p_xx = pe.computePs(1,1,ts,J_i,D_i,offSiteList,Lx,Ly)
    p_yy = pe.computePs(2,2,ts,J_i,D_i,offSiteList,Lx,Ly)
    p_xy = pe.computePs(1,2,ts,J_i,D_i,offSiteList,Lx,Ly)
    fac0 = 1#2      #Need to change this in notes -> counting of sites from 2 to 1 sites per UC
    fac1 = 1#2
    fac2 = 2#4
    ham = np.zeros((2*Ns,2*Ns),dtype=complex)
    #p_zz sums over nn but is on-site -> problem when geometry is not square
    #diagonal
    ham[:Ns,:Ns] = abs(h_i)/fac0*np.cos(theta) - S/fac1*np.diag(np.sum(p_zz[0],axis=0))
    ham[Ns:,Ns:] = abs(h_i)/fac0*np.cos(theta) - S/fac1*np.diag(np.sum(p_zz[0],axis=0))
    #off_diag 1 - nn
    off_diag_1_nn = S/fac2*(p_xx[0]+p_yy[0])
    ham[:Ns,:Ns] += off_diag_1_nn
    ham[Ns:,Ns:] += off_diag_1_nn
    #off_diag 2 - nn
    off_diag_2_nn = S/fac2*(p_xx[0]-p_yy[0]+2*1j*p_xy[0])
    ham[:Ns,Ns:] += off_diag_2_nn
    ham[Ns:,:Ns] += off_diag_2_nn.T.conj().T
    #Remove offSiteList
    indexesToRemove = []
    for offTerm in offSiteList:
        indexesToRemove.append(offTerm[0]*Ly + offTerm[1])
        indexesToRemove.append(offTerm[0]*Ly + offTerm[1] + Lx*Ly)
    ham = np.delete(ham,indexesToRemove,axis=0)
    ham = np.delete(ham,indexesToRemove,axis=1)
    return ham

def bogoliubovTransformation(Lx,Ly,Ns,nP,gFinal,hInitial,S,offSiteList,**kwargs):
    """ Compute the Bogoliubov transformation for the real-space Hamiltonian.

    Parameters
    ----------
    Lx,Ly : int, rectangular linear sizes.
    Ns : int, number of sites.
    nP : int, number of steps from initial to final state.
    gFinal, hInitial : floats
    S : float, spin size.
    offSiteList : list of 2-tuple.
        Coordinates of sites which are turned off.
    **kwargs

    Returns
    -------
    U_, V_, evals : bogoliubov transformation matrices U and V and eigenvalues.
    """
    g1_t_i,g2_t_i,d1_t_i,h_t_i = getHamiltonianParameters(Lx,Ly,nP,gFinal,hInitial)   #parameters of Hamiltonian which depend on time
    saveWf = kwargs.get('saveWf',False)
    excludeZeroMode = kwargs.get('excludeZeroMode',False)
    verbose = kwargs.get('verbose',False)

    argsFn = ('bogoliubov_rs',Lx,Ly,Ns)
    dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
    transformationFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npz')
    if not Path(transformationFn).is_file():
        U_ = np.zeros((nP,Ns,Ns),dtype=complex)
        V_ = np.zeros((nP,Ns,Ns),dtype=complex)
        evals = np.zeros((nP,Ns))
        iterBog = tqdm(range(nP),desc="Computing Bogoliubov transformation") if verbose else range(nP)
        for i_sr in iterBog:
            # Hamiltonian parameters
            J_i = (g1_t_i[i_sr,:,:],g2_t_i)  #site-dependent hopping
            D_i = (d1_t_i[i_sr,:,:],np.zeros((Ns,Ns)))
            h_i = h_t_i[i_sr,:,:]
            theta,phi = pe.quantizationAxis(S,J_i,D_i,h_i)
            ts = pe.computeTs(theta,phi)       #All t-parameters for A and B sublattice
            parameters = (S,Lx,Ly,h_i,ts,theta,phi,J_i,D_i,offSiteList)
            #
            hamiltonian = computeHamiltonianRs(*parameters)
            if np.max(np.absolute(hamiltonian-hamiltonian.conj()))>1e-5:
                print("Hamiltonian is not real! Procedure might be wrong")
            # Para-diagonalization, see notes (appendix) for details
            A = hamiltonian[:Ns,:Ns]
            B = hamiltonian[:Ns,Ns:]
            try:
                K = scipy.linalg.cholesky(A-B)
            except:
                K = scipy.linalg.cholesky(A-B+np.identity(Ns)*1e-5)
            lam2,chi_ = scipy.linalg.eigh(K@(A+B)@K.T.conj())
            evals[i_sr] = np.sqrt(lam2)         #dispersion -> positive
            #
            chi = chi_ / evals[i_sr]**(1/2)     #normalized eigenvectors: divide each column of chi_ by the corresponding eigenvalue -> of course for the gapless mode there is a problem here
            phi_ = K.T.conj()@chi
            psi_ = (A+B)@phi_/evals[i_sr]       # Problem also here
            U_[i_sr] = 1/2*(phi_+psi_)
            V_[i_sr] = 1/2*(phi_-psi_)
        # Save
        if saveWf:
            if not Path(dataDn).is_dir():
                print("Creating 'Data/' folder in home directory.")
                os.system('mkdir '+dataDn)
            np.savez(transformationFn,awesomeU=U_,awesomeV=V_,evals=evals)
    else:
        if verbose:
            print("Loading Bogoliubov transformation from file")
        U_ = np.load(transformationFn)['awesomeU']
        V_ = np.load(transformationFn)['awesomeV']
        evals = np.load(transformationFn)['evals']

    if excludeZeroMode:       #Put to 0 the eigenstate corresponding to the zero energy mode -> a bit far fetched
        for i_sr in range(2,nP):
            U_[i_sr,:,0] *= 0
            V_[i_sr,:,0] *= 0
    return U_, V_, evals

def extendFunction(func,Lx,Ly,offSiteList,indexesMap):
    """
    Here is for plotting:
        We take an Ns-sized vector and map it to Lx,Ly grid putting nans in offSites.
    """
    fullFunc = np.zeros(Lx*Ly)
    for ix in range(Lx):
        for iy in range(Ly):
            if (ix,iy) in offSiteList:
                fullFunc[ix*Ly+iy] = np.nan
            else:
                fullFunc[ix*Ly+iy] = func[indexesMap.index((ix,iy))]
    return fullFunc.reshape(Lx,Ly)

def computeCorrelator(Lx,Ly,Ns,nP,measureTimeList,gFinal,hInitial,S,U_,V_,evals,offSiteList,site0,perturbationSite,includeList,**kwargs):
    """ Compute real space correlator.
    """
    excludeZeroMode = kwargs.get('excludeZeroMode',False)
    correlatorType = kwargs.get('correlatorType','zz')
    saveCorrelator = kwargs.get('saveCorrelator',False)
    verbose = kwargs.get('verbose',False)

    indexesMap = mapSiteIndex(Lx,Ly,offSiteList)
    perturbationIndex = indexesMap.index(perturbationSite) #site_j[1] + site_j[0]*Ly
    g1_t_i,g2_t_i,d1_t_i,h_t_i = getHamiltonianParameters(Lx,Ly,nP,gFinal,hInitial)   #parameters of Hamiltonian which depend on time
    Ntimes = len(measureTimeList)
    txtZeroEnergy = 'without0energy' if excludeZeroMode else 'with0energy'

    argsFn = ('correlator_rs',Lx,Ly,Ns,txtZeroEnergy)
    dataDn = pf.getHomeDirname(str(Path.cwd()),'Data/')
    correlatorFn = pf.getFilename(*argsFn,dirname=dataDn,extension='.npy')
    if not Path(correlatorFn).is_file():
        correlator = np.zeros((nP,Lx,Ly,Ntimes),dtype=complex)
        iterCorr = tqdm(range(nP),desc="Computing correlator") if verbose else range(nP)
        for i_sr in iterCorr:
            # Hamiltonian parameters
            J_i = (g1_t_i[i_sr,:,:],np.zeros((Ns,Ns)))  #site-dependent hopping
            D_i = (d1_t_i[i_sr,:,:],np.zeros((Ns,Ns)))
            h_i = h_t_i[i_sr,:,:]
            theta,phi = pe.quantizationAxis(S,J_i,D_i,h_i)
            ts = pe.computeTs(theta,phi)       #All t-parameters for A and B sublattice
            #
            U = np.zeros((2*Ns,2*Ns),dtype=complex)
            U[:Ns,:Ns] = U_[i_sr]
            U[:Ns,Ns:] = V_[i_sr]
            U[Ns:,:Ns] = V_[i_sr]
            U[Ns:,Ns:] = U_[i_sr]
            #Correlator -> can make this faster, we actually only need U_ and V_
            exp_e = np.exp(-1j*2*np.pi*measureTimeList[:,None]*evals[i_sr,None,:])
            A = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[Ns:,Ns:],optimize=True)
            B = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[:Ns,Ns:],optimize=True)
            G = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[:Ns,Ns:],optimize=True)
            H = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[Ns:,Ns:],optimize=True)
            #
            for ind_i in range(Lx*Ly):
                if (ind_i//Ly,ind_i%Ly) in offSiteList:
                    continue
                correlator[i_sr,ind_i//Ly,ind_i%Ly] = dicCorrelators[correlatorType](
                    S,Lx,Ly,
                    ts,
                    site0,
                    indexesMap.index((ind_i//Ly,ind_i%Ly)),
                    perturbationIndex,
                    offSiteList,
                    indexesMap,
                    A,B,G,H,
                    includeList
                )
        if saveCorrelator:
            if not Path(dataDn).is_dir():
                print("Creating 'Data/' folder in home directory.")
                os.system('mkdir '+dataDn)
            np.save(correlatorFn,correlator)
    else:
        print("Loading real-space correlator from file")
        correlator = np.load(correlatorFn)
    return correlator

def correlator_zz(S,Lx,Ly,ts,site0,measurementIndex,perturbationIndex,offSiteList,indexesMap,A,B,G,H,includeList=[2,4,6,8]):
    """
    Compute real space <[Z_i(t),Z_j(0)]> correlator.
    A,B,G and H are Ns,Ns.
    The correlator is Lx,Ly.
    """
    ts_i = ts[(site0+indexesMap[measurementIndex][0]+indexesMap[measurementIndex][1])%2]
    ts_j = ts[(site0+indexesMap[perturbationIndex][0]+indexesMap[perturbationIndex][1])%2]
    ts_list = [ts_i, ts_j]
    ZZ = np.zeros(A[0,0].shape,dtype=complex)
    for ops in ['XX','ZZ']:
        original_op = 'ZZ'
        list_terms = compute_combinations(ops,[measurementIndex,perturbationIndex],'t0',S)
        coeff_t = compute_coeff_t(ops,original_op,ts_list)
        for t in list_terms:
            ZZ += coeff_t * t[0] * compute_contraction(t[1],t[2],t[3],A,B,G,H,includeList)
    return 2*1j*np.imag(ZZ)

def correlator_ze(S,Lx,Ly,ts,site0,ind_i,perturbationIndex,offSiteList,A,B,G,H,includeList=[2,4,6,8]):
    """
    Compute real space <[Z_i(t),E_j(0)]> correlator.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+perturbationIndex//Ly+perturbationIndex%Ly)%2]
    ZE = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_j = get_nn(perturbationIndex,Lx,Ly)
    for i in offSiteList:
        if i[0]*Ly+i[1] in ind_nn_j:
            ind_nn_j.remove(i[0]*Ly+i[1])
    for ind_s in ind_nn_j:
        ts_s = ts[(site0+ind_s//Ly+ind_s%Ly)%2]
        ts_list = [ts_i, ts_j, ts_s]
        for ops in ['ZXX','XZX','XXZ','ZZZ','ZYY']:
            original_op = 'ZXX'
            if ops=='ZYY':
                original_op = 'ZYY'
            list_terms = compute_combinations(ops,[ind_i,perturbationIndex,ind_s],'t00',S)
            coeff_t = compute_coeff_t(ops,original_op,ts_list)
            for t in list_terms:
                ZE += coeff_t * t[0] * compute_contraction(t[1],t[2],t[3],A,B,G,H,includeList)
    return 2*1j/len(ind_nn_j)*np.imag(ZE)

def correlator_ez(S,Lx,Ly,ts,site0,ind_i,perturbationIndex,offSiteList,A,B,G,H,includeList=[2,4,6,8]):
    """
    Compute real space <[E_i(t),Z_j(0)]> correlator.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+perturbationIndex//Ly+perturbationIndex%Ly)%2]
    EZ = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_i = get_nn(ind_i,Lx,Ly)
    for i in offSiteList:
        if i[0]*Ly+i[1] in ind_nn_i:
            ind_nn_i.remove(i[0]*Ly+i[1])
    for ind_r in ind_nn_i:
        ts_r = ts[(site0+ind_r//Ly+ind_r%Ly)%2]
        ts_list = [ts_i, ts_r, ts_j]
        for ops in ['XXZ','XZX','ZXX','ZZZ','YYZ']:
            original_op = 'XXZ'
            if ops=='YYZ':
                original_op = 'YYZ'
            list_terms = compute_combinations(ops,[ind_i,ind_r,perturbationIndex],'tt0',S)
            coeff_t = compute_coeff_t(ops,original_op,ts_list)
            for t in list_terms:
                EZ += coeff_t * t[0] * compute_contraction(t[1],t[2],t[3],A,B,G,H,includeList)
    return 2*1j/len(ind_nn_i)*np.imag(EZ)

def correlator_ee(S,Lx,Ly,ts,site0,ind_i,perturbationIndex,offSiteList,A,B,G,H,includeList=[2,4,6,8]):
    """
    Compute real space <[E_i(t),E_j(0)]> correlator.
    Site j is where the E perturbation is applied -> we assume it is
    somewhere in the middle which has all 4 nearest neighbors and average
    over them.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+perturbationIndex//Ly+perturbationIndex%Ly)%2]
    EE = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_i = get_nn(ind_i,Lx,Ly)
    ind_nn_j = [perturbationIndex+Ly,]
#    ind_nn_j = get_nn(perturbationIndex,Lx,Ly)
    for i in offSiteList:
        if i[0]*Ly+i[1] in ind_nn_i:
            ind_nn_i.remove(i[0]*Ly+i[1])
        if i[0]*Ly+i[1] in ind_nn_j:
            ind_nn_j.remove(i[0]*Ly+i[1])
    for ind_r in ind_nn_i:
        ts_r = ts[(site0+ind_r//Ly+ind_r%Ly)%2]
        for ind_s in ind_nn_j:
            ts_s = ts[(site0+ind_s//Ly+ind_s%Ly)%2]
            ts_list = [ts_i,ts_r,ts_j,ts_s]
            for ops in ['XXXX','ZZZZ','XXZZ','ZZXX','XZXZ','ZXZX','ZXXZ','XZZX','XXYY','ZZYY','YYXX','YYZZ','YYYY']:
                original_op = 'XXXX'
                if ops in ['XXYY','ZZYY']:
                    original_op = 'XXYY'
                if ops in ['YYXX','YYZZ']:
                    original_op = 'YYXX'
                if ops == 'YYYY':
                    original_op = 'YYYY'
                list_terms = compute_combinations(ops,[ind_i,ind_r,perturbationIndex,ind_s],'tt00',S)
                coeff_t = compute_coeff_t(ops,original_op,ts_list)
                for t in list_terms:
                    contraction = compute_contraction(t[1],t[2],t[3],A,B,G,H,includeList)
                    EE += coeff_t * t[0] * contraction
    return 2*1j/len(ind_nn_i)/len(ind_nn_j)*np.imag(EE)

def correlator_xx(S,Lx,Ly,ts,site0,ind_i,perturbationIndex,offSiteList,A,B,G,H,includeList=[2,4,6,8]):
    """
    Compute real space <[X_i(t),X_j(0)]> correlator.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+perturbationIndex//Ly+perturbationIndex%Ly)%2]
    ts_list = [ts_i, ts_j]
    XX = np.zeros(A[0,0].shape,dtype=complex)
    for ops in ['XX','ZZ']:
        original_op = 'XX'
        list_terms = compute_combinations(ops,[ind_i,perturbationIndex],'t0',S)
        coeff_t = compute_coeff_t(ops,original_op,ts_list)
        for t in list_terms:
            XX += coeff_t * t[0] * compute_contraction(t[1],t[2],t[3],A,B,G,H,includeList)
    return 2*1j*np.imag(XX)

def correlator_jj(S,Lx,Ly,ts,site0,ind_i,perturbationIndex,offSiteList,A,B,G,H,includeList=[2,4,6,8]):
    """
    Compute real space <[J_i(t),J_j(0)]> correlator.
    Site j is where the J perturbation is applied -> we assume it is
    somewhere in the middle which has all 4 nearest neighbors and average
    over them.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+perturbationIndex//Ly+perturbationIndex%Ly)%2]
    JJ = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_i = get_nn(ind_i,Lx,Ly)
    ind_nn_j = [perturbationIndex+Ly,]
#    ind_nn_j = get_nn(perturbationIndex,Lx,Ly)
    for i in offSiteList:
        if i[0]*Ly+i[1] in ind_nn_i:
            ind_nn_i.remove(i[0]*Ly+i[1])
        if i[0]*Ly+i[1] in ind_nn_j:
            ind_nn_j.remove(i[0]*Ly+i[1])
    term_list = ['XYXY','ZYZY','XYYX','ZYYZ','YXXY','YZZY','YXYX','YZYZ']
    for ind_r in ind_nn_i:
        ts_r = ts[(site0+ind_r//Ly+ind_r%Ly)%2]
        for ind_s in ind_nn_j:
            ts_s = ts[(site0+ind_s//Ly+ind_s%Ly)%2]
            ts_list = [ts_i,ts_r,ts_j,ts_s]
            for i,ops in enumerate(term_list):
                original_op = ops if i%2==0 else term_list[i-1]
                list_terms = compute_combinations(ops,[ind_i,ind_r,perturbationIndex,ind_s],'tt00',S)
                coeff_t = compute_coeff_t(ops,original_op,ts_list)
                if i in [2,3,4,5]:
                    coeff_t *= -1
                for t in list_terms:
                    contraction = compute_contraction(t[1],t[2],t[3],A,B,G,H,includeList)
                    JJ += coeff_t * t[0] * contraction
    return 2*1j/len(ind_nn_i)/len(ind_nn_j)*np.imag(JJ)

def generate_pairings(elements):
    """
    Here we get all the possible permutation lists for the Wick contraction -> perfect matchings.
    """
    if len(elements) == 0:
        return [[]]
    pairings = []
    a = elements[0]
    for i in range(1, len(elements)):
        b = elements[i]
        rest = elements[1:i] + elements[i+1:]
        for rest_pairing in generate_pairings(rest):
            pairings.append([(a, b)] + rest_pairing)
    return pairings

permutation_lists = {}
for i in range(2,16,2):
    permutation_lists[i] = generate_pairings(list(range(i)))

def compute_contraction(op_list,ind_list,time_list,A,B,G,H,includeList=[2,4,6,8]):
    """
    Here we compute the contractions using Wick decomposition of the single operator list `op_list`, with the given sites and times.
    First we compute all the 2-operator terms.
    len(op) = 2 -> 1 term
    len(op) = 4 -> 3 terms
    len(op) = 6 -> 15 terms
    len(op) = 8 -> 105 terms
    etc..
    """
    ops_dic = {'aa':B,'bb':A,'ab':H,'ba':G}
    if len(op_list) in includeList:
        perm_list = permutation_lists[len(op_list)]
        result = 0
        for i in range(len(perm_list)):
            temp = 1
            for j in range(len(perm_list[i])):
                op_ = op_list[perm_list[i][j][0]]+op_list[perm_list[i][j][1]]
                ind_ =  [ ind_list[perm_list[i][j][0]], ind_list[perm_list[i][j][1]] ]
                time_ = time_list[perm_list[i][j][0]]!=time_list[perm_list[i][j][1]]
                op = ops_dic[op_][ind_[0],ind_[1]]
                temp *= op if time_ else op[0]
            result += temp
        return result
    else:
        return 0

def compute_coeff_t(op_t,op_o,ts_list):
    """
    Compute the product of t-coefficients given the original operator `op_o` and the transformed one `op_t`
    """
    ind_t_dic = {'Z':0, 'X':1, 'Y':2}
    ind_o_dic = {'X':0, 'Y':1, 'Z':2}
    coeff = 1
    for i in range(len(op_t)):
        ind_transformed = ind_t_dic[op_t[i]]
        ind_original = ind_o_dic[op_o[i]]
        coeff *= ts_list[i][ind_transformed][ind_original]
    return coeff

def compute_combinations(op_list,ind_list,time_list,S):
    """
    Here we compute symbolically all terms of the HP expansion of the operator list `op`.
    a -> a, a^dag -> b.
    Need also to keep sites information
    X_j = sqrt(S/2)(a_j+b_j)
    Y_j = -i*sqrt(S/2)(a_j-b_j)
    Z_j = S-b_ja_j
    return a list of 3-tuple, with first element a coefficient (pm (i) S**n) second element an operator list ('abba..') and third element a list of sites ([ind_i,ind_j,..]) of same length as the operator string
    """
    op_dic = {'X':[np.sqrt(S/2),'a'], 'Y':'(a-b)', 'Z':'S-ba'}
    coeff_dic = {'X':np.sqrt(S/2), 'Y':1j*np.sqrt(S/2), 'Z':1}
    terms = []
    coeff = 1
    for i in range(len(op_list)):
        if op_list[i]=='X':
            terms.append([ [np.sqrt(S/2),'a',[ind_list[i]], time_list[i]] , [np.sqrt(S/2),'b',[ind_list[i]], time_list[i] ]])
        if op_list[i]=='Y':
            terms.append([ [-1j*np.sqrt(S/2),'a',[ind_list[i]], time_list[i]] , [1j*np.sqrt(S/2),'b',[ind_list[i] ], time_list[i] ]])
        if op_list[i]=='Z':
            terms.append([ [S,'',[],''] , [-1,'ba',[ind_list[i],ind_list[i]], time_list[i]+time_list[i] ]])
    for i in range(len(op_list)-1): #n-1 multiplications
        new_terms = []
        mult = []
        for j in range(len(terms[0])):
            for l in range(len(terms[1])):
                mult.append( [terms[0][j][0]*terms[1][l][0], terms[0][j][1]+terms[1][l][1], terms[0][j][2]+terms[1][l][2], terms[0][j][3]+terms[1][l][3] ]  )
        new_terms.append(mult)
        #remaining part
        for j in range(2,len(terms)):
            new_terms.append(terms[j])
        terms = list(new_terms)
    return terms[0]

def get_nn(ind,Lx,Ly):
    """
    Compute indices of nearest neighbors of site ind.
    """
    result= []
    if ind+Ly<=(Lx*Ly-1):        #right neighbor
        result.append(ind+Ly)
    if ind-Ly>=0:
        result.append(ind-Ly)   #left neighbor
    if (ind+1)//Ly==ind//Ly:    #up neighbor
        result.append(ind+1)
    if (ind-1)//Ly==ind//Ly:    #bottom neighbor
        result.append(ind-1)
    return result

dicCorrelators = {'zz':correlator_zz,'xx':correlator_xx,'ze':correlator_ze,'ez':correlator_ez, 'ee':correlator_ee, 'jj':correlator_jj}

def fourier_fft(correlator_xt,*args):
    """
    Compute the standard 2D Fourier transform.
    In time we always use fft.
    """
    n_sr, Lx, Ly, Nt = correlator_xt.shape
    N_omega = args[0]
    correlator_kw = np.zeros((n_sr,Lx,Ly,N_omega),dtype=complex)
    for i_sr in range(n_sr):
        temp = np.zeros((Lx,Ly,Nt),dtype=complex)
        for it in range(Nt):
            temp[:,:,it] = fftshift(fft2(correlator_xt[i_sr,:,:,it],norm='ortho'))
        for ix in range(Lx):
            for iy in range(Ly):
                correlator_kw[i_sr,ix,iy] = fftshift(fft(temp[ix,iy],n=N_omega))
    # Momenta
    return correlator_kw

def fourier_dst(correlator_xt,*args):
    """
    Compute the Discrete Sin Transform since we have open BC.
    In time we always use fft.
    correlator_xt has shape (n_sr, Lx, Ly, Nt) with n_sr number of stop ratios (10) and Nt number of time steps (400) in the measurement
    """
    type_dst = 1
    N_omega = args[0]
    n_sr, Lx, Ly, Nt = correlator_xt.shape
    correlator_kw = np.zeros((n_sr,Lx,Ly,N_omega),dtype=complex)
    temp = np.zeros((n_sr,Lx,Ly,Nt),dtype=complex)
    for i_sr in range(n_sr):
        for it in range(Nt):
            temp[i_sr,:,:,it] = dstn(correlator_xt[i_sr,:,:,it], type=type_dst, norm='ortho')
        for ix in range(Lx):
            for iy in range(Ly):
                correlator_kw[i_sr,ix,iy] = fftshift(fft(temp[i_sr,ix,iy],n=N_omega))
    return correlator_kw

def fourier_dct(correlator_xt,*args):
    """
    Compute the Discrete Sin Transform since we have open BC.
    In time we always use fft.
    correlator_xt has shape (n_sr, Lx, Ly, Nt) with n_sr number of stop ratios (10) and Nt number of time steps (400) in the measurement
    """
    type_dct = 2
    N_omega = args[0]
    n_sr, Lx, Ly, Nt = correlator_xt.shape
    correlator_kw = np.zeros((n_sr,Lx,Ly,N_omega),dtype=complex)
    temp = np.zeros((n_sr,Lx,Ly,Nt),dtype=complex)
    for i_sr in range(n_sr):
        for it in range(Nt):
            temp[i_sr,:,:,it] = dctn(correlator_xt[i_sr,:,:,it], type=type_dct, norm='ortho')
        for ix in range(Lx):
            for iy in range(Ly):
                correlator_kw[i_sr,ix,iy] = fftshift(fft(temp[i_sr,ix,iy],n=N_omega))
    return correlator_kw

def fourier_dat(correlator_xt,*args):
    """
    Compute the Discrete Amazing Transform with the Bogoliubov functions.
    In time we always use fft.
    correlator_xt has shape (n_sr, Lx, Ly, Nt) with n_sr number of stop ratios (10) and Nt number of time steps (401) in the measurement
    U_ and V_ are (n_sr,Ns,Ns) matrices -> (x,m)
    """
    N_omega, U_, V_, perturbationSite, offSiteList = args
    n_sr, Lx, Ly, Ntimes = correlator_xt.shape
    indexesMap = mapSiteIndex(Lx,Ly,offSiteList)
    perturbationIndex = indexesMap.index(perturbationSite) #site_j[1] + site_j[0]*Ly
    correlator_xt = correlator_xt.reshape(n_sr,Lx*Ly,Ntimes)
    if 0:    #remove indeces for non-rectangular shape
        listRemovedInds = []
        for site in offSiteList:
            listRemovedInds.append(site[0]*Ly+site[1])
        correlator_xt = np.delete(correlator_xt,listRemovedInds,axis=1)     # n_sr,Ns,Ntimes
        Ns = correlator_xt.shape[1]
    else:
        Ns = Lx*Ly
    correlator_kw = np.zeros((n_sr,Lx,Ly,N_omega),dtype=complex)
    temp = np.zeros((n_sr,Lx,Ly,Ntimes),dtype=complex)
    for i_sr in range(n_sr):
        A_ik = np.real(U_[i_sr] - V_[i_sr])
        B_ik = np.real(U_[i_sr] + V_[i_sr])
        Bj_k = B_ik[perturbationIndex]
        phi_ik = A_ik #/ Bj_k[None,:]
        for i in range(Ns):
            if len(indexesMap)==0:
                ix,iy = i//Ly, i%Ly
            else:
                ix,iy = indexesMap[i]
            phi_ik[i,:] *= 2/np.pi*(-1)**(ix+iy+1)
        for k in range(Ns):
            kx,ky = get_momentum_Bogoliubov3(phi_ik[:,k].reshape(Lx,Ly))
            temp[i_sr,kx,ky] = np.sum(phi_ik[:,k,None]*correlator_xt[i_sr,:,:],axis=0)
            correlator_kw[i_sr,kx,ky] = fftshift(fft(temp[i_sr,kx,ky],n=N_omega))
    return correlator_kw

fourierTransform = {'fft':fourier_fft, 'dct':fourier_dct, 'dst':fourier_dst, 'dat':fourier_dat}

def get_momentum_Bogoliubov(f_in,type_dct=2):
    """
    Compute the peak of the dctn of input function to extract the momentum.
    f_in has shape (Lx,Ly).
    """
    Lx,Ly = f_in.shape
    ps = dctn(f_in,type=type_dct)
    ind = np.argmax(np.absolute(ps))
    kx,ky = ind//Ly, ind%Ly
    return kx,ky

def get_momentum_Bogoliubov3(f_in,ik=0,type_dct=2):
    """
    Compute the peak of the dctn of input function to extract the momentum.
    f_in has shape (Lx,Ly).
    """
    Lx,Ly = f_in.shape
    f_in = f_in.T
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
    #Renormalize the awsome functions by the cosine
    abs_f = np.absolute(f_in)
    res = np.zeros((Lx,Ly))
    for kx in range(Lx):
        for ky in range(Ly):
            cosx = np.cos(np.pi*kx*(2*X+1)/2/Lx)
            cosy = np.cos(np.pi*ky*(2*Y+1)/2/Ly)
            abs_cosx = np.absolute(cosx)
            abs_cosx[abs_cosx==0] = 1
            abs_cosy = np.absolute(cosy)
            abs_cosy[abs_cosy==0] = 1
            res[kx,ky] = np.sum(f_in*abs_f*cosx/abs_cosx*cosy/abs_cosy)
            if 0:
                print(res[kx,ky])
                fig = plt.figure(figsize=(14,10))
                ax = fig.add_subplot(121,projection='3d')
                ax.plot_surface(X,Y,f_in)
                ax = fig.add_subplot(122,projection='3d')
                ax.plot_surface(X,Y,abs_f*cosx/abs_cosx*cosy/abs_cosy)
                plt.suptitle("ks:%d,%d"%(kx,ky))
                plt.show()
    if 0:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X,Y,np.absolute(res).T,cmap='plasma')
        ax.set_title(ik)
        plt.show()
    ind = np.argmax(np.absolute(res))
    kx,ky = ind//Ly, ind%Ly
    return kx,ky

class SqrtNorm(mcolors.Normalize):
    def __call__(self, value, clip=None):
        return (super().__call__(value, clip))**(1/2)

def correlatorPlot(correlator_kw, **kwargs):
    """
    Plot frequency over mod k for the different stop ratios.
    correlator_kw has shape (n_sr, Lx, Ly, Nomega) with n_sr number of stop ratios (10) and Nomega number of frequency stps (2000)
    """
    f_type = kwargs['fourier_type']
    if f_type in ['fft','dst','dct','dat']:
        n_sr,Lx,Ly,N_omega = correlator_kw.shape
        if f_type=='fft':
            kx = fftshift(fftfreq(Lx,d=1)*2*np.pi)
            ky = fftshift(fftfreq(Ly,d=1)*2*np.pi)
        elif f_type=='dst':
            kx = np.pi * np.arange(1, Lx + 1) / (Lx + 1)
            ky = np.pi * np.arange(1, Ly + 1) / (Ly + 1)
        elif f_type in ['dct','dat']:
            kx = np.pi * np.arange(0, Lx ) / (Lx )
            ky = np.pi * np.arange(0, Ly ) / (Ly )
        #
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K_mag = np.sqrt(KX**2 + KY**2)
        K_flat = K_mag.ravel()

    freqs = fftshift(fftfreq(N_omega,0.8/400))

    # Define k bins
    if 'n_bins' in kwargs.keys():
        num_k_bins = kwargs['n_bins']
    else:
        num_k_bins = 100
    k_bins = np.linspace(0,np.sqrt(2)*np.pi, num_k_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    #
    K_mesh, W_mesh = np.meshgrid(k_centers, freqs, indexing='ij')
    # Figure
    fig = plt.figure(figsize=(20.8,8))
    #fig = plt.figure(figsize=(8,8))
    if 'title' in kwargs.keys():
        plt.suptitle(kwargs['title'],fontsize=20)
    cbar_label = kwargs['fourier_type']
    P_k_omega_sr = np.zeros((n_sr,num_k_bins,N_omega))
    for i_sr in range(n_sr):
        if f_type=='dat' and 0:
            corr_flat = correlator_kw[i_sr,:,0,:]
            K_flat = correlator_kw[i_sr,:,1,0]
        else:
            corr_flat = correlator_kw[i_sr].reshape(Lx*Ly, N_omega)
        for i in range(num_k_bins):
            mask = (K_flat >= k_bins[i]) & (K_flat < k_bins[i+1])
            if np.any(mask):
                P_k_omega_sr[i_sr, i, :] = np.mean(np.abs(corr_flat[mask, :]), axis=0)
    vmax = np.max(P_k_omega_sr)
    for i_sr in range(n_sr):     # Plotting
        vmax = np.max(P_k_omega_sr[i_sr])
        P_k_omega = P_k_omega_sr[i_sr]
        ax = fig.add_subplot(2,5,i_sr+1)
        #ax = fig.add_subplot()
        ax.set_facecolor('black')
        mesh = ax.pcolormesh(K_mesh, W_mesh, P_k_omega,
                             shading='auto',
                             cmap='inferno',
                             norm=SqrtNorm(vmin=0,vmax=vmax)
                            )
        ax.set_ylim(-70,70)
        ax.set_xlim(0,np.sqrt(2)*np.pi)
        #plt.title("Spectral power in $|k|$ vs $\\omega$")
        cbar = fig.colorbar(mesh, ax=ax)
        if i_sr in [4,9]:
            cbar.set_label(cbar_label,fontsize=15)
        if i_sr in [0,5]:
            ax.set_ylabel(r'$\omega$',fontsize=15)
        if i_sr in [5,6,7,8,9]:
            ax.set_xlabel(r'$|k|$',fontsize=15)
    #
    plt.subplots_adjust(wspace=0.112, hspace=0.116, left=0.035, right=0.982, bottom=0.076, top=0.94)

    if 'showfig' in kwargs.keys():
        if kwargs['showfig']:
            plt.show()



