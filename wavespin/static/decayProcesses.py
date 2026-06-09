""" Computation of scattering and decay rates at different orders for the different processes.
openHamiltonian elements.

The vertex as well as eigenvalues are supposed to be already given by the system.
"""

import numpy as np
from wavespin.tools.functions import lorentz, lorentz_n

def rate_1to2_1(system):
    """ Decay rate of 1 to 2 process at first order.
    """
    # Vertex, T and evals
    Vn_lm = system.vertex1to2[1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.sca_broadening * np.mean(evals[1:] - evals[:-1])
    en = evals[:,None,None]
    el = evals[None,:,None]
    em = evals[None,None,:]
    ### 1 -> 2
    arg = en - el -  em
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*(em+el)) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1)
        Gamma_1to2 = 2 * np.pi * np.einsum('nlm,nlm,nlm->n',Vn_lm**2,delta_vals,bose_factor)
    else:
        Gamma_1to2 = 2 * np.pi * np.einsum('nlm,nlm->n',Vn_lm**2,delta_vals)
    ### 1 <- 2
    arg = en + em - el
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*el) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1)
        Gamma_2to1 = 4 * np.pi * np.einsum('lnm,nlm,nlm->n',Vn_lm**2,delta_vals,bose_factor)
    else:
        Gamma_2to1 = np.zeros(system.Ns-1)
    # Final
    Gamma_n = Gamma_1to2 + Gamma_2to1
    return Gamma_n

def rate_1to2_2(system):
    """ Decay rate of 1 to 2 process at second order.
    """
    # Vertex, T and evals
    Vn_lm = system.vertex1to2[1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.sca_broadening * np.mean(evals[1:] - evals[:-1])
    en = evals[:,None]
    el = evals[None,:]
    ### 2 -> 1
    arg = 2*en - el
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*2*en)) * np.exp(beta*el) / (np.exp(beta*el)-1)
        Gamma_n = np.pi * np.einsum('lnn,nl,nl->n',Vn_lm**2,delta_vals,bose_factor)
    else:
        Gamma_n = np.pi * np.einsum('lnn,nl->n',Vn_lm**2,delta_vals)
    return Gamma_n

def rate_2to2_1(system):
    """ Decay rate of 2 to 2 process at first order.
    """
    # Vertex, T and evals
    Vnl_mp = system.vertex2to2[1:,1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.sca_broadening * np.mean(evals[1:] - evals[:-1])
    #
    en = evals[:,None,None,None]
    el = evals[None,:,None,None]
    em = evals[None,None,:,None]
    ep = evals[None,None,None,:]
    ### 2 -> 2
    arg = en + el - em - ep
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*(em+ep)) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1) / (np.exp(beta*ep)-1)
        Gamma_n = 4 * np.pi * np.einsum('nlmp,nlmp,nlmp->n',Vnl_mp**2,delta_vals,bose_factor)
    else:
        Gamma_n = np.zeros(system.Ns-1)
    return Gamma_n

def rate_2to2_2(system):
    """ Decay rate of 2 to 2 process at second order.
    """
    # Vertex, T and evals
    Vnl_mp = system.vertex2to2[1:,1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.sca_broadening * np.mean(evals[1:] - evals[:-1])
    en = evals[:,None,None]
    el = evals[None,:,None]
    em = evals[None,None,:]
    ### 2 -> 2
    arg = 2*en - el - em
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*2*en)) * np.exp(beta*(el+em)) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1)
        Gamma_n = 2 * np.pi * np.einsum('nnlm,nlm,nlm->n',Vnl_mp**2,delta_vals,bose_factor)
    else:
        Gamma_n = 2 * np.pi * np.einsum('nnlm,nlm->n',Vnl_mp**2,delta_vals)
    return Gamma_n

def rate_1to3_1(system):
    """ Decay rate of 1 to 3 process at first order.
    """
    # Vertex, T and evals
    Vn_lmp = system.vertex1to3[1:,1:,1:,1:]     #remove 0-energy mode from each mode index
    T = system.p.sca_temperature
    evals = system.evals[1:]                    #remove 0-energy eigenvalue
    # Broadening
    gamma = system.p.sca_broadening * np.mean(evals[1:] - evals[:-1])
    en = evals[:,None,None,None]
    el = evals[None,:,None,None]
    em = evals[None,None,:,None]
    ep = evals[None,None,None,:]
    ### 1 -> 3
    arg = en - el - em - ep
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*(el+em+ep)) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1) / (np.exp(beta*ep)-1)
        Gamma_1to3 = 6 * np.pi * np.einsum('nlmp,nlmp,nlmp->n',Vn_lmp**2,delta_vals,bose_factor)
    else:
        Gamma_1to3 = 6 * np.pi * np.einsum('nlmp,nlmp->n',Vn_lmp**2,delta_vals)
    ### 1 <- 3
    arg = en + el + em - ep
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*ep) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1) / (np.exp(beta*ep)-1)
        Gamma_3to1 = 18 * np.pi * np.einsum('pnlm,nlmp,nlmp->n',Vn_lmp**2,delta_vals,bose_factor)
    else:
        Gamma_3to1 = np.zeros(system.Ns-1)
    # Final
    Gamma_n = Gamma_1to3 + Gamma_3to1
    return Gamma_n

def rate_1to3_2(system):
    """ Decay rate of 1 to 3 process at second order.
    """
    # Vertex, T and evals
    Vn_lmp = system.vertex1to3[1:,1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.sca_broadening * np.mean(evals[1:] - evals[:-1])
    en = evals[:,None,None]
    el = evals[None,:,None]
    em = evals[None,None,:]
    ### 3 -> 1
    arg = 2*en + el - em
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*2*en)) * np.exp(beta*em) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1)
        Gamma_n = 9 * np.pi * np.einsum('mnnl,nlm,nlm->n',Vn_lmp**2,delta_vals,bose_factor)
    else:
        Gamma_n = np.zeros(system.Ns-1)
    return Gamma_n

def rate_1to3_3(system):
    """ Decay rate of 1 to 3 process at third order.
    """
    # Vertex, T and evals
    Vn_lmp = system.vertex1to3[1:,1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.sca_broadening * np.mean(evals[1:] - evals[:-1])
    en = evals[:,None]
    el = evals[None,:]
    ### 3 -> 1
    arg = 3*en - el
    delta_vals = lorentz(arg, gamma)
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*3*en)) * np.exp(beta*el) / (np.exp(beta*el)-1)
        Gamma_n = np.pi * np.einsum('lnnn,nl,nl->n',Vn_lmp**2,delta_vals,bose_factor)
    else:
        Gamma_n = np.pi * np.einsum('lnnn,nl->n',Vn_lmp**2,delta_vals)
    return Gamma_n

def rate_2to2_1_sc(system):
    """ Decay rate of 2 to 2 process at first order.
    Compute self-consistently the rate -> start from constant broadening for each mode.
    """
    # Vertex, T and evals
    Vnl_mp = system.vertex2to2[1:,1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    #
    en = evals[:,None,None,None]
    el = evals[None,:,None,None]
    em = evals[None,None,:,None]
    ep = evals[None,None,None,:]
    if T != 0:
        beta = 1/T
        bose_factor = (1-np.exp(-beta*en)) * np.exp(beta*(em+ep)) / (np.exp(beta*el)-1) / (np.exp(beta*em)-1) / (np.exp(beta*ep)-1)
    else:
        return np.zeros(system.Ns-1)
    # Broadening
    gamma_0 = system.p.sca_broadening * np.mean(evals[1:] - evals[:-1]) * np.ones(system.Ns-1)
    ### 2 -> 2
    arg_delta = np.array(en + el - em - ep)
    gamma_story = []
    while True:
        if len(gamma_story)==0:
            delta_vals = lorentz_n(arg_delta, gamma_0)
        else:
            delta_vals = lorentz_n(arg_delta, gamma_story[-1]/2)
        Gamma_n = 4 * np.pi * np.einsum('nlmp,nlmp,nlmp->n',Vnl_mp**2,delta_vals,bose_factor)       # Factor 4??
        gamma_story.append(Gamma_n)
        if len(gamma_story)==1:
            continue
        if np.sum(np.absolute(gamma_story[-1]-gamma_story[-2]))<1e-3:
            break
    print("Steps for convergence: %d"%len(gamma_story))
    gamma_story = np.array(gamma_story)
    if 0:
        print("%d steps"%gamma_story.shape[0])
        import matplotlib.pyplot as plt
        if 0:
            fig1 = plt.figure(figsize=(15,10))
            for i in range(system.Ns-1):
                ax = fig1.add_subplot(system.Lx,system.Ly,i+2)
                ax.plot(np.arange(gamma_story.shape[0]),gamma_story[:,i])
                ax.set_title("Mode %d"%(i+1))
            fig1.tight_layout()
        #
        fig2 = plt.figure(figsize=(10,10))
        ax = fig2.add_subplot()
        ax.scatter(np.arange(system.Ns-1)+1,gamma_story[0],color='r',label='initial')
        ax.scatter(np.arange(system.Ns-1)+1,gamma_story[-1],color='b',label='final')
        ax.legend()
        fig2.tight_layout()
        plt.show()
        exit()
    return gamma_story[-1]

dic_processes = {
    '1to2_1':rate_1to2_1,
    '1to2_2':rate_1to2_2,
    '2to2_1':rate_2to2_1,
    '2to2_2':rate_2to2_2,
    '1to3_1':rate_1to3_1,
    '1to3_2':rate_1to3_2,
    '1to3_3':rate_1to3_3,
}
