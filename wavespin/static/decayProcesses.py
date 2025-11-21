""" Computation of scattering and decay rates at different orders for the different processes.
openHamiltonian elements.

The vertex as well as eigenvalues are supposed to be already given by the system.
"""

import numpy as np
from wavespin.tools.functions import lorentz

def rate_1to2_1(system):
    """ Decay rate of 1 to 2 process at first order.
    """
    # Vertex, T and evals
    Vn_lm = system.vertex1to2[1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    # Broadening
    edif = evals[2:] - evals[1:-1]
    gamma = system.p.sca_broadening * np.mean(edif)
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
    edif = evals[2:] - evals[1:-1]
    gamma = system.p.sca_broadening * np.mean(edif)
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
    edif = evals[2:] - evals[1:-1]
    gamma = system.p.sca_broadening * np.mean(edif)
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
        Gamma_n = 4 * np.pi * np.einsum('nlmp,nlmp,nlmp->n',Vnl_mp**2,delta_vals,bose_factor)       # Factor 4??
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
    edif = evals[2:] - evals[1:-1]
    gamma = system.p.sca_broadening * np.mean(edif)
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
    Vn_lmp = system.vertex1to3[1:,1:,1:,1:]
    T = system.p.sca_temperature
    evals = system.evals[1:]
    # Broadening
    edif = evals[2:] - evals[1:-1]
    gamma = system.p.sca_broadening * np.mean(edif)
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
    edif = evals[2:] - evals[1:-1]
    gamma = system.p.sca_broadening * np.mean(edif)
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
    edif = evals[2:] - evals[1:-1]
    gamma = system.p.sca_broadening * np.mean(edif)
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

dic_processes = {
    '1to2_1':rate_1to2_1,
    '1to2_2':rate_1to2_2,
    '2to2_1':rate_2to2_1,
    '2to2_2':rate_2to2_2,
    '1to3_1':rate_1to3_1,
    '1to3_2':rate_1to3_2,
    '1to3_3':rate_1to3_3,
}
