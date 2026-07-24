""" Computation of scattering and decay rates at different orders for the different processes.
openHamiltonian elements.

The vertex as well as eigenvalues are supposed to be already given by the system.
"""

import numpy as np
from wavespin.tools.functions import lorentz, lorentz_n

def rate_1to2_1(system):
    """Decay rate of 1 magnon to 2 magnons at first order.

    Applies Fermi's Golden Rule:

    .. math::
        \\Gamma_n = 2\\pi \\sum_{l,m} |V_{nlm}|^2 \\, \\delta(E_n - E_l - E_m)
        \\, (1 + n_l + n_m)

    plus the reverse process :math:`2 \\to 1`.  The energy-conserving delta
    is broadened with a Lorentzian whose width is proportional to the mean
    level spacing.

    Parameters
    ----------
    system : openHamiltonian
        Must have ``evals``, ``U_``, ``V_``, and ``vertex1to2`` populated
        (the vertex is a rank-3 tensor of shape ``(Ns, Ns, Ns)``).

    Returns
    -------
    (Ns-1,) ndarray
        Decay rate :math:`\\Gamma_n` for each mode (zero mode excluded).
    """
    # Vertex, T and evals
    Vn_lm = system.vertex1to2[1:,1:,1:]
    T = system.p.scattering.temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.scattering.broadening * np.mean(evals[1:] - evals[:-1])
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
    """Decay rate of 1 magnon to 2 magnons at second order.

    A single-channel process :math:`2 \\to 1` at second order in the
    interaction vertex.

    Parameters
    ----------
    system : openHamiltonian
        Must have ``evals`` and ``vertex1to2``.

    Returns
    -------
    (Ns-1,) ndarray
        Decay rate :math:`\\Gamma_n` per mode.
    """
    # Vertex, T and evals
    Vn_lm = system.vertex1to2[1:,1:,1:]
    T = system.p.scattering.temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.scattering.broadening * np.mean(evals[1:] - evals[:-1])
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
    """Decay rate of 2 magnons to 2 magnons at first order.

    .. math::
        \\Gamma_n = 4\\pi \\sum_{l,m,p} |V_{nlmp}|^2
        \\, \\delta(E_n + E_l - E_m - E_p)
        \\, (1 + n_n)(1 + n_l) n_m n_p / (\\ldots)

    Contributions vanish at T = 0.

    Parameters
    ----------
    system : openHamiltonian
        Must have ``evals`` and ``vertex2to2``
        (rank-4 tensor, shape ``(Ns, Ns, Ns, Ns)``).

    Returns
    -------
    (Ns-1,) ndarray
        Decay rate :math:`\\Gamma_n` per mode.
    """
    # Vertex, T and evals
    Vnl_mp = system.vertex2to2[1:,1:,1:,1:]
    T = system.p.scattering.temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.scattering.broadening * np.mean(evals[1:] - evals[:-1])
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
    """Decay rate of 2 magnons to 2 magnons at second order.

    Single-channel process :math:`2n \\to l + m` at second order.

    Parameters
    ----------
    system : openHamiltonian
        Must have ``evals`` and ``vertex2to2``.

    Returns
    -------
    (Ns-1,) ndarray
        Decay rate :math:`\\Gamma_n` per mode.
    """
    # Vertex, T and evals
    Vnl_mp = system.vertex2to2[1:,1:,1:,1:]
    T = system.p.scattering.temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.scattering.broadening * np.mean(evals[1:] - evals[:-1])
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
    """Decay rate of 1 magnon to 3 magnons at first order.

    .. math::
        \\Gamma_n = 6\\pi \\sum_{l,m,p} |V_{nlmp}|^2
        \\, \\delta(E_n - E_l - E_m - E_p)
        \\, (1 + n_l + n_m + n_p)

    plus the reverse process :math:`3 \\to 1`.

    Parameters
    ----------
    system : openHamiltonian
        Must have ``evals`` and ``vertex1to3``
        (rank-4 tensor, shape ``(Ns, Ns, Ns, Ns)``).

    Returns
    -------
    (Ns-1,) ndarray
        Decay rate :math:`\\Gamma_n` per mode.
    """
    # Vertex, T and evals
    Vn_lmp = system.vertex1to3[1:,1:,1:,1:]     #remove 0-energy mode from each mode index
    T = system.p.scattering.temperature
    evals = system.evals[1:]                    #remove 0-energy eigenvalue
    # Broadening
    gamma = system.p.scattering.broadening * np.mean(evals[1:] - evals[:-1])
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
    """Decay rate of 1 magnon to 3 magnons at second order.

    Single-channel process :math:`3 \\to 1` at second order.

    Parameters
    ----------
    system : openHamiltonian
        Must have ``evals`` and ``vertex1to3``.

    Returns
    -------
    (Ns-1,) ndarray
        Decay rate :math:`\\Gamma_n` per mode.
    """
    # Vertex, T and evals
    Vn_lmp = system.vertex1to3[1:,1:,1:,1:]
    T = system.p.scattering.temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.scattering.broadening * np.mean(evals[1:] - evals[:-1])
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
    """Decay rate of 1 magnon to 3 magnons at third order.

    Single-channel process :math:`3 \\to 1` at third order.

    Parameters
    ----------
    system : openHamiltonian
        Must have ``evals`` and ``vertex1to3``.

    Returns
    -------
    (Ns-1,) ndarray
        Decay rate :math:`\\Gamma_n` per mode.
    """
    # Vertex, T and evals
    Vn_lmp = system.vertex1to3[1:,1:,1:,1:]
    T = system.p.scattering.temperature
    evals = system.evals[1:]
    # Broadening
    gamma = system.p.scattering.broadening * np.mean(evals[1:] - evals[:-1])
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
    """Self-consistent 2-to-2 rate at first order.

    Iterates the rate formula, using the rate itself as the broadening
    width for the Lorentzian energy delta, until convergence.

    Parameters
    ----------
    system : openHamiltonian
        Must have ``evals`` and ``vertex2to2``.

    Returns
    -------
    (Ns-1,) ndarray
        Self-consistent decay rate :math:`\\Gamma_n` per mode.
    """
    # Vertex, T and evals
    Vnl_mp = system.vertex2to2[1:,1:,1:,1:]
    T = system.p.scattering.temperature
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
    gamma_0 = system.p.scattering.broadening * np.mean(evals[1:] - evals[:-1]) * np.ones(system.Ns-1)
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
