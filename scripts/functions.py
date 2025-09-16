import numpy as np

# General
def theta_they(*pars):
    J,S,lam,H = pars
    Hc = 4*J*S*(1-lam)
    if abs(H/Hc)<1:
        return np.arcsin(H/Hc)
    else:
        return np.pi/2
def theta_my(*pars):
    J,S,D,H = pars
    Hc = 4*J*S*(1-D)
    if abs(H/Hc)<1:
        return np.arccos(H/Hc)
    else:
        return 0
# Numeric
def gamma(k,*pars):
    if len(k.shape)==1:
        return 1/2*(np.cos(k[0]) + np.cos(k[1]))
    elif len(k.shape)==2:
        return 1/2*(np.cos(k[:,0]) + np.cos(k[:,1]))
    else:
        print("error in gamma")
        exit()
def epsilon_they(k,gk,*pars):
    J,S,lam,H = pars
    th = theta_they(*pars)
    return 4*J*S*np.sqrt((1-gk)*(1-gk*(lam*np.cos(th)**2+np.sin(th)**2)))
def epsilon_my(k,gk,*pars):
    J,S,D,H = pars
    th = theta_my(*pars)
    return 4*J*S*np.sqrt((1+gk)*(1-gk*(D*np.sin(th)**2+np.cos(th)**2)))
def Ak_they(k,gk,*pars):
    J,S,lam,H = pars
    th = theta_they(*pars)
    return 4*J*S*(1-1/2*gk*(1+lam*np.cos(th)**2+np.sin(th)**2))
def Ak_my(k,gk,*pars):
    J,S,D,H = pars
    th = theta_my(*pars)
    return 4*J*S*(1+1/2*gk*np.sin(th)**2*(1-D))
def uk_(k,gk,Ak_,epsilon_,*pars):
    ak = Ak_(k,gk,*pars)
    ek = epsilon_(k,gk,*pars)
    result = np.zeros_like(ek, dtype=float)
    mask = ek != 0
    result[mask] = np.sqrt((ak[mask] + ek[mask]) / (2 * ek[mask]))
    return result
def vk_(k,gk,Ak_,epsilon_,*pars):
    ak = Ak_(k,gk,*pars)
    ek = epsilon_(k,gk,*pars)
    absGk = np.absolute(gk)
    result = np.zeros_like(ek, dtype=float)
    mask = (ek != 0) & (absGk != 0) & (ak-ek>0)
    result[mask] = -np.sqrt((ak[mask] - ek[mask]) / (2 * ek[mask]))*gk[mask]/absGk[mask]
    return result
def Gamma3_(k,eps_k,K_grid,eps_q,eta,Ak_,epsilon_,th,*pars):
    L = int(np.sqrt(K_grid.shape[0]))
    gkmq = gamma(k-K_grid,*pars)
    eps_kmq = epsilon_(k-K_grid,gkmq,*pars)             # shape (L^2) -> can make this faster
    arg = eps_k - (eps_q + eps_kmq)         # shape (L^2)
    # evaluate delta and sum
    delta_vals = lorentz(arg, eta)            # shape (L^2)
    V_k_q = V3(k,K_grid,Ak_,epsilon_,th,*pars)              # shape (L^2)
    S = np.pi/2*np.sum(V_k_q**2 * delta_vals)
    return S
def gauss(x,eta):
    return np.exp(-0.5*(x/eta)**2) / (np.sqrt(2*np.pi)*eta)
def lorentz(x,eta):
    return (1.0/np.pi) * (eta / (x**2 + eta**2))
def V3(k,q,Ak_,epsilon_,th,*pars):
    J,S,_,H = pars
    factor = np.cos(th) if Ak_==Ak_they else np.sin(th)
    #
    gk = gamma(k,*pars)
    uk = uk_(k,gk,Ak_,epsilon_,*pars)
    vk = vk_(k,gk,Ak_,epsilon_,*pars)
    #
    gq = gamma(q,*pars)
    uq = uk_(q,gq,Ak_,epsilon_,*pars)
    vq = vk_(q,gq,Ak_,epsilon_,*pars)
    #
    gkmq = gamma(k-q,*pars)
    ukmq = uk_(k-q,gkmq,Ak_,epsilon_,*pars)
    vkmq = vk_(k-q,gkmq,Ak_,epsilon_,*pars)
    return H*factor/np.sqrt(2*S) * (  gk*(uk+vk)*(uq*vkmq+vq*ukmq)
                                        + gq*(uq+vq)*(uk*ukmq+vk*vkmq)
                                        + gkmq*(ukmq+vkmq)*(uk*uq+vk*vq)   )

from tqdm import tqdm
def Gamma4a_(k,eps_k,K_grid,eps_p,eta,Ak_,epsilon_,theta_,*pars):
    L = int(np.sqrt(K_grid.shape[0]))
    result = 0
    for ind_q in tqdm(range(L**2)):
        q = K_grid[ind_q]
        eps_q = eps_p[ind_q]
        gkmqmp = gamma(k-q-K_grid,*pars)
        eps_kmqmp = epsilon_(k-q-K_grid,gkmqmp,*pars)           # shape (L^2) -> can make this faster
        arg = eps_k - (eps_q + eps_p + eps_kmqmp)       # shape (L^2)
        delta_vals = lorentz(arg, eta)                  # shape (L^2)
        U_k_qp = U4a(k,q,K_grid,Ak_,epsilon_,theta_,*pars)                  # shape (L^2)
        result += np.pi/6*np.sum(U_k_qp**2 * delta_vals)
    return result
def U4a(k,q,p,Ak_,epsilon_,theta_,*pars):
    J,S,D,H = pars
    th = theta_(*pars)
    if Ak_==Ak_they:
        f1 = np.sin(th)**2+1+D*np.cos(th)**2
        f2 = f1
        f3 = -4*(np.cos(th)**2+D*np.sin(th)**2)
    else:
        print("Not implemented")
        exit()
    #
    gk = gamma(k,*pars)
    uk = uk_(k,gk,Ak_,epsilon_,*pars)
    vk = vk_(k,gk,Ak_,epsilon_,*pars)
    #
    gq = gamma(q,*pars)
    uq = uk_(q,gq,Ak_,epsilon_,*pars)
    vq = vk_(q,gq,Ak_,epsilon_,*pars)
    #
    gp = gamma(p,*pars)
    up = uk_(p,gp,Ak_,epsilon_,*pars)
    vp = vk_(p,gp,Ak_,epsilon_,*pars)
    #
    gkmqmp = gamma(k-q-p,*pars)
    ukmqmp = uk_(k-q-p,gkmqmp,Ak_,epsilon_,*pars)
    vkmqmp = vk_(k-q-p,gkmqmp,Ak_,epsilon_,*pars)
    #
    gkmp = gamma(k-p,*pars)
    #
    gppq = gamma(p+q,*pars)
    return 3/2*J*(
        f1*( (gk+gp)*(uq*vkmqmp+vq*ukmqmp)*(vk*up+uk*vp) + gq*(vk*vkmqmp+uk*ukmqmp)*(vq*vp+uq*up) + gkmqmp*(vp*vkmqmp+up*ukmqmp)*(vk*vq+uk*uq) ) +
        f2*( gk*(vk*uq+uk*vq)*(vp*vkmqmp+up*ukmqmp) + gq*(vk*vp+uk*uq)*(uq*vkmqmp+vq*ukmqmp) + (gp+gkmqmp)*(vp*ukmqmp+up*vkmqmp)*(vk*vq+uk*uq) ) +
        f3*( gppq*(vk*vkmqmp+uk*ukmqmp)*(uq*vp+vq*up) + gkmp*(uq*vkmqmp+vq*ukmqmp)*(vk*vp+uk*up) )
    )

def _getQs3(k,k_grid,shellFactor=1e-3,*pars):
    """ Compute which q values are on shell with k. """
    ek = epsilon(k,*pars)
    Eq = epsilon(k_grid,*pars)
    Ekmq = epsilon(k-k_grid,*pars)
    inds = np.where(abs(ek-Eq-Ekmq)<shellFactor)
    if len(inds)>0:
        return k_grid[ inds[0] ]
    else:
        return np.array([])

def _getQs4a(k,k_grid,shellFactor=1e-3,*pars):
    """ Compute which q values are on shell with k. """
    ek = epsilon(k,*pars)
    Eq = epsilon(k_grid,*pars)
    Ekmq = epsilon(k-k_grid,*pars)
    inds = np.where(abs(ek-Eq-Ekmq)<shellFactor)
    if len(inds)>0:
        return k_grid[ inds[0] ]
    else:
        return np.array([])

# Analytic
def c_(*pars):
    J,S,D,H = pars
    th = theta(*pars)
    return 2*J*S*np.sqrt(1-D)*np.cos(th)
def phi_(k):
    if len(k.shape)==1:
        return np.arctan2(k[1],k[0])
    elif len(k.shape)==2:
        return np.arctan2(k[:,1],k[:,0])
def alpha_(k,*pars):
    J,S,D,H = pars
    th = theta(*pars)
    c = c_(*pars)
    phi = phi_(k)
    return c/8*(1/(np.cos(th)**2*(1-D)) - (15+np.cos(4*phi))/12)
def analyticGamma3(k_grid,*pars):
    J,S,D,H = pars
    th = theta(*pars)
    c = c_(*pars)
    alpha = alpha_(k_grid,*pars)
    return 3*J/16/np.pi*np.tan(th)**2*np.sqrt(c/6/alpha) * np.linalg.norm(k_grid,axis=1)**3













































