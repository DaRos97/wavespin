import numpy as np

# General
def theta_fm(*pars):
    J,S,lam,H = pars
    Hc = 4*J*S*(1-lam)
    if abs(H/Hc)<1:
        return np.arcsin(H/Hc)
    else:
        return np.pi/2
def theta_afm(*pars):
    J,S,D,H = pars
    Hc = 4*J*S*(1-D)
    if abs(H/Hc)<1:
        return np.arccos(H/Hc)
    else:
        return 0
def p_afm(*pars):
    J,S,D,H = pars
    th = theta_afm(*pars)
    return (-J*S**2*(np.cos(th)**2+D*np.sin(th)**2),
            J*S**2,
            -J*S**2*(np.sin(th)**2+D*np.cos(th)**2),
            -4*J*S**2*(1-D)*np.cos(th)**2
            )
def p_fm(*pars):
    J,S,lam,H = pars
    ph = theta_fm(*pars)
    return (-J*S**2*(np.sin(ph)**2+lam*np.cos(ph)**2),
            -J*S**2,
            -J*S**2*(np.cos(ph)**2+lam*np.cos(ph)**2),
            -4*J*S**2*(1-lam)*np.sin(ph)**2
            )
def gamma(k):
    if len(k.shape)==1:
        return 1/2*(np.cos(k[0]) + np.cos(k[1]))
    elif len(k.shape)==2:
        return 1/2*(np.cos(k[:,0]) + np.cos(k[:,1]))
    elif len(k.shape)==3:
        return 1/2*(np.cos(k[:,:,0]) + np.cos(k[:,:,1]))
    else:
        print("error in gamma")
        exit()
def N11(gk,ps,*pars):
    S = pars[1]
    return 2 * ( gk*(ps[0]+ps[1])/S - 2*ps[2]/S - ps[3]/2/S )
def N12(gk,ps,*pars):
    S = pars[1]
    return 2*gk*(ps[0]-ps[1])/S
def epsilon(gk,ps,*pars):
    return np.sqrt(N11(gk,ps,*pars)**2 - N12(gk,ps,*pars)**2)
def bogoliubovFunctions(gk,eps,ps,*pars):
    Ak = N11(gk,ps,*pars)
    uk = np.zeros_like(eps, dtype=float)
    mask_u = eps != 0
    uk[mask_u] = np.sqrt((Ak[mask_u] + eps[mask_u]) / (2 * eps[mask_u]))
    #
    vk = np.zeros_like(eps, dtype=float)
    absGk = np.absolute(gk)
    mask_v = (eps != 0) & (absGk != 0) & (Ak-eps>0)
    vk[mask_v] = -np.sqrt((Ak[mask_v] - eps[mask_v]) / (2 * eps[mask_v]))*gk[mask_v]/absGk[mask_v]
    return uk, vk
# 1 -> 2 decay
def Gamma3_(k,eps_k,K_grid,eps_q,eta,Ak_,epsilon_,th,*pars):
    L = int(np.sqrt(K_grid.shape[0]))
    gkmq = gamma(k-K_grid)
    eps_kmq = epsilon_(gkmq,*pars)             # shape (L^2) -> can make this faster
    arg = eps_k - (eps_q + eps_kmq)         # shape (L^2)
    # evaluate delta and sum
    delta_vals = lorentz(arg, eta)            # shape (L^2)
    V_k_q = V3(k,K_grid,Ak_,epsilon_,th,*pars)              # shape (L^2)
    S = np.pi/2*np.sum(V_k_q**2 * delta_vals)
    return S
def V3(k,q,Ak_,epsilon_,th,*pars):
    J,S,_,H = pars
    factor = np.cos(th) if Ak_==Ak_fm else np.sin(th)
    #
    gk = gamma(k)
    uk = uk_(gk,Ak_,epsilon_,*pars)
    vk = vk_(gk,Ak_,epsilon_,*pars)
    #
    gq = gamma(q)
    uq = uk_(gq,Ak_,epsilon_,*pars)
    vq = vk_(gq,Ak_,epsilon_,*pars)
    #
    gkmq = gamma(k-q)
    ukmq = uk_(gkmq,Ak_,epsilon_,*pars)
    vkmq = vk_(gkmq,Ak_,epsilon_,*pars)
    return H*factor/np.sqrt(2*S) * (  gk*(uk+vk)*(uq*vkmq+vq*ukmq)
                                        + gq*(uq+vq)*(uk*ukmq+vk*vkmq)
                                        + gkmq*(ukmq+vkmq)*(uk*uq+vk*vq)   )
# 1 -> 3 decay
from time import time as ttt
def Gamma4a_(k,K_grid,gk,g_grid,gppq,eps_k,eps_grid,eta,ps,*pars):
    gkmqmp = gamma(k-K_grid[:,None,:]-K_grid[None,:,:])
    eps_kmqmp = epsilon(gkmqmp,ps,*pars)
    # Compute delta
    arg = eps_k - (eps_grid[:,None] + eps_grid[None,:] + eps_kmqmp)   # shape (L^2,L^2)
    delta_vals = lorentz(arg, eta)              # shape (L^2,L^2)
    # Computation of U_k(q,p)
    f1 = -(ps[0]+ps[1])/(8*pars[1]**2)
    f2 = (ps[1]-ps[0])/(8*pars[1]**2)
    f3 = ps[2]/pars[1]**2
    uk,vk = bogoliubovFunctions(gk,eps_k,ps,*pars)
    t1 = ttt()
    uq,vq = bogoliubovFunctions(g_grid,eps_grid,ps,*pars)
    uq = uq[:,None]
    vq = vq[:,None]
    up,vp = bogoliubovFunctions(g_grid,eps_grid,ps,*pars)
    up = up[None,:]
    vp = vp[None,:]
    ukmqmp,vkmqmp = bogoliubovFunctions(gkmqmp,eps_kmqmp,ps,*pars)
    gkm = gamma(k-K_grid)
    gq = g_grid[:,None]
    gp = g_grid[None,:]
    gkmq = gkm[:,None]
    gkmp = gkm[None,:]
    uuv = uq*up*vkmqmp + uq*vp*ukmqmp + vq*up*ukmqmp
    vvu = uq*vp*vkmqmp + vq*vp*ukmqmp + vq*up*vkmqmp
    guuv = gq*uq*up*vkmqmp + gkmqmp*uq*vp*ukmqmp + gp*vq*up*ukmqmp
    gvvu = gp*uq*vp*vkmqmp + gq*vq*vp*ukmqmp + gkmqmp*vq*up*vkmqmp
    U_k_qp = (  gk * ( (f1*uk+f2*vk)*uuv + (f1*vk+f2*uk)*vvu)
                + (f1*uk+f2*vk)*gvvu + (f1*vk+f2*uk)*guuv
                + gkmq * (uk*uq + vk*vq) * (up*vkmqmp + vp*ukmqmp)
                + gkmp * (uk*up + vk*vp) * (uq*vkmqmp + vq*ukmqmp)
                + gppq * (uk*ukmqmp + vk*vkmqmp) * (up*vq + vp*uq)
                )
    result = np.pi/6*np.sum(U_k_qp**2 * delta_vals)
    return result

def gauss(x,eta):
    return np.exp(-0.5*(x/eta)**2) / (np.sqrt(2*np.pi)*eta)
def lorentz(x,eta):
    return (1.0/np.pi) * (eta / (x**2 + eta**2))

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













































