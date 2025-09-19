import numpy as np

# General
def theta_fm(*pars):
    J,S,lam,H = pars
    Hc = 4*J*S*(1-lam)
    if abs(H/Hc) < 1:
        return np.arcsin(H/Hc)
    else:
        return np.pi/2

def theta_afm(*pars):
    J,S,D,H = pars
    Hc = 4*J*S*(1-D)
    if abs(H/Hc) < 1:
        return np.arccos(H/Hc)
    else:
        return 0

def p_afm(*pars):
    J,S,D,H = pars
    th = theta_afm(*pars)
    return (-J*S**2*(np.cos(th)**2+D*np.sin(th)**2),
            J*S**2,
            -J*S**2*(np.sin(th)**2+D*np.cos(th)**2),
            -4*J*S**2*(1-D)*np.cos(th)**2)

def p_fm(*pars):
    J,S,lam,H = pars
    ph = theta_fm(*pars)
    return (-J*S**2*(np.sin(ph)**2+lam*np.cos(ph)**2),
            -J*S**2,
            -J*S**2*(np.cos(ph)**2+lam*np.cos(ph)**2),
            -4*J*S**2*(1-lam)*np.sin(ph)**2)

def gamma(k):
    """k can be (...,2) or (2,) etc. Returns 1/2*(cos(kx)+cos(ky))"""
    k = np.asarray(k)
    if k.ndim == 1 and k.shape[0] == 2:
        return 0.5*(np.cos(k[0]) + np.cos(k[1]))
    elif k.ndim >= 2 and k.shape[-1] == 2:
        return 0.5*(np.cos(k[...,0]) + np.cos(k[...,1]))
    else:
        raise ValueError("gamma: expected last dimension 2")

def N11(gk,ps,*pars):
    S = pars[1]
    return 2 * ( gk*(ps[0]+ps[1])/S - 2*ps[2]/S - ps[3]/(2*S) )

def N12(gk,ps,*pars):
    S = pars[1]
    return 2*gk*(ps[0]-ps[1])/S

def epsilon(gk,ps,*pars):
    # ensure non-negative inside sqrt via clamp to 0
    val = N11(gk,ps,*pars)**2 - N12(gk,ps,*pars)**2
    # numerical roundoff protection
    val = np.where(val < 0, 0.0, val)
    return np.sqrt(val)

def bogoliubovFunctions(gk,eps,ps,*pars):
    Ak = N11(gk,ps,*pars)
    uk = np.zeros_like(eps, dtype=np.float64)
    mask_u = eps != 0
    uk[mask_u] = np.sqrt((Ak[mask_u] + eps[mask_u]) / (2 * eps[mask_u]))
    #
    vk = np.zeros_like(eps, dtype=np.float64)
    absGk = np.absolute(gk)
    mask_v = (eps != 0) & (absGk != 0) & (Ak-eps > 0)
    vk[mask_v] = -np.sqrt((Ak[mask_v] - eps[mask_v]) / (2 * eps[mask_v])) * gk[mask_v] / absGk[mask_v]
    return uk.astype(np.float64), vk.astype(np.float64)

def gauss(x,eta):
    return np.exp(-0.5*(x/eta)**2) / (np.sqrt(2*np.pi)*eta)

def lorentz(x,eta):
    return (1.0/np.pi) * (eta / (x**2 + eta**2))

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
                + f3 * gkmq * (uk*uq + vk*vq) * (up*vkmqmp + vp*ukmqmp)
                + f3 * gkmp * (uk*up + vk*vp) * (uq*vkmqmp + vq*ukmqmp)
                + f3 * gppq * (uk*ukmqmp + vk*vkmqmp) * (up*vq + vp*uq)
                )
    result = np.pi/6*np.sum(U_k_qp**2 * delta_vals)
    return result

# ---------- Blocked Gamma4 evaluator ----------
def Gamma4a_blocked(k, K_grid, gk, g_grid, eps_k, eps_grid, uk_grid, vk_grid,
                    eta, ps, *pars, block=1024):
    """
    Evaluate Gamma4a_ for a single k, but in memory-safe blocks.
    - k: (2,) array
    - K_grid: (N,2) array of momenta
    - gk: scalar gamma(k)
    - g_grid: (N,) gamma values for K_grid
    - eps_k: scalar epsilon(k)
    - eps_grid: (N,) epsilon for K_grid
    - returns scalar result (same as original function)
    - block: block size for q (and p). tune to available memory.
    """
    # Ensure float32 throughout to save memory
    dtype = np.float64
    k = np.asarray(k, dtype=dtype)
    K_grid = np.asarray(K_grid, dtype=dtype)
    g_grid = np.asarray(g_grid, dtype=dtype)
    eps_grid = np.asarray(eps_grid, dtype=dtype)
    eps_k = np.float64(eps_k)
    gk = np.float64(gk)

    N = K_grid.shape[0]     #L^2
    result = np.float64(0.0)   # accumulate in double for safety

    # Precompute constants used in U_k_qp
    f1 = -(ps[0] + ps[1]) / (8 * pars[1]**2)
    f2 = (ps[1] - ps[0]) / (8 * pars[1]**2)
    f3 = ps[2] / pars[1]**2
    # uk and vk
    uk_k, vk_k = bogoliubovFunctions(np.array([gk], dtype=np.float64),
                                     np.array([eps_k], dtype=np.float64),
                                     ps, *pars)
    uk_k = uk_k[0].astype(dtype)
    vk_k = vk_k[0].astype(dtype)
    a1 = (f1*uk_k + f2*vk_k)   # scalar
    a2 = (f1*vk_k + f2*uk_k)   # scalar

    # Loop over blocks for q and p
    for i in range(0, N, block):
        i1 = i
        i2 = min(N, i + block)
        q_block = K_grid[i1:i2]                # (b,2)
        gq = g_grid[i1:i2].astype(dtype)       # (b,)
        uq = uk_grid[i1:i2].astype(dtype)      # (b,)
        vq = vk_grid[i1:i2].astype(dtype)      # (b,)
        eps_q = eps_grid[i1:i2].astype(dtype)  # (b,)
        for j in range(0, N, block):
            j1 = j
            j2 = min(N, j + block)
            p_block = K_grid[j1:j2]                # (c,2)
            gp = g_grid[j1:j2].astype(dtype)       # (c,)
            up = uk_grid[j1:j2].astype(dtype)      # (c,)
            vp = vk_grid[j1:j2].astype(dtype)      # (c,)
            eps_p = eps_grid[j1:j2].astype(dtype)  # (c,)
            # compute local arrays of shape (b, c)
            # gppq = gamma(q + p)  shape (b, c)
            qp_sum = q_block[:,None,:] + p_block[None,:,:]   # (b,c,2)
            gppq_block = gamma(qp_sum).astype(dtype)        # (b,c)
            # k - q - p  shape (b,c,2)
            k_minus_qp = (k[None,None,:] - q_block[:,None,:] - p_block[None,:,:]).astype(dtype)
            gkmqmp = gamma(k_minus_qp).astype(dtype)        # (b,c)
            eps_kmqmp = epsilon(gkmqmp, ps, *pars).astype(dtype)  # (b,c)
            # other helper quantities
            # gkm for q and p used later as 1D arrays
            gkmq = gamma(k - q_block).astype(dtype)   # (b,)
            gkmp = gamma(k - p_block).astype(dtype)   # (c,)
            # Broadcasted versions for formula
            # shapes:
            # uq[:,None] (b,1), up[None,:] (1,c)
            uq_b = uq[:,None]
            vq_b = vq[:,None]
            up_b = up[None,:]
            vp_b = vp[None,:]
            # ukmqmp, vkmqmp from bogoliubovFunctions for gkmqmp
            ukmqmp, vkmqmp = bogoliubovFunctions(gkmqmp, eps_kmqmp, ps, *pars)
            # gq[:,None], gp[None,:], gkmq etc
            gq_b = gq[:,None]
            gp_b = gp[None,:]
            gkmq_b = gkmq[:,None]     # (b,1)
            gkmp_b = gkmp[None,:]     # (1,c)
            #
            uuv = uq_b * up_b * vkmqmp + uq_b * vp_b * ukmqmp + vq_b * up_b * ukmqmp
            vvu = uq_b * vp_b * vkmqmp + vq_b * vp_b * ukmqmp + vq_b * up_b * vkmqmp
            guuv = gq_b * (uq_b * up_b * vkmqmp) + gkmqmp * (uq_b * vp_b * ukmqmp) + gp_b * (vq_b * up_b * ukmqmp)
            gvvu = gp_b * (uq_b * vp_b * vkmqmp) + gq_b * (vq_b * vp_b * ukmqmp) + gkmqmp * (vq_b * up_b * vkmqmp)
            # Now build U_k_qp
            U_k_qp = ( gk * ( a1 * uuv + a2 * vvu )
                       + a1 * gvvu + a2 * guuv
                       + f3 * gkmq_b * (uk_k * uq_b + vk_k * vq_b) * (up_b * vkmqmp + vp_b * ukmqmp)
                       + f3 * gkmp_b * (uk_k * up_b + vk_k * vp_b) * (uq_b * vkmqmp + vq_b * ukmqmp)
                       + f3 * gppq_block * (uk_k * ukmqmp + vk_k * vkmqmp) * (up_b * vq_b + vp_b * uq_b)
                     ).astype(dtype)
            # delta function (lorentz) argument
            arg = eps_k - (eps_q[:,None] + eps_p[None,:] + eps_kmqmp)
            delta_vals = lorentz(arg, eta).astype(dtype)
            # accumulate (use float64 accumulator)
            block_sum = np.sum((U_k_qp**2 * delta_vals).astype(np.float64))
            result += block_sum
    #
    return (np.pi/6.0) * result

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













































