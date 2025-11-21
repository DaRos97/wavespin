""" Here we compute the magnon decay rate.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions as fs
from pathlib import Path
import sys

save = True

if 0: # Old  #Plot curvature at different fields
    D_list = [0.]#,0.3,0.7,0.9]
    fH_list = [0,0.5,0.9,1]
    Ngm = 500
    Nmx = int(Ngm/np.sqrt(2))
    Nk = Ngm+Nmx
    k_gm = np.column_stack((np.linspace(0,np.pi,Ngm,endpoint=False),np.linspace(0,np.pi,Ngm,endpoint=False)))
    k_mx = np.column_stack((np.ones(Nmx)*np.pi,np.linspace(np.pi,0,Nmx)))
    k_to_compute = np.append(k_gm,k_mx,axis=0)
    #Their model
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(121)
    for D in D_list:
        lam = -D
        Hc = 4*S*J*(1-lam)
        for fH in fH_list:
            parameters = (J,S,lam,fH*Hc)
            gk = fs.gamma(k_to_compute)
            ek = fs.epsilon_they(gk,*parameters)
            ax.plot(np.arange(Nk),ek,label=r"H/Hc=%.2f, $\lambda$=%.2f"%(fH,lam))
    ax.axvline(Ngm,color='r',ls='--')
    ax.set_xticks([0,Ngm,Nk-1],[r'$\Gamma$',r'$M$',r'$X$'],size=20)
    ax.set_ylim(0,4)
    ax.legend()
    ax.set_title("Uniform field")

    ax = fig.add_subplot(122)
    for D in D_list:
        Hc = 4*S*J*(1-D)
        for fH in fH_list:
            parameters = (J,S,D,fH*Hc)
            gk = fs.gamma(k_to_compute)
            ek = fs.epsilon_my(gk,*parameters)
            ax.plot(np.arange(Nk),ek,label=r"H/Hc=%.2f, D=%.2f"%(fH,D))
    ax.axvline(Ngm,color='r',ls='--')
    ax.set_xticks([0,Ngm,Nk-1],[r'$\Gamma$',r'$M$',r'$X$'],size=20)
    ax.legend()
    ax.set_title("Staggered field")
    ax.set_ylim(0,4)
    plt.show()
    exit()
if 0: #Old
    """ Compute phase space of 1->2 magnon process """
    H = 0.1
    if 0:
        lam = 0.5
        parameters = (J,S,lam,H)
        epsilon_ = fs.epsilon_they
    else:
        D = 0.1
        parameters = (J,S,D,H)
        epsilon_ = fs.epsilon_my
    res = np.zeros(L**2)
    gq = fs.gamma(K_grid)
    eq = epsilon_(gq,*parameters)
    for ik in tqdm(range(L**2)):
        k = K_grid[ik]
        ek = eq[ik]
        gkmq = fs.gamma(k-K_grid)
        ekmq = epsilon_(gkmq,*parameters)
        spread = fs.lorentz(ek - (eq + ekmq),eta)
        vals = spread[spread>1/np.pi/eta-0.1]
        res[ik] = np.sign(len(vals)-2)
    res[res<0.5] = np.nan
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(KX,KY,res.reshape(L,L),cmap='plasma')
    plt.show()
    exit()
if 0: #Old
    """ Compute decay rate of 1->2 magnon process """
    J = 1
    S = 1/2
    D_list = [-0.5,]
    fH_list = [0.3,0.5,0.7,0.9]
    Ham = 'FM'      #FM or AFM
    Ngm = 100
    L = 200
    eta = 0.001
    k_list = np.linspace(0,np.pi,L,endpoint=False)
    KX,KY = np.meshgrid(k_list,k_list,indexing='ij')
    K_grid = np.column_stack(([KX.ravel(),KY.ravel()]))
    print("Parameters: J=%d"%J)
    print("fH values: ",fH_list)
    print("D values: ",D_list)
    print("BZ linear size: ",L)
    print("Gaussian broadening: ",eta)
    Nmx = int(Ngm/np.sqrt(2))
    Nk = Ngm+Nmx
    k_gm = np.column_stack((np.linspace(0,np.pi,Ngm,endpoint=False),np.linspace(0,np.pi,Ngm,endpoint=False)))
    k_mx = np.column_stack((np.ones(Nmx)*np.pi,np.linspace(np.pi,0,Nmx)))
    k_to_compute = np.append(k_gm,k_mx,axis=0)
    Gamma3 = np.zeros((len(D_list),len(fH_list),Nk))
    for idd,D in enumerate(D_list):
        for ih,fH in enumerate(fH_list):
            filename = "Data/1to2decay_%s_J%.2f_D%.2f_H%.2f_Nk%d_L%d_eta%.7f.npy"%(Ham,J,D,fH,Ngm,L,eta)
            if Path(filename).is_file():
                print("Loading from file D=%.2f, fH=%.2f, %s"%(D,fH,Ham))
                Gamma3 = np.load(filename)
            else:
                if Ham=='AFM':
                    Hc = 4*J*S*(1-D)
                    parameters = (J,S,D,fH*Hc)
                    theta = fs.theta_afm(*parameters)
                    epsilon_ = fs.epsilon_afm
                    Ak_ = fs.Ak_afm
                else:
                    lam = -D
                    Hc = 4*J*S*(1-lam)
                    parameters = (J,S,lam,fH*Hc)
                    theta = fs.theta_fm(*parameters)
                    epsilon_ = fs.epsilon_fm
                    Ak_ = fs.Ak_fm
                gk = fs.gamma(k_to_compute)
                ek = epsilon_(gk,*parameters)
                gq = fs.gamma(K_grid)
                eq = epsilon_(gq,*parameters)
                for ik in tqdm(range(Nk)):
                    k = k_to_compute[ik]
                    Gamma3[idd,ih,ik] = fs.Gamma3_(k,ek[ik],K_grid,eq,eta,Ak_,epsilon_,theta,*parameters)/L**2
            if save:
                np.save(filename,Gamma3)

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot()
    x = np.arange(Nk)
    colors = ['gray','orange','lime','aqua']
    for idd,D in enumerate(D_list):
        for ih,fH in enumerate(fH_list):
            if Ham=='FM':
                txt, v = (r'$\lambda$',-D)
            else:
                txt, v = ('D',D)
            ax.plot(x,abs(Gamma3[idd,ih]),label='%s, %s=%.2f, H/Hc=%.2f'%(Ham,txt,v,fH))
    ax.axvline(Ngm,color='r',ls='--')
    ax.set_xlim(0,Nk)
    ax.set_xticks([0,Ngm,Nk-1],[r'$\Gamma$',r'$M$',r'$X$'],size=20)
    ax.set_ylabel("Decay rate",size=20)
    ax.set_title(r"1->2 decay process, L=%d, eta=%.7f"%(L,eta),size=30)
#        ax.set_ylim(0,0.7)
    ax.legend(fontsize=20)
    figname = "Figures/1to2decay_%s_J%.2f_Nk%d_L%d_eta%.5f_D"%(Ham,J,Ngm,L,eta)
    for D in D_list:
        figname += "{:.3f}".format(D)
        if not D==D_list[-1]:
            figname += '_'
        else:
            figname += '_H'
    for fH in fH_list:
        figname += "{:.2f}".format(fH)
        if not fH==fH_list[-1]:
            figname += '_'
        else:
            figname += '.png'
    if save:
        fig.savefig(figname)
    plt.show()
    exit()
if 0: # Old
    """ Compute phase space of 1->3 magnon process """
    H = 0.
    if 1:
        lam = 0.5
        parameters = (J,S,lam,H)
        epsilon_ = fs.epsilon_they
    else:
        D = 0.1
        parameters = (J,S,D,H)
        epsilon_ = fs.epsilon_my
    res = np.zeros(L**2)
    gq = fs.gamma(K_grid)
    ep = epsilon_(gq,*parameters)
    for ik in tqdm(range(L**2)):
        k = K_grid[ik]
        ek = ep[ik]
        for iq in range(L**2):
            q = K_grid[iq]
            eq = ep[iq]
            gkmqmp = fs.gamma(k-q-K_grid)
            ekmqmp = epsilon_(gkmqmp,*parameters)
            spread = fs.lorentz(ek - (eq + ep + ekmqmp),eta)
            vals = spread[spread>1/np.pi/eta-0.1]
            res[ik] += len(vals)
#    res[res<0.5] = np.nan
    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(KX,KY,np.log(res.reshape(L,L)),cmap='plasma')
    plt.show()
    exit()
if 1:
    """ Compute decay rate of 1->3 magnon process """
    J = 1
    S = 1/2
    D_list = [-0.3,]
    fH_list = [0.]
    Ham = 'AFM'      #FM or AFM
    Nk = 31
    ind = int(sys.argv[1])
    L_list = [50,70,100,120,150,200,250,300]
    eta_list = [1e-4,1e-5,1e-6,1e-7,1e-8]
    L = L_list[ind//len(eta_list)]
    eta = eta_list[ind%len(eta_list)]
    print("Parameters: J=%d"%J)
    print("fH values: ",fH_list)
    print("D values: ",D_list)
    print("BZ linear size: ",L)
    print("Gaussian broadening: ",eta)
    print("Points in G-M cut: ",Nk)
    # use float32 everywhere for memory savings
    dtype = np.float64
    k_list = np.linspace(0, np.pi, L, endpoint=False, dtype=dtype)
    KX,KY = np.meshgrid(k_list, k_list, indexing='ij')
    K_grid = np.column_stack(([KX.ravel(), KY.ravel()])).astype(dtype)
    vals = np.linspace(0, np.pi, Nk, endpoint=True, dtype=dtype)
    k_to_compute = np.column_stack((vals,vals)).astype(dtype)
    Gamma4 = np.zeros((len(D_list), len(fH_list), Nk), dtype=np.float64)
    # Pre-compute gammas for all parameters
    gk = fs.gamma(k_to_compute).astype(dtype)               # shape (Nk,)
    g_grid = fs.gamma(K_grid).astype(dtype)                 # shape (L^2,)
    for idd,D in enumerate(D_list):
        for ih,fH in enumerate(fH_list):
            filename = f"Data/1to3decay_%s_J%.2f_D%.2f_H%.2f_Nk%d_L%d_eta{eta}.npy"%(Ham,J,D,fH,Nk,L)
            if Path(filename).is_file():
                print("Loading from file D=%.2f, fH=%.2f, %s"%(D,fH,Ham))
                Gamma4 = np.load(filename)
            else:
                if Ham=='AFM':
                    Hc = 4*J*S*(1-D)
                    parameters = (J,S,D,fH*Hc)
                    ps = fs.p_afm(*parameters)
                else:
                    lam = -D
                    Hc = 4*J*S*(1-lam)
                    parameters = (J,S,lam,fH*Hc)
                    ps = fs.p_fm(*parameters)
                # precompute gamma and epsilon for k_to_compute and entire grid
                eps_k_all = fs.epsilon(gk, ps, *parameters).astype(dtype)
                eps_grid = fs.epsilon(g_grid, ps, *parameters).astype(dtype)
                uk_grid, vk_grid = fs.bogoliubovFunctions(g_grid, eps_grid, ps, *parameters)
                # choose block size (tune to memory). For L=200 (N=40000), block=512 or 1024 is reasonable.
                # block^2 elements exist temporarily -> memory ~ block^2 * 4 bytes (float32)
                block = 1024
                for ik in tqdm(range(Nk)):
                    k = k_to_compute[ik].astype(dtype)
                    eps_k = eps_k_all[ik].astype(dtype)
                    gk_val = gk[ik].astype(dtype)
                    val = fs.Gamma4a_blocked(k, K_grid, gk_val, g_grid, eps_k, eps_grid, uk_grid, vk_grid,
                                             eta, ps, *parameters, block=block)
                    Gamma4[idd, ih, ik] = val / (L**4)
                    print(k[0]/np.pi)
                    print(Gamma4[idd,ih,ik])
                    print('--------------------------')
            if save:
                np.save(filename,Gamma4)

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot()
    x = np.arange(Nk)
    colors = ['gray','orange','lime','aqua']
    for idd,D in enumerate(D_list):
        for ih,fH in enumerate(fH_list):
            if Ham=='FM':
                txt, v = (r'$\lambda$',-D)
            else:
                txt, v = ('D',D)
            ax.plot(x,abs(Gamma4[idd,ih]),'k*-',label='%s, %s=%.2f, H/Hc=%.2f'%(Ham,txt,v,fH))
    ax.set_xlim(0,Nk)
    ax.set_xticks([0,Nk/4,Nk/2,Nk*3/4,Nk-1],[r'$\Gamma$','','','',r'$M$'],size=20)
    ax.set_xlim(Nk*k_to_compute[0][0]/np.pi,Nk*k_to_compute[-1][0]/np.pi)
    ax.set_ylabel("Decay rate",size=20)
    ax.set_title(r"1->3 decay process, L=%d, eta=%.7f"%(L,eta),size=30)
#        ax.set_ylim(0,0.7)
    ax.legend(fontsize=20)
    figname = f"Figures/1to3decay_%s_J%.2f_Nk%d_L%d_eta{eta}_D"%(Ham,J,Nk,L)
    for D in D_list:
        figname += "{:.3f}".format(D)
        if not D==D_list[-1]:
            figname += '_'
        else:
            figname += '_H'
    for fH in fH_list:
        figname += "{:.2f}".format(fH)
        if not fH==fH_list[-1]:
            figname += '_'
        else:
            figname += '.png'
    if save:
        fig.savefig(figname)
    plt.show()
    exit()










































