""" Here we compute the magnon decay rate.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions as fs
from pathlib import Path
import sys

save = True

if 0:   #Plot curvature at different fields
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
if 0:
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
if 0:
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
if 0:
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
    L_list = [50,70,100,120,150]
    eta_list = [1e-4,1e-5,1e-6,1e-7]
    L = L_list[ind//len(eta_list)]
    eta = eta_list[ind%len(eta_list)]
    k_list = np.linspace(0,np.pi,L,endpoint=False)
    KX,KY = np.meshgrid(k_list,k_list,indexing='ij')
    K_grid = np.column_stack(([KX.ravel(),KY.ravel()]))
    print("Parameters: J=%d"%J)
    print("fH values: ",fH_list)
    print("D values: ",D_list)
    print("BZ linear size: ",L)
    print("Gaussian broadening: ",eta)
    print("Points in G-M cut: ",Nk)
    vals = np.linspace(0*np.pi,np.pi,Nk,endpoint=1)#True)
    k_to_compute = np.column_stack((vals,vals))
    Gamma4 = np.zeros((len(D_list),len(fH_list),Nk))
    for idd,D in enumerate(D_list):
        for ih,fH in enumerate(fH_list):
            filename = "Data/1to3decay_%s_J%.2f_D%.2f_H%.2f_Nk%d_L%d_eta%.7f.npy"%(Ham,J,D,fH,Nk,L,eta)
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
                gk = fs.gamma(k_to_compute)
                eps_k = fs.epsilon(gk,ps,*parameters)
                #
                g_grid = fs.gamma(K_grid)
                gq = g_grid[:,None]
                gp = g_grid[None,:]
                gppq = fs.gamma(K_grid[:,None] + K_grid[None,:])
                eps_grid = fs.epsilon(g_grid,ps,*parameters)
                for ik in tqdm(range(Nk)):      #this maybe can be also parallelized
                    k = k_to_compute[ik]
                    if k[0]==np.pi:
                        continue
                    Gamma4[idd,ih,ik] = fs.Gamma4a_(k,K_grid,gk[ik],g_grid,gppq,eps_k[ik],eps_grid,eta,ps,*parameters)/L**4
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
    figname = "Figures/1to3decay_%s_J%.2f_Nk%d_L%d_eta%.5f_D"%(Ham,J,Nk,L,eta)
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










































