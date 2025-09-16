""" Here we compute the magnon decay rate.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions as fs
from pathlib import Path
import sys

save = True

J = 1
S = 1/2
D_list = [0.0,]
fH_list = [0.1,0.5,0.9]
prec0 = 1e-8
if 0:       #Decide size and eta from input line
    L_list = [20,50,70,100,150,200,300,400]
    eta_list = [0.1,0.05,0.01,0.005,0.001,0.0005]

    ind = int(sys.argv[1])
    il = ind//len(eta_list)
    ie = ind%len(eta_list)
    L = L_list[il]
    eta = eta_list[ie]
else:
    L = 700
    eta = 0.005

print("Parameters: J=%d"%J)
print("fH values: ",fH_list)
print("D values: ",D_list)
print("BZ linear size: ",L)
print("Gaussian broadening: ",eta)

k_list = np.linspace(0,np.pi,L,endpoint=False)
KX,KY = np.meshgrid(k_list,k_list,indexing='ij')
K_grid = np.column_stack(([KX.ravel(),KY.ravel()]))

if 0:   #Plot curvature at different fields
    D_list = [0.,0.3,0.7,0.9]
    fH_list = [0,]#0.5,0.9]
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
            gk = fs.gamma(k_to_compute,*parameters)
            ek = fs.epsilon_they(k_to_compute,gk,*parameters)
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
            gk = fs.gamma(k_to_compute,*parameters)
            ek = fs.epsilon_my(k_to_compute,gk,*parameters)
            ax.plot(np.arange(Nk),ek,label=r"H/Hc=%.2f, D=%.2f"%(fH,D))
    ax.axvline(Ngm,color='r',ls='--')
    ax.set_xticks([0,Ngm,Nk-1],[r'$\Gamma$',r'$M$',r'$X$'],size=20)
    ax.legend()
    ax.set_title("Staggered field")
    ax.set_ylim(0,4)
    plt.show()
    exit()

if 1:
    """ Compute decay rate of 1->2 magnon processes """
    Ngm = 100
    Nmx = int(Ngm/np.sqrt(2))
    Nk = Ngm+Nmx
    k_gm = np.column_stack((np.linspace(0,np.pi,Ngm,endpoint=False),np.linspace(0,np.pi,Ngm,endpoint=False)))
    k_mx = np.column_stack((np.ones(Nmx)*np.pi,np.linspace(np.pi,0,Nmx)))
    k_to_compute = np.append(k_gm,k_mx,axis=0)
    Gamma3_my = np.zeros((len(D_list),len(fH_list),Nk))
    Gamma3_they = np.zeros((len(D_list),len(fH_list),Nk))
    for idd,D in enumerate(D_list):
        for ih,fH in enumerate(fH_list):
            filename = "Data/1to2decay_J%.2f_D%.2f_H%.2f_Nk%d_L%d_eta%.7f.npz"%(J,D,fH,Ngm,L,eta)
            if Path(filename).is_file():
                print("Loading from file")
                Gamma3_my = np.load(filename)['my']
                Gamma3_they = np.load(filename)['they']
            else:
                Hc = 4*J*S*(1-D)
                parameters_my = (J,S,D,fH*Hc)
                theta_my = fs.theta_my(*parameters_my)
                gk_my = fs.gamma(k_to_compute,*parameters_my)
                ek_my = fs.epsilon_my(k_to_compute,gk_my,*parameters_my)
                gq_my = fs.gamma(K_grid,*parameters_my)
                eq_my = fs.epsilon_my(K_grid,gq_my,*parameters_my)
                #
                lam = -D
                Hc = 4*J*S*(1-lam)
                parameters_they = (J,S,lam,fH*Hc)
                theta_they = fs.theta_my(*parameters_they)
                gk_they = fs.gamma(k_to_compute,*parameters_they)
                ek_they = fs.epsilon_they(k_to_compute,gk_they,*parameters_they)
                gq_they = fs.gamma(K_grid,*parameters_they)
                eq_they = fs.epsilon_they(K_grid,gq_they,*parameters_they)
                for ik in tqdm(range(Nk)):
                    k = k_to_compute[ik]
                    if not np.linalg.norm(k-np.array([np.pi,np.pi]))<prec0:     #skip k=0 -> gapless -> problems
                        Gamma3_my[idd,ih,ik] = fs.Gamma3_(k,ek_my[ik],K_grid,eq_my,eta,fs.Ak_my,fs.epsilon_my,theta_my,*parameters_my)/L**2
                    if not np.linalg.norm(k)<prec0:     #skip k=0 -> gapless -> problems
                        Gamma3_they[idd,ih,ik] = fs.Gamma3_(k,ek_they[ik],K_grid,eq_they,eta,fs.Ak_they,fs.epsilon_they,theta_they,*parameters_they)/L**2
            if save:
                np.savez(filename,my=Gamma3_my,they=Gamma3_they)

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot()
    x = np.arange(Nk)
    colors_my = ['gray','orange','lime','aqua']
    colors_they = ['k','r','g','b']
    for idd,D in enumerate(D_list):
        for ih,fH in enumerate(fH_list):
            ax.plot(x,abs(Gamma3_my[idd,ih]),label='AFM-staggered D=%.2f, H/Hc=%.2f'%(D,fH))
    if 0:
        for idd,D in enumerate(D_list):
            lam = -D
            for ih,fH in enumerate(fH_list):
                ax.plot(x,abs(Gamma3_they[idd,ih]),label=r'FM-uniform $\lambda$=%.2f, H/Hc=%.2f'%(lam,fH))
    ax.axvline(Ngm,color='r',ls='--')
    ax.set_xlim(0,Nk)
    ax.set_xticks([0,Ngm,Nk-1],[r'$\Gamma$',r'$M$',r'$X$'],size=20)
    ax.set_ylabel("Decay rate",size=20)
    ax.set_title(r"1->2 decay process, L=%d, eta=%.7f"%(L,eta),size=30)
#        ax.set_ylim(0,0.7)
    ax.legend(fontsize=20)
    figname = "Figures/1to2decay_J%.2f_Nk%d_L%d_eta%.5f_D"%(J,Ngm,L,eta)
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

if 0:
    """ Compute decay rate of 1->2 magnon processes """
    Nk = 30
    k_to_compute_they = np.column_stack((np.linspace(0,np.pi,Nk,endpoint=False),np.linspace(0,np.pi,Nk,endpoint=False)))
    k_to_compute_my = k_to_compute_they#np.column_stack((np.linspace(0,np.pi,Ngm,endpoint=False),np.linspace(0,np.pi,Ngm,endpoint=False)))
    filename = "Data/1to3decay_J%.2f_H%.2f_Nk%d_L%d_eta%.5f_D"%(J,H,Nk,L,eta)
    for D in D_list:
        filename += "{:.3f}".format(D)
        if not D==D_list[-1]:
            filename += '_'
        else:
            filename += '.npz'
    if not Path(filename).is_file():
        """ Compute decay rate of 1->2 magnon processes """
        Gamma4a_my = np.zeros((len(H_list),Nk))
        Gamma4a_they = np.zeros((len(H_list),Nk))
        for ind_d,D in enumerate(D_list):
            parameters = (J,S,D,H)
            gk_they = fs.gamma(k_to_compute_they,*parameters)
            ek_they = fs.epsilon_they(k_to_compute_they,gk_they,*parameters)
            gq_they = fs.gamma(K_grid_my,*parameters)
            eq_they = fs.epsilon_they(K_grid_they,gq_they,*parameters)
            for ind in tqdm(range(Nk)):
                k_they = k_to_compute_they[ind]
                if not np.linalg.norm(k_they)<prec0:     #skip k=0 -> gapless -> problems
                    Gamma4a_they[ind_d,ind] = fs.Gamma4a_(k_they,ek_they[ind],K_grid_they,eq_they,eta,fs.Ak_they,fs.epsilon_they,fs.theta_they,*parameters)/L**4
        if save:
            np.savez(filename,my=Gamma4a_my,they=Gamma4a_they)
    else:
        print("Loading from file")
        Gamma4a_my = np.load(filename)['my']
        Gamma4a_they = np.load(filename)['they']

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot()
    x = np.arange(Nk)
    colors_my = ['gray','orange','lime','aqua']
    colors_they = ['k','r','g']
    for ind_d,D in enumerate(D_list):
#        ax.plot(x,abs(Gamma4a_my[ind_d]),color=colors_my[ind_d],label='AFM-staggered H: %.2f'%H)
        ax.plot(x,abs(Gamma4a_they[ind_d]),marker='^',color=colors_they[ind_d],label='FM-uniform H: %.2f'%H)
    ax.set_xlim(0,Nk)
    ax.set_xticks([0,Nk-1],[r'$\Gamma$',r'$M$'],size=20)
    ax.set_ylabel("Decay rate",size=20)
    ax.set_title("1->3 decay process, L=%d, eta=%.7f"%(L,eta),size=30)
#    ax.set_ylim(0,0.7)
    ax.legend(fontsize=20)
    figname = "Figures/1to3decay_J%.2f_D%.2f_Nk%d_L%d_eta%.5f_H"%(J,D,Nk,L,eta)
    for H in H_list:
        figname += "{:.3f}".format(H)
        if not H==H_list[-1]:
            figname += '_'
        else:
            figname += '.png'
    fig.savefig(figname)
#    plt.show()










































