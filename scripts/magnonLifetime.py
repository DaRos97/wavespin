""" Here we compute the magnon decay rate.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import functions as fs
from pathlib import Path
import sys

J = 1
S = 1/2
#Hc = 4*J*S*(1-D)
#H_list = [i*Hc for i in [0.3,0.5,0.7,0.9]]
D_list = [0.3,]#0.5,0.8]
D = D_list[0]
H_list = [0.]
H = H_list[0]

L_list = [20,50,70,100,150,200,300,400]
eta_list = [0.1,0.05,0.01,0.005,0.001,0.0005]

ind = int(sys.argv[1])
il = ind//len(eta_list)
ie = ind%len(eta_list)

prec0 = 1e-8
L = L_list[il]
eta = eta_list[ie]
save = True
print("Parameters: J=%d, D=%.2f"%(J,D))
print("H values: ",H_list)
print("D values: ",D_list)
print("BZ linear size: ",L)
print("Gaussian broadening: ",eta)

k_list = np.linspace(-np.pi*0,np.pi,L,endpoint=False)
KX,KY = np.meshgrid(k_list,k_list,indexing='ij')
K_grid_they = np.column_stack(([KX.ravel(),KY.ravel()]))
#k_list = np.linspace(0,2*np.pi,L,endpoint=False)
#KX,KY = np.meshgrid(k_list,k_list,indexing='ij')
K_grid_my = K_grid_they#np.column_stack(([KX.ravel(),KY.ravel()]))

if 0:   # Plot dispersion
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(121,projection='3d')
    energies_they = fs.epsilon_they(K_grid_they,*parameters)
    ax.plot_surface(KX,KY,energies_they.reshape(L,L),cmap='plasma')
    ax = fig.add_subplot(122,projection='3d')
    #energies_my = fs.epsilon_my(K_grid,*parameters)
    energies_my = fs.Ak_they(K_grid_they,*parameters) - energies_they
    ax.plot_surface(KX,KY,energies_my.reshape(L,L),cmap='plasma')
    plt.show()
    exit()
    ### Check shifts
    ax = fig.add_subplot(132,projection='3d')
    ind = 6220
    print(ind)
    k = K_grid[ind]
    ek2 = fs.epsilon(k-K_grid,*parameters)
    print(np.argmin(ek2))
    ax.plot_surface(KX,KY,ek2.reshape(L,L),cmap='plasma')
    ax = fig.add_subplot(133,projection='3d')
    ind0 = L*L//2-ind
    i0, j0 = ind0//L, ind0%L
    ek3 = np.roll(np.roll(ek.reshape(L,L),-i0,axis=0),-j0,axis=1).reshape(-1)
    print(np.argmin(ek3))
    ax.plot_surface(KX,KY,ek3.reshape(L,L),cmap='plasma')
    plt.show()
    exit()

if 0:
    """ Compute decay rate of 1->2 magnon processes """
    Ngm = 200
    Nmx = int(Ngm/np.sqrt(2))
    Nk = Ngm+Nmx
    k_gm = np.column_stack((np.linspace(0,np.pi,Ngm,endpoint=False),np.linspace(0,np.pi,Ngm,endpoint=False)))
    k_mx = np.column_stack((np.ones(Nmx)*np.pi,np.linspace(np.pi,0,Nmx)))
    k_to_compute_they = np.append(k_gm,k_mx,axis=0)
    k_gm = np.column_stack((np.linspace(np.pi,0*np.pi,Ngm,endpoint=False),np.linspace(np.pi,0*np.pi,Ngm,endpoint=False)))
    k_mx = np.column_stack((np.ones(Nmx)*0*np.pi,np.linspace(0*np.pi,np.pi,Nmx)))
    k_to_compute_my = k_to_compute_they#np.append(k_gm,k_mx,axis=0)
    filename = "Data/1to2decay_J%.2f_D%.2f_Nk%d_L%d_eta%.5f_H"%(J,D,Ngm,L,eta)
    for H in H_list:
        filename += "{:.3f}".format(H)
        if not H==H_list[-1]:
            filename += '_'
        else:
            filename += '.npz'
    if not Path(filename).is_file():
        """ Compute decay rate of 1->2 magnon processes """
        Gamma3_my = np.zeros((len(H_list),Nk))
        Gamma3_they = np.zeros((len(H_list),Nk))
        for ih,H in enumerate(H_list):
            parameters = (J,S,D,H)
            gk_my = fs.gamma(k_to_compute_my,*parameters)
            ek_my = fs.epsilon_my(k_to_compute_my,gk_my,*parameters)
            gq_my = fs.gamma(K_grid_my,*parameters)
            eq_my = fs.epsilon_my(K_grid_my,gq_my,*parameters)
            gk_they = fs.gamma(k_to_compute_they,*parameters)
            ek_they = fs.epsilon_they(k_to_compute_they,gk_they,*parameters)
            gq_they = fs.gamma(K_grid_my,*parameters)
            eq_they = fs.epsilon_they(K_grid_they,gq_they,*parameters)
            for ind in tqdm(range(Nk)):
                k_my = k_to_compute_my[ind]
                k_they = k_to_compute_they[ind]
                if not np.linalg.norm(k_my-np.array([np.pi,np.pi]))<prec0:     #skip k=0 -> gapless -> problems
                    Gamma3_my[ih,ind] = fs.Gamma3_(k_my,ek_my[ind],K_grid_my,eq_my,eta,fs.Ak_my,fs.epsilon_my,fs.theta_my,*parameters)/L**2
                if not np.linalg.norm(k_they)<prec0:     #skip k=0 -> gapless -> problems
                    Gamma3_they[ih,ind] = fs.Gamma3_(k_they,ek_they[ind],K_grid_they,eq_they,eta,fs.Ak_they,fs.epsilon_they,fs.theta_they,*parameters)/L**2
        if save:
            np.savez(filename,my=Gamma3_my,they=Gamma3_they)
    else:
        print("Loading from file")
        Gamma3_my = np.load(filename)['my']
        Gamma3_they = np.load(filename)['they']

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot()
    x = np.arange(Nk)
    colors_my = ['gray','orange','lime','aqua']
    colors_they = ['k','r','g','b']
    for ih,H in enumerate(H_list):
        ax.plot(x,abs(Gamma3_my[ih]),color=colors_my[ih],label='AFM-staggered H: %.2f'%H)
        ax.plot(x,abs(Gamma3_they[ih]),color=colors_they[ih],label='FM-uniform H: %.2f'%H)
    ax.axvline(Ngm,color='r',ls='--')
    ax.set_xlim(0,Nk)
    ax.set_xticks([0,Ngm,Nk-1],[r'$\Gamma$',r'$M$',r'$X$'],size=20)
    ax.set_ylabel("Decay rate",size=20)
    ax.set_title("1->2 decay process, L=%d, eta=%.7f"%(L,eta),size=30)
#        ax.set_ylim(0,0.7)
    ax.legend(fontsize=20)
    figname = "Figures/1to2decay_J%.2f_D%.2f_Nk%d_L%d_eta%.5f_H"%(J,D,Ngm,L,eta)
    for H in H_list:
        figname += "{:.3f}".format(H)
        if not H==H_list[-1]:
            figname += '_'
        else:
            figname += '.png'
    fig.savefig(figname)
    plt.show()

if 1:
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










































