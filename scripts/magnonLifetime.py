import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def theta(*pars):
    J,S,D,H = pars
    Hc = 4*J*S*(1-D)
    return np.arcsin(H/Hc)
def gamma(k,*pars):
    J,S,D,H = pars
    if len(k.shape)==1:
        return 1/2*(np.cos(k[0]) + np.cos(k[1]))
    elif len(k.shape)==2:
        return 1/2*(np.cos(k[:,0]) + np.cos(k[:,1]))
    else:
        print("error in gamma")
        exit()
def epsilon(k,*pars):
    J,S,D,H = pars
    gk = gamma(k,*pars)
    th = theta(*pars)
    return 4*J*S*np.sqrt((1-gk)*(1-gk*(D*np.cos(th)**2+np.sin(th)**2)))
def Ak(k,*pars):
    J,S,D,H = pars
    gk = gamma(k,*pars)
    th = theta(*pars)
    return 4*J*S*(1-1/2*gk*(1+D*np.cos(th)**2+np.sin(th)**2))
def uk_(k,*pars):
    ak = Ak(k,*pars)
    ek = epsilon(k,*pars)
    if ek==0:
        print("zero dispersion for k=",k)
    return np.sqrt((ak+ek)/2/ek)
def vk_(k,*pars):
    ak = Ak(k,*pars)
    ek = epsilon(k,*pars)
    gk = gamma(k,*pars)
    absGk = np.absolute(gk)
    if absGk==0:
        absGk = 1
    if ek==0:
        print("zero dispersion for k=",k)
    return -np.sqrt((ak-ek)/2/ek)*gk/absGk

def V3(k,q,*pars):
    J,S,D,H = pars
    th = theta(*pars)
    gk = gamma(k,*pars)
    uk = uk_(k,*pars)
    vk = vk_(k,*pars)
    gq = gamma(q,*pars)
    uq = uk_(q,*pars)
    vq = vk_(q,*pars)
    gkmq = gamma(k-q,*pars)
    ukmq = uk_(k-q,*pars)
    vkmq = vk_(k-q,*pars)
    return H*np.cos(th)/np.sqrt(2*S) * (  gk*(uk+vk)*(uq*vkmq+vq*ukmq)
                                        + gq*(uq+vq)*(uk*ukmq+vk*vkmq)
                                        + gkmq*(ukmq+vkmq)*(uk*uq+vk*vq)   )

def getQs3(k,k_list,*pars):
    """ Compute which q values are on shell with k. """
    factor = 1e-1
    L = len(k_list)
    ek = epsilon(k,*pars)
    Ek = epsilon()
    Qs = []
    for ind in range(L**2):
        ix,iy = ind//L, ind%L
        q = np.array([k_list[ix],k_list[iy]])
        eq = epsilon(q,*pars)
        ekmq = epsilon(k-q,*pars)
        if abs(ek-eq-ekmq)<factor:
            Qs.append(q)
    return Qs

J = 10
S = 1/2
D = 1/J
H = 1
parameters = (J,S,D,H)

L = 30
k_list = np.linspace(0,np.pi/10,L)

disp = False

Gamma3 = np.zeros((L,L))
for ind in tqdm(range(L**2)):
    ix,iy = ind//L, ind%L
    k = np.array([k_list[ix],k_list[iy]])
    allowedQs = getQs3(k,k_list,*parameters)
    for q in allowedQs:
        Gamma3[ix,iy] += np.pi/2*V3(k,q,*parameters)**2
    if disp:
        print("k: ",k)
        print(allowedQs)
        print(Gamma3[ix,iy])
        input()

KX,KY = np.meshgrid(k_list,k_list,indexing='ij')

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(KX,KY,Gamma3,cmap='plasma')

plt.show()

