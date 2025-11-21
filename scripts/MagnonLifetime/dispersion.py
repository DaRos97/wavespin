import functions as fs
import matplotlib.pyplot as plt
import numpy as np

J = 20
S = 0.5
D = 0
H = 0
L = 200

pars = (J,S,D,H)
ps = fs.p_afm(*pars)

dtype = np.float64
k_list = np.linspace(0, 2*np.pi, L, endpoint=False, dtype=dtype)
KX,KY = np.meshgrid(k_list, k_list, indexing='ij')
K_grid = np.column_stack(([KX.ravel(), KY.ravel()])).astype(dtype)

gk = fs.gamma(K_grid)

energies = fs.epsilon(gk,ps,*pars)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(KX,KY,energies.reshape(L,L),cmap='viridis')
ax.set_xlabel("kx")
ax.set_ylabel("ky")
ax.set_zlabel("energy")
fig.tight_layout()
plt.show()
