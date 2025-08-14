import numpy as np
from wavespin.classicSpins.montecarlo import *


# --------------------- Example usage ---------------------
if __name__ == "__main__":
    p = Params(
        L=10, J1=1.0, J2=0., g=0., h=0,
        T_max=3.0, T_min=1e-3, alpha=0.95,
        sweeps_per_T_high=20, sweeps_per_T_low=100,
        overrelax_every=1, proposal_step=0.35, seed=42,
    )
    sim = XXZJ1J2MC(p)
    hist = sim.anneal(verbose=True)
    E_over_N = sim.total_energy() / sim.N
    m = sim.magnetization()
    ms = sim.staggered_mz()
    print("Final results:")
    print(f"E/N = {E_over_N:.8f}")
    print(f"m = {m}")
    print(f"m_staggered_z = {ms:.6f}")

    # Optional: locate dominant ordering wavevector from Szz(q)
    S_q, kx, ky = sim.structure_factor(component='z')
    # Find the peak location (rough)
    idx = np.unravel_index(np.argmax(S_q), S_q.shape)
    qx, qy = kx[idx[1]], ky[idx[0]]
    print(f"Peak in Szz(q) near q=({qx:.3f}, {qy:.3f}) (expect (pi,pi) for staggered z-order)")

