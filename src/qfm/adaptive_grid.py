"""
adaptive_grid.py — Dynamic Precision Curvature-Based Time Stepping for QFM

Instead of uniform tau spacing (e.g., 0.0, 0.1, 0.2 ... 1.0), this algorithm 
evaluates the local state velocity ||dρ/dτ||_Bures. If the velocity exceeds
a threshold, it inserts intermediate steps to maintain fidelity, while skipping 
over regions where the Hamiltonian changes slowly.
"""

import torch
import numpy as np
from .metrics import bures_distance

def adaptive_tau_schedule(H_fn, initial_ensemble, n_qubits, base_steps=10, max_steps=30, v_threshold=0.2):
    """
    Simulates a fast proxy-trajectory (e.g. ground states of H_fn) 
    to estimate Bures velocity and generate an adaptive non-uniform tau grid.
    
    H_fn: callable H(tau) taking tau in [0, 1].
    base_steps: initial uniform grid resolution
    v_threshold: maximum allowed Bures distance between steps.
    """
    # 1. Start with a base uniform grid
    grid = list(np.linspace(0.0, 1.0, base_steps + 1))
    
    # 2. Compute ground states at each grid point
    def get_gs_density(tau):
        H = H_fn(tau)
        ev, evec = torch.linalg.eigh(H)
        gs = evec[:, 0]
        return torch.outer(gs, gs.conj())
        
    rhos = [get_gs_density(t) for t in grid]
    
    # 3. Adaptively subdivide intervals until all velocities are below threshold
    changed = True
    iterations = 0
    while changed and len(grid) < max_steps and iterations < 5:
        changed = False
        new_grid = [grid[0]]
        new_rhos = [rhos[0]]
        
        for i in range(len(grid) - 1):
            t0, t1 = grid[i], grid[i+1]
            r0, r1 = rhos[i], rhos[i+1]
            
            # Compute Bures velocity
            d_B = float(bures_distance(r0, r1))
            
            if d_B > v_threshold:
                # Subdivide
                t_mid = (t0 + t1) / 2.0
                r_mid = get_gs_density(t_mid)
                
                new_grid.extend([t_mid, t1])
                new_rhos.extend([r_mid, r1])
                changed = True
            else:
                new_grid.append(t1)
                new_rhos.append(r1)
                
        grid = new_grid
        rhos = new_rhos
        iterations += 1
        
    return grid
