import torch
from ..utils import state_vector_to_density_matrix

def generate_initial_ensemble(M=100):
    """
    S_0 = { |psi_0^m> = e^{-i \sigma_x G_m} |0> , G_m ~ U(0, 2\pi) }
    Returns tensor of pure state vectors.
    """
    G_m = torch.rand(M) * 2 * torch.pi
    states = []
    for g in G_m:
        # e^{-i \sigma_x G_m} = cos(G_m)I - i sin(G_m)\sigma_x
        # Applied to |0> -> cos(G_m)|0> - i sin(G_m)|1>
        state = torch.tensor([torch.cos(g), -1j * torch.sin(g)], dtype=torch.complex128)
        states.append(state)
    return torch.stack(states), G_m

def get_target_ensemble_rhos(initial_states_pure, tau, T_steps=20):
    """
    Target density matrices at step tau rotated around Z axis.
    """
    total_rot = torch.pi * tau / T_steps
    c = torch.tensor(-1j * total_rot, dtype=torch.complex128)
    
    # e^{-i \sigma_z G_tau} = diag(e^{-i G_tau}, e^{i G_tau})
    rot_mat = torch.tensor([
        [torch.exp(c), 0],
        [0, torch.exp(-c)]
    ], dtype=torch.complex128)
    
    target_rhos = []
    for state in initial_states_pure:
        rotated_state = torch.matmul(rot_mat, state)
        target_rhos.append(state_vector_to_density_matrix(rotated_state))
    return torch.stack(target_rhos)
