"""
dynamical_systems.py — Nonlinear dynamics and stability analysis for QFM.

Implements trajectory action, Lyapunov exponent approximations, and 
Loschmidt Echo for evaluating the stability of learned flows.
"""
import torch
import numpy as np
from typing import List, Tuple
from .metrics import bures_distance, von_neumann_entropy, uhlmann_fidelity

def qfm_trajectory_action(rhos: List[torch.Tensor]) -> float:
    """
    Computes the Information Geometric Action of the discrete trajectory:
       A = sum_{t=0}^{T-1} d_Bures(ρ_t, ρ_{t+1})^2
    This represents the kinetic energy / transport cost of the flow.
    """
    action = 0.0
    for tau in range(len(rhos) - 1):
        d_B = bures_distance(rhos[tau], rhos[tau+1])
        action += float(d_B)**2
    return action

def optimal_bures_geodesic_cost(rho_0: torch.Tensor, rho_T: torch.Tensor, T_steps: int) -> float:
    """
    Computes the minimum theoretically possible action (geodesic cost) 
    between the start and end states across T discrete steps:
        A_optimal = d_Bures(ρ_0, ρ_T)^2 / T
    """
    d_B = float(bures_distance(rho_0, rho_T))
    return (d_B ** 2) / T_steps

def entropy_production(rhos: List[torch.Tensor]) -> List[float]:
    """
    Computes the incremental change in von Neumann entropy at each step:
        ΔS(τ) = S(ρ_{τ+1}) - S(ρ_τ)
    Positive values indicate irreversible mixing / heat generation.
    Returns list of length T.
    """
    delta_S = []
    for tau in range(len(rhos) - 1):
        S_t0 = von_neumann_entropy(rhos[tau])
        S_t1 = von_neumann_entropy(rhos[tau+1])
        delta_S.append(float(S_t1 - S_t0))
    return delta_S

def loschmidt_echo_perturbation(trainer, initial_pure: torch.Tensor, target_fn, T_steps: int, epsilon: float = 1e-3) -> List[float]:
    """
    Measure chaotic divergence by applying a small random unitary perturbation 
    to the initial state, and pushing both through the exact same frozen QFM unitaries.
    
    Returns: The Uhlmann fidelity over time (Loschmidt Echo) F(ρ_t(0), ρ_t(ε)).
    Decay of echo indicates chaotic instability in the learned flow.
    """
    dim = initial_pure.shape[-1]
    n_data = int(np.log2(dim))
    
    # 1. Create a perturbed initial state U(eps) |psi>
    H_rand = torch.randn((dim, dim), dtype=torch.complex128)
    H_rand = 0.5 * (H_rand + H_rand.conj().T)
    U_eps = torch.linalg.matrix_exp(-1j * epsilon * H_rand)
    
    psi_eps = torch.matmul(initial_pure, U_eps.T) # Batched matrix mult
    psi_eps = psi_eps / torch.linalg.vector_norm(psi_eps, dim=-1, keepdim=True)
    
    # 2. Replay the sequence using the learned models in trainer
    current_orig = initial_pure.clone()
    current_pert = psi_eps.clone()
    
    fidelities = []
    
    from .utils import state_vector_to_density_matrix, partial_trace_pure_to_mixed
    
    for tau in range(T_steps):
        model, n_ancilla_used = trainer.models[tau]
        
        if n_ancilla_used == 0:
            out_orig = model(current_orig)
            out_pert = model(current_pert)
            
            rho_orig = torch.stack([state_vector_to_density_matrix(p) for p in out_orig]).mean(dim=0)
            rho_pert = torch.stack([state_vector_to_density_matrix(p) for p in out_pert]).mean(dim=0)
            
            current_orig = torch.stack([p.detach() for p in out_orig])
            current_pert = torch.stack([p.detach() for p in out_pert])
            
        else: # U_n_a used
            ancilla_rot = (tau+1) * torch.pi / T_steps
            # Padding
            pad_orig = trainer.pad_state_with_ancilla(current_orig, n_data, n_ancilla_used)
            pad_pert = trainer.pad_state_with_ancilla(current_pert, n_data, n_ancilla_used)
            
            out_orig_full = model(pad_orig, ancilla_rot=ancilla_rot)
            out_pert_full = model(pad_pert, ancilla_rot=ancilla_rot)
            
            rho_orig = torch.stack([partial_trace_pure_to_mixed(p, n_data, n_ancilla_used) for p in out_orig_full]).mean(dim=0)
            rho_pert = torch.stack([partial_trace_pure_to_mixed(p, n_data, n_ancilla_used) for p in out_pert_full]).mean(dim=0)
            
            # Collapse for next step
            dim_data = 2**n_data
            col_orig, col_pert = [], []
            for p_o, p_p in zip(out_orig_full, out_pert_full):
                branch_o = p_o[:dim_data]; col_orig.append(branch_o / torch.linalg.vector_norm(branch_o))
                branch_p = p_p[:dim_data]; col_pert.append(branch_p / torch.linalg.vector_norm(branch_p))
            current_orig = torch.stack(col_orig).detach()
            current_pert = torch.stack(col_pert).detach()
            
        F = float(uhlmann_fidelity(rho_orig, rho_pert))
        fidelities.append(F)
        
    return fidelities

def quantum_work_heat_breakdown(rhos: List[torch.Tensor], Hs: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
    """
    Computes the Work and Heat breakdown via the First Law of Quantum Thermodynamics:
        ΔE = ΔW + ΔQ
        ΔW = Tr(ρ_t (H_{t+1} - H_t))
        ΔQ = Tr((ρ_{t+1} - ρ_t) H_{t+1})
    """
    work = []
    heat = []
    for t in range(len(rhos) - 1):
        dW = torch.real(torch.trace(rhos[t] @ (Hs[t+1] - Hs[t])))
        dQ = torch.real(torch.trace((rhos[t+1] - rhos[t]) @ Hs[t+1]))
        work.append(float(dW))
        heat.append(float(dQ))
    return work, heat

def calculate_ergotropy(rho: torch.Tensor, H: torch.Tensor) -> float:
    """
    Computes Ergotropy: the maximum work extractable from a state via unitary operations.
    W_ext = Tr(ρ H) - min_U Tr(U ρ U† H)
    The minimum is achieved by a 'passive state' σ_ρ (eigenvalues of ρ in reverse order of H).
    """
    ev_rho, _ = torch.linalg.eigh(rho)
    ev_H, _ = torch.linalg.eigh(H)
    
    # Sort rho eigenvalues descending, H eigenvalues ascending
    rho_sorted = torch.sort(ev_rho.real, descending=True).values
    H_sorted = torch.sort(ev_H.real, descending=False).values
    
    passive_energy = torch.sum(rho_sorted * H_sorted)
    current_energy = torch.real(torch.trace(rho @ H))
    
    return float(current_energy - passive_energy)

def coherence_evolution(rhos: List[torch.Tensor]) -> List[float]:
    """
    Computes the l1-norm of coherence: sum of absolute values of off-diagonal elements in the energy basis.
    Approximated here in the computational basis.
    """
    coherences = []
    for rho in rhos:
        off_diag = rho.clone()
        off_diag.fill_diagonal_(0)
        coherences.append(float(torch.sum(torch.abs(off_diag))))
    return coherences
