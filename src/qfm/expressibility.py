"""
expressibility.py — Evaluation of Ansatz capacity and Haar-randomness.

Follows the foundations of Sim et al. (2019) to quantify 
expressibility (KL divergence to Haar) and entangling capability.
"""
import torch
import numpy as np
from typing import Optional


def sample_random_states(qnode, n_params: int, N: int = 1000) -> torch.Tensor:
    """
    Sample N random output states from the circuit with uniform random parameters.
    Returns Tensor of shape (N, d).
    """
    states = []
    for _ in range(N):
        theta = torch.rand(n_params) * 2 * np.pi
        psi   = qnode(theta).detach()
        states.append(psi)
    return torch.stack(states)


def pairwise_fidelities(states: torch.Tensor, N_pairs: int = 5000) -> np.ndarray:
    """
    Compute N_pairs random pairwise fidelities F_{ab} = |⟨ψ_a|ψ_b⟩|²
    from the provided state set.
    """
    N = states.shape[0]
    fids = []
    idx_a = np.random.randint(0, N, size=N_pairs)
    idx_b = np.random.randint(0, N, size=N_pairs)
    for a, b in zip(idx_a, idx_b):
        if a == b:
            continue
        overlap = torch.vdot(states[a], states[b])
        fids.append(float(torch.abs(overlap) ** 2))
    return np.array(fids)


def haar_pdf(F: np.ndarray, d: int) -> np.ndarray:
    """Haar random fidelity distribution for d-dimensional Hilbert space."""
    return (d - 1) * (1 - F) ** (d - 2)


def expressibility_kl(
    qnode,
    params_shape: tuple,
    n_samples: int = 1000,
    n_bins: int = 75,
    return_histograms: bool = False,
):
    """
    Compute the expressibility as KL(P_PQC || P_Haar).
    Lower value = more expressive (closer to Haar random).
    If return_histograms is True, returns (kl, p_pqc, p_haar, bin_centers).
    """
    n_params = int(np.prod(params_shape))
    d = None  # infer from circuit output

    # Sample states
    states_list = []
    
    # Generate flat params then reshape to match EHA ansatz requirement
    if isinstance(params_shape, int):
        params_shape = (params_shape,)
        
    for _ in range(n_samples):
        theta = (torch.rand(params_shape) * 2 * np.pi)
        psi   = qnode(theta).detach()
        if d is None:
            d = psi.shape[0]
        states_list.append(psi)
    states = torch.stack(states_list)

    # Pairwise fidelities
    N_pairs = min(n_samples * (n_samples - 1) // 2, 5000)
    fids    = pairwise_fidelities(states, N_pairs=N_pairs)

    # Bin the empirical distribution
    bins       = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    p_pqc, _   = np.histogram(fids, bins=bins, density=True)
    p_haar      = haar_pdf(bin_centers, d)

    # Compute KL Divergence
    eps    = 1e-10
    p_pqc_norm  = p_pqc + eps
    p_haar_norm = p_haar + eps
    p_pqc_norm  = p_pqc_norm / p_pqc_norm.sum()
    p_haar_norm = p_haar_norm / p_haar_norm.sum()

    kl = float(np.sum(p_pqc_norm * np.log(p_pqc_norm / p_haar_norm)))
    
    if return_histograms:
        return kl, p_pqc, p_haar, bin_centers
    return kl


def meyer_wallach_entanglement(
    qnode,
    params_shape: tuple,
    n_samples: int = 500,
) -> float:
    """
    Compute the Meyer-Wallach entanglement measure Q ∈ [0, 1].
    Q = (4/N) Σ_θ [1 − (1/n) Σ_k Tr(ρ_k²)]
    where ρ_k = Tr_{¬k}(|ψ⟩⟨ψ|) is the k-th qubit's reduced state.
    """
    Q_total = 0.0
    n_qubits = None

    for _ in range(n_samples):
        theta = torch.rand(params_shape) * 2 * np.pi
        psi   = qnode(theta).detach()
        d     = psi.shape[0]
        if n_qubits is None:
            n_qubits = int(np.log2(d))

        # For each qubit k, compute Tr(ρ_k²) = 1 - S_lin(ρ_k)
        purities = []
        for k in range(n_qubits):
            # Reshape psi into (2^k, 2, 2^{n-k-1})
            psi_r = psi.reshape(2**k, 2, 2**(n_qubits - k - 1))
            # ρ_k[a, b] = Σ_{i,l} psi_r[i, a, l] * conj(psi_r[i, b, l])
            rho_k = torch.einsum('ial,ibl->ab', psi_r, psi_r.conj())
            purity_k = float(torch.real(torch.trace(rho_k @ rho_k)))
            purities.append(purity_k)

        Q_total += 1.0 - np.mean(purities)

    return float(4.0 * Q_total / n_samples)


def expressibility_report(
    qnode,
    params_shape: tuple,
    n_samples: int = 500,
) -> dict:
    """Full expressibility and entanglement report."""
    kl = expressibility_kl(qnode, params_shape, n_samples=n_samples)
    Q  = meyer_wallach_entanglement(qnode, params_shape, n_samples=n_samples)
    return {
        "expressibility_kl":            kl,
        "expressibility_label":         "excellent" if kl < 0.05 else ("good" if kl < 0.3 else "limited"),
        "meyer_wallach_Q":              Q,
        "entangling_capability_label":  "high" if Q > 0.7 else ("medium" if Q > 0.3 else "low"),
    }
