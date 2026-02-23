"""
metrics.py — Statistical and geometric metrics for quantum states.

Includes implementations for Uhlmann fidelity, Bures distance, 
von Neumann entropy, and quantum Wasserstein distances.
"""
import torch
import numpy as np


# Matrix square root (via eigendecomposition)

def matrix_sqrt(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the principal matrix square root of a positive semi-definite Hermitian matrix A.
    Uses eigendecomposition: A = V Λ V†  →  A^{1/2} = V Λ^{1/2} V†
    Numerically stable and differentiable.
    """
    ev, V = torch.linalg.eigh(A)
    ev_safe = torch.clamp(ev.real, min=0.0).to(torch.complex128)
    return (V * ev_safe.sqrt().unsqueeze(0)) @ V.conj().T


# Exact Uhlmann Fidelity

def uhlmann_fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Exact Uhlmann fidelity F(ρ, σ) = (Tr √(√ρ σ √ρ))².
    Reduces to |⟨ψ|φ⟩|² for pure states.
    """
    sqrt_rho = matrix_sqrt(rho)
    M        = sqrt_rho @ sigma @ sqrt_rho
    return torch.real(torch.trace(matrix_sqrt(M))) ** 2


def uhlmann_fidelity_ensemble(rhos1: torch.Tensor, rhos2: torch.Tensor) -> torch.Tensor:
    """Average Uhlmann fidelity over matched pairs."""
    return torch.mean(torch.stack([uhlmann_fidelity(r, s) for r, s in zip(rhos1, rhos2)]))


# Bures Metric

def bures_distance(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Bures distance: d_B(ρ, σ) = √(2 − 2√F(ρ, σ)).
    This is a Riemannian metric on the manifold of density matrices.
    d_B ∈ [0, √2]; d_B = 0 iff ρ = σ; d_B = √2 for orthogonal pure states.
    """
    F = uhlmann_fidelity(rho, sigma)
    return torch.sqrt(torch.clamp(2.0 - 2.0 * torch.sqrt(F.clamp(min=0.0)), min=0.0))


def bures_angle(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Bures angle Θ(ρ, σ) = arccos(√F(ρ, σ)).
    Also known as the quantum angle; Θ ∈ [0, π/2].
    """
    F = uhlmann_fidelity(rho, sigma)
    return torch.arccos(torch.sqrt(F.clamp(0.0, 1.0)))


# Quantum Wasserstein (Sinkhorn approximation)

def _cost_matrix_bures(rhos1: torch.Tensor, rhos2: torch.Tensor) -> torch.Tensor:
    """Pairwise Bures distance matrix C[i,j] = d_B(ρ_i, σ_j)."""
    M1, M2 = len(rhos1), len(rhos2)
    C = torch.zeros(M1, M2, dtype=torch.float64)
    for i, r in enumerate(rhos1):
        for j, s in enumerate(rhos2):
            C[i, j] = bures_distance(r, s)
    return C


def sinkhorn_wasserstein(
    rhos1: torch.Tensor,
    rhos2: torch.Tensor,
    epsilon: float = 0.01,
    n_iter: int = 100,
) -> float:
    """
    Sinkhorn-regularized quantum Wasserstein W_ε(μ, ν) using Bures cost.

    Solves: min_{T ≥ 0, T1=a, Tᵀ1=b} ⟨C, T⟩ + ε H(T)
    where H(T) = -Σ T_{ij}(log T_{ij} - 1)  (entropic regularization).

    Returns the regularized transport cost (approximates W_1 for small ε).
    """
    C   = _cost_matrix_bures(rhos1, rhos2).double()
    M1, M2 = C.shape
    a   = torch.ones(M1, dtype=torch.float64) / M1
    b   = torch.ones(M2, dtype=torch.float64) / M2

    # Gibbs kernel
    K   = torch.exp(-C / epsilon)
    u   = torch.ones(M1, dtype=torch.float64)
    v   = torch.ones(M2, dtype=torch.float64)

    for _ in range(n_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)

    T = torch.diag(u) @ K @ torch.diag(v)
    return float(torch.sum(T * C))


# Rényi Divergence D_α(ρ||σ)

def renyi_divergence(rho: torch.Tensor, sigma: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    Sandwiched Rényi divergence D̃_α(ρ||σ):
        D̃_α = (1/(α−1)) log Tr[(σ^{(1-α)/2α} ρ σ^{(1-α)/2α})^α]
    For α=2: D̃_2(ρ||σ) = log Tr[σ^{-1/2} ρ σ^{-1/2} ρ].
    We use the simpler Petz Rényi divergence for numerical stability:
        D_α^Petz = (1/(α−1)) log Tr[ρ^α σ^{1-α}]
    """
    if abs(alpha - 1.0) < 1e-6:
        # Von Neumann relative entropy D_1(ρ||σ) = Tr(ρ(log ρ - log σ))
        ev_rho, V_rho   = torch.linalg.eigh(rho)
        ev_sig, V_sig   = torch.linalg.eigh(sigma)
        ev_rho  = ev_rho.clamp(min=1e-12)
        ev_sig  = ev_sig.clamp(min=1e-12)
        log_rho = (V_rho * torch.log(ev_rho).unsqueeze(0)) @ V_rho.conj().T
        log_sig = (V_sig * torch.log(ev_sig).unsqueeze(0)) @ V_sig.conj().T
        return torch.real(torch.trace(rho @ (log_rho - log_sig)))

    ev_rho, V_rho = torch.linalg.eigh(rho)
    ev_sig, V_sig = torch.linalg.eigh(sigma)
    ev_rho = ev_rho.clamp(min=1e-12).to(torch.complex128)
    ev_sig = ev_sig.clamp(min=1e-12).to(torch.complex128)

    rho_alpha   = (V_rho * (ev_rho ** alpha).unsqueeze(0)) @ V_rho.conj().T
    sig_1malpha = (V_sig * (ev_sig ** (1.0 - alpha)).unsqueeze(0)) @ V_sig.conj().T
    tr_val      = torch.real(torch.trace(rho_alpha @ sig_1malpha))
    return torch.log2(tr_val.clamp(min=1e-12)) / (alpha - 1.0)



# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------

def purity(rho: torch.Tensor) -> torch.Tensor:
    """γ(ρ) = Tr(ρ²).  Pure state ⟹ γ=1; maximally mixed ⟹ γ=1/d."""
    return torch.real(torch.trace(torch.matmul(rho, rho)))


# ---------------------------------------------------------------------------
# Von Neumann Entropy  (already in utils, aliased here for convenience)
# ---------------------------------------------------------------------------

def von_neumann_entropy(rho: torch.Tensor) -> torch.Tensor:
    """S(ρ) = −Tr(ρ log₂ ρ)."""
    ev = torch.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return -torch.sum(ev * torch.log2(ev))


# ---------------------------------------------------------------------------
# Rényi Entropy
# ---------------------------------------------------------------------------

def renyi_entropy(rho: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    S_α(ρ) = 1/(1−α) · log₂ Tr(ρ^α).
    α=2  ⟹ collision entropy.
    α→1  ⟹ Von Neumann entropy (use von_neumann_entropy instead).
    """
    if abs(alpha - 1.0) < 1e-6:
        return von_neumann_entropy(rho)
    ev = torch.linalg.eigvalsh(rho).clamp(min=1e-12)
    tr_rho_alpha = torch.sum(ev ** alpha)
    return torch.log2(tr_rho_alpha) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Purity-based Rényi
# ---------------------------------------------------------------------------

def purity_entropy(rho: torch.Tensor) -> torch.Tensor:
    """S₂ = −log₂ Tr(ρ²) = −log₂ γ(ρ)."""
    return -torch.log2(purity(rho).clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Negativity — entanglement witness via partial transpose
# ---------------------------------------------------------------------------

def partial_transpose(rho: torch.Tensor, n_qubits_A: int, n_qubits_B: int) -> torch.Tensor:
    """
    Partial transpose on subsystem B of a bipartite density matrix.
    ρ has shape (dA·dB, dA·dB).
    """
    dA = 2 ** n_qubits_A
    dB = 2 ** n_qubits_B
    # reshape to (dA, dB, dA, dB)
    rho_r = rho.reshape(dA, dB, dA, dB)
    # transpose on B indices: (dA, dB, dA, dB) → (dA, dB', dA, dB) with B↔B'
    rho_pt = rho_r.permute(0, 3, 2, 1).reshape(dA * dB, dA * dB)
    return rho_pt


def negativity(rho: torch.Tensor, n_qubits_A: int = 1, n_qubits_B: int = 1) -> torch.Tensor:
    """
    Negativity N(ρ) = (||ρ^{T_B}||₁ − 1) / 2.
    Ranges [0, 0.5] for a 2-qubit state; 0 iff separable.
    """
    rho_pt = partial_transpose(rho, n_qubits_A, n_qubits_B)
    # Nuclear norm = sum of singular values (= sum |λ_i| for Hermitian)
    ev = torch.linalg.eigvalsh(rho_pt)
    trace_norm = torch.sum(torch.abs(ev))
    return (trace_norm - 1.0) / 2.0


# ---------------------------------------------------------------------------
# Frobenius / MMD kernel utilities
# ---------------------------------------------------------------------------

def fidelity_overlap(rho1: torch.Tensor, rho2: torch.Tensor) -> torch.Tensor:
    """Tr(ρ₁ ρ₂) — differentiable fidelity proxy."""
    return torch.real(torch.trace(torch.matmul(rho1, rho2)))


def ensemble_purity(rhos: torch.Tensor) -> torch.Tensor:
    """Average purity over a batch of density matrices (shape N×d×d)."""
    return torch.mean(torch.stack([purity(r) for r in rhos]))


def ensemble_von_neumann(rhos: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.stack([von_neumann_entropy(r) for r in rhos]))


def ensemble_renyi(rhos: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return torch.mean(torch.stack([renyi_entropy(r, alpha) for r in rhos]))


def ensemble_negativity(rhos: torch.Tensor, nA: int = 1, nB: int = 1) -> torch.Tensor:
    return torch.mean(torch.stack([negativity(r, nA, nB) for r in rhos]))
