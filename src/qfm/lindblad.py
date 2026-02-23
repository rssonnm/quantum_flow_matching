"""
lindblad.py — Numerical solvers for the Lindblad Master Equation.
"""

import torch
import numpy as np
from typing import List, Tuple


# Superoperator construction

def _vec(rho: torch.Tensor) -> torch.Tensor:
    """Column-stack vectorization: |ρ⟩⟩ = vec(ρ)."""
    return rho.reshape(-1)


def _unvec(v: torch.Tensor, d: int) -> torch.Tensor:
    """Inverse vectorization."""
    return v.reshape(d, d)


def _commutator_superop(H: torch.Tensor) -> torch.Tensor:
    """
    Superoperator for -i[H, ρ]:
        -i(H ⊗ I - I ⊗ H^T)   (vec ordering: row-to-col stacking)
    """
    d = H.shape[0]
    I = torch.eye(d, dtype=torch.complex128)
    Ht = H.T.contiguous()
    return -1j * (torch.kron(H, I) - torch.kron(I, Ht))


def _dissipator_superop(L: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Superoperator for a single Lindblad term:
        D[L](ρ) = γ(LρL† - ½L†Lρ - ½ρL†L)
    as a (d²×d²) superoperator.
    """
    d = L.shape[0]
    I = torch.eye(d, dtype=torch.complex128)
    Lc  = L.conj().T.contiguous()
    LcL = (Lc @ L).contiguous()
    LcL_T = LcL.T.contiguous()
    Lconj = L.conj().contiguous()
    return gamma * (
        torch.kron(L, Lconj)
        - 0.5 * torch.kron(LcL, I)
        - 0.5 * torch.kron(I, LcL_T)
    )


def build_lindblad_superoperator(
    H: torch.Tensor,
    jump_ops: List[torch.Tensor],
    gammas: List[float],
) -> torch.Tensor:
    """
    Full Lindblad superoperator L = L_H + Σ_k D[L_k].
    Shape: (d², d²), complex128.
    """
    L_sup = _commutator_superop(H)
    for L_k, gk in zip(jump_ops, gammas):
        L_sup = L_sup + _dissipator_superop(L_k, gk)
    return L_sup


def lindblad_evolve(
    rho0: torch.Tensor,
    H: torch.Tensor,
    jump_ops: List[torch.Tensor],
    gammas: List[float],
    times: List[float],
) -> List[torch.Tensor]:
    """
    Evolve rho0 under Lindblad dynamics at each time step.
    Uses matrix exponentiation: ρ(t) = unvec(expm(t·L) vec(ρ(0))).
    Returns list of density matrices at each t in `times`.
    """
    from scipy.linalg import expm

    d   = rho0.shape[0]
    L   = build_lindblad_superoperator(H, jump_ops, gammas)
    L_np = L.detach().cpu().numpy()
    v0   = _vec(rho0).detach().cpu().numpy()

    rhos = []
    for t in times:
        Lt  = expm(t * L_np)
        vt  = Lt @ v0
        rho_t = torch.tensor(vt.reshape(d, d), dtype=torch.complex128)
        # enforce Hermiticity (numerical errors)
        rho_t = 0.5 * (rho_t + rho_t.conj().T)
        rhos.append(rho_t)
    return rhos


# Standard jump operators

def amplitude_damping_jump(n_qubits: int, target_qubit: int = 0) -> torch.Tensor:
    """Single-qubit amplitude damping: L = |0⟩⟨1| on target_qubit."""
    L_1q = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex128)
    I     = torch.eye(2, dtype=torch.complex128)
    ops   = [L_1q if i == target_qubit else I for i in range(n_qubits)]
    result = ops[0]
    for op in ops[1:]:
        result = torch.kron(result, op)
    return result


def dephasing_jump(n_qubits: int, target_qubit: int = 0) -> torch.Tensor:
    """Single-qubit dephasing: L = σ_z on target_qubit."""
    Z     = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    I     = torch.eye(2, dtype=torch.complex128)
    ops   = [Z if i == target_qubit else I for i in range(n_qubits)]
    result = ops[0]
    for op in ops[1:]:
        result = torch.kron(result, op)
    return result


# QFM vs Lindblad comparison metric

def trajectory_bures_distance(
    qfm_rhos: List[torch.Tensor],
    lindblad_rhos: List[torch.Tensor],
) -> List[float]:
    """
    Compute Bures distance at each step between QFM and Lindblad trajectories.
    d_B(ρ, σ) = sqrt(2 - 2 sqrt(F_Uhlmann(ρ, σ)))
    """
    from .metrics import uhlmann_fidelity
    distances = []
    for rho_qfm, rho_lin in zip(qfm_rhos, lindblad_rhos):
        F = uhlmann_fidelity(rho_qfm, rho_lin)
        d_B = float(torch.sqrt(torch.clamp(2.0 - 2.0 * torch.sqrt(F), min=0.0)))
        distances.append(d_B)
    return distances
