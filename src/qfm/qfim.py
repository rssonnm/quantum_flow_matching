"""
qfim.py — Quantum Fisher Information Matrix (QFIM) computation.

Mathematical foundations for measuring the geometric curvature of 
parameterized quantum state manifolds.
"""
import torch
import numpy as np
import numpy as np
from typing import Optional


def _shifted_state(qnode, params: torch.Tensor, param_idx: int,
                   input_state: torch.Tensor, shift: float = np.pi / 2):
    """
    Returns the output state with the (l, k)-th parameter shifted by ±shift.
    Used for finite-difference QFIM approximation.
    """
    flat_params = params.clone().detach().flatten()
    shifted_plus  = flat_params.clone(); shifted_plus[param_idx]  += shift
    shifted_minus = flat_params.clone(); shifted_minus[param_idx] -= shift
    shape = params.shape
    s_plus  = qnode(input_state, shifted_plus.reshape(shape))
    s_minus = qnode(input_state, shifted_minus.reshape(shape))
    return s_plus.detach(), s_minus.detach()


def compute_qfim_fd(
    qnode,
    params: torch.Tensor,
    input_state: torch.Tensor,
    shift: float = np.pi / 2,
) -> torch.Tensor:
    """
    Compute the QFIM via finite difference (parameter shift rule extended):
        F_ij = Re[⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩] * 4

    Uses: ∂_k |ψ⟩ ≈ (|ψ(+s)⟩ - |ψ(-s)⟩) / (2 sin(s))
    For shift = π/2 this is exact for generators with eigenvalues ±1/2.
    """
    P_flat = params.numel()
    psi0 = qnode(input_state, params).detach()

    # Build derivative states
    d_psi = []
    for k in range(P_flat):
        s_plus, s_minus = _shifted_state(qnode, params, k, input_state, shift)
        deriv = (s_plus - s_minus) / (2.0 * np.sin(shift))
        d_psi.append(deriv)

    F = torch.zeros((P_flat, P_flat), dtype=torch.float64)
    for i in range(P_flat):
        for j in range(i, P_flat):
            inner_ij = torch.vdot(d_psi[i], d_psi[j])
            inner_i0 = torch.vdot(d_psi[i], psi0)
            inner_0j = torch.vdot(psi0, d_psi[j])
            F_ij = 4.0 * torch.real(inner_ij - inner_i0 * inner_0j)
            F[i, j] = F_ij
            F[j, i] = F_ij

    return F


def compute_qfim_ensemble(
    qnode,
    params: torch.Tensor,
    ensemble: torch.Tensor,    # (M, d) pure state vectors
    shift: float = np.pi / 2,
) -> torch.Tensor:
    """
    Ensemble-averaged QFIM: F̄ = (1/M) Σ_m F(|ψ_m⟩).
    This is the relevant metric for QFM training dynamics.
    """
    F_sum = None
    M = ensemble.shape[0]
    for psi in ensemble:
        F_m = compute_qfim_fd(qnode, params, psi, shift)
        F_sum = F_m if F_sum is None else F_sum + F_m
    return F_sum / M


def effective_dimension(F: torch.Tensor, threshold: float = 1e-6) -> float:
    """
    Effective dimension: fraction of eigenvalues above threshold.
        d_eff = |{λ_i > threshold}| / P
    """
    ev = torch.linalg.eigvalsh(F)
    n_active = int((ev > threshold).sum())
    return n_active / F.shape[0]


def barren_plateau_report(F: torch.Tensor) -> dict:
    """
    Returns a dictionary of QFIM diagnostics:
      - eigenvalues (sorted descending)
      - max_eigenvalue
      - min_nonzero_eigenvalue
      - effective_dimension
      - condition_number
    """
    ev = torch.linalg.eigvalsh(F).real
    ev_sorted = torch.sort(ev, descending=True).values
    nonzero   = ev_sorted[ev_sorted > 1e-10]
    cond      = (ev_sorted[0] / nonzero[-1]).item() if len(nonzero) > 0 else float('inf')
    return {
        "eigenvalues":           ev_sorted.tolist(),
        "max_eigenvalue":        float(ev_sorted[0]),
        "min_nonzero":           float(nonzero[-1]) if len(nonzero) > 0 else 0.0,
        "effective_dimension":   effective_dimension(F),
        "condition_number":      cond,
        "rank":                  int((ev_sorted > 1e-10).sum()),
        "barren_plateau_flag":   bool(ev_sorted[0] < 1e-5),
    }
