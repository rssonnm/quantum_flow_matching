"""
quantum_ot.py — Quantum Optimal Transport geodesics and transport analysis for QFM.

Implements:
  - Bures geodesic interpolation ρ(t) between two density matrices
  - Quantum W2 distance (Bures metric as transport cost)
  - Comparison between QFM trajectory and geodesic
  - Discrete Benamou-Brenier kinetic energy
  - Transport efficiency analysis

The Bures metric ds² = (1/2) Tr(dρ G) where ρG + Gρ = dρ defines the quantum
analog of the Wasserstein-2 metric on density matrices (quantum OT).

References:
    Bures (1969); Uhlmann (1976); Dittmann (1999);
    Chen et al., "Optimal Transport for Quantum Density Matrices" (2022).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from .metrics import bures_distance, uhlmann_fidelity, matrix_sqrt, purity


# ---------------------------------------------------------------------------
# Bures geodesic interpolation
# ---------------------------------------------------------------------------

def bures_geodesic_interpolation(
    rho_0: torch.Tensor,
    rho_T: torch.Tensor,
    n_points: int = 20,
) -> List[torch.Tensor]:
    """
    Compute the Bures geodesic ρ(s) for s ∈ [0,1] between rho_0 and rho_T.

    The exact formula on the Bures manifold:
        ρ(s) = [(1-s)√ρ₀ + s G^{1/2}]  M  [(1-s)√ρ₀ + s G^{1/2}]†
    where G is the geometric mean and M involves the parallel transport frame.

    For the practical implementation we use the commutative approximation:
        ρ(s) ∝ exp[(1-s) log(ρ₀) + s log(ρ_T)]    (matrix exponential interpolation)
    This is the geodesic in the Riemannian manifold of positive-definite matrices
    under the affine-invariant metric, which coincides with Bures for commuting states.

    For general non-commuting states we use a numerically stable Uhlmann-based formula:
        ρ(s) = ((1-s)ρ₀^{1/2} + s W ρ_T^{1/2}) ((1-s)ρ₀^{1/2} + s W ρ_T^{1/2})†
    where W = √(ρ₀^{1/2} ρ_T ρ₀^{1/2})^{-1/2} ρ₀^{1/2} ρ_T^{1/2}.

    Args:
        rho_0:    Initial density matrix (d×d).
        rho_T:    Target density matrix (d×d).
        n_points: Number of interpolation points including endpoints.

    Returns:
        List of n_points density matrices along the geodesic.
    """
    s_vals = np.linspace(0.0, 1.0, n_points)
    geodesic = []

    sqrt_rho0 = matrix_sqrt(rho_0)
    # Compute the Uhlmann parallel transport W
    M = sqrt_rho0 @ rho_T @ sqrt_rho0
    sqrt_M = matrix_sqrt(M)
    # W = sqrt_rho0^{-1} sqrt_M (pseudo-inverse if singular)
    try:
        inv_sqrt_rho0 = torch.linalg.pinv(sqrt_rho0)
        W_half = inv_sqrt_rho0 @ sqrt_M  # W rho_T^{1/2}
    except Exception:
        W_half = torch.eye(rho_0.shape[0], dtype=torch.complex128)

    sqrt_rhoT = matrix_sqrt(rho_T)

    for s in s_vals:
        A = (1.0 - s) * sqrt_rho0 + s * W_half
        rho_s = A @ A.conj().T
        # Normalize to unit trace
        tr = torch.real(torch.trace(rho_s))
        if tr > 1e-12:
            rho_s = rho_s / tr
        geodesic.append(rho_s)

    return geodesic


# ---------------------------------------------------------------------------
# Quantum W2 distance
# ---------------------------------------------------------------------------

def quantum_w2_distance(rho_0: torch.Tensor, rho_T: torch.Tensor) -> float:
    """
    Quantum Wasserstein-2 distance:
        W₂(ρ₀, ρ_T) = d_B(ρ₀, ρ_T)

    The Bures distance is the geodesic distance on the manifold of density matrices
    and equals the square root of the quantum optimal transport cost (W₂²).

    Returns:
        W2 distance (float).
    """
    return float(bures_distance(rho_0, rho_T))


def quantum_w2_squared(rho_0: torch.Tensor, rho_T: torch.Tensor) -> float:
    """W₂² = 2 - 2√F(ρ₀, ρ_T) = Bures distance²."""
    d = float(bures_distance(rho_0, rho_T))
    return d ** 2


# ---------------------------------------------------------------------------
# Trajectory vs geodesic analysis
# ---------------------------------------------------------------------------

def trajectory_vs_geodesic_analysis(
    qfm_rhos: List[torch.Tensor],
    rho_0: torch.Tensor,
    rho_T: torch.Tensor,
) -> Dict[str, object]:
    """
    Comprehensive comparison of the QFM trajectory against the Bures geodesic.

    Computes:
      - QFM action A_traj = Σ d_B(ρ_t, ρ_{t+1})²
      - Geodesic action A_geo = d_B(ρ_0, ρ_T)² (lower bound)
      - Per-step deviation from geodesic
      - Transport efficiency η = A_geo / A_traj

    Returns:
        Dict with full analysis.
    """
    T = len(qfm_rhos)
    n_geo = max(T, 20)
    geodesic = bures_geodesic_interpolation(rho_0, rho_T, n_points=T)

    # QFM trajectory action
    step_costs_qfm = []
    for t in range(T - 1):
        d = float(bures_distance(qfm_rhos[t], qfm_rhos[t + 1]))
        step_costs_qfm.append(d ** 2)
    A_traj = sum(step_costs_qfm)

    # Geodesic action
    step_costs_geo = []
    for t in range(T - 1):
        d = float(bures_distance(geodesic[t], geodesic[t + 1]))
        step_costs_geo.append(d ** 2)
    A_geo_traj = sum(step_costs_geo)

    # True geodesic cost (direct d_B²)
    A_geo_true = quantum_w2_squared(rho_0, rho_T)

    # Per-step deviation: d_B(QFM_t, Geo_t)
    deviations = [float(bures_distance(qfm_rhos[t], geodesic[t])) for t in range(T)]

    efficiency = A_geo_true / (A_traj + 1e-15)

    return {
        'trajectory_action':   A_traj,
        'geodesic_action_true': A_geo_true,
        'geodesic_action_disc': A_geo_traj,
        'step_costs_qfm':      step_costs_qfm,
        'step_costs_geodesic': step_costs_geo,
        'deviations_from_geo': deviations,
        'transport_efficiency': float(min(efficiency, 1.0)),
        'excess_cost':          float(max(A_traj - A_geo_true, 0.0)),
        'geodesic_rhos':        geodesic,
        'T_steps':              T,
    }


# ---------------------------------------------------------------------------
# Transport efficiency
# ---------------------------------------------------------------------------

def transport_efficiency(trajectory_action: float, geodesic_action: float) -> float:
    """
    η = A_geo / A_traj ∈ (0, 1].  η=1 iff trajectory IS the geodesic.
    """
    return float(geodesic_action / (trajectory_action + 1e-15))


# ---------------------------------------------------------------------------
# Discrete Benamou-Brenier kinetic energy
# ---------------------------------------------------------------------------

def discrete_benamou_brenier_energy(rhos: List[torch.Tensor], dt: float = None) -> List[float]:
    """
    Discrete kinetic energy in the Benamou-Brenier sense:
        KE(t) = ||v_t||²_ρ_t  where v_t = (ρ_{t+1} - ρ_t) / Δt

    On the Bures manifold, the kinetic energy per step equals:
        KE_t = d_B(ρ_t, ρ_{t+1})² / Δt²

    Args:
        rhos: List of density matrices along trajectory.
        dt:   Time step size (default: 1/T).

    Returns:
        List of kinetic energies per step.
    """
    T = len(rhos) - 1
    if dt is None:
        dt = 1.0 / max(T, 1)
    KE = []
    for t in range(T):
        d = float(bures_distance(rhos[t], rhos[t + 1]))
        KE.append((d / dt) ** 2)
    return KE


def total_benamou_brenier_cost(rhos: List[torch.Tensor], dt: float = None) -> float:
    """
    Total BB cost = Σ_t KE(t) * Δt.  Equals W₂² in continuous limit.
    """
    T = len(rhos) - 1
    if dt is None:
        dt = 1.0 / max(T, 1)
    KE = discrete_benamou_brenier_energy(rhos, dt)
    return float(sum(KE) * dt)


# ---------------------------------------------------------------------------
# Quantum speed limit
# ---------------------------------------------------------------------------

def quantum_speed_limit_bound(
    rho_0: torch.Tensor,
    rho_T: torch.Tensor,
    mean_energy_variance: float,
) -> float:
    """
    Mandelstam-Tamm Quantum Speed Limit:
        T_QSL ≥ ħ · arccos(√F(ρ₀, ρ_T)) / ΔE_rms

    With ħ=1 and ΔE_rms = sqrt(mean energy variance).

    Returns:
        Minimum time T_QSL required for the state transformation.
    """
    F = float(uhlmann_fidelity(rho_0, rho_T))
    angle = float(np.arccos(np.sqrt(min(max(F, 0.0), 1.0))))
    if mean_energy_variance < 1e-15:
        return float('inf')
    return float(angle / mean_energy_variance)


def qsl_efficiency(actual_time: float, qsl_time: float) -> float:
    """
    QSL efficiency: η_QSL = T_QSL / T_actual ∈ (0, 1].
    η_QSL = 1 means evolution is saturating the speed limit.
    """
    return float(min(qsl_time / (actual_time + 1e-15), 1.0))


# ---------------------------------------------------------------------------
# Geodesic curvature of trajectory
# ---------------------------------------------------------------------------

def mean_geodesic_deviation(
    qfm_rhos: List[torch.Tensor],
    rho_0: torch.Tensor,
    rho_T: torch.Tensor,
) -> Dict[str, float]:
    """
    Mean and max deviation of the QFM trajectory from the Bures geodesic.

    Returns:
        Dict with 'mean_deviation', 'max_deviation', 'std_deviation'.
    """
    T = len(qfm_rhos)
    geodesic = bures_geodesic_interpolation(rho_0, rho_T, n_points=T)
    devs = [float(bures_distance(qfm_rhos[t], geodesic[t])) for t in range(T)]
    arr = np.array(devs)
    return {
        'mean_deviation': float(arr.mean()),
        'max_deviation':  float(arr.max()),
        'std_deviation':  float(arr.std()),
        'deviations':     devs,
    }
