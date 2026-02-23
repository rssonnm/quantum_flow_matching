"""
flow_geometry.py — Geometric analysis of QFM vector fields on the density matrix manifold.

Implements:
  - Quantum vector field extraction from trajectory
  - Lindblad-like generator inference from consecutive density matrices
  - Bures sectional curvature tensor components
  - Geodesic curvature of the QFM trajectory
  - Parallel transport deviation analysis
  - Flow divergence and expansion rate
  - Holonomy (geometric phase) estimation

The Bures manifold (P_1(H), g_B) is a Riemannian manifold. Its geometry
determines the optimal transport structure and the differential-geometric
properties of QFM as a continuous flow on this manifold.

References:
    Bures (1969); Dittmann (1993); Uhlmann (1976, 1993);
    Luenberger "Optimization on Riemannian Manifolds" (2007).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from .metrics import bures_distance, uhlmann_fidelity, matrix_sqrt, von_neumann_entropy


# ---------------------------------------------------------------------------
# Quantum vector field extraction
# ---------------------------------------------------------------------------

def quantum_vector_field(
    rhos: List[torch.Tensor],
    dt: float = None,
    normalize: bool = False,
) -> List[torch.Tensor]:
    """
    Extract the discrete quantum vector field from a trajectory:
        V_t = (ρ_{t+1} - ρ_t) / Δt

    V_t is a Hermitian, traceless matrix (tangent to the density matrix manifold).

    Args:
        rhos:      Trajectory of density matrices.
        dt:        Time step (default: 1/T).
        normalize: If True, normalize each V_t to unit Hilbert-Schmidt norm.

    Returns:
        List of T tangent matrices V_t.
    """
    T = len(rhos) - 1
    if dt is None:
        dt = 1.0 / max(T, 1)
    vectors = []
    for t in range(T):
        V = (rhos[t + 1] - rhos[t]) / dt
        if normalize:
            norm = float(torch.real(torch.trace(V.conj().T @ V)).sqrt())
            if norm > 1e-12:
                V = V / norm
        vectors.append(V)
    return vectors


def hilbert_schmidt_norm(M: torch.Tensor) -> float:
    """||M||_HS = sqrt(Tr(M† M))."""
    return float(torch.real(torch.trace(M.conj().T @ M)).sqrt())


def vector_field_magnitude(rhos: List[torch.Tensor], dt: float = None) -> List[float]:
    """
    HS norm of the vector field at each step: ||V_t||_HS.
    """
    Vs = quantum_vector_field(rhos, dt)
    return [hilbert_schmidt_norm(V) for V in Vs]


# ---------------------------------------------------------------------------
# Lindblad-like generator inference
# ---------------------------------------------------------------------------

def generator_from_consecutive(
    rho_t: torch.Tensor,
    rho_tp1: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Infer the Lindblad-type generator L such that:
        dρ/dt = L(ρ) ≈ (ρ_{t+1} - ρ_t) / Δt

    For unitary evolution: L = -i[H, ρ] → H ≈ inferred effective Hamiltonian.
    For open evolution:    L = -i[H, ρ] + Σ_k (L_k ρ L_k† - {L_k†L_k, ρ}/2).

    Simplified: extract the effective generator Ĝ = (ρ_{t+1} - ρ_t) / Δt
    and decompose it into anti-Hermitian (Hamiltonian) and Hermitian (dissipative) parts.

    Returns:
        Dict with 'G_total', 'G_antiherm' (coherent), 'G_herm' (dissipative),
        'effective_H' (inferred Hamiltonian).
    """
    G = (rho_tp1 - rho_t) / dt
    G_antiherm = 0.5 * (G - G.conj().T)   # pure imaginary diagonal: -i[H, ρ]
    G_herm     = 0.5 * (G + G.conj().T)   # relaxation / decoherence terms
    # Effective H from -i[H, ρ] = G_antiherm → H ≈ ?
    # For simplicity: project G_antiherm as -i H_eff ρ + iρ H_eff
    # Not exactly solvable, but we can report the skew-Hermitian part
    H_eff = 1j * G_antiherm  # approximate: assume ρ ≈ I/d to extract H
    return {
        'G_total':      G,
        'G_antiherm':   G_antiherm,
        'G_herm':       G_herm,
        'effective_H':  H_eff,
        'coherent_rate': float(hilbert_schmidt_norm(G_antiherm)),
        'dissipative_rate': float(hilbert_schmidt_norm(G_herm)),
    }


def generator_spectrum_trajectory(
    rhos: List[torch.Tensor],
    dt: float = None,
) -> Dict[str, list]:
    """
    Compute generator decomposition at each step of the trajectory.

    Returns:
        Dict with 'coherent_rates', 'dissipative_rates', 'total_rates'.
    """
    T = len(rhos) - 1
    if dt is None:
        dt = 1.0 / max(T, 1)
    coh, dis, tot = [], [], []
    for t in range(T):
        g = generator_from_consecutive(rhos[t], rhos[t+1], dt)
        coh.append(g['coherent_rate'])
        dis.append(g['dissipative_rate'])
        tot.append(float(hilbert_schmidt_norm(g['G_total'])))
    return {
        'coherent_rates':   coh,
        'dissipative_rates': dis,
        'total_rates':       tot,
    }


# ---------------------------------------------------------------------------
# Bures sectional curvature
# ---------------------------------------------------------------------------

def sectional_curvature_bures(
    rho: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
) -> float:
    """
    Approximation of the Bures sectional curvature K(X, Y) at ρ.

    The exact formula involves the super-operator equation ρG + Gρ = X (SLD equation).
    For a qubit (d=2), K ≤ 0 everywhere on the Bures manifold.

    We compute the discrete approximation via the commutator norm:
        K_approx(X, Y) = ||[X, Y]||_HS² / (||X||_HS² ||Y||_HS² - |⟨X,Y⟩_HS|²)

    Note: ||·||_HS is the Hilbert-Schmidt norm; ⟨A,B⟩_HS = Re Tr(A†B).

    Args:
        rho: Base density matrix (not used in this approximation, but part of the manifold point).
        X:   First tangent vector (Hermitian traceless matrix).
        Y:   Second tangent vector (Hermitian traceless matrix).

    Returns:
        Approximate sectional curvature K(X, Y).
    """
    comm = X @ Y - Y @ X
    comm_norm_sq = float(torch.real(torch.trace(comm.conj().T @ comm)))
    norm_X_sq = float(torch.real(torch.trace(X.conj().T @ X)))
    norm_Y_sq = float(torch.real(torch.trace(Y.conj().T @ Y)))
    inner_XY  = float(torch.real(torch.trace(X.conj().T @ Y)))
    denom = norm_X_sq * norm_Y_sq - inner_XY ** 2
    if abs(denom) < 1e-12:
        return 0.0
    return float(-comm_norm_sq / (4.0 * denom))


def curvature_along_trajectory(
    rhos: List[torch.Tensor],
    dt: float = None,
) -> List[float]:
    """
    Compute sectional curvature at each interior step of the trajectory
    using consecutive tangent vectors as the span (X=V_t, Y=V_{t+1}).

    Returns:
        List of curvature values (length T-1).
    """
    Vs = quantum_vector_field(rhos, dt)
    curvatures = []
    for t in range(len(Vs) - 1):
        K = sectional_curvature_bures(rhos[t + 1], Vs[t], Vs[t + 1])
        curvatures.append(K)
    return curvatures


# ---------------------------------------------------------------------------
# Geodesic curvature of the trajectory
# ---------------------------------------------------------------------------

def geodesic_curvature(rhos: List[torch.Tensor], dt: float = None) -> List[float]:
    """
    Geodesic curvature κ(τ) = ||acceleration||_HS / ||velocity||_HS²

    Acceleration: a_t ≈ (V_{t+1} - V_t) / Δt

    A geodesic has κ ≡ 0. High κ means the flow is deviating from a geodesic.

    Returns:
        List of geodesic curvature values (length T-1).
    """
    T = len(rhos) - 1
    if dt is None:
        dt = 1.0 / max(T, 1)
    Vs = quantum_vector_field(rhos, dt)
    curvatures = []
    for t in range(len(Vs) - 1):
        accel = (Vs[t + 1] - Vs[t]) / dt
        speed = hilbert_schmidt_norm(Vs[t])
        kappa = hilbert_schmidt_norm(accel) / (speed ** 2 + 1e-12)
        curvatures.append(float(kappa))
    return curvatures


# ---------------------------------------------------------------------------
# Parallel transport deviation
# ---------------------------------------------------------------------------

def parallel_transport_deviation(
    rhos: List[torch.Tensor],
    dt: float = None,
) -> List[float]:
    """
    Measure how much the vector field V_t deviates from parallel transport.

    Parallel transport condition on the Bures manifold:
        ∇_{V_t} V = 0

    Approximated as:
        δ_t = ||V_{t+1} - V_t||_HS / ||V_t||_HS

    A small δ_t means the vector field is approximately parallel transported.

    Returns:
        List of deviation values per step.
    """
    Vs = quantum_vector_field(rhos, dt)
    deviations = []
    for t in range(len(Vs) - 1):
        diff = Vs[t + 1] - Vs[t]
        speed = hilbert_schmidt_norm(Vs[t])
        dev   = hilbert_schmidt_norm(diff) / (speed + 1e-12)
        deviations.append(float(dev))
    return deviations


# ---------------------------------------------------------------------------
# Flow divergence
# ---------------------------------------------------------------------------

def flow_divergence(
    rhos: List[torch.Tensor],
    dt: float = None,
) -> List[float]:
    """
    Discrete divergence of the quantum vector field.
    
    Approximated by the trace of the Jacobian of V with respect to ρ:
        div V_t ≈ Tr(∂V_t/∂ρ)
    
    Simple discrete estimate: 
        div(t) = Tr(V_{t+1}) - Tr(V_t) ≈ changes in flow "volume"
    
    For density matrices, Tr(V) = 0 (traceless tangent vectors), so
    we use the Frobenius structure change as a proxy:
        div_approx(t) = ||V_{t+1}||_HS - ||V_t||_HS

    Returns:
        List of flow divergence proxies per step.
    """
    Vs = quantum_vector_field(rhos, dt)
    mags = [hilbert_schmidt_norm(V) for V in Vs]
    divs = [mags[t+1] - mags[t] for t in range(len(mags) - 1)]
    return divs


# ---------------------------------------------------------------------------
# Geometric phase (holonomy) estimation
# ---------------------------------------------------------------------------

def geometric_phase_estimate(rhos: List[torch.Tensor]) -> float:
    """
    Estimate the Pancharatnam-Berry geometric phase accumulated along the trajectory.

    For a cyclic trajectory (ρ_0 ≈ ρ_T), the geometric phase is:
        γ_geo = arg(⟨ψ_0|ψ_1⟩⟨ψ_1|ψ_2⟩...⟨ψ_{T-1}|ψ_T⟩)

    For mixed states (using Uhlmann phase):
        The Uhlmann phase ϕ_U = arg Tr√(√ρ_0 ρ_T √ρ_0) over the cyclic path.

    We compute the phase angle of the product of overlap amplitudes.
    This is an approximation; exact Uhlmann phase requires purification.

    Returns:
        Geometric phase estimate in radians ∈ [-π, π].
    """
    phase_product = complex(1.0, 0.0)
    for t in range(len(rhos) - 1):
        # Overlap: Tr(√ρ_t ρ_{t+1} √ρ_t)^{1/2}
        sqrt_rho = matrix_sqrt(rhos[t])
        M = sqrt_rho @ rhos[t + 1] @ sqrt_rho
        # sqrt of trace
        overlap = torch.real(torch.trace(matrix_sqrt(M)))
        # Phase contribution from Uhlmann: approximate as arg of Tr(ρ_t ρ_{t+1})
        inner = torch.trace(rhos[t].conj().T @ rhos[t + 1])
        if abs(float(torch.abs(inner))) > 1e-12:
            phase_product *= complex(float(inner.real), float(inner.imag))

    return float(np.angle(phase_product))


# ---------------------------------------------------------------------------
# Full geometry summary
# ---------------------------------------------------------------------------

def full_geometry_report(
    rhos: List[torch.Tensor],
    dt: float = None,
) -> Dict[str, object]:
    """
    Compute all geometric quantities for the QFM trajectory.

    Returns:
        Comprehensive geometry report dict.
    """
    T = len(rhos) - 1
    if dt is None:
        dt = 1.0 / max(T, 1)

    gen_spectrum  = generator_spectrum_trajectory(rhos, dt)
    geo_curvature = geodesic_curvature(rhos, dt)
    pt_deviations = parallel_transport_deviation(rhos, dt)
    vf_magnitudes = vector_field_magnitude(rhos, dt)
    curv_traj     = curvature_along_trajectory(rhos, dt)
    div_traj      = flow_divergence(rhos, dt)
    geo_phase     = geometric_phase_estimate(rhos)

    return {
        'generator_coherent_rates':   gen_spectrum['coherent_rates'],
        'generator_dissipative_rates': gen_spectrum['dissipative_rates'],
        'geodesic_curvature':          geo_curvature,
        'parallel_transport_deviation': pt_deviations,
        'vector_field_magnitudes':     vf_magnitudes,
        'sectional_curvatures':        curv_traj,
        'flow_divergence':             div_traj,
        'geometric_phase':             geo_phase,
        'mean_curvature':             float(np.mean(geo_curvature)) if geo_curvature else 0.0,
        'mean_pt_deviation':          float(np.mean(pt_deviations)) if pt_deviations else 0.0,
    }
