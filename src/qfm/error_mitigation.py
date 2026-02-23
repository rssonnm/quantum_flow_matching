"""
error_mitigation.py — Hardware-ready error mitigation for NISQ-era QFM.

Implements:
  - Zero-Noise Extrapolation (ZNE) with Richardson and polynomial extrapolation
  - Gate count and circuit depth analysis for EHA ansatz
  - Effective noise rate modeling (T1/T2 decoherence + gate errors)
  - Noise-scaled circuit simulation via Pauli noise channel amplification
  - Mitigated expectation value estimation

References:
    Li & Benjamin, PRX 2017 (ZNE); Temme et al., PRL 2017;
    Giurgica-Tiron et al., QST 2020 (digital ZNE).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Dict, Callable, Optional

from .metrics import uhlmann_fidelity, bures_distance


# ---------------------------------------------------------------------------
# Gate count analysis
# ---------------------------------------------------------------------------

def gate_decomposition_cost(
    n_qubits: int,
    n_layers: int,
    n_ancilla: int = 0,
    include_ancilla: bool = True,
) -> Dict[str, int]:
    """
    Estimate native gate counts for an EHA-style ansatz after decomposition.

    EHA structure per layer:
      - n_q single-qubit Ry rotations (1 parameter each)
      - n_q single-qubit Rz rotations (1 parameter each)
      - (n_q - 1) CZ entangling gates (ring topology)
    Total for n_layers:
      - Single-qubit gates: 2 * n_q * n_layers
      - CZ gates:           (n_q - 1) * n_layers

    Args:
        n_qubits:        Number of data qubits.
        n_layers:        Number of variational layers.
        n_ancilla:       Number of ancilla qubits (Stinespring).
        include_ancilla: Whether to count ancilla gates.

    Returns:
        Dict with counts: 'single_qubit_gates', 'cx_gates', 'total_gates',
        'circuit_depth', 'two_qubit_fraction'.
    """
    n_q = n_qubits + (n_ancilla if include_ancilla else 0)
    single_q = 2 * n_q * n_layers
    cx       = (n_q - 1) * n_layers if n_q > 1 else 0
    total    = single_q + cx
    # Simplified depth estimate (series of layers)
    depth    = n_layers * (2 + 1)  # 2 single-q layers + 1 CZ layer per variational block
    return {
        'n_qubits_total':    n_q,
        'n_layers':          n_layers,
        'single_qubit_gates': single_q,
        'cx_gates':           cx,
        'total_gates':        total,
        'circuit_depth':      depth,
        'two_qubit_fraction': float(cx / (total + 1)),
    }


# ---------------------------------------------------------------------------
# Effective noise rate modeling
# ---------------------------------------------------------------------------

def effective_noise_rate(
    T1_us: float,
    T2_us: float,
    gate_time_ns: float,
    n_cx_gates: int,
    cx_error_rate: float = 1e-2,
    single_q_error_rate: float = 1e-3,
) -> Dict[str, float]:
    """
    Compute effective noise parameters for a circuit on realistic hardware.

    Models:
      - Depolarizing noise per gate
      - T1/T2 decoherence over total circuit time
      - Combined effective error rate

    Args:
        T1_us:              Relaxation time (microseconds).
        T2_us:              Dephasing time (microseconds).
        gate_time_ns:       Gate execution time (nanoseconds).
        n_cx_gates:         Number of 2-qubit gates in circuit.
        cx_error_rate:      Per-gate 2-qubit error probability.
        single_q_error_rate: Per-gate 1-qubit error probability.

    Returns:
        Dict with decoherence contributions and total effective error.
    """
    # Total circuit time (nanoseconds → microseconds)
    circuit_time_ns = n_cx_gates * gate_time_ns * 3  # ~3x overhead for CX
    circuit_time_us = circuit_time_ns / 1000.0

    # T1 / T2 error probability
    p_T1 = 1.0 - np.exp(-circuit_time_us / (T1_us + 1e-12))
    p_T2 = 1.0 - np.exp(-circuit_time_us / (T2_us + 1e-12))

    # Gate error accumulation (conservative: independent channels)
    p_gate = 1.0 - (1.0 - cx_error_rate) ** n_cx_gates

    # Combined effective noise (independent channels → additive rates for small p)
    p_eff = 1.0 - (1.0 - p_T1) * (1.0 - p_T2) * (1.0 - p_gate)

    # Expected fidelity after noise
    expected_fidelity = 1.0 - p_eff

    return {
        'circuit_time_us':    float(circuit_time_us),
        'p_T1_error':         float(p_T1),
        'p_T2_error':         float(p_T2),
        'p_gate_error':       float(p_gate),
        'p_effective':        float(min(p_eff, 1.0)),
        'expected_fidelity':  float(max(expected_fidelity, 0.0)),
        'T1_us':              T1_us,
        'T2_us':              T2_us,
    }


# ---------------------------------------------------------------------------
# Pauli noise channel for ZNE scaling
# ---------------------------------------------------------------------------

def apply_depolarizing_noise(
    rho: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """
    Apply n-qubit depolarizing channel E_p(ρ) = (1-p) ρ + p (I/d).

    Args:
        rho: Density matrix (d×d).
        p:   Depolarizing probability p ∈ [0, 1].

    Returns:
        Noisy density matrix.
    """
    d = rho.shape[0]
    identity = torch.eye(d, dtype=torch.complex128) / d
    return (1.0 - p) * rho + p * identity


def apply_dephasing_noise(
    rho: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """
    Apply n-qubit dephasing channel (phase damping).
    For a 1-qubit state: E(ρ) = [[ρ₀₀, (1-p)ρ₀₁], [(1-p)ρ₁₀, ρ₁₁]].
    Generalized here as diagonal-preserving damping.

    Args:
        rho: Density matrix.
        p:   Dephasing probability p ∈ [0, 1].

    Returns:
        Dephased density matrix.
    """
    mask = torch.ones_like(rho)
    d = rho.shape[0]
    for i in range(d):
        for j in range(d):
            if i != j:
                mask[i, j] = (1.0 - p)
    return rho * mask


# ---------------------------------------------------------------------------
# Zero-Noise Extrapolation (ZNE)
# ---------------------------------------------------------------------------

def zero_noise_extrapolation(
    observable_fn: Callable[[float], float],
    noise_factors: List[float] = None,
    method: str = 'richardson',
) -> Dict[str, float]:
    """
    Perform Zero-Noise Extrapolation to estimate the noiseless expectation value.

    The circuit is scaled by noise_factors (gate folding or noise amplification),
    and the results are extrapolated back to noise_factor = 0.

    Args:
        observable_fn: Function taking a noise_factor (float, ≥1) and returning
                       the observed expectation value (float).
        noise_factors: List of noise amplification factors (default: [1, 2, 3]).
        method:        'richardson' | 'polynomial' | 'exponential'.

    Returns:
        Dict with 'mitigated_value', 'raw_values', 'noise_factors', 'method'.
    """
    if noise_factors is None:
        noise_factors = [1.0, 2.0, 3.0]

    raw_values = [observable_fn(lam) for lam in noise_factors]

    if method == 'richardson':
        mitigated = _richardson_extrapolation(raw_values, noise_factors)
    elif method == 'polynomial':
        mitigated = _polynomial_extrapolation(raw_values, noise_factors)
    elif method == 'exponential':
        mitigated = _exponential_extrapolation(raw_values, noise_factors)
    else:
        raise ValueError(f"Unknown ZNE method: {method}")

    return {
        'mitigated_value': float(mitigated),
        'raw_values':      raw_values,
        'noise_factors':   noise_factors,
        'method':          method,
    }


def _richardson_extrapolation(values: List[float], factors: List[float]) -> float:
    """
    Richardson extrapolation to λ=0 using polynomial coefficients.
    For n points, the formula gives exact cancellation up to O(λ^n).
    """
    n = len(values)
    lam = np.array(factors, dtype=float)
    y   = np.array(values,  dtype=float)
    # Polynomial interpolation at λ=0
    # Coefficients from Vandermonde system
    V = np.vander(lam, n, increasing=True)
    try:
        c = np.linalg.solve(V.T @ V, V.T @ y)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(V, y, rcond=None)
    return float(c[0])  # constant term = value at λ=0


def _polynomial_extrapolation(values: List[float], factors: List[float]) -> float:
    """Polynomial fit, evaluate at λ=0."""
    deg = min(len(values) - 1, 3)
    coeffs = np.polyfit(factors, values, deg)
    return float(np.polyval(coeffs, 0.0))


def _exponential_extrapolation(values: List[float], factors: List[float]) -> float:
    """Fit y = A * exp(-b * λ) + c, evaluate at λ=0."""
    if len(values) < 3:
        return _richardson_extrapolation(values, factors)
    lam = np.array(factors, dtype=float)
    y   = np.array(values,  dtype=float)
    # Log-linear fit on (y - y[-1])
    offset = min(y.min() - 1e-6, 0)
    y_shifted = y - offset
    valid = y_shifted > 1e-12
    if valid.sum() < 2:
        return float(y[0])
    log_y = np.log(y_shifted[valid])
    A_mat = np.column_stack([np.ones(valid.sum()), lam[valid]])
    coef, *_ = np.linalg.lstsq(A_mat, log_y, rcond=None)
    return float(np.exp(coef[0]) + offset)


# ---------------------------------------------------------------------------
# ZNE on density matrix evolution
# ---------------------------------------------------------------------------

def zne_fidelity_mitigation(
    rho_noisy_fn: Callable[[float], torch.Tensor],
    rho_target: torch.Tensor,
    noise_factors: List[float] = None,
    method: str = 'richardson',
) -> Dict[str, float]:
    """
    Apply ZNE to estimate the noiseless fidelity F(ρ_ideal, ρ_target).

    Args:
        rho_noisy_fn: Function taking noise_factor → noisy density matrix.
        rho_target:   Target (ideal) density matrix.
        noise_factors: Noise amplification factors.
        method:        Extrapolation method.

    Returns:
        Dict with 'mitigated_fidelity', 'raw_fidelities', etc.
    """
    if noise_factors is None:
        noise_factors = [1.0, 2.0, 3.0]

    def fid_fn(lam: float) -> float:
        rho = rho_noisy_fn(lam)
        return float(uhlmann_fidelity(rho, rho_target))

    result = zero_noise_extrapolation(fid_fn, noise_factors, method)
    result['mitigated_fidelity'] = min(max(result['mitigated_value'], 0.0), 1.0)
    result['raw_fidelities'] = result.pop('raw_values')
    return result


# ---------------------------------------------------------------------------
# Mitigated trajectory analysis
# ---------------------------------------------------------------------------

def noisy_trajectory(
    clean_rhos: List[torch.Tensor],
    noise_per_step: float,
    noise_type: str = 'depolarizing',
) -> List[torch.Tensor]:
    """
    Simulate a noisy trajectory by applying noise channel at each step.

    Args:
        clean_rhos:     Ideal trajectory density matrices.
        noise_per_step: Noise probability p per step.
        noise_type:     'depolarizing' | 'dephasing'.

    Returns:
        List of noisy density matrices.
    """
    out = []
    rho = clean_rhos[0].clone()
    out.append(rho)
    for tau in range(1, len(clean_rhos)):
        rho = clean_rhos[tau].clone()
        if noise_type == 'depolarizing':
            rho = apply_depolarizing_noise(rho, noise_per_step)
        elif noise_type == 'dephasing':
            rho = apply_dephasing_noise(rho, noise_per_step)
        out.append(rho)
    return out


def mitigated_fidelity_trajectory(
    clean_rhos: List[torch.Tensor],
    rho_target: torch.Tensor,
    noise_levels: List[float] = None,
) -> Dict[str, list]:
    """
    Compute raw, noisy, and ZNE-mitigated fidelity curves for the full trajectory.

    Args:
        clean_rhos:   Ideal trajectory.
        rho_target:   Target state.
        noise_levels: List of noise rates to analyse [default: 3 levels].

    Returns:
        Dict with lists for 'ideal', 'noisy', 'mitigated' fidelities per step.
    """
    if noise_levels is None:
        noise_levels = [0.02, 0.04, 0.06]

    T = len(clean_rhos)
    ideal_fids   = [float(uhlmann_fidelity(r, rho_target)) for r in clean_rhos]

    # Noisy at first noise level for visualization
    noisy_rhos = noisy_trajectory(clean_rhos, noise_levels[0])
    noisy_fids = [float(uhlmann_fidelity(r, rho_target)) for r in noisy_rhos]

    # ZNE-mitigated per step
    mitigated_fids = []
    for tau in range(T):
        obs_vals = []
        for p in noise_levels:
            rho_n = apply_depolarizing_noise(clean_rhos[tau], p)
            obs_vals.append(float(uhlmann_fidelity(rho_n, rho_target)))
        mit = _richardson_extrapolation(obs_vals, noise_levels)
        mitigated_fids.append(float(min(max(mit, 0.0), 1.0)))

    return {
        'ideal_fidelities':    ideal_fids,
        'noisy_fidelities':    noisy_fids,
        'mitigated_fidelities': mitigated_fids,
        'noise_levels':         noise_levels,
    }
