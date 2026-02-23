"""
entanglement_scaling.py — Multi-qubit entanglement analysis for Quantum Flow Matching.

Implements:
  - Bipartite entanglement entropy (reduced density matrix von Neumann entropy)
  - Quantum mutual information I(A:B) = S(A) + S(B) - S(AB)
  - Concurrence for 2-qubit mixed states (Wootters 1998)
  - GHZ and W state preparation fidelity
  - Entanglement generation rate along QFM trajectories
  - Scaling analysis over multiple N-qubit configurations

These results form the entanglement scaling experiment that validates QFM
as a genuine quantum advantage over classical approaches.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Dict, Callable, Optional

from .metrics import von_neumann_entropy, purity, uhlmann_fidelity, partial_transpose


# ---------------------------------------------------------------------------
# Partial trace (generalized)
# ---------------------------------------------------------------------------

def partial_trace(
    rho: torch.Tensor,
    keep: List[int],
    n_qubits: int,
) -> torch.Tensor:
    """
    Compute the reduced density matrix by tracing out all qubits NOT in 'keep'.

    Args:
        rho:      Full density matrix of shape (2^n, 2^n).
        keep:     List of qubit indices to retain (0-indexed).
        n_qubits: Total number of qubits.

    Returns:
        Reduced density matrix of shape (2^|keep|, 2^|keep|).
    """
    d_full = 2 ** n_qubits
    assert rho.shape == (d_full, d_full), f"Expected ({d_full},{d_full}), got {rho.shape}"

    trace_out = [i for i in range(n_qubits) if i not in keep]
    if not trace_out:
        return rho

    # Reshape to tensor form: (2, 2, ..., 2, 2, 2, ..., 2)  [2n indices]
    shape = [2] * (2 * n_qubits)
    rho_t = rho.reshape(shape)

    # Trace out each qubit in trace_out (from largest index to avoid index shifting)
    for q in sorted(trace_out, reverse=True):
        # Trace: contract indices q and q + n_qubits (after current reshape)
        curr_n = rho_t.ndim // 2
        rho_t = torch.einsum(
            _trace_einsum_str(curr_n, q),
            rho_t,
        )

    d_keep = 2 ** len(keep)
    return rho_t.reshape(d_keep, d_keep)


def _trace_einsum_str(n: int, q: int) -> str:
    """Build einsum string that traces over qubit q in a 2n-index tensor."""
    row_idx = list(range(n))
    col_idx = list(range(n, 2 * n))
    # Contract row_idx[q] and col_idx[q] with the same dummy label
    dummy = 2 * n  # new label
    # Build the einsum string
    all_idx = row_idx + col_idx
    all_idx[q] = dummy
    all_idx[n + q] = dummy

    def to_char(i):
        return chr(ord('a') + i)

    in_str  = ''.join(to_char(i) for i in all_idx)
    out_idx = [i for i in (row_idx + col_idx) if i != dummy and all_idx.count(i) == 1]
    out_str = ''.join(to_char(i) for i in out_idx)
    return f'{in_str}->{out_str}'


# ---------------------------------------------------------------------------
# Bipartite entanglement entropy
# ---------------------------------------------------------------------------

def entanglement_entropy_bipartite(
    rho: torch.Tensor,
    n_qubits: int,
    partition_A: List[int],
) -> float:
    """
    Bipartite entanglement entropy S(A) = -Tr(ρ_A log₂ ρ_A).

    Args:
        rho:         Full n-qubit density matrix.
        n_qubits:    Total number of qubits.
        partition_A: Qubit indices in subsystem A.

    Returns:
        Entanglement entropy S(A) in bits.
    """
    rho_A = partial_trace(rho, partition_A, n_qubits)
    return float(von_neumann_entropy(rho_A))


def mutual_information_bipartite(
    rho: torch.Tensor,
    n_qubits: int,
    partition_A: List[int],
) -> float:
    """
    Quantum mutual information:
        I(A:B) = S(A) + S(B) - S(AB)

    For a pure global state: I(A:B) = 2 S(A).

    Args:
        rho:         Full density matrix.
        n_qubits:    Total qubits.
        partition_A: Qubit indices in A (complement is B).

    Returns:
        Mutual information I(A:B) in bits.
    """
    partition_B = [i for i in range(n_qubits) if i not in partition_A]
    S_A  = entanglement_entropy_bipartite(rho, n_qubits, partition_A)
    S_B  = entanglement_entropy_bipartite(rho, n_qubits, partition_B)
    S_AB = float(von_neumann_entropy(rho))
    return float(S_A + S_B - S_AB)


# ---------------------------------------------------------------------------
# Concurrence (Wootters 1998) for 2-qubit states
# ---------------------------------------------------------------------------

def concurrence(rho_2q: torch.Tensor) -> float:
    """
    Concurrence C(ρ) for a 2-qubit mixed state:
        C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    where λᵢ are the square roots of eigenvalues of ρ(σ_y⊗σ_y)ρ*(σ_y⊗σ_y)
    in decreasing order.

    Reference: Wootters, PRL 80, 2245 (1998).

    Args:
        rho_2q: 4×4 density matrix of a 2-qubit system.

    Returns:
        Concurrence C ∈ [0, 1]. C=0 separable, C=1 maximally entangled.
    """
    assert rho_2q.shape == (4, 4), "concurrence requires 4x4 density matrix"
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    sysy = torch.kron(sy, sy)
    rho_np = rho_2q.detach().cpu().numpy().astype(complex)
    sysy_np = sysy.detach().cpu().numpy().astype(complex)
    # Tilde matrix
    rho_tilde = sysy_np @ rho_np.conj() @ sysy_np
    R = rho_np @ rho_tilde
    # Eigenvalues of R (guaranteed non-negative for valid density matrices)
    eigvals = np.linalg.eigvals(R)
    eigvals_r = np.real(eigvals)
    eigvals_r = np.clip(eigvals_r, 0, None)
    lambdas = np.sort(np.sqrt(eigvals_r))[::-1]
    C = max(0.0, float(lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))
    return C


def formation_entropy(concurrence_val: float) -> float:
    """
    Entanglement of formation E_F(C):
        E_F = h((1 + √(1-C²))/2)
    where h is the binary entropy.
    """
    C = float(min(max(concurrence_val, 0.0), 1.0))
    x = (1.0 + np.sqrt(max(1.0 - C**2, 0.0))) / 2.0
    if x <= 0 or x >= 1:
        return 0.0
    return float(-x * np.log2(x) - (1-x) * np.log2(1-x + 1e-15))


# ---------------------------------------------------------------------------
# GHZ and W state targets
# ---------------------------------------------------------------------------

def ghz_density_matrix(n_qubits: int) -> torch.Tensor:
    """
    |GHZ_N⟩ = (|0...0⟩ + |1...1⟩) / √2
    Returns the density matrix (2^N × 2^N).
    """
    d = 2 ** n_qubits
    psi = torch.zeros(d, dtype=torch.complex128)
    psi[0]    = 1.0 / np.sqrt(2)
    psi[d - 1] = 1.0 / np.sqrt(2)
    return psi.unsqueeze(1) @ psi.unsqueeze(0).conj()


def w_density_matrix(n_qubits: int) -> torch.Tensor:
    """
    |W_N⟩ = (|10...0⟩ + |010...0⟩ + ... + |0...01⟩) / √N
    Returns the density matrix.
    """
    d = 2 ** n_qubits
    psi = torch.zeros(d, dtype=torch.complex128)
    for k in range(n_qubits):
        idx = 2 ** (n_qubits - 1 - k)  # qubit k in |1⟩, rest |0⟩
        psi[idx] = 1.0 / np.sqrt(n_qubits)
    return psi.unsqueeze(1) @ psi.unsqueeze(0).conj()


def cluster_state_density_matrix(n_qubits: int) -> torch.Tensor:
    """
    Linear cluster state |C_N⟩ on n qubits.
    Prepared by: H^N → CZ on (0,1),(1,2),...,(N-2,N-1).
    Returns the density matrix by building it from scratch.
    """
    d = 2 ** n_qubits
    # Start from |+⟩^N = H^N |0...0⟩
    psi = torch.ones(d, dtype=torch.complex128) / np.sqrt(d)
    # Apply CZ gates (controlled-Z between adjacent qubits)
    for q in range(n_qubits - 1):
        # CZ_{q, q+1}: flip sign if both qubits are |1⟩
        for i in range(d):
            bit_q   = (i >> (n_qubits - 1 - q))     & 1
            bit_qp1 = (i >> (n_qubits - 2 - q))     & 1
            if bit_q == 1 and bit_qp1 == 1:
                psi[i] = -psi[i]
    return psi.unsqueeze(1) @ psi.unsqueeze(0).conj()


# ---------------------------------------------------------------------------
# State-specific fidelities
# ---------------------------------------------------------------------------

def ghz_fidelity(rho: torch.Tensor, n_qubits: int) -> float:
    """Fidelity of rho with |GHZ_N⟩."""
    rho_ghz = ghz_density_matrix(n_qubits)
    return float(uhlmann_fidelity(rho, rho_ghz))


def w_fidelity(rho: torch.Tensor, n_qubits: int) -> float:
    """Fidelity of rho with |W_N⟩."""
    rho_w = w_density_matrix(n_qubits)
    return float(uhlmann_fidelity(rho, rho_w))


def cluster_fidelity(rho: torch.Tensor, n_qubits: int) -> float:
    """Fidelity of rho with linear cluster state |C_N⟩."""
    rho_c = cluster_state_density_matrix(n_qubits)
    return float(uhlmann_fidelity(rho, rho_c))


# ---------------------------------------------------------------------------
# Entanglement generation rate along trajectory
# ---------------------------------------------------------------------------

def entanglement_generation_rate(
    rhos: List[torch.Tensor],
    n_qubits: int,
    partition_A: Optional[List[int]] = None,
) -> List[float]:
    """
    Compute ΔS_E(τ) = S_E(ρ_{τ+1}) - S_E(ρ_τ) at each step.

    Positive values indicate entanglement is being generated by QFM.

    Args:
        rhos:        Trajectory of density matrices.
        n_qubits:    System size.
        partition_A: Subsystem A (default: first half).

    Returns:
        List of entanglement generation rates per step.
    """
    if partition_A is None:
        partition_A = list(range(n_qubits // 2))
    entropies = [entanglement_entropy_bipartite(r, n_qubits, partition_A) for r in rhos]
    rates = [entropies[t + 1] - entropies[t] for t in range(len(rhos) - 1)]
    return rates


# ---------------------------------------------------------------------------
# Scaling analysis
# ---------------------------------------------------------------------------

def scaling_analysis(
    n_qubit_range: List[int],
    metric_fn: Callable[[int], float],
) -> Dict[str, list]:
    """
    Evaluate a metric over multiple system sizes and fit a scaling law.

    Args:
        n_qubit_range: List of n_qubit values to evaluate.
        metric_fn:     Function n_qubits → metric_value.

    Returns:
        Dict with 'n_values', 'metric_values', 'scaling_exponent', 'scaling_prefactor'.
    """
    vals = [metric_fn(n) for n in n_qubit_range]
    
    # Fit log(metric) = log(A) + alpha * log(n)
    n_arr = np.array(n_qubit_range, dtype=float)
    v_arr = np.array(vals, dtype=float)
    valid = v_arr > 1e-15
    exponent = 0.0
    prefactor = 0.0
    if valid.sum() >= 2:
        log_n = np.log(n_arr[valid])
        log_v = np.log(v_arr[valid])
        A_mat = np.column_stack([np.ones_like(log_n), log_n])
        coef, *_ = np.linalg.lstsq(A_mat, log_v, rcond=None)
        prefactor = float(np.exp(coef[0]))
        exponent  = float(coef[1])

    return {
        'n_values':        list(n_qubit_range),
        'metric_values':   vals,
        'scaling_exponent': exponent,
        'scaling_prefactor': prefactor,
    }
