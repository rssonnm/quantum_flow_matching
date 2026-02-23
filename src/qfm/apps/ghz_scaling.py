"""
ghz_scaling.py — GHZ, W, and Cluster state preparation scaling studies for QFM.

Provides:
  - GHZ, W-state, and cluster state density matrices for arbitrary N-qubit sizes
  - Maximum entanglement verification utilities
  - Scaling experiment runner: fidelity vs N_qubits
  - Circuit resource estimates for state preparation

These states serve as challenging targets for QFM's state preparation experiments.
GHZ states are maximally entangled (saturate entanglement bounds), making them
the hardest test of QFM's ability to generate entanglement from separable sources.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Dict, Tuple

from ..metrics import uhlmann_fidelity, von_neumann_entropy, purity, negativity
from ..entanglement_scaling import (
    ghz_density_matrix, w_density_matrix, cluster_state_density_matrix,
    entanglement_entropy_bipartite, concurrence,
)


# ---------------------------------------------------------------------------
# State preparation verification
# ---------------------------------------------------------------------------

def verify_ghz_properties(n_qubits: int) -> Dict[str, float]:
    """
    Verify that the constructed GHZ state has correct entanglement properties.

    Expected:
      - Purity: 1.0 (pure state)
      - Von Neumann entropy: 0 (pure)
      - Bipartite entanglement entropy: 1 bit (maximally entangled bipartition)
      - Negativity: 0.5 (2-qubit subsystem, max entangled)

    Returns:
        Dict of verified properties.
    """
    rho = ghz_density_matrix(n_qubits)
    pur  = float(purity(rho))
    svn  = float(von_neumann_entropy(rho))
    # Bipartite: A = qubit 0, B = rest
    S_E  = entanglement_entropy_bipartite(rho, n_qubits, [0])
    # For 2-qubit we can compute concurrence
    neg  = None
    if n_qubits == 2:
        neg = float(negativity(rho, 1, 1))

    result = {
        'n_qubits':              n_qubits,
        'purity':                pur,
        'von_neumann_entropy':   svn,
        'bipartite_entropy_S_E': S_E,
        'expected_S_E':          1.0,   # 1 ebit for GHZ
        'is_pure':               abs(pur - 1.0) < 1e-6,
    }
    if neg is not None:
        result['negativity'] = neg
    return result


def verify_w_properties(n_qubits: int) -> Dict[str, float]:
    """
    Verify W-state entanglement properties.

    W states have S_E = log₂(n) / n (single-qubit bipartition),
    and are robust under particle loss (unlike GHZ).
    """
    rho = w_density_matrix(n_qubits)
    pur  = float(purity(rho))
    svn  = float(von_neumann_entropy(rho))
    S_E  = entanglement_entropy_bipartite(rho, n_qubits, [0])
    return {
        'n_qubits':              n_qubits,
        'purity':                pur,
        'von_neumann_entropy':   svn,
        'bipartite_entropy_S_E': S_E,
        'theoretical_S_E':       float(-1/n_qubits * np.log2(1/n_qubits) - (1 - 1/n_qubits) * np.log2(max(1 - 1/n_qubits, 1e-15))),
    }


# ---------------------------------------------------------------------------
# Fidelity vs N_qubits scaling
# ---------------------------------------------------------------------------

def ghz_fidelity_scaling(
    trained_rhos: Dict[int, torch.Tensor],
) -> Dict[str, list]:
    """
    Compute GHZ fidelity for trained states at different system sizes.

    Args:
        trained_rhos: Dict mapping n_qubits → final trained density matrix.

    Returns:
        Dict with 'n_values', 'fidelities', 'entropies'.
    """
    n_vals, fids, entropies = [], [], []
    for n, rho in sorted(trained_rhos.items()):
        rho_ghz = ghz_density_matrix(n)
        fid = float(uhlmann_fidelity(rho, rho_ghz))
        S_E = entanglement_entropy_bipartite(rho, n, [0])
        n_vals.append(n)
        fids.append(fid)
        entropies.append(S_E)
    return {
        'n_values':       n_vals,
        'fidelities':     fids,
        'entropies':      entropies,
    }


def w_fidelity_scaling(trained_rhos: Dict[int, torch.Tensor]) -> Dict[str, list]:
    """W-state fidelity scaling analogous to ghz_fidelity_scaling."""
    n_vals, fids = [], []
    for n, rho in sorted(trained_rhos.items()):
        rho_w = w_density_matrix(n)
        fids.append(float(uhlmann_fidelity(rho, rho_w)))
        n_vals.append(n)
    return {'n_values': n_vals, 'fidelities': fids}


def cluster_fidelity_scaling(trained_rhos: Dict[int, torch.Tensor]) -> Dict[str, list]:
    """Cluster state fidelity scaling."""
    n_vals, fids = [], []
    for n, rho in sorted(trained_rhos.items()):
        rho_c = cluster_state_density_matrix(n)
        fids.append(float(uhlmann_fidelity(rho, rho_c)))
        n_vals.append(n)
    return {'n_values': n_vals, 'fidelities': fids}


# ---------------------------------------------------------------------------
# Circuit resource estimates for state preparation
# ---------------------------------------------------------------------------

def ghz_circuit_resources(n_qubits: int) -> Dict[str, int]:
    """
    Theoretical minimum circuit resource for GHZ preparation:
      H gate on qubit 0, then CNOT cascade: (n-1) CNOTs.
    Total depth: 2 (H layer + CNOT cascade in series).

    For comparison with QFM's EHA circuit.
    """
    return {
        'hadamard_gates': 1,
        'cnot_gates':     n_qubits - 1,
        'total_gates':    n_qubits,
        'circuit_depth':  2,  # H then cascade
        'two_qubit_fraction': (n_qubits - 1) / n_qubits,
    }


def w_circuit_resources(n_qubits: int) -> Dict[str, int]:
    """
    Recursive W-state circuit: requires O(n) Ry rotations + O(n) CNOTs.
    """
    return {
        'ry_gates':          n_qubits,
        'cnot_gates':        n_qubits - 1,
        'total_gates':       2 * n_qubits - 1,
        'circuit_depth':     n_qubits,
        'two_qubit_fraction': (n_qubits - 1) / (2 * n_qubits - 1),
    }


# ---------------------------------------------------------------------------
# Entanglement spectrum analysis
# ---------------------------------------------------------------------------

def entanglement_spectrum(rho: torch.Tensor, n_qubits: int, partition_A: List[int]) -> List[float]:
    """
    Compute the entanglement spectrum: eigenvalues of the reduced density matrix ρ_A.

    The entanglement spectrum {λ_i} satisfies:
        S_E = -Σ_i λ_i log₂(λ_i)  (Von Neumann entropy)
        S_2 = -log₂(Σ_i λ_i²)     (Rényi-2 entropy)

    Returns:
        Sorted eigenvalues of ρ_A (descending).
    """
    from ..entanglement_scaling import partial_trace
    rho_A = partial_trace(rho, partition_A, n_qubits)
    ev = torch.linalg.eigvalsh(rho_A).real
    ev_pos = torch.clamp(ev, min=0.0)
    return sorted(ev_pos.tolist(), reverse=True)


def entanglement_gap(spectrum: List[float]) -> float:
    """
    Entanglement gap: λ_1 - λ_2 (gap between two largest Schmidt values).
    A large gap indicates a near-product state; small gap → strongly entangled.
    """
    if len(spectrum) < 2:
        return float(spectrum[0]) if spectrum else 0.0
    return float(spectrum[0] - spectrum[1])


def multi_partite_entanglement_measure(rho: torch.Tensor, n_qubits: int) -> Dict[str, float]:
    """
    Genuine multipartite entanglement (GME) measure:
    Compute bipartite entanglement entropy for all balanced bipartitions
    and report statistics.

    Returns:
        Dict with 'mean_S_E', 'min_S_E', 'max_S_E', 'is_gme_candidate'.
    """
    from ..entanglement_scaling import partial_trace
    from itertools import combinations

    half = n_qubits // 2
    all_S_E = []
    for part_A in combinations(range(n_qubits), half):
        S_E = entanglement_entropy_bipartite(rho, n_qubits, list(part_A))
        all_S_E.append(S_E)

    mean_SE = float(np.mean(all_S_E))
    min_SE  = float(np.min(all_S_E))
    max_SE  = float(np.max(all_S_E))
    # GME candidate: all bipartitions have S_E > threshold
    is_gme  = min_SE > 0.1

    return {
        'mean_S_E':          mean_SE,
        'min_S_E':           min_SE,
        'max_S_E':           max_SE,
        'all_partitions_SE': all_S_E,
        'is_gme_candidate':  is_gme,
    }
