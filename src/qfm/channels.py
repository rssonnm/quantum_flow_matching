"""
channels.py — Quantum Channel representations and transformations.

Provides tools for extracting Kraus operators, computing process matrices (Chi),
and Pauli Transfer Matrices (PTM) for CPTP maps.
"""
import torch
import numpy as np
from typing import List, Tuple


# Pauli basis (for PTM and chi-matrix)
_I  = torch.eye(2, dtype=torch.complex128)
_X  = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)
_Y  = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex128)
_Z  = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
_PAULI_1Q = [_I, _X, _Y, _Z]


def _pauli_basis_n(n: int) -> List[torch.Tensor]:
    """Returns a (normalized) n-qubit Pauli basis of 4^n matrices."""
    from functools import reduce
    basis_1 = [p / np.sqrt(2) for p in _PAULI_1Q]   # trace-ortho normalization
    if n == 1:
        return basis_1
    basis = basis_1
    for _ in range(n - 1):
        basis = [torch.kron(a, b) for a in basis for b in basis_1]
    return basis


# Kraus operator extraction

def kraus_from_unitary(U: torch.Tensor, n_data: int, n_ancilla: int) -> List[torch.Tensor]:
    """
    Extract Kraus operators from a joint unitary U over (data ⊗ ancilla),
    assuming ancilla starts in |0⟩ and is measured in the computational basis.

      K_r = ⟨r|_A  U  |0⟩_A   for r = 0, …, 2^n_ancilla - 1

    U shape: (d_data * d_ancilla, d_data * d_ancilla)
    Returns list of K_r tensors each of shape (d_data, d_data).
    """
    d_data    = 2 ** n_data
    d_ancilla = 2 ** n_ancilla

    # U reshaped to (d_data, d_ancilla, d_data, d_ancilla)
    # indices: (row_data, row_ancilla, col_data, col_ancilla)
    U_r = U.reshape(d_data, d_ancilla, d_data, d_ancilla)

    kraus_ops = []
    for r in range(d_ancilla):
        # K_r[i, j] = U[i, r, j, 0]  (ancilla starts in |0⟩, measured in |r⟩)
        K_r = U_r[:, r, :, 0]  # shape (d_data, d_data)
        kraus_ops.append(K_r)
    return kraus_ops


def kraus_from_eigenvectors(
    circuit_fn,
    params: torch.Tensor,
    n_data: int,
    n_ancilla: int,
) -> List[torch.Tensor]:
    """
    Numerically extract the unitary matrix from a PennyLane QNode returning
    qml.state(), then call kraus_from_unitary.
    """
    import pennylane as qml

    d_data    = 2 ** n_data
    d_ancilla = 2 ** n_ancilla
    d_total   = d_data * d_ancilla

    # Run the circuit on each basis state to build the unitary column-by-column
    U_cols = []
    for k in range(d_total):
        basis_state = torch.zeros(d_total, dtype=torch.complex128)
        basis_state[k] = 1.0
        col = circuit_fn(basis_state, params)
        U_cols.append(col.detach())
    U = torch.stack(U_cols, dim=1)  # (d_total, d_total)
    return kraus_from_unitary(U, n_data, n_ancilla)


# Completeness verification

def kraus_completeness_error(kraus_ops: List[torch.Tensor]) -> float:
    """||Σ_k K_k†K_k - I||_F  (should be ≈ 0 for a valid CPTP map)."""
    d = kraus_ops[0].shape[0]
    total = sum(K.conj().T @ K for K in kraus_ops)
    diff  = total - torch.eye(d, dtype=torch.complex128)
    return float(torch.linalg.matrix_norm(diff, ord='fro').real)


def apply_channel(kraus_ops: List[torch.Tensor], rho: torch.Tensor) -> torch.Tensor:
    """Apply the CPTP map E(rho) = Σ_k K_k rho K_k†."""
    return sum(K @ rho @ K.conj().T for K in kraus_ops)


# Process Matrix (chi-matrix / Choi matrix)

def choi_matrix(kraus_ops: List[torch.Tensor]) -> torch.Tensor:
    """
    Choi–Jamiołkowski isomorphism:
        C = Σ_k |K_k⟩⟩⟨⟨K_k|
    where |K⟩⟩ = vec(K) is the column-stacking vectorization.
    C has shape (d², d²).
    """
    d = kraus_ops[0].shape[0]
    C = torch.zeros((d * d, d * d), dtype=torch.complex128)
    for K in kraus_ops:
        vec_K = K.reshape(-1, 1)           # column vectorization
        C    += vec_K @ vec_K.conj().T
    return C


def chi_matrix(kraus_ops: List[torch.Tensor]) -> torch.Tensor:
    """
    Process matrix χ in the Pauli basis:
        χ_{mn} = Tr(K_m† K_n)   (over all pairs of Kraus operators)
    Returns (4^n × 4^n) matrix.
    """
    n_qubits = int(np.log2(kraus_ops[0].shape[0]))
    basis    = _pauli_basis_n(n_qubits)
    B        = len(basis)
    chi      = torch.zeros((B, B), dtype=torch.complex128)
    for m, P_m in enumerate(basis):
        for n, P_n in enumerate(basis):
            # chi_{mn} = sum_k Tr(P_m† K_k) * conj(Tr(P_n† K_k))
            for K in kraus_ops:
                tm = torch.trace(P_m.conj().T @ K)
                tn = torch.trace(P_n.conj().T @ K)
                chi[m, n] += tm * tn.conj()
    return chi


def channel_fidelity(chi1: torch.Tensor, chi2: torch.Tensor) -> float:
    """
    Process fidelity between two channels with chi-matrices chi1, chi2:
        F_proc = Tr(chi1 chi2) / d²
    """
    d2 = chi1.shape[0]
    return float(torch.real(torch.trace(chi1 @ chi2)) / d2)


# Pauli Transfer Matrix

def pauli_transfer_matrix(kraus_ops: List[torch.Tensor]) -> torch.Tensor:
    """
    PTM_mn = Tr(P_m  E(P_n)) / d   — real matrix if channel is unital.
    Eigenvalues ∈ [-1, 1]; PTM of identity channel = I.
    """
    n_qubits = int(np.log2(kraus_ops[0].shape[0]))
    d        = 2 ** n_qubits
    basis    = _pauli_basis_n(n_qubits)
    B        = len(basis)
    R        = torch.zeros((B, B), dtype=torch.complex128)
    for n_idx, P_n in enumerate(basis):
        EP_n = apply_channel(kraus_ops, P_n)
        for m_idx, P_m in enumerate(basis):
            R[m_idx, n_idx] = torch.trace(P_m.conj().T @ EP_n) / d
    return torch.real(R)


# Diamond Norm Upper Bound

def diamond_norm_upper_bound(kraus_ops_1: List[torch.Tensor],
                              kraus_ops_2: List[torch.Tensor]) -> float:
    """
    Upper bound on diamond norm via PTM operator norm:
        ||E1 - E2||_◇ ≤ d · ||PTM(E1) - PTM(E2)||_1
    This is a loose but analytically tractable bound.
    """
    R1 = pauli_transfer_matrix(kraus_ops_1)
    R2 = pauli_transfer_matrix(kraus_ops_2)
    d  = kraus_ops_1[0].shape[0]
    diff_norm = float(torch.linalg.matrix_norm(R1 - R2, ord=1))
    return d * diff_norm
