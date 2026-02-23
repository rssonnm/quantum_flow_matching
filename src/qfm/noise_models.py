"""
noise_models.py — Physical noise models for NISQ simulations.
"""
import torch

def depolarizing_channel(rho: torch.Tensor, p: float) -> torch.Tensor:
    """
    Applies a global depolarizing channel to an n-qubit density matrix.
    E(ρ) = (1-p)ρ + p/d I
    where d = 2^n.
    """
    if p == 0.0:
        return rho
    
    d = rho.shape[-1]
    I = torch.eye(d, dtype=rho.dtype, device=rho.device) / d
    return (1.0 - p) * rho + p * I

def amplitude_damping_channel_1q(rho: torch.Tensor, gamma: float, target_qubit: int, n_qubits: int) -> torch.Tensor:
    """
    Applies amplitude damping to a single target_qubit.
    Kraus operators: K0 = [[1, 0], [0, sqrt(1-gamma)]], K1 = [[0, sqrt(gamma)], [0, 0]]
    """
    if gamma == 0.0:
        return rho
        
    K0_1q = torch.tensor([[1.0, 0.0], [0.0, (1.0 - gamma)**0.5]], dtype=torch.complex128)
    K1_1q = torch.tensor([[0.0, gamma**0.5], [0.0, 0.0]], dtype=torch.complex128)
    
    I = torch.eye(2, dtype=torch.complex128)
    
    K0_ops = [K0_1q if i == target_qubit else I for i in range(n_qubits)]
    K1_ops = [K1_1q if i == target_qubit else I for i in range(n_qubits)]
    
    K0 = K0_ops[0]
    K1 = K1_ops[0]
    for i in range(1, n_qubits):
        K0 = torch.kron(K0, K0_ops[i])
        K1 = torch.kron(K1, K1_ops[i])
        
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

def apply_noise(rho_batch: torch.Tensor, noise_type: str, p: float, n_qubits: int) -> torch.Tensor:
    """
    Applies a selected noise model to a batch of density matrices.
    """
    if p == 0.0:
         return rho_batch
         
    noisy_rhos = []
    for rho in rho_batch:
        if noise_type == "depolarizing":
            r = depolarizing_channel(rho, p)
        elif noise_type == "amplitude_damping":
            # Apply uniformly to all qubits for simplicity
            r = rho
            for i in range(n_qubits):
                r = amplitude_damping_channel_1q(r, p, i, n_qubits)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        noisy_rhos.append(r)
        
    return torch.stack(noisy_rhos)
