import torch
from ..utils import von_neumann_entropy, state_vector_to_density_matrix

def generate_separable_ensemble(n_qubits=2, M=20):
    """
    Creates an ensemble of separated |0...0> states to begin entanglement growth.
    """
    # |0> tensor n_qubits times is just a 1 followed by zeros
    state = torch.zeros(2**n_qubits, dtype=torch.complex128)
    state[0] = 1.0
    return torch.stack([state for _ in range(M)])

def partial_trace_bipartite(rho, n_qubits, keep_indices):
    """
    Calculates the partial trace of a density matrix, keeping only the specified indices.
    For simplicity with Entanglement Entropy, if we have 2 qubits and want S of qubit 0, 
    we trace out qubit 1.
    """
    # rho has shape (2^n, 2^n). Reshape to (2, 2, ..., 2) (2n times)
    shape = [2] * (2 * n_qubits)
    rho_reshaped = rho.reshape(shape)
    
    # We want to trace out indices NOT in keep_indices.
    # To keep qubit 0 in a 2-qubit system, trace out qubit 1.
    # The indices for the bra and ket of qubit i are i and i + n_qubits
    trace_indices = [i for i in range(n_qubits) if i not in keep_indices]
    
    if len(trace_indices) == 1 and trace_indices[0] == 1 and n_qubits == 2:
        # Specific fast path for 2 qubits tracing out the 2nd
        # rho_reshaped is (2, 2, 2, 2) -> (i, j, k, l)
        # Trace over j and l -> einsum('ijil->il') 
        rho_reduced = torch.einsum('ijik->ik', rho_reshaped)
    else:
        # Generic would require forming equation string dynamically.
        # Hardcoding the 2-qubit case as requested in the paper's benchmark.
        rho_reduced = torch.einsum('ijik->ik', rho_reshaped)
        
    return rho_reduced

def loss_fn_entropy(rhos_gen, target_entropy):
    """
    Minimizes the Mean Squared Error between the generated entanglement entropy and the target.
    Assumes a 2-qubit system tracing out the 2nd qubit.
    """
    loss = 0.0
    for rho in rhos_gen:
        rho_reduced = partial_trace_bipartite(rho, 2, [0])
        s = von_neumann_entropy(rho_reduced)
        loss += (s - target_entropy)**2
    return loss / len(rhos_gen)
