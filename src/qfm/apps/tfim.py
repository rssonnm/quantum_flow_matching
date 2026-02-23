import torch
from ..utils import expectation_mixed, state_vector_to_density_matrix

def build_tfim_hamiltonian(n_qubits, g_field):
    """
    Builds the Transverse-Field Ising Model Hamiltonian matrix.
    H = - \sum Z_i Z_{i+1} - g \sum X_i
    """
    dim = 2**n_qubits
    H = torch.zeros((dim, dim), dtype=torch.complex128)
    
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)
    
    # ZZ terms
    for i in range(n_qubits - 1):
        term = torch.eye(1, dtype=torch.complex128)
        for j in range(n_qubits):
            if j == i or j == i + 1:
                term = torch.kron(term, Z)
            else:
                term = torch.kron(term, I)
        H -= term
        
    # X terms
    for i in range(n_qubits):
        term = torch.eye(1, dtype=torch.complex128)
        for j in range(n_qubits):
            if j == i:
                term = torch.kron(term, X)
            else:
                term = torch.kron(term, I)
        H -= g_field * term
        
    return H

def get_ground_state_rho(H):
    """ returns the ground state density matrix of H """
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    gs = eigenvectors[:, 0]
    return state_vector_to_density_matrix(gs)

def loss_fn_energy(rhos_gen, H_target):
    """
    Minimize energy of generated states with respect to H_target.
    Average <H> over the ensemble of density matrices.
    """
    loss = 0.0
    for rho in rhos_gen:
        e = expectation_mixed(rho, H_target)
        loss += e
    return loss / len(rhos_gen)

def magnetization_mixed(rho, n_qubits):
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)
    mag = 0.0
    for i in range(n_qubits):
        term = torch.eye(1, dtype=torch.complex128)
        for j in range(n_qubits):
            if j == i:
                term = torch.kron(term, Z)
            else:
                term = torch.kron(term, I)
        mag += expectation_mixed(rho, term)
    return mag / n_qubits
