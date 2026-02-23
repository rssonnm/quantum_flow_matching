"""
utils.py — General utility functions for Quantum Flow Matching.
"""
import torch

def get_device():
    """Returns the optimal PyTorch device (MPS if available, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def state_vector_to_density_matrix(state_vector):
    """Converts a pure state vector into a density matrix."""
    return torch.outer(state_vector, torch.conj(state_vector))

def partial_trace_pure_to_mixed(state_vector, n_data_qubits, n_ancilla_qubits):
    """
    Computes the partial trace over the ancilla qubits of a pure state vector or a batch of them.
    Returns the reduced density matrix rho_data (dim: 2^n_data x 2^n_data).
    Assumes ancilla qubits are the LAST n_ancilla_qubits.
    """
    dim_data = 2**n_data_qubits
    dim_ancilla = 2**n_ancilla_qubits
    
    # Check if we have a batch dimension
    if state_vector.dim() == 1:
        psi_reshaped = state_vector.view(dim_data, dim_ancilla)
        rho_data = torch.matmul(psi_reshaped, torch.conj(psi_reshaped).T)
    else:
        # Batch mode
        batch_size = state_vector.shape[0]
        psi_reshaped = state_vector.view(batch_size, dim_data, dim_ancilla)
        # B x D x A matmul B x A x D
        rho_data = torch.bmm(psi_reshaped, torch.conj(psi_reshaped).transpose(1, 2))
        
    return rho_data

def fidelity_mixed(rho1, rho2):
    """
    Computes the Uhlmann fidelity between two density matrices rho1 and rho2.
    F(rho1, rho2) = (Tr \sqrt{\sqrt{rho1} rho2 \sqrt{rho1}})^2
    For simplicity and gradient stability in optimizations, we use the Frobenius inner product overlap:
    F_overlap = Tr(rho1 @ rho2) / (Tr(rho1) * Tr(rho2)) which simplifies to Tr(rho1 @ rho2) if traces are 1.
    If one state is pure (rho1 = |psi><psi|), F = <psi|rho2|psi>.
    """
    # Using the strict overlap Tr(rho1 @ rho2) as a highly differentiable proxy for MMD kernel
    return torch.real(torch.trace(torch.matmul(rho1, rho2)))

def mmd_loss(rho_gen, rho_target):
    """
    Maximum Mean Discrepancy (MMD) using the fidelity kernel between two state ensembles
    represented simply by their average density matrices, or for single matched density matrices.
    D_MMD = <F(rho_gen, rho_gen)> + <F(rho_target, rho_target)> - 2 <F(rho_gen, rho_target)>
    """
    f_gen_gen = fidelity_mixed(rho_gen, rho_gen)
    f_tar_tar = fidelity_mixed(rho_target, rho_target)
    f_gen_tar = fidelity_mixed(rho_gen, rho_target)
    return f_gen_gen + f_tar_tar - 2.0 * f_gen_tar

def expectation_mixed(rho, observable_matrix):
    """
    Computes Tr(rho @ O).
    """
    return torch.real(torch.trace(torch.matmul(rho, observable_matrix)))

def von_neumann_entropy(rho):
    """
    Calculates the Von Neumann entanglement entropy of a density matrix.
    S = -Tr(rho * log2(rho))
    """
    # Use eigh since rho is Hermitian
    eigenvalues = torch.linalg.eigvalsh(rho)
    # Filter eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues))
    return entropy

def set_academic_style():
    """
    Configures matplotlib to use a clean, publication-ready academic style with a white background.
    Run this at the beginning of any visualization script to globally override the dark theme.
    """
    import matplotlib.pyplot as plt
    plt.style.use('default')
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "lines.linewidth": 2.0,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "grid.color": "gray",
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })
    return True
