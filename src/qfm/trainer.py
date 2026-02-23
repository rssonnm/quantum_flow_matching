import logging
import torch
import torch.optim as optim
from .ansatz import EHA_Circuit
from .utils import partial_trace_pure_to_mixed, state_vector_to_density_matrix

logger = logging.getLogger(__name__)

class QFMTrainer:
    """
    Trainer for Quantum Flow Matching using density matrix formulations.

    Attributes:
        n_data (int): Number of system qubits.
        n_ancilla (int): Number of ancilla qubits for Stinespring dilation.
        n_layers (int): Number of layers in the EHA ansatz.
        T_steps (int): Total number of discrete flow steps.
        threshold (float): Convergence threshold for training energy.
        lr (float): Learning rate for optimization.
        models (list): List of trained models for each step tau.
    """
    def __init__(self, n_data, n_ancilla=1, n_layers=5, T_steps=20, threshold=0.05, lr=0.01):
        self.n_data = n_data
        self.n_ancilla = n_ancilla
        self.n_layers = n_layers
        self.T_steps = T_steps
        self.threshold = threshold
        self.lr = lr
        
        # We store the learned models for each step tau
        self.models = []

    def pad_state_with_ancilla(self, states_data_only, n_data, n_ancilla):
        """
        Pad n_data qubit pure states with n_ancilla qubits in |0> state.

        Args:
            states_data_only (torch.Tensor): Tensor of system states.
            n_data (int): Number of system qubits.
            n_ancilla (int): Number of ancilla qubits.

        Returns:
            torch.Tensor: Padded states |psi> \otimes |0>_a.
        """
        padded = []
        for state in states_data_only:
            pad = torch.zeros(2**(n_data + n_ancilla), dtype=torch.complex128)
            pad[:len(state)] = state
            padded.append(pad)
        return torch.stack(padded)

    def train_step(self, tau, initial_ensemble_pure, loss_fn, target_fn, max_epochs=100, prev_model=None):
        """
        Train the model for a specific flow step tau.

        Args:
            tau (int): Current step index.
            initial_ensemble_pure (torch.Tensor): Tensor of pure states |psi_{tau-1}>.
            loss_fn (callable): Loss function (rhos_gen, target).
            target_fn (callable): Target function returning density matrices or observables.
            max_epochs (int): Maximum number of optimization epochs.
            prev_model (EHA_Circuit, optional): Previous model for warm-starting.

        Returns:
            tuple: (final_ensemble, epoch_losses, model)
        """
        model_n = EHA_Circuit(self.n_data, self.n_layers, n_ancilla=self.n_ancilla)
        if prev_model is not None:
            model_n.load_state_dict(prev_model.state_dict())
            
        optimizer = optim.Adam(model_n.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=self.lr/100)
        
        inputs_n = initial_ensemble_pure
        target = target_fn(tau)
        
        logger.info(f"Starting Training Step {tau}/{self.T_steps}")
        epoch_losses = []
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            out_pure_states = model_n(inputs_n)
            
            # Convert to density matrices
            if self.n_ancilla > 0:
                stacked_pure = torch.stack([psi for psi in out_pure_states])
                out_rhos = partial_trace_pure_to_mixed(stacked_pure, self.n_data, self.n_ancilla)
            else:
                out_rhos = torch.stack([state_vector_to_density_matrix(psi) for psi in out_pure_states])
            
            loss = loss_fn(out_rhos, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            if loss.item() < self.threshold:
                logger.debug(f"Convergence reached at epoch {epoch} (Loss: {loss.item():.6f})")
                self.models.append((model_n, 0)) 
                return [s.detach() for s in out_pure_states], epoch_losses, model_n
                
        self.models.append((model_n, 0))
        return [s.detach() for s in out_pure_states], epoch_losses, model_n
