"""
ansatz.py — Entanglement-varied Hardware-efficient Ansatz (EHA).

This module implements the EHA architecture using PennyLane and PyTorch.
The EHA is designed for efficient state preparation on NISQ devices.
"""
import pennylane as qml
import torch
import torch.nn as nn

class EHA_Circuit(nn.Module):
    """
    Hardware-efficient ansatz with parameterized rotations and entangling gates.
    """
    def __init__(self, n_data, n_layers, n_ancilla=0, dev_name="default.qubit"):
        super().__init__()
        self.n_data = n_data
        self.n_ancilla = n_ancilla
        self.n_layers = n_layers
        self.total_qubits = n_data + n_ancilla
        # Using default.qubit for fast backprop on pure vectors. We will TRACE out ancillas in PyTorch.
        self.dev = qml.device(dev_name, wires=self.total_qubits)
        
        # 3 rotation params per qubit + 3 coupling params per pairs
        self.num_pairs = self.total_qubits - 1 if self.total_qubits > 1 else 0
        self.params_per_layer = self.total_qubits * 3 + self.num_pairs * 3
        
        init_radius = 1.0 / self.n_layers
        self.theta = nn.Parameter(torch.empty(self.n_layers, self.params_per_layer).uniform_(-init_radius, init_radius))
        
        # Define the QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, params, ancilla_rot=0.0):
            # QubitStateVector prepares the |psi_data> |0_ancilla>
            qml.StatePrep(inputs, wires=range(self.total_qubits))
            
            if self.n_ancilla > 0:
                for i in range(self.n_data, self.total_qubits):
                    qml.RY(ancilla_rot, wires=i)
                    
            # Apply EHA layers
            for l in range(self.n_layers):
                idx = 0
                for w in range(self.total_qubits):
                    qml.RX(params[l, idx], wires=w)
                    qml.RY(params[l, idx+1], wires=w)
                    qml.RZ(params[l, idx+2], wires=w)
                    idx += 3
                for w in range(self.total_qubits - 1):
                    qml.IsingXX(params[l, idx], wires=[w, w+1])
                    qml.IsingYY(params[l, idx+1], wires=[w, w+1])
                    qml.IsingZZ(params[l, idx+2], wires=[w, w+1])
                    idx += 3
            return qml.state()
            
        self.qnode = _circuit

    def forward(self, x, ancilla_rot=0.0):
        out_states = []
        for state in x:
            out_states.append(self.qnode(state, self.theta, ancilla_rot))
        return torch.stack(out_states)
