"""
circuit_viz.py — EHA circuit diagram visualizations.
"""
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
import torch

logger = logging.getLogger(__name__)

def draw_eha_circuit(
    n_data: int = 4,
    n_ancilla: int = 1,
    n_layers: int = 3,
    out_path: str = "results/eha_circuit.png",
):
    """
    Draws the EHA circuit as a matplotlib figure.
    """
    total_qubits = n_data + n_ancilla
    dev = qml.device("default.qubit", wires=total_qubits)
    num_pairs = total_qubits - 1 if total_qubits > 1 else 0
    params_per_layer = total_qubits * 3 + num_pairs * 3
    params = torch.zeros((n_layers, params_per_layer))

    @qml.qnode(dev)
    def circuit(params):
        if n_ancilla > 0:
            for i in range(n_data, total_qubits):
                qml.RY(0.0, wires=i)
                
        for l in range(n_layers):
            idx = 0
            for w in range(total_qubits):
                qml.RX(params[l, idx], wires=w)
                qml.RY(params[l, idx+1], wires=w)
                qml.RZ(params[l, idx+2], wires=w)
                idx += 3
            for w in range(num_pairs):
                qml.IsingXX(params[l, idx], wires=[w, w+1])
                qml.IsingYY(params[l, idx+1], wires=[w, w+1])
                qml.IsingZZ(params[l, idx+2], wires=[w, w+1])
                idx += 3
        return qml.state()

    fig, ax = qml.draw_mpl(circuit, decimals=2, style="pennylane")(params)
    ax.set_title(f"EHA Circuit (n_data={n_data}, n_ancilla={n_ancilla}, L={n_layers})")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved circuit diagram to {out_path}")
