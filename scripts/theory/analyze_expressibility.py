import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import pennylane as qml
import matplotlib.pyplot as plt

from src.qfm.ansatz import EHA_Circuit
from src.qfm.expressibility import expressibility_report

def run_expressibility_analysis(n_qubits=2, layer_depths=[1, 2, 4, 8], out_dir="results"):
    
    print(f"EHA Expressibility Analysis ({n_qubits} qubits)")
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    axes = axes.flatten()

    kl_vals, Q_vals = [], []
    
    for i, L in enumerate(layer_depths):
        circuit = EHA_Circuit(n_data=n_qubits, n_layers=L, n_ancilla=0)
        n_params = circuit.theta.numel()
        zero_state = torch.zeros(2**n_qubits, dtype=torch.complex128)
        zero_state[0] = 1.0
        def _wrapped_qnode(theta):
            return circuit.qnode(zero_state, theta.reshape(L, -1))
        kl, p_pqc, p_haar, bin_centers = expressibility_report(
            _wrapped_qnode,
            (n_params,), 
            n_samples=1500
        )
        from src.qfm.expressibility import expressibility_kl, meyer_wallach_entanglement
        kl, p_pqc, p_haar, bin_centers = expressibility_kl(_wrapped_qnode, (n_params,), n_samples=1500, return_histograms=True)
        Q = meyer_wallach_entanglement(_wrapped_qnode, (n_params,), n_samples=500)

        kl_vals.append(kl)
        Q_vals.append(Q)
        print(f"L={L}: KL={kl:.4f}, Q={Q:.4f}")
        ax = axes[i]
        width = bin_centers[1] - bin_centers[0]
        ax.bar(bin_centers, p_pqc, width=width*0.9, color="C1", label=f"EHA (L={L})")
        ax.plot(bin_centers, p_haar, color="C0", lw=2.5, ls="--", label="Haar Random")
        
        ax.set_title(f"Layers $L={L}$\nKL = {kl:.3f}, $Q$ = {Q:.3f}")
        ax.set_xlabel("Fidelity $F$")
        if i == 0:
            ax.set_ylabel("Probability Density $P(F)$")
        ax.legend(facecolor="white")
        ax.grid(True, ls="--")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, max(max(p_pqc), max(p_haar)) * 1.1)
    fig.suptitle(f"Quantum Neural Network Expressibility (Entanglement-varied Hardware-efficient Ansatz)\n{n_qubits} Qubits vs Haar-Random Distributions", y=1.05)
                 
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "expressibility_analysis.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved expressive analysis plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_expressibility_analysis(out_dir=args.out)
