import sys
import os
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.qfm.trainer import QFMTrainer
from src.qfm.utils import state_vector_to_density_matrix
from src.qfm.apps.tfim import build_tfim_hamiltonian, loss_fn_energy, magnetization_mixed
from src.qfm.metrics import ensemble_purity
from src.qfm.visualization.phase_diagram import plot_tfim_phase_diagram
from src.qfm.visualization.loss_curves import plot_loss_landscape

def analyze_tfim(n_qubits=4, T_steps=15, M=20, n_layers=5, threshold=0.05, lr=0.1, out_dir="results"):
    print("  TFIM PHASE TRANSITION  (%d qubits, T=%d)" % (n_qubits, T_steps))

    H_0 = build_tfim_hamiltonian(n_qubits, 0.0)
    ev, evec = torch.linalg.eigh(H_0)
    gs_pure = evec[:, 0]
    initial_states_pure = torch.stack([gs_pure for _ in range(M)])

    trainer = QFMTrainer(
        n_data=n_qubits, n_ancilla=0, n_layers=n_layers,
        T_steps=T_steps, threshold=threshold, lr=lr
    )

    g_values, mag_values, pur_values = [], [], []
    loss_history, switch_tau = [], []
    current_ensemble = initial_states_pure
    prev_model = None

    for tau in range(1, T_steps + 1):
        g_tau = 1.5 * tau / T_steps
        H_target = build_tfim_hamiltonian(n_qubits, g_tau)

        class _EpochRecorder:
            def __call__(self, rhos_gen, H_tgt):
                return loss_fn_energy(rhos_gen, H_tgt)

        recorder = _EpochRecorder()
        current_ensemble, epoch_losses, prev_model = trainer.train_step(
            tau, current_ensemble, recorder,
            lambda t: H_target, max_epochs=50, prev_model=prev_model
        )

        rhos_t = torch.stack([state_vector_to_density_matrix(s) for s in current_ensemble])
        avg_mag  = float(torch.mean(torch.stack([magnetization_mixed(r, n_qubits) for r in rhos_t])))
        avg_pur  = float(ensemble_purity(rhos_t))

        g_values.append(g_tau)
        mag_values.append(avg_mag)
        pur_values.append(avg_pur)
        loss_history.append(epoch_losses if epoch_losses else [0.0])

        if trainer.models[-1][1] > 0:
            switch_tau.append(tau)

        print(f"  τ={tau:2d}  g={g_tau:.2f}  |M|={avg_mag:.4f}  γ={avg_pur:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    plot_tfim_phase_diagram(g_values, mag_values, n_qubits=n_qubits,
                            purity_values=pur_values,
                            out_path=os.path.join(out_dir, "tfim_phase_diagram.png"))
    plot_loss_landscape(loss_history, switch_tau=switch_tau,
                        out_path=os.path.join(out_dir, "tfim_loss_landscape.png"),
                        title="TFIM — Energy Loss Landscape")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    analyze_tfim(out_dir=args.out)
