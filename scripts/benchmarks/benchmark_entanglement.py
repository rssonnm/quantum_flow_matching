import sys
import os
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.qfm.trainer import QFMTrainer
from src.qfm.utils import state_vector_to_density_matrix
from src.qfm.apps.entanglement import generate_separable_ensemble, loss_fn_entropy
from src.qfm.metrics import ensemble_von_neumann, ensemble_negativity
from src.qfm.visualization.entanglement_viz import plot_entropy_growth
from src.qfm.visualization.loss_curves import plot_loss_landscape

def analyze_entanglement(n_qubits=2, T_steps=10, M=20, n_layers=10, threshold=0.01, lr=0.1, out_dir="results"):
    print("  ENTANGLEMENT ENTROPY GROWTH  (%d qubits, T=%d)" % (n_qubits, T_steps))

    initial_states_pure = generate_separable_ensemble(n_qubits, M)
    trainer = QFMTrainer(
        n_data=n_qubits, n_ancilla=0, n_layers=n_layers,
        T_steps=T_steps, threshold=threshold, lr=lr
    )

    target_entropies = [tau / T_steps for tau in range(1, T_steps + 1)]
    rhos_per_tau = []
    loss_history, switch_tau = [], []
    current_ensemble = initial_states_pure
    prev_model = None

    for tau in range(1, T_steps + 1):
        tgt_ent = tau / T_steps

        class _EpochRecorder:
            def __call__(self, rhos_gen, tgt):
                return loss_fn_entropy(rhos_gen, tgt)

        current_ensemble, epoch_losses, prev_model = trainer.train_step(
            tau, current_ensemble, _EpochRecorder(),
            lambda t: tgt_ent, max_epochs=60, prev_model=prev_model
        )

        rhos_t = torch.stack([state_vector_to_density_matrix(s) for s in current_ensemble])
        rhos_per_tau.append(rhos_t)
        loss_history.append(epoch_losses if epoch_losses else [0.0])

        if trainer.models[-1][1] > 0:
            switch_tau.append(tau)

        avg_vn = float(ensemble_von_neumann(rhos_t))
        avg_neg = float(ensemble_negativity(rhos_t, 1, 1))
        print(f"  τ={tau:2d}  S(ρ)={avg_vn:.4f}  target={tgt_ent:.2f}  N(ρ)={avg_neg:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    plot_entropy_growth(rhos_per_tau, target_entropies, n_qubits_A=1, n_qubits_B=1,
                        out_path=os.path.join(out_dir, "entanglement_entropy_growth.png"))
    plot_loss_landscape(loss_history, switch_tau=switch_tau,
                        out_path=os.path.join(out_dir, "entanglement_loss_landscape.png"),
                        title="Entanglement Growth — Loss Landscape")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    analyze_entanglement(out_dir=args.out)
