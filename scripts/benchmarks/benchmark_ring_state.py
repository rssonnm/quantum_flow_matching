import sys
import os
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.qfm.trainer import QFMTrainer
from src.qfm.utils import mmd_loss, state_vector_to_density_matrix
from src.qfm.apps.ring_state import generate_initial_ensemble, get_target_ensemble_rhos
from src.qfm.visualization.bloch_sphere import plot_bloch_trajectory, animate_bloch_sphere
from src.qfm.visualization.density_matrix_viz import plot_density_matrix, plot_eigenspectrum_sequence
from src.qfm.visualization.loss_curves import plot_loss_landscape

def analyze_ring_state(T_steps=20, M=100, n_layers=5, threshold=0.01, lr=0.1, out_dir="results"):
    print("  RING STATE EVOLUTION  (1 qubit, T=%d, M=%d)" % (T_steps, M))

    initial_states_pure, _ = generate_initial_ensemble(M)
    trainer = QFMTrainer(
        n_data=1, n_ancilla=1, n_layers=n_layers,
        T_steps=T_steps, threshold=threshold, lr=lr
    )

    ensemble_snapshots  = [initial_states_pure]
    rhos_per_tau        = []
    loss_history        = []
    switch_tau          = []

    current_ensemble = initial_states_pure
    import src.qfm.trainer as _tr
    _orig = _tr.QFMTrainer.train_step

    for tau in range(1, T_steps + 1):
        target_rhos = get_target_ensemble_rhos(initial_states_pure, tau, T_steps)
        epoch_losses_ref = []

        def mmd_loss_wrapper(rhos_gen, targets):
            loss = 0.0
            for g, t in zip(rhos_gen, targets):
                loss += mmd_loss(g, t)
            return loss / len(rhos_gen)

        def _patched_train_step(self, tau_arg, ens, loss_fn, target_fn, max_epochs=100, prev_model=None):
            from src.qfm.ansatz import EHA_Circuit
            import torch.optim as optim
            model_n = EHA_Circuit(self.n_data, self.n_layers, n_ancilla=0)
            if prev_model is not None:
                model_n.load_state_dict(prev_model.state_dict())
            opt = optim.Adam(model_n.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=self.lr/100)
            
            inputs_n = ens
            target = target_fn(tau_arg)

            for epoch in range(max_epochs):
                opt.zero_grad()
                out_pure = model_n(inputs_n)
                out_rhos = torch.stack([state_vector_to_density_matrix(p) for p in out_pure])
                loss = loss_fn(out_rhos, target)
                loss.backward()
                opt.step()
                scheduler.step()
                epoch_losses_ref.append(abs(float(loss.item())))
                if loss.item() < self.threshold:
                    self.models.append((model_n, 0))
                    return [s.detach() for s in out_pure], epoch_losses_ref, model_n
            self.models.append((model_n, 0))
            return [s.detach() for s in out_pure], epoch_losses_ref, model_n

        _tr.QFMTrainer.train_step = _patched_train_step

        current_ensemble, epoch_losses, _ = trainer.train_step(
            tau, current_ensemble, mmd_loss_wrapper,
            lambda t: get_target_ensemble_rhos(initial_states_pure, t, T_steps),
            max_epochs=50
        )
        
        print(f"  τ={tau:2d}  epochs={len(epoch_losses):3d}  best_loss={min(epoch_losses):.5f}")
        loss_history.append(epoch_losses)

        rhos_t = torch.stack([state_vector_to_density_matrix(psi) for psi in current_ensemble])
        rhos_per_tau.append(rhos_t)
        ensemble_snapshots.append(current_ensemble)

        if trainer.models[-1][1] > 0:
            switch_tau.append(tau)

    _tr.QFMTrainer.train_step = _orig
    os.makedirs(out_dir, exist_ok=True)
    plot_bloch_trajectory(ensemble_snapshots, out_path=os.path.join(out_dir, "bloch_trajectory.png"))
    try:
        animate_bloch_sphere(ensemble_snapshots, out_path=os.path.join(out_dir, "bloch_animation.gif"))
    except Exception:
        pass
    for tau in [1, T_steps // 2, T_steps]:
        avg_rho = torch.mean(rhos_per_tau[tau - 1], dim=0)
        plot_density_matrix(
            avg_rho, tau,
            out_path=os.path.join(out_dir, f"density_matrix_tau{tau}.png"),
            title_prefix="Ring State | "
        )
    plot_loss_landscape(loss_history, switch_tau=switch_tau,
                        out_path=os.path.join(out_dir, "ring_loss_landscape.png"),
                        title="Ring State — MMD Loss Landscape")
    repr_rhos = [torch.mean(r, dim=0) for r in rhos_per_tau]
    plot_eigenspectrum_sequence(repr_rhos, out_path=os.path.join(out_dir, "ring_eigenspectrum.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    analyze_ring_state(out_dir=args.out)
