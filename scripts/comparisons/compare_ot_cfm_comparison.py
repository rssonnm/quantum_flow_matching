
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import torch
import numpy as np
from src.classical_fm import CFMTrainer, sample_gaussian, sample_ring
from src.qfm.trainer import QFMTrainer
from src.qfm.utils import state_vector_to_density_matrix, mmd_loss
from src.qfm.apps.ring_state import generate_initial_ensemble, get_target_ensemble_rhos
from src.qfm.metrics import purity
from src.classical_fm.comparison_viz import plot_fm_vs_qfm

def sliced_wasserstein_2(x: torch.Tensor, y: torch.Tensor, n_proj=200) -> float:
    
    d = x.shape[1]
    total = 0.0
    for _ in range(n_proj):
        v = torch.randn(d); v = v / v.norm()
        px = (x @ v).sort().values
        py = (y @ v).sort().values
        total += float(((px - py) ** 2).mean())
    return (total / n_proj) ** 0.5

def run_classical_fm(n_iter=2000, batch=512, n_traj_steps=30, n_traj_samples=200):
    print("  CLASSICAL FLOW MATCHING  (OT-CFM, R²)")

    t0 = time.time()
    trainer = CFMTrainer(lr=3e-4)
    cfm_losses = trainer.train(n_iter=n_iter, batch=batch, use_ot=True, log_every=400)
    t_train = time.time() - t0
    print(f"  Training time: {t_train:.1f}s")
    x0_fixed = sample_gaussian(n_traj_samples)
    traj_list = [x0_fixed.clone()]
    x = x0_fixed.clone()
    dt = 1.0 / n_traj_steps
    trainer.model.eval()
    with torch.no_grad():
        for s in range(n_traj_steps):
            t_cur = torch.tensor(s * dt).expand(n_traj_samples)
            v = trainer.model(x, t_cur)
            x = x + dt * v
            traj_list.append(x.clone())
    x_gen = traj_list[-1]
    x_tgt = sample_ring(n_traj_samples)
    sw2 = sliced_wasserstein_2(x_gen, x_tgt, n_proj=200)
    n_params = sum(p.numel() for p in trainer.model.parameters())

    print(f"  Final Sliced-W₂(gen, target): {sw2:.4f}")
    print(f"  MLP parameter count:          {n_params:,}")

    return traj_list, cfm_losses, float(sw2), n_params, t_train

def run_quantum_fm(T_steps=20, M=40, n_layers=5, threshold=0.01, lr=0.1):
    print("  QUANTUM FLOW MATCHING  (PQC, Bloch sphere)")

    t0 = time.time()
    initial_pure, _ = generate_initial_ensemble(M)
    trainer = QFMTrainer(n_data=1, n_ancilla=1, n_layers=n_layers,
                         T_steps=T_steps, threshold=threshold, lr=lr)

    qfm_losses = []
    trajectory_states = [list(initial_pure)] 
    current = initial_pure

    def mmd_loss_fn(rhos_gen, targets):
        loss = sum(mmd_loss(g, t) for g, t in zip(rhos_gen, targets)) / len(rhos_gen)
        qfm_losses.append(float(loss))
        return loss

    prev_model = None
    for tau in range(1, T_steps + 1):
        current, epoch_losses, prev_model = trainer.train_step(
            tau, current, mmd_loss_fn,
            lambda t: get_target_ensemble_rhos(initial_pure, t, T_steps),
            max_epochs=30,
            prev_model=prev_model
        )
        trajectory_states.append(list(current))
        qfm_losses.extend(epoch_losses)
        rhos_t = [state_vector_to_density_matrix(s) for s in current]
        avg_pur = np.mean([float(purity(r)) for r in rhos_t])
        print(f"  τ={tau:2d}/{T_steps}  final epoch loss={epoch_losses[-1]:.5f}  purity={avg_pur:.4f}")

    t_train = time.time() - t0
    print(f"  Training time: {t_train:.1f}s")
    rhos_gen = [state_vector_to_density_matrix(s) for s in current]
    rhos_tgt = list(get_target_ensemble_rhos(initial_pure, T_steps, T_steps))
    final_mmd = float(sum(mmd_loss(g, t).item() for g, t in zip(rhos_gen, rhos_tgt)) / len(rhos_gen))

    from src.qfm.ansatz import EHA_Circuit
    circuit   = EHA_Circuit(1, n_layers, n_ancilla=1)
    n_params  = sum(p.numel() for p in circuit.parameters())

    print(f"  Final MMD:                    {final_mmd:.5f}")
    print(f"  PQC parameter count:          {n_params}")

    return trajectory_states, qfm_losses, float(final_mmd), n_params, t_train

def print_comparison_report(cfm_sw2, qfm_mmd, cfm_params, qfm_params,
                              cfm_time, qfm_time, cfm_losses, qfm_losses):
    print("  COMPARISON REPORT: Classical FM vs Quantum FM")
    print(f"{'Metric':<30} {'Classical FM':>15} {'Quantum FM':>15}")
    print("-" * 60)
    print(f"{'State space':<30} {'R² (Gaussian)':>15} {'S(H) Bloch':>15}")
    print(f"{'Parameters':<30} {cfm_params:>15,} {qfm_params:>15,}")
    print(f"{'Training time (s)':<30} {cfm_time:>15.1f} {qfm_time:>15.1f}")
    print(f"{'Best loss':<30} {min(cfm_losses):>15.5f} {min(qfm_losses):>15.5f}")
    print(f"{'Distribution dist.':<30} {'W₂='}{cfm_sw2:>9.4f} {'MMD='}{qfm_mmd:>8.5f}")
    print(f"{'Constraints':<30} {'∫p dx=1':>15} {'Tr[E(ρ)]=1, CP':>15}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfm-iter",  type=int, default=2000)
    parser.add_argument("--qfm-steps", type=int, default=20)
    parser.add_argument("--qfm-m",     type=int, default=40)
    args = parser.parse_args()
    cfm_traj, cfm_losses, cfm_sw2, cfm_params, cfm_time = run_classical_fm(
        n_iter=args.cfm_iter
    )
    qfm_traj, qfm_losses, qfm_mmd, qfm_params, qfm_time = run_quantum_fm(
        T_steps=args.qfm_steps, M=args.qfm_m
    )
    print_comparison_report(cfm_sw2, qfm_mmd, cfm_params, qfm_params,
                             cfm_time, qfm_time, cfm_losses, qfm_losses)
    print("\n[viz] Generating comparison figure …")
    plot_fm_vs_qfm(
        cfm_trajectories=cfm_traj,
        qfm_trajectories=qfm_traj,
        cfm_losses=cfm_losses,
        qfm_losses=qfm_losses,
        cfm_w2=cfm_sw2,
        qfm_mmd=qfm_mmd,
    )
    print("\n[✓] Done — results/fm_vs_qfm_comparison.png")
