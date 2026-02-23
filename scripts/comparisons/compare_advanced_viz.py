import sys
import os
import argparse
import torch
import numpy as np
import pennylane as qml
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.qfm.trainer import QFMTrainer
from src.qfm.utils import state_vector_to_density_matrix, mmd_loss
from src.qfm.metrics import purity, von_neumann_entropy

from src.qfm.apps.ring_state import generate_initial_ensemble, get_target_ensemble_rhos
from src.qfm.apps.entanglement import generate_separable_ensemble, loss_fn_entropy
from src.qfm.apps.tfim import build_tfim_hamiltonian, loss_fn_energy

from src.qfm.lindblad import (
    build_lindblad_superoperator, lindblad_evolve,
    amplitude_damping_jump, dephasing_jump
)

from src.qfm.visualization.flow_vs_diffusion import (
    plot_flow_vs_diffusion, _rho_to_vec, _fit_pca
)
from src.qfm.visualization.wigner_viz import plot_wigner_sequence, animate_wigner
from src.qfm.visualization.state_manifold import (
    plot_state_manifold_2d, plot_state_manifold_3d
)
from src.qfm.visualization.quantum_velocity_field import plot_velocity_field
from src.qfm.visualization.spectrum_dynamics import plot_spectrum_dynamics

def simulate_ring_state(T_steps=15, M=40, n_layers=5, threshold=0.01, lr=0.1):
    
    print("\n[sim] Running Ring State for advanced viz …")
    initial_pure, _ = generate_initial_ensemble(M)
    trainer = QFMTrainer(n_data=1, n_ancilla=0, n_layers=n_layers,
                         T_steps=T_steps, threshold=threshold, lr=lr)

    ensemble_snapshots = [[state_vector_to_density_matrix(s) for s in initial_pure]]
    current = initial_pure

    def mmd_loss_fn(rhos_gen, targets):
        return sum(mmd_loss(g, t) for g, t in zip(rhos_gen, targets)) / len(rhos_gen)

    for tau in range(1, T_steps + 1):
        target_rhos = get_target_ensemble_rhos(initial_pure, tau, T_steps)
        current, epoch_losses, prev_model = trainer.train_step(
            tau, current, mmd_loss_fn,
            lambda t: get_target_ensemble_rhos(initial_pure, t, T_steps),
            max_epochs=50, prev_model=None
        )
        snap = [state_vector_to_density_matrix(s) for s in current]
        ensemble_snapshots.append(snap)
        print(f"  τ={tau:2d}/{T_steps}  avg_purity={np.mean([float(purity(r)) for r in snap]):.4f}")

    return ensemble_snapshots

def run_flow_vs_diffusion(ensemble_snapshots, out_dir):
    print("\n[viz] Flow vs Diffusion …")
    T = len(ensemble_snapshots) - 1
    n_qubits = 1
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)
    H_np = (-Z).numpy()                            
    jump_ops_np = [amplitude_damping_jump(1, 0).numpy()]
    gammas = [0.04]
    M_traj = min(len(ensemble_snapshots[0]), 20)
    qfm_trajectories = []
    for m in range(M_traj):
        traj = [ensemble_snapshots[tau][m] for tau in range(T + 1)]
        qfm_trajectories.append(traj)

    plot_flow_vs_diffusion(
        qfm_trajectories, H_np, jump_ops_np, gammas,
        n_diff_trajectories=15,
        out_path=os.path.join(out_dir, "flow_vs_diffusion.png")
    )

def run_wigner(ensemble_snapshots, out_dir):
    print("\n[viz] Wigner & Husimi …")
    mean_rhos = [
        torch.mean(torch.stack(snap), dim=0)
        for snap in ensemble_snapshots
    ]
    plot_wigner_sequence(mean_rhos, n_panels=5,
                         out_path=os.path.join(out_dir, "wigner_sequence.png"))
    try:
        animate_wigner(mean_rhos, out_path=os.path.join(out_dir, "wigner_animation.gif"))
    except Exception as e:
        print(f"  [warn] animation skipped: {e}")

def run_manifold(ensemble_snapshots, out_dir):
    print("\n[viz] State Manifold PCA …")
    plot_state_manifold_2d(ensemble_snapshots,
                           out_path=os.path.join(out_dir, "state_manifold_2d.png"))
    plot_state_manifold_3d(ensemble_snapshots,
                           out_path=os.path.join(out_dir, "state_manifold_3d.png"))

def run_velocity_field(ensemble_snapshots, out_dir):
    print("\n[viz] Velocity Field …")
    plot_velocity_field(ensemble_snapshots,
                        out_path=os.path.join(out_dir, "quantum_velocity_field.png"))

def run_spectrum(ensemble_snapshots, out_dir):
    print("\n[viz] Spectral Dynamics (3-Qubit Condensation) …")
    from src.qfm.apps.tfim import build_tfim_hamiltonian, loss_fn_energy
    import torch
    
    nq = 3
    H_s = build_tfim_hamiltonian(nq, 0.0)
    H_t = build_tfim_hamiltonian(nq, 1.0)
    T_steps = 15
    M_states = 32
    
    trainer = QFMTrainer(n_data=nq, n_ancilla=0, n_layers=4, T_steps=T_steps, lr=0.08)
    torch.manual_seed(42)
    current = torch.randn(M_states, 2**nq, dtype=torch.complex128)
    current = current / torch.linalg.norm(current, dim=1, keepdim=True)
    
    rhos = []
    rhos.append(sum(state_vector_to_density_matrix(c) for c in current) / M_states)
    
    prev = None
    for tau in range(1, T_steps + 1):
        g_tau = tau / T_steps
        H_tau = (1 - g_tau) * H_s + g_tau * H_t
        current, epoch_losses, prev = trainer.train_step(
            tau, current, lambda r, t: loss_fn_energy(r, t), lambda t: H_tau, max_epochs=45, prev_model=prev
        )
        rho_tau = sum(state_vector_to_density_matrix(c) for c in current) / M_states
        rhos.append(rho_tau)
        print(f"  τ={tau:2d}/{T_steps}  avg_loss={np.mean(epoch_losses):.4f}")
        
    plot_spectrum_dynamics(rhos, n_qubits_A=nq, n_qubits_B=0,
                           out_path=os.path.join(out_dir, "spectrum_dynamics.png"))

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--skip-flow",     action="store_true")
    parser.add_argument("--skip-wigner",   action="store_true")
    parser.add_argument("--skip-manifold", action="store_true")
    parser.add_argument("--skip-velocity", action="store_true")
    parser.add_argument("--skip-spectrum", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ensemble_snapshots = simulate_ring_state()

    if not args.skip_flow:     run_flow_vs_diffusion(ensemble_snapshots, args.out)
    if not args.skip_wigner:   run_wigner(ensemble_snapshots, args.out)
    if not args.skip_manifold: run_manifold(ensemble_snapshots, args.out)
    if not args.skip_velocity: run_velocity_field(ensemble_snapshots, args.out)
    if not args.skip_spectrum: run_spectrum(ensemble_snapshots, args.out)

    print("  ALL ADVANCED VISUALIZATIONS COMPLETE")
    print(f"  Check {args.out}/ for all figures")
