import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.qfm.trainer import QFMTrainer
from src.qfm.utils import state_vector_to_density_matrix, partial_trace_pure_to_mixed
from src.qfm.metrics import bures_distance, von_neumann_entropy
from src.qfm.apps.tfim import build_tfim_hamiltonian, loss_fn_energy
from src.qfm.dynamical_systems import (
    qfm_trajectory_action, 
    optimal_bures_geodesic_cost, 
    entropy_production, 
    loschmidt_echo_perturbation,
    quantum_work_heat_breakdown,
    calculate_ergotropy,
    coherence_evolution
)

def run_dynamics_analysis(n_qubits=3, T_steps=15, M=20, threshold=0.04, out_dir="results"):
    print(f"Advanced Thermodynamic Dashboard (n={n_qubits}, T={T_steps})")

    H_0 = build_tfim_hamiltonian(n_qubits, 0.0)
    ev0, evec0 = torch.linalg.eigh(H_0)
    initial_pure = torch.stack([evec0[:, 0] for _ in range(M)])

    trainer = QFMTrainer(n_data=n_qubits, n_ancilla=0, T_steps=T_steps, threshold=threshold)
    
    rhos = [torch.stack([state_vector_to_density_matrix(s) for s in initial_pure]).mean(dim=0)]
    Hs = [H_0]
    energies = [float(torch.real(torch.trace(rhos[0] @ H_0)))]
    
    current = initial_pure
    prev = None
    for tau in range(1, T_steps + 1):
        H_t = build_tfim_hamiltonian(n_qubits, tau/T_steps)
        Hs.append(H_t)
        current, _, prev = trainer.train_step(tau, current, loss_fn_energy, lambda t: H_t, max_epochs=50, prev_model=prev)
        rho_t = torch.stack([state_vector_to_density_matrix(s) for s in current]).mean(dim=0)
        rhos.append(rho_t)
        energies.append(float(torch.real(torch.trace(rho_t @ H_t))))
    work, heat = quantum_work_heat_breakdown(rhos, Hs)
    c_work = np.cumsum([0] + work)
    c_heat = np.cumsum([0] + heat)
    entropies = [float(von_neumann_entropy(r)) for r in rhos]
    ergotropies = [calculate_ergotropy(r, h) for r, h in zip(rhos, Hs)]
    coherences = coherence_evolution(rhos)
    echo = loschmidt_echo_perturbation(trainer, initial_pure, None, T_steps, epsilon=0.01)
    echo = [1.0] + echo # start at 1.0

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    axes[0, 0].plot(energies, 'k-o', label="Total Energy", lw=2)
    axes[0, 0].plot(c_work, '--', color="C6", label="Cumulative Work")
    axes[0, 0].plot(c_heat, ':', color="C4", label="Cumulative Heat")
    axes[0, 0].set_title("(a) First Law: dE = dW + dQ")
    axes[0, 0].set_xlabel("Step T"); axes[0, 0].legend()
    axes[0, 1].plot(entropies, 'o-', color="C3", label="Entropy S(ρ)")
    ax2_twin = axes[0, 1].twinx()
    ax2_twin.plot([float(torch.real(torch.trace(r @ r))) for r in rhos], 's--', color="C2", label="Purity")
    axes[0, 1].set_title("(b) Entropic Production & Purity")
    axes[0, 1].legend(loc='upper left'); ax2_twin.legend(loc='upper right')
    axes[1, 0].fill_between(range(len(ergotropies)), ergotropies, color="C1", label="Ergotropy")
    axes[1, 0].plot(ergotropies, color="C1", lw=2)
    axes[1, 0].plot(coherences, 'v-', color="C0", label="Coherence (C_l1)")
    axes[1, 0].set_title("(c) Quantum Resource Dynamics")
    axes[1, 0].legend()
    axes[1, 1].plot(echo, 'D-', color="C5", label="Loschmidt Echo")
    axes[1, 1].set_ylim(0.9, 1.01)
    axes[1, 1].set_title("(d) Dynamical Stability (Perturbation Resistance)")
    axes[1, 1].set_ylabel("Fidelity F(ρ, ρ_eps)"); axes[1, 1].legend()

    fig.suptitle(f"QFM Quantum Thermodynamic Analysis: {n_qubits} Qubits")
    out_path = os.path.join(out_dir, "dynamics_thermo_analysis.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved advanced dashboard to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results_demo")
    args = parser.parse_args()
    run_dynamics_analysis(out_dir=args.out)
