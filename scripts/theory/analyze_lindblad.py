import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.qfm.lindblad import lindblad_evolve, amplitude_damping_jump
from src.qfm.apps.tfim import build_tfim_hamiltonian, loss_fn_energy
from src.qfm.utils import state_vector_to_density_matrix as sv2rho, partial_trace_pure_to_mixed
from src.qfm.trainer import QFMTrainer
from src.qfm.metrics import bures_distance, von_neumann_entropy, purity

def run_lindblad_comparison(n_qubits=2, gamma=0.05, T_steps=10, out_dir="results"):
    print(f"Lindblad vs QFM Comparison (n={n_qubits}, gamma={gamma})")
    
    H = build_tfim_hamiltonian(n_qubits, 0.5)
    jump_ops = [amplitude_damping_jump(n_qubits, k) for k in range(n_qubits)]
    gammas   = [gamma] * n_qubits
    psi0 = torch.ones(2**n_qubits, dtype=torch.complex128) / np.sqrt(2**n_qubits)
    rho0 = sv2rho(psi0)

    times = [t / T_steps for t in range(T_steps + 1)]
    lindblad_rhos = lindblad_evolve(rho0, H, jump_ops, gammas, times)

    trainer = QFMTrainer(n_data=n_qubits, n_ancilla=0, T_steps=T_steps)
    M = 10; current = torch.stack([psi0 for _ in range(M)])
    qfm_rhos = [rho0]
    for tau in range(1, T_steps + 1):
        H_t = build_tfim_hamiltonian(n_qubits, 0.5)
        current, _, _ = trainer.train_step(tau, current, loss_fn_energy, lambda t: H_t, max_epochs=20)
        qfm_rhos.append(torch.mean(torch.stack([sv2rho(s) for s in current]), dim=0))
    lind_purities, qfm_purities = [], []
    lind_entropies, qfm_entropies = [], []
    
    for rl, rq in zip(lindblad_rhos, qfm_rhos):
        lind_purities.append(float(torch.real(torch.trace(rl @ rl))))
        qfm_purities.append(float(torch.real(torch.trace(rq @ rq))))
        evals_l = torch.linalg.eigvalsh(rl)
        evals_l = evals_l[evals_l > 1e-12]
        lind_entropies.append(float(-torch.sum(evals_l * torch.log2(evals_l))))
        
        evals_q = torch.linalg.eigvalsh(rq)
        evals_q = evals_q[evals_q > 1e-12]
        qfm_entropies.append(float(-torch.sum(evals_q * torch.log2(evals_q))))

    bures = [float(bures_distance(rq, rl)) for rq, rl in zip(qfm_rhos, lindblad_rhos)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(times, bures, "o-", color="C0", lw=2.5, label="QFM $\\leftrightarrow$ Exact Lindblad")
    ax.set_title("Bures Distance $\\mathcal{D}_B(\\rho_{\\mathrm{QFM}}, \\rho_{\\mathrm{Lindblad}})$")
    ax.set_xlabel("Evolution Time $t$")
    ax.set_ylabel("Distance")
    ax.grid(True, ls="--")
    ax.legend(facecolor="white")
    ax = axes[1]
    ax.plot(times, lind_purities, "--", color="C2", lw=2, label="Exact Object $\\mathcal{L}$")
    ax.plot(times, qfm_purities, "s-", color="C1", lw=2.5, label="QFM Generative")
    ax.set_title("State Purity $\\mathrm{Tr}(\\rho^2)$")
    ax.set_xlabel("Evolution Time $t$")
    ax.grid(True, ls="--")
    ax.legend(facecolor="white")
    ax = axes[2]
    ax.plot(times, lind_entropies, "--", color="C2", lw=2, label="Exact Object $\\mathcal{L}$")
    ax.plot(times, qfm_entropies, "d-", color="C3", lw=2.5, label="QFM Generative")
    ax.set_title("Von Neumann Entropy $S(\\rho)$")
    ax.set_xlabel("Evolution Time $t$")
    ax.grid(True, ls="--")
    ax.legend(facecolor="white")

    fig.suptitle(f"Open Quantum System Trajectories: Lindblad Master Equation vs Generative QFM\n($n={n_qubits}$ qubits, Damping $\\gamma={gamma}$)", y=1.05)

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "lindblad_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved advanced Lindblad comparison to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_lindblad_comparison(out_dir=args.out)
