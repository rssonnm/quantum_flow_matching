import argparse
import os
import sys

# Standard path insertion
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.qfm.trainer import QFMTrainer
from src.qfm.apps.tfim import build_tfim_hamiltonian, loss_fn_energy
from src.qfm.utils import state_vector_to_density_matrix

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'primary':   "C4",
    'secondary': "C7",
    'tertiary':  "C6",
    'accent':    "C1",
    'dark':      "C2",
    'light':     "C7",
}

def train_qfm_collect_all(n_qubits=3, T_steps=15, M=20, threshold=0.05):
    print(f"Training QFM on TFIM (n={n_qubits}, T={T_steps}, M={M})")

    H_0 = build_tfim_hamiltonian(n_qubits, 0.0)
    ev0, evec0 = torch.linalg.eigh(H_0)
    gs0 = evec0[:, 0]
    initial_pure = torch.stack([gs0 for _ in range(M)])

    trainer = QFMTrainer(
        n_data=n_qubits, n_ancilla=0, n_layers=5,
        T_steps=T_steps, threshold=threshold, lr=0.1
    )

    rhos = [torch.stack([state_vector_to_density_matrix(s) for s in initial_pure]).mean(dim=0)]
    rhos_per_member = [[state_vector_to_density_matrix(s) for s in initial_pure]]
    target_rhos = []
    all_losses = []
    g_values = [0.0]

    current = initial_pure
    prev_model = None

    for tau in range(1, T_steps + 1):
        g_tau = 1.0 * tau / T_steps
        g_values.append(g_tau)
        H_target = build_tfim_hamiltonian(n_qubits, g_tau)

        ev_t, evec_t = torch.linalg.eigh(H_target)
        gs_t = evec_t[:, 0]
        rho_target = state_vector_to_density_matrix(gs_t)
        target_rhos.append(rho_target)

        current, losses, prev_model = trainer.train_step(
            tau, current,
            lambda rhos_gen, tgt: loss_fn_energy(rhos_gen, tgt),
            lambda t: H_target, max_epochs=40, prev_model=prev_model
        )

        rho_t = torch.stack([state_vector_to_density_matrix(s) for s in current]).mean(dim=0)
        rhos.append(rho_t)
        rhos_per_member.append([state_vector_to_density_matrix(s) for s in current])
        all_losses.append(losses)

    return trainer, rhos, rhos_per_member, target_rhos, all_losses, g_values, n_qubits, T_steps
