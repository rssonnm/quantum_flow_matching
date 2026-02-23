import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.entanglement_scaling import (
    entanglement_entropy_bipartite, mutual_information_bipartite,
    concurrence, ghz_density_matrix, w_density_matrix,
    entanglement_generation_rate, ghz_fidelity, w_fidelity,
)
from src.qfm.metrics import von_neumann_entropy, purity, uhlmann_fidelity


def run_entanglement_scaling_analysis(out_dir="results"):
    print("Running Entanglement Scaling Analysis")
    os.makedirs(out_dir, exist_ok=True)

    n_qubit_range = [2, 3, 4]
    all_results = {}

    for nq in n_qubit_range:
        print(f"  Simulating {nq}-qubit QFM...")
        _, rhos, _, _, _, g_vals, nq_actual, T = train_qfm_collect_all(
            n_qubits=nq, T_steps=8, M=6
        )
        all_results[nq] = {
            'rhos': rhos,
            'g_vals': g_vals,
        }

    # ===== FIGURE: 4-panel Entanglement Analysis =====
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    colors_nq = ['C0', 'C2', 'C3', 'C5']

    # (a) Bipartite entanglement entropy S_E vs flow step for each N
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, nq in enumerate(n_qubit_range):
        rhos = all_results[nq]['rhos']
        half = [0]  # partition A = qubit 0
        S_E_curve = [entanglement_entropy_bipartite(r, nq, half) for r in rhos]
        tau_arr = np.linspace(0, 1, len(rhos))
        ax1.plot(tau_arr, S_E_curve, 'o-', color=colors_nq[idx], lw=2.5, ms=7,
                 markerfacecolor='white', markeredgewidth=2, label=f'$N={nq}$')
        # GHZ upper bound
        ax1.axhline(1.0, color='gray', ls=':', lw=1.2, alpha=0.5)

    ax1.set_xlabel('Normalized Time $\\tau$')
    ax1.set_ylabel('$S_E$ (bits)')
    ax1.set_title('(a) Bipartite Entanglement Entropy $S_E(\\tau)$\n(Subsystem: qubit 0 vs rest)', pad=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.axhline(1.0, color='red', ls='--', lw=1.5, alpha=0.7, label='GHZ max ($S_E=1$)')

    # (b) Entanglement generation rate dS_E/dτ
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, nq in enumerate(n_qubit_range):
        rhos = all_results[nq]['rhos']
        rates = entanglement_generation_rate(rhos, nq, [0])
        tau_mid = np.linspace(0, 1, len(rates))
        ax2.bar(tau_mid + idx * 0.02, rates, 1.0 / len(rates) * 0.9, alpha=0.75,
                color=colors_nq[idx], edgecolor='black', linewidth=0.5, label=f'$N={nq}$')
    ax2.axhline(0, color='black', lw=1)
    ax2.set_xlabel('Flow Step $\\tau$')
    ax2.set_ylabel('$\\Delta S_E / \\Delta\\tau$ (bits/step)')
    ax2.set_title('(b) Entanglement Generation Rate $\\dot{S}_E(\\tau)$', pad=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', ls='--', alpha=0.5)

    # (c) GHZ and W-state fidelity at final step vs N
    ax3 = fig.add_subplot(gs[1, 0])
    ghz_fids = []
    w_fids   = []
    for nq in n_qubit_range:
        rho_final = all_results[nq]['rhos'][-1]
        ghz_fids.append(ghz_fidelity(rho_final, nq))
        w_fids.append(w_fidelity(rho_final, nq))

    x = np.arange(len(n_qubit_range))
    ax3.bar(x - 0.2, ghz_fids, 0.35, color=COLORS['primary'], alpha=0.85,
            edgecolor='black', linewidth=0.8, label='GHZ Fidelity $F_{GHZ}$')
    ax3.bar(x + 0.2, w_fids, 0.35, color=COLORS['secondary'], alpha=0.85,
            edgecolor='black', linewidth=0.8, label='W-State Fidelity $F_W$')
    for xi, (fg, fw) in zip(x, zip(ghz_fids, w_fids)):
        ax3.text(xi - 0.2, fg + 0.015, f'{fg:.3f}', ha='center', fontsize=9, color=COLORS['primary'])
        ax3.text(xi + 0.2, fw + 0.015, f'{fw:.3f}', ha='center', fontsize=9, color=COLORS['secondary'])
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'$N={nq}$' for nq in n_qubit_range])
    ax3.set_ylabel('Final Fidelity $F$')
    ax3.set_title('(c) Final Step: GHZ vs W-State Fidelity', pad=12)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 1.15)
    ax3.grid(True, axis='y', ls='--', alpha=0.5)

    # (d) Mutual information I(A:B) evolution
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, nq in enumerate(n_qubit_range):
        rhos = all_results[nq]['rhos']
        MI_curve = [mutual_information_bipartite(r, nq, [0]) for r in rhos]
        tau_arr  = np.linspace(0, 1, len(rhos))
        ax4.plot(tau_arr, MI_curve, 'o-', color=colors_nq[idx], lw=2.5, ms=7,
                 markerfacecolor='white', markeredgewidth=2, label=f'$N={nq}$')

    ax4.set_xlabel('Normalized Time $\\tau$')
    ax4.set_ylabel('Mutual Information $I(A:B)$ (bits)')
    ax4.set_title('(d) Quantum Mutual Information $I(A:B)(\\tau)$', pad=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, ls='--', alpha=0.5)

    fig.suptitle('Quantum Flow Matching — Multi-Qubit Entanglement Scaling',
                 y=0.98, fontsize=14, color=COLORS['primary'])
    path = os.path.join(out_dir, 'qfm_entanglement_scaling.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    print(f"\n[entanglement_scaling] Summary:")
    for nq in n_qubit_range:
        rho_f = all_results[nq]['rhos'][-1]
        SE_f  = entanglement_entropy_bipartite(rho_f, nq, [0])
        print(f"  N={nq}: S_E(final)={SE_f:.4f}, GHZ_F={ghz_fidelity(rho_f, nq):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_entanglement_scaling_analysis(args.out)

if __name__ == "__main__":
    main()
