import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.classical_baseline import (
    ClassicalFlowMatchingBaseline, compare_qfm_vs_cfm,
)
from src.qfm.metrics import uhlmann_fidelity, bures_distance, von_neumann_entropy, purity


def run_classical_vs_quantum_analysis(n_qubits=2, T_steps=8, M=6, out_dir="results"):
    print("Running Classical FM vs QFM Benchmark")
    os.makedirs(out_dir, exist_ok=True)

    trainer, rhos, losses_all, _, _, g_vals, nq, T = train_qfm_collect_all(
        n_qubits=n_qubits, T_steps=T_steps, M=M
    )

    rho_0 = rhos[0]
    rho_T = rhos[-1]
    d = 2 ** n_qubits

    # --- Train Classical FM baseline
    print("  Training Classical Flow Matching baseline...")
    # Use random pure states as source ensemble
    source_rhos = [rho_0] * M
    target_rhos = [rho_T] * M
    cfm_n_params_default = 64  # hidden=64, layers=3
    cfm = ClassicalFlowMatchingBaseline(d=d, hidden_dim=64, n_hidden_layers=3, lr=1e-3)
    cfm_losses = cfm.train(source_rhos, target_rhos, n_epochs=150, n_time_samples=6, verbose=False)
    cfm_trajectory = cfm.generate_trajectory(rho_0, T_steps=T_steps)

    # --- QFM trajectory: rhos
    # --- QFM params = sum of params in all models
    qfm_n_params = sum(p.numel() for p in trainer.models[0][0].parameters()) * len(trainer.models)
    cfm_n_params  = sum(p.numel() for p in cfm.model.parameters())

    # --- Head-to-head comparison
    cmp = compare_qfm_vs_cfm(rhos, cfm_trajectory, rho_T, qfm_n_params, cfm_n_params)

    tau_arr = np.linspace(0, 1, T_steps + 1)

    # ===== FIGURE: 4-panel Classical vs QFM Benchmark =====
    fig = plt.figure(figsize=(20, 15))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # (a) Training loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    # Flatten losses_all robustly (each step may contain tensors or scalars)
    qfm_loss_flat = []
    for step_l in losses_all:
        for l in step_l:
            try:
                qfm_loss_flat.append(float(l.item()) if hasattr(l, 'item') else float(l))
            except Exception:
                pass
    n_show = min(len(qfm_loss_flat), len(cfm_losses), 150)
    ax1.semilogy(range(n_show), qfm_loss_flat[:n_show], color=COLORS['primary'],
                 lw=2.5, label=f'QFM ($P={qfm_n_params}$ params)')
    ax1.semilogy(range(len(cfm_losses[:n_show])), cfm_losses[:n_show], '--',
                 color=COLORS['secondary'], lw=2.5, label=f'CFM ($P={cfm_n_params}$ params)')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('(a) Training Loss: QFM vs Classical FM', pad=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, ls='--', alpha=0.5, which='both')

    # (b) Fidelity to target along trajectory
    ax2 = fig.add_subplot(gs[0, 1])
    qfm_fids = cmp['qfm_purity_curve']  # placeholder using purity from compare
    qfm_fid_curve = [float(uhlmann_fidelity(r, rho_T)) for r in rhos]
    cfm_fid_curve = [float(uhlmann_fidelity(r, rho_T)) for r in cfm_trajectory]
    ax2.plot(tau_arr, qfm_fid_curve, 'o-', color=COLORS['primary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label=f'QFM (final F={cmp["qfm_final_fidelity"]:.4f})')
    ax2.plot(tau_arr, cfm_fid_curve, 's--', color=COLORS['secondary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label=f'CFM (final F={cmp["cfm_final_fidelity"]:.4f})')
    ax2.fill_between(tau_arr, qfm_fid_curve, cfm_fid_curve,
                     where=np.array(qfm_fid_curve) >= np.array(cfm_fid_curve),
                     alpha=0.15, color=COLORS['primary'], label='QFM advantage')
    ax2.fill_between(tau_arr, qfm_fid_curve, cfm_fid_curve,
                     where=np.array(qfm_fid_curve) < np.array(cfm_fid_curve),
                     alpha=0.15, color=COLORS['secondary'], label='CFM advantage')
    ax2.set_xlabel('Normalized Time $\\tau$')
    ax2.set_ylabel('Uhlmann Fidelity $F(\\rho_\\tau, \\rho_T)$')
    ax2.set_title('(b) Fidelity to Target: QFM vs Classical FM', pad=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.set_ylim(-0.05, 1.05)

    # (c) Transport cost (step-wise Bures distance)
    ax3 = fig.add_subplot(gs[1, 0])
    qfm_step_d = [float(bures_distance(rhos[t], rhos[t+1])) for t in range(len(rhos)-1)]
    cfm_step_d = [float(bures_distance(cfm_trajectory[t], cfm_trajectory[t+1])) for t in range(len(cfm_trajectory)-1)]
    tau_steps = np.arange(len(qfm_step_d))
    ax3.plot(tau_steps, qfm_step_d, 'o-', color=COLORS['primary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label=f'QFM (A={cmp["qfm_action"]:.4f})')
    ax3.plot(tau_steps[:len(cfm_step_d)], cfm_step_d, 's--', color=COLORS['secondary'],
             lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2,
             label=f'CFM (A={cmp["cfm_action"]:.4f})')
    ax3.set_xlabel('Flow Step $\\tau$')
    ax3.set_ylabel('Step Bures Distance $d_B(\\rho_\\tau, \\rho_{\\tau+1})$')
    ax3.set_title(f'(c) Per-Step Transport Cost\n(Action Advantage: {cmp["action_advantage"]:.4f})', pad=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, ls='--', alpha=0.5)

    # (d) Summary bar: final metrics comparison
    ax4 = fig.add_subplot(gs[1, 1])
    metric_names = [
        'Final\nFidelity', 'Bures\nDistance (×10)', 'Action\n(1-norm)', 'Param\nEfficiency (×100)'
    ]
    qfm_vals = [
        cmp['qfm_final_fidelity'],
        min(cmp['qfm_final_bures'] * 10, 1.0),
        min(cmp['qfm_action'], 1.0),
        min(cmp['qfm_param_efficiency'] * 100, 1.0),
    ]
    cfm_vals = [
        cmp['cfm_final_fidelity'],
        min(cmp['cfm_final_bures'] * 10, 1.0),
        min(cmp['cfm_action'], 1.0),
        min(cmp['cfm_param_efficiency'] * 100, 1.0),
    ]
    x = np.arange(len(metric_names))
    ax4.bar(x - 0.2, qfm_vals, 0.35, color=COLORS['primary'], alpha=0.85,
            edgecolor='black', linewidth=0.8, label='QFM')
    ax4.bar(x + 0.2, cfm_vals, 0.35, color=COLORS['secondary'], alpha=0.85,
            edgecolor='black', linewidth=0.8, label='Classical FM')
    for xi, (qv, cv) in zip(x, zip(qfm_vals, cfm_vals)):
        ax4.text(xi - 0.2, qv + 0.02, f'{qv:.3f}', ha='center', fontsize=8, color=COLORS['primary'])
        ax4.text(xi + 0.2, cv + 0.02, f'{cv:.3f}', ha='center', fontsize=8, color=COLORS['secondary'])
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names)
    ax4.set_ylabel('Metric Value (normalized)')
    ax4.set_title('(d) Head-to-Head: QFM vs Classical FM', pad=12)
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, 1.3)
    ax4.grid(True, axis='y', ls='--', alpha=0.5)

    fig.suptitle('Quantum Flow Matching — Benchmark: QFM vs Classical Flow Matching',
                 y=0.98, fontsize=14, color=COLORS['primary'])
    path = os.path.join(out_dir, 'qfm_classical_vs_quantum.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    print(f"\n[classical_vs_quantum] Summary:")
    print(f"  QFM  final fidelity = {cmp['qfm_final_fidelity']:.4f}")
    print(f"  CFM  final fidelity = {cmp['cfm_final_fidelity']:.4f}")
    print(f"  Fidelity advantage (QFM - CFM) = {cmp['fidelity_advantage']:.4f}")
    print(f"  QFM params={qfm_n_params}, CFM params={cfm_n_params}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--ensemble", type=int, default=6)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_classical_vs_quantum_analysis(args.qubits, args.steps, args.ensemble, args.out)

if __name__ == "__main__":
    main()
