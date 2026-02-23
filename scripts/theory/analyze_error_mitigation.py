import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.error_mitigation import (
    gate_decomposition_cost, effective_noise_rate,
    apply_depolarizing_noise, mitigated_fidelity_trajectory,
    noisy_trajectory,
)
from src.qfm.metrics import uhlmann_fidelity, bures_distance


def run_error_mitigation_analysis(n_qubits=1, T_steps=8, M=6, out_dir="results"):
    print("Running Error Mitigation & Hardware Analysis")
    os.makedirs(out_dir, exist_ok=True)

    _, rhos, losses_all, _, _, g_vals, nq, T = train_qfm_collect_all(
        n_qubits=n_qubits, T_steps=T_steps, M=M
    )

    rho_target = rhos[-1]

    # --- Gate cost analysis over different n_qubits and n_layers
    nq_range = [1, 2, 3, 4, 5]
    nl_range = [1, 2, 3, 4, 5]
    cx_matrix = np.zeros((len(nq_range), len(nl_range)))
    depth_matrix = np.zeros((len(nq_range), len(nl_range)))
    for i, nq_v in enumerate(nq_range):
        for j, nl_v in enumerate(nl_range):
            gc = gate_decomposition_cost(nq_v, nl_v, n_ancilla=1)
            cx_matrix[i, j] = gc['cx_gates']
            depth_matrix[i, j] = gc['circuit_depth']

    # --- IBM Brisbane-like hardware parameters
    hw_params = [
        {'T1_us': 100, 'T2_us': 80,  'gate_time_ns': 300, 'label': 'NISQ-Near'},
        {'T1_us': 200, 'T2_us': 150, 'gate_time_ns': 200, 'label': 'NISQ-Mid'},
        {'T1_us': 500, 'T2_us': 300, 'gate_time_ns': 100, 'label': 'NISQ-Best'},
    ]
    gc_current = gate_decomposition_cost(n_qubits, T_steps, n_ancilla=1)
    n_cx = gc_current['cx_gates']

    hw_results = {}
    for hw in hw_params:
        r = effective_noise_rate(hw['T1_us'], hw['T2_us'], hw['gate_time_ns'], n_cx)
        hw_results[hw['label']] = r

    # --- ZNE analysis (noise_levels as proxy for scaling)
    noise_levels = [0.02, 0.04, 0.06, 0.08]
    mit_result = mitigated_fidelity_trajectory(rhos, rho_target, noise_levels=noise_levels[:3])

    # --- Noisy trajectories at different noise levels for individual curves
    noisy_fid_curves = {}
    for p in noise_levels:
        noisy_rhos = noisy_trajectory(rhos, p)
        fids = [float(uhlmann_fidelity(r, rho_target)) for r in noisy_rhos]
        noisy_fid_curves[p] = fids

    tau_arr = np.linspace(0, 1, len(rhos))

    # ===== FIGURE: 4-panel Error Mitigation =====
    fig = plt.figure(figsize=(20, 15))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # (a) Fidelity curves: ideal, noisy, ZNE-mitigated
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tau_arr, mit_result['ideal_fidelities'], 'o-', color=COLORS['primary'],
             lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='Ideal (noiseless)', zorder=4)
    ax1.plot(tau_arr, mit_result['noisy_fidelities'], 's--', color='red',
             lw=2, ms=6, markerfacecolor='white', markeredgewidth=2, label=f'Noisy ($p={noise_levels[0]:.2f}$)')
    ax1.plot(tau_arr, mit_result['mitigated_fidelities'], 'D-', color=COLORS['secondary'],
             lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='ZNE-Mitigated', zorder=5)
    ax1.fill_between(tau_arr, mit_result['noisy_fidelities'], mit_result['ideal_fidelities'],
                     alpha=0.15, color=COLORS['primary'], label='Mitig. gain')
    ax1.set_xlabel('Normalized Time $\\tau$')
    ax1.set_ylabel('Uhlmann Fidelity $F(\\rho_\\tau, \\rho_T)$')
    ax1.set_title('(a) ZNE Error Mitigation: Ideal vs Noisy vs Mitigated', pad=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.set_ylim(-0.05, 1.05)

    # (b) Fidelity degradation at different noise levels
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = plt.cm.Reds
    for i, (p, fids) in enumerate(noisy_fid_curves.items()):
        color = cmap(0.3 + 0.6 * i / len(noisy_fid_curves))
        ax2.plot(tau_arr, fids, lw=2, ms=5, label=f'$p={p:.2f}$', color=color)
    ax2.plot(tau_arr, mit_result['ideal_fidelities'], 'k-', lw=2.5, label='Ideal', zorder=5)
    ax2.set_xlabel('Normalized Time $\\tau$')
    ax2.set_ylabel('Fidelity $F(\\rho_\\tau, \\rho_T)$')
    ax2.set_title('(b) Fidelity vs Depolarizing Noise Level $p$', pad=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.set_ylim(-0.05, 1.05)

    # (c) Gate count: CX gates heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(cx_matrix, cmap='Blues', aspect='auto')
    ax3.set_xticks(range(len(nl_range)))
    ax3.set_xticklabels([f'L={l}' for l in nl_range])
    ax3.set_yticks(range(len(nq_range)))
    ax3.set_yticklabels([f'n={n}' for n in nq_range])
    for i in range(len(nq_range)):
        for j in range(len(nl_range)):
            color = 'white' if cx_matrix[i, j] > cx_matrix.max() / 2 else 'black'
            ax3.text(j, i, int(cx_matrix[i, j]), ha='center', va='center', fontsize=11, color=color)
    plt.colorbar(im, ax=ax3, shrink=0.85).set_label('CNOT Count')
    ax3.set_xlabel('Layers $L$')
    ax3.set_ylabel('Qubits $n$')
    ax3.set_title('(c) CNOT Gate Count Heatmap (n × L)', pad=12)

    # (d) Hardware noise rate comparison
    ax4 = fig.add_subplot(gs[1, 1])
    hw_labels = list(hw_results.keys())
    p_T1  = [hw_results[l]['p_T1_error'] for l in hw_labels]
    p_T2  = [hw_results[l]['p_T2_error'] for l in hw_labels]
    p_gate = [hw_results[l]['p_gate_error'] for l in hw_labels]
    p_eff  = [hw_results[l]['p_effective'] for l in hw_labels]
    x = np.arange(len(hw_labels))
    w = 0.2
    ax4.bar(x - 1.5*w, p_T1,   w, label='$p_{T_1}$',   color=COLORS['primary'], alpha=0.85, edgecolor='black')
    ax4.bar(x - 0.5*w, p_T2,   w, label='$p_{T_2}$',   color=COLORS['secondary'], alpha=0.85, edgecolor='black')
    ax4.bar(x + 0.5*w, p_gate, w, label='$p_{gate}$',  color='orange', alpha=0.85, edgecolor='black')
    ax4.bar(x + 1.5*w, p_eff,  w, label='$p_{eff}$',   color='red', alpha=0.85, edgecolor='black')
    for xi, pe in zip(x, p_eff):
        ax4.text(xi + 1.5*w, pe + 0.005, f'{pe:.3f}', ha='center', color='red', fontsize=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels(hw_labels)
    ax4.set_ylabel('Error Probability $p$')
    ax4.set_title(f'(d) Hardware Noise Profile ({nq_setting} qubits, T={T_steps} steps)', pad=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, axis='y', ls='--', alpha=0.5)
    ax4.set_ylim(0, min(max(p_eff) * 1.4, 1.0))

    fig.suptitle('Quantum Flow Matching — Error Mitigation & NISQ Hardware Analysis',
                 y=0.98, fontsize=14, color=COLORS['primary'])
    path = os.path.join(out_dir, 'qfm_error_mitigation.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    print(f"\n[error_mitigation] Summary:")
    final_ideal   = mit_result['ideal_fidelities'][-1]
    final_noisy   = mit_result['noisy_fidelities'][-1]
    final_mitig   = mit_result['mitigated_fidelities'][-1]
    print(f"  Final fidelity: Ideal={final_ideal:.4f} | Noisy={final_noisy:.4f} | Mitigated={final_mitig:.4f}")
    print(f"  Mitigation gain: {final_mitig - final_noisy:.4f}")
    for label, r in hw_results.items():
        print(f"  [{label}] p_eff={r['p_effective']:.4f}, expected_F={r['expected_fidelity']:.4f}")


nq_setting = 1  # module-level default for title usage

def main():
    global nq_setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--ensemble", type=int, default=6)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    nq_setting = args.qubits
    run_error_mitigation_analysis(args.qubits, args.steps, args.ensemble, args.out)

if __name__ == "__main__":
    main()
