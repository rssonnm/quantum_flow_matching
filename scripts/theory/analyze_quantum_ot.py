import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.quantum_ot import (
    bures_geodesic_interpolation, trajectory_vs_geodesic_analysis,
    discrete_benamou_brenier_energy, total_benamou_brenier_cost,
    quantum_speed_limit_bound, mean_geodesic_deviation,
)
from src.qfm.metrics import bures_distance, uhlmann_fidelity


def run_quantum_ot_analysis(n_qubits=2, T_steps=12, M=8, out_dir="results"):
    print("Running Quantum OT Geodesic Analysis")
    os.makedirs(out_dir, exist_ok=True)

    _, rhos, losses_all, _, _, g_vals, nq, T = train_qfm_collect_all(
        n_qubits=n_qubits, T_steps=T_steps, M=M
    )

    rho_0 = rhos[0]
    rho_T = rhos[-1]

    # --- Full OT analysis
    ot_result = trajectory_vs_geodesic_analysis(rhos, rho_0, rho_T)
    geodesic   = ot_result['geodesic_rhos']

    # --- Benamou-Brenier kinetic energy
    KE_qfm = discrete_benamou_brenier_energy(rhos)
    KE_geo = discrete_benamou_brenier_energy(geodesic)

    # --- Deviation from geodesic
    dev_result = mean_geodesic_deviation(rhos, rho_0, rho_T)

    # --- Bures distance curves
    d_qfm_to_target = [float(bures_distance(r, rho_T)) for r in rhos]
    d_geo_to_target = [float(bures_distance(r, rho_T)) for r in geodesic]
    tau_arr = np.linspace(0, 1, len(rhos))

    # --- QFM step costs vs Geodesic step costs
    sc_qfm = ot_result['step_costs_qfm']
    sc_geo = ot_result['step_costs_geodesic']

    # ===== FIGURE: 4-panel Quantum OT Analysis =====
    fig = plt.figure(figsize=(20, 15))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # (a) Distance to target: QFM vs Geodesic
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tau_arr, d_qfm_to_target, 'o-', color=COLORS['primary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label='QFM trajectory')
    ax1.plot(tau_arr, d_geo_to_target, 's--', color=COLORS['secondary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label='Bures geodesic')
    ax1.fill_between(tau_arr, d_qfm_to_target, d_geo_to_target,
                     color=COLORS['accent'], alpha=0.2, label='Excess transport')
    ax1.scatter(tau_arr[0], d_qfm_to_target[0], color=COLORS['accent'], s=200, zorder=5)
    ax1.scatter(tau_arr[-1], d_qfm_to_target[-1], color='red', marker='*', s=300, zorder=5)
    ax1.set_xlabel('Normalized Time $\\tau$')
    ax1.set_ylabel('Bures Distance to Target $d_B(\\rho_\\tau, \\rho_T)$')
    ax1.set_title('(a) QFM vs Geodesic: Distance to Target', pad=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, ls='--', alpha=0.5)

    # (b) Per-step action cost: QFM vs Geodesic
    ax2 = fig.add_subplot(gs[0, 1])
    tau_inner = np.arange(len(sc_qfm))
    ax2.bar(tau_inner - 0.2, sc_qfm, 0.35, color=COLORS['primary'], alpha=0.8,
            edgecolor='black', linewidth=0.5, label=f'QFM (total={ot_result["trajectory_action"]:.4f})')
    ax2.bar(tau_inner + 0.2, sc_geo, 0.35, color=COLORS['secondary'], alpha=0.8,
            edgecolor='black', linewidth=0.5, label=f'Geodesic (total={ot_result["geodesic_action_disc"]:.4f})')
    ax2.axhline(ot_result['geodesic_action_true'] / max(len(sc_qfm), 1), color='gray',
                ls=':', lw=2, label='True geodesic / T')
    ax2.set_xlabel('Flow Step $\\tau$')
    ax2.set_ylabel('Step Action $d_B^2(\\rho_\\tau, \\rho_{\\tau+1})$')
    ax2.set_title(f'(b) Per-Step Action Cost\\n(Efficiency $\\eta={ot_result["transport_efficiency"]:.3f}$)', pad=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', ls='--', alpha=0.5)

    # (c) Benamou-Brenier kinetic energy
    ax3 = fig.add_subplot(gs[1, 0])
    tau_ke = np.arange(len(KE_qfm))
    ax3.plot(tau_ke, KE_qfm, 'D-', color=COLORS['primary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label='QFM kinetic energy')
    ax3.plot(tau_ke[:len(KE_geo)], KE_geo[:len(KE_qfm)], 's--', color=COLORS['secondary'],
             lw=2.0, ms=6, markerfacecolor='white', markeredgewidth=2, label='Geodesic kinetic energy')
    ax3.fill_between(tau_ke[:len(KE_geo)], KE_qfm[:len(KE_geo)], KE_geo[:len(KE_geo)],
                     alpha=0.2, color=COLORS['accent'])
    ax3.set_xlabel('Flow Step $\\tau$')
    ax3.set_ylabel('Kinetic Energy $|\\dot{\\rho}_\\tau|^2_{HS}$')
    ax3.set_title(f'(c) Benamou–Brenier Kinetic Energy\n$E_{{BB}}^{{QFM}}={total_benamou_brenier_cost(rhos):.4f}$', pad=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, ls='--', alpha=0.5)

    # (d) Geodesic deviation along trajectory
    ax4 = fig.add_subplot(gs[1, 1])
    devs = dev_result['deviations']
    ax4.plot(tau_arr, devs, 'o-', color=COLORS['tertiary'] if 'tertiary' in COLORS else 'C2',
             lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2)
    ax4.axhline(dev_result['mean_deviation'], color=COLORS['primary'], ls='--', lw=2,
                label=f'Mean dev: {dev_result["mean_deviation"]:.4f}')
    ax4.axhline(dev_result['max_deviation'], color='red', ls=':', lw=2,
                label=f'Max dev: {dev_result["max_deviation"]:.4f}')
    ax4.fill_between(tau_arr, devs, dev_result['mean_deviation'],
                     where=np.array(devs) > dev_result['mean_deviation'],
                     alpha=0.2, color='red')
    ax4.fill_between(tau_arr, devs, dev_result['mean_deviation'],
                     where=np.array(devs) <= dev_result['mean_deviation'],
                     alpha=0.2, color='green')
    ax4.set_xlabel('Normalized Time $\\tau$')
    ax4.set_ylabel('$d_B(\\rho^{QFM}_\\tau, \\rho^{geo}_\\tau)$')
    ax4.set_title('(d) Point-wise Deviation from Bures Geodesic', pad=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, ls='--', alpha=0.5)

    fig.suptitle('Quantum Flow Matching — Optimal Transport Analysis & Geodesic Comparison',
                 y=0.98, fontsize=14, color=COLORS['primary'])
    path = os.path.join(out_dir, 'qfm_quantum_ot.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    print(f"\n[quantum_ot] Summary:")
    print(f"  Trajectory action A = {ot_result['trajectory_action']:.6f}")
    print(f"  Geodesic action A*  = {ot_result['geodesic_action_true']:.6f}")
    print(f"  Transport efficiency η = {ot_result['transport_efficiency']:.4f}")
    print(f"  Mean geodesic deviation = {dev_result['mean_deviation']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_quantum_ot_analysis(args.qubits, args.steps, args.ensemble, args.out)

if __name__ == "__main__":
    main()
