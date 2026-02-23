import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.flow_geometry import (
    quantum_vector_field, vector_field_magnitude,
    generator_spectrum_trajectory, geodesic_curvature,
    parallel_transport_deviation, curvature_along_trajectory,
    full_geometry_report, geometric_phase_estimate,
)


def run_flow_geometry_analysis(n_qubits=2, T_steps=12, M=8, out_dir="results"):
    print("Running Flow Geometry Analysis (Bures Manifold)")
    os.makedirs(out_dir, exist_ok=True)

    _, rhos, losses_all, _, _, g_vals, nq, T = train_qfm_collect_all(
        n_qubits=n_qubits, T_steps=T_steps, M=M
    )

    # Compute full geometry report
    geo = full_geometry_report(rhos)

    tau_arr       = np.linspace(0, 1, len(rhos))
    tau_steps     = np.linspace(0, 1, len(rhos) - 1)
    tau_inner     = np.linspace(0, 1, max(len(rhos) - 2, 1))

    # ===== FIGURE: 4-panel Flow Geometry =====
    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    # (a) Coherent vs Dissipative generator rates
    ax1 = fig.add_subplot(gs[0, 0])
    coh = geo['generator_coherent_rates']
    dis = geo['generator_dissipative_rates']
    tot = [c + d for c, d in zip(coh, dis)]
    ax1.stackplot(tau_steps, coh, dis,
                  labels=['Coherent (unitary) $||G_{\\mathrm{coh}}||_{HS}$',
                          'Dissipative (decoherence) $||G_{\\mathrm{diss}}||_{HS}$'],
                  colors=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
    ax1.plot(tau_steps, tot, 'k-', lw=1.5, label='Total $||G||_{HS}$')
    ax1.set_xlabel('Normalized Time $\\tau$')
    ax1.set_ylabel('Generator Rate (Hilbert-Schmidt norm)')
    ax1.set_title('(a) Lindblad Generator Decomposition\nCoherent vs Dissipative Dynamics', pad=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, ls='--', alpha=0.5)

    # (b) Vector field magnitude ||V_t||_HS
    ax2 = fig.add_subplot(gs[0, 1])
    vf_mags = geo['vector_field_magnitudes']
    ax2.plot(tau_steps, vf_mags, 'o-', color=COLORS['primary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label='QFM Vector Field')
    ax2.fill_between(tau_steps, vf_mags, alpha=0.2, color=COLORS['primary'])
    # Highlight max speed location
    max_idx = int(np.argmax(vf_mags))
    ax2.scatter(tau_steps[max_idx], vf_mags[max_idx], color='red', s=200, zorder=5,
                edgecolors='black', label=f'Max speed ($\\tau$={tau_steps[max_idx]:.2f})')
    ax2.set_xlabel('Normalized Time $\\tau$')
    ax2.set_ylabel('$\\|\\|v_\\tau\\|\\|_{HS}$')
    ax2.set_title('(b) QFM Vector Field Magnitude (Flow Speed)', pad=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, ls='--', alpha=0.5)

    # (c) Geodesic curvature κ and parallel transport deviation δ
    ax3 = fig.add_subplot(gs[1, 0])
    geo_curv  = geo['geodesic_curvature']
    pt_dev    = geo['parallel_transport_deviation']
    n_inner   = min(len(geo_curv), len(pt_dev))
    tau_inner_arr = np.linspace(0, 1, n_inner) if n_inner > 0 else []

    if n_inner > 0:
        ax3.plot(tau_inner_arr, geo_curv[:n_inner], 'D-', color=COLORS['primary'], lw=2.5, ms=7,
                 markerfacecolor='white', markeredgewidth=2, label='Geodesic curvature $\\kappa(\\tau)$')
        ax3b = ax3.twinx()
        ax3b.plot(tau_inner_arr, pt_dev[:n_inner], 's--', color=COLORS['secondary'], lw=2.0, ms=6,
                  markerfacecolor='white', markeredgewidth=2, label='PT deviation $\\delta(\\tau)$')
        ax3b.set_ylabel('Parallel Transport Deviation $\\delta$', color=COLORS['secondary'])
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    else:
        ax3.text(0.5, 0.5, 'Not enough steps\nfor curvature', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12)

    ax3.set_xlabel('Normalized Time $\\tau$')
    ax3.set_ylabel('Geodesic Curvature $\\kappa$', color=COLORS['primary'])
    ax3.set_title('(c) Riemannian Curvature & Parallel Transport\non the Bures Manifold', pad=12)
    ax3.grid(True, ls='--', alpha=0.5)

    # (d) Sectional curvature K(V_t, V_{t+1})
    ax4 = fig.add_subplot(gs[1, 1])
    sec_curv = geo['sectional_curvatures']
    if sec_curv:
        tau_sec = np.linspace(0, 1, len(sec_curv))
        colors_c = ['red' if k > 0 else COLORS['primary'] for k in sec_curv]
        bars = ax4.bar(tau_sec, sec_curv, width=0.8 / max(len(sec_curv), 1),
                       color=colors_c, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax4.axhline(0, color='black', lw=1.5, label='Flat manifold $K=0$')
        ax4.axhline(geo['mean_curvature'], color=COLORS['secondary'], ls='--', lw=2,
                    label=f'Mean $K={geo["mean_curvature"]:.4f}$')
        geo_phase = geo['geometric_phase']
        ax4.set_title(f'(d) Sectional Curvature $K(V_t, V_{{t+1}})$\nGeometric Phase: $\\phi_{{geo}}={geo_phase:.4f}$ rad', pad=12)
    else:
        ax4.text(0.5, 0.5, 'Insufficient steps', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('(d) Sectional Curvature', pad=12)

    ax4.set_xlabel('Normalized Time $\\tau$')
    ax4.set_ylabel('Sectional Curvature $K$')
    ax4.legend(fontsize=10)
    ax4.grid(True, axis='y', ls='--', alpha=0.5)

    fig.suptitle('Quantum Flow Matching — Differential Geometry of the Bures Manifold',
                 y=0.98, fontsize=14, color=COLORS['primary'])
    path = os.path.join(out_dir, 'qfm_flow_geometry.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    print(f"\n[flow_geometry] Summary:")
    print(f"  Geometric phase φ_geo = {geo['geometric_phase']:.6f} rad")
    print(f"  Mean geodesic curvature κ = {geo['mean_curvature']:.6f}")
    print(f"  Mean parallel transport deviation δ = {geo['mean_pt_deviation']:.6f}")
    if geo['vector_field_magnitudes']:
        print(f"  Max flow speed = {max(geo['vector_field_magnitudes']):.6f}")
    coh = geo['generator_coherent_rates']
    dis = geo['generator_dissipative_rates']
    if coh:
        coh_frac = np.mean(coh) / (np.mean(coh) + np.mean(dis) + 1e-12)
        print(f"  Mean coherent fraction = {coh_frac:.4f} (1.0 = purely unitary flow)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_flow_geometry_analysis(args.qubits, args.steps, args.ensemble, args.out)

if __name__ == "__main__":
    main()
