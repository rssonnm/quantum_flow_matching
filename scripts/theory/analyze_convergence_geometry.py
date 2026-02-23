import argparse
import os
import sys

# Standard path insertion
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.metrics import uhlmann_fidelity, bures_distance, von_neumann_entropy

def fidelity_dashboard(rhos, rhos_per_member, target_rhos, T_steps, output_dir):
    print("Generating Advanced Fidelity Convergence Dashboard")
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.30, height_ratios=[1, 1, 1])

    steps = list(range(1, T_steps + 1))
    steps_full = list(range(T_steps + 1))
    fidelities = [float(uhlmann_fidelity(rhos[t+1], target_rhos[t])) for t in range(T_steps)]
    bures_dists = [float(bures_distance(rhos[t+1], target_rhos[t])) for t in range(T_steps)]
    step_bures  = [float(bures_distance(rhos[i], rhos[i+1])) for i in range(T_steps)]
    purities    = [float(torch.real(torch.trace(rhos[t] @ rhos[t]))) for t in range(T_steps + 1)]
    entropies   = [float(von_neumann_entropy(rhos[t])) for t in range(T_steps + 1)]
    
    n_members = min(15, len(rhos_per_member[0]))
    fid_mat = np.zeros((n_members, T_steps))
    for t in range(T_steps):
        for m in range(n_members):
            fid_mat[m, t] = float(uhlmann_fidelity(rhos_per_member[t+1][m], target_rhos[t]))
    
    final_fids = [float(uhlmann_fidelity(r, target_rhos[-1])) for r in rhos_per_member[-1]]
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, fidelities, 'o-', color=COLORS['primary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label='Ensemble Mean $\\mathcal{F}$')
    fid_means, fid_stds = np.mean(fid_mat, axis=0), np.std(fid_mat, axis=0)
    ax1.fill_between(steps, fid_means - fid_stds, fid_means + fid_stds, color=COLORS['primary'], alpha=0.3, label='$\\pm 1\\sigma$ band')
    ax1.axhline(0.99, color=COLORS['primary'], ls='--', lw=1.5, label='99% Threshold')
    ax1.axhspan(0.99, 1.05, facecolor=COLORS['primary'], alpha=0.1)
    
    ymin_fid = max(0, min(fidelities) - 0.08)
    ax1.set_ylim(ymin_fid, 1.05)
    ax1.set_xlabel('Integration Step $\\tau$')
    ax1.set_ylabel('Uhlmann Fidelity $\\mathcal{F}$')
    ax1.set_title('(a)  Mean Trajectory Convergence')
    ax1.grid(True, ls='--')
    ax1.legend(fontsize=9, loc='lower right')
    
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(steps, bures_dists, color=COLORS['secondary'], edgecolor='black', linewidth=0.5, label='$D_B(\\rho_\\tau, \\rho_{target})$')
    ax2b = ax2.twinx()
    cum_path = np.cumsum([0] + step_bures)
    direct = float(bures_distance(rhos[0], rhos[-1]))
    ax2b.plot(steps_full, cum_path, 's-', color="C8", lw=2.0, ms=5, markerfacecolor='white', markeredgewidth=1.5, label='Cumulative Path')
    ax2b.axhline(direct, color='gray', ls=':', lw=1.5, label=f'Direct Geodesic = {direct:.3f}')
    efficiency = direct / (cum_path[-1] + 1e-12)
    ax2.text(0.02, 0.95, f"Path Efficiency: {efficiency:.2%}", transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc="white", alpha=0.8))

    ax2.set_xlabel('Integration Step $\\tau$')
    ax2.set_ylabel('Bures Distance $D_B$', color=COLORS['secondary'])
    ax2b.set_ylabel('Cumulative Bures Length', color="C8")
    ax2.set_title('(b)  Bures Error & Geodesic Path')
    ax2.grid(axis='y', ls='--')
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps_full, purities, 'D-', color="C9", lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='$\\gamma = \\mathrm{tr}(\\rho^2)$')
    ax3.axhline(1.0, color="C4", ls='--', lw=1.5, label='Pure state ($\\gamma=1$)')
    ax3.axhline(1.0 / 8, color="C2", ls=':', lw=1.5, label=f'Maximally mixed')
    ax3.set_xlabel('Integration Step $\\tau$')
    ax3.set_ylabel('State Purity $\\gamma$')
    ax3.set_ylim(0, 1.15)
    ax3.grid(True, ls='--')
    ax3.legend(fontsize=9, loc='lower left')
    ax3.set_title('(c)  Quantum State Purity Evolution')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps_full, entropies, 'o-', color="C5", lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='$S(\\rho)$')
    max_entropy = np.log(8)
    ax4.axhline(max_entropy, color="C2", ls=':', lw=1.5, label=f'$S_{{max}} = {max_entropy:.2f}$')
    ax4b = ax4.twinx()
    dS = np.diff(entropies)
    ax4b.bar(steps, dS, color="C8", width=0.4, label='$\\Delta S / \\Delta \\tau$', alpha=0.6)
    ax4.set_xlabel('Integration Step $\\tau$')
    ax4.set_ylabel('Von Neumann Entropy $S(\\rho)$', color="C5")
    ax4b.set_ylabel('Entropy Rate $\\Delta S$', color="C8")
    ax4.grid(True, ls='--')
    ax4.set_title('(d)  Entropy & Information Dynamics')

    ax5 = fig.add_subplot(gs[2, 0])
    im = ax5.imshow(fid_mat, aspect='auto', cmap='magma', extent=[0.5, T_steps + 0.5, n_members - 0.5, -0.5], vmin=max(0, fid_mat.min() - 0.05), vmax=min(1.0, fid_mat.max() + 0.02))
    fig.colorbar(im, ax=ax5, pad=0.02, shrink=0.9).set_label('Individual $\\mathcal{F}$')
    ax5.set_xlabel('Integration Step $\\tau$')
    ax5.set_ylabel('Ensemble Member $m$')
    ax5.set_title('(e)  Monte Carlo Trajectory Profiles')
    ax5.set_xticks(steps[::2])

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(final_fids, bins=min(20, max(5, len(final_fids) // 2)), color="C6", edgecolor='black', linewidth=0.8, density=True, label='Empirical PDF')
    
    mean_f, std_f, med_f = np.mean(final_fids), np.std(final_fids), np.median(final_fids)
    ax6.axvline(mean_f, color="C2", ls='--', lw=2, label=f'Mean = {mean_f:.4f}')
    ax6.axvline(med_f, color="C8", ls=':', lw=2, label=f'Median = {med_f:.4f}')
    ax6.set_xlabel('Final Target Fidelity $\\mathcal{F}_T$')
    ax6.set_ylabel('Probability Density')
    ax6.set_title(f'(f)  Sink State Distribution ($M={len(final_fids)}$)')
    ax6.grid(True, ls='--')
    ax6.legend(fontsize=9, loc='upper left')

    fig.suptitle("Quantum Flow Matching — Comprehensive Convergence Analysis Dashboard", y=0.99, color="C0")
    path = os.path.join(output_dir, 'qfm_fidelity_dashboard.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

def geodesic_deviation(rhos, T_steps, output_dir):
    print("Generating Geodesic Deviation Analysis")
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.30, height_ratios=[1, 1, 1])

    steps = list(range(1, T_steps + 1))
    steps_full = list(range(T_steps + 1))
    step_dists = [float(bures_distance(rhos[i], rhos[i+1])) for i in range(T_steps)]
    cum_path   = np.cumsum([0.0] + step_dists)
    direct     = float(bures_distance(rhos[0], rhos[-1]))
    actions    = [d**2 for d in step_dists]                  
    cum_action = np.cumsum([0.0] + actions)
    direct_dists = [float(bures_distance(rhos[0], rhos[t])) for t in range(T_steps + 1)]
    geodesic_interp = np.linspace(0, direct, T_steps + 1)
    curvature = cum_path - geodesic_interp
    velocities = np.array(step_dists)                        
    acceleration = np.diff(velocities) if len(velocities) > 1 else np.array([0.0])
    
    efficiency_ratio = np.zeros(T_steps)
    for t in range(T_steps):
        efficiency_ratio[t] = direct_dists[t + 1] / cum_path[t + 1] if cum_path[t + 1] > 1e-12 else 1.0

    ax1 = fig.add_subplot(gs[0, 0])
    colors_bar = [plt.cm.RdYlBu_r(d / (max(step_dists) + 1e-9)) for d in step_dists]
    ax1.bar(steps, step_dists, color=colors_bar, edgecolor='black', linewidth=0.5)
    mean_dist = np.mean(step_dists)
    ax1.axhline(mean_dist, color="C2", ls='--', lw=2, label=f'Mean = {mean_dist:.3f}')
    ax1.set_xlabel('Integration Step $\\tau$')
    ax1.set_ylabel('$D_B(\\rho_\\tau, \\rho_{\\tau+1})$')
    ax1.set_title('(a)  Per-Step Bures Distance')
    ax1.grid(True, ls='--', axis='y')
    ax1.legend(fontsize=9, loc='upper left')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps_full, cum_path, 'o-', color="C4", lw=2.5, ms=6, markerfacecolor='white', markeredgewidth=1.5, label='QFM Cumulative Path')
    ax2.plot(steps_full, geodesic_interp, '--', color="C4", lw=2.0, label=f'Ideal Geodesic ($D_B^{{direct}}$ = {direct:.3f})')
    ax2.set_xlabel('Integration Step $\\tau$')
    ax2.set_ylabel('Cumulative Bures Length')
    ax2.set_title('(b)  Path vs Geodesic Comparison')
    ax2.grid(True, ls='--')
    ax2.legend(fontsize=9, loc='lower right')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(steps, actions, color="C6", edgecolor='black', linewidth=0.5, label='$\\mathcal{A}_\\tau = D_B^2$')
    ax3b = ax3.twinx()
    ax3b.plot(steps_full, cum_action, 'D-', color="C8", lw=2.0, ms=5, markerfacecolor='white', markeredgewidth=1.5, label=f'$\\Sigma \\mathcal{{A}}$ = {cum_action[-1]:.3f}')
    ax3.set_xlabel('Integration Step $\\tau$')
    ax3.set_ylabel('Action $\\mathcal{A}_\\tau = D_B^2(\\rho_\\tau, \\rho_{\\tau+1})$')
    ax3b.set_ylabel('Cumulative Action $\\sum \\mathcal{A}$', color="C8")
    ax3.set_title('(c)  Benamou-Brenier Action Density')
    ax3.grid(True, ls='--', axis='y')
    
    ax4 = fig.add_subplot(gs[1, 1])
    colors_curv = ["C2" if c > 0 else "C4" for c in curvature]
    ax4.bar(steps_full, curvature, color=colors_curv, edgecolor='black', linewidth=0.5)
    ax4.axhline(0, color='black', lw=1)
    ax4.set_xlabel('Integration Step $\\tau$')
    ax4.set_ylabel('Geodesic Deviation $\\delta_\\tau$')
    ax4.set_title('(d)  Local Geodesic Curvature')
    ax4.grid(True, ls='--')

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(steps, velocities, 'o-', color="C4", lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='Velocity $v_\\tau = D_B / \\Delta\\tau$')
    ax5b = ax5.twinx()
    acc_steps = list(range(2, T_steps + 1))
    if len(acceleration) > 0:
        ax5b.bar(acc_steps, acceleration, color="C8", width=0.4, label='Acceleration $a_\\tau$', alpha=0.6)
    ax5.set_xlabel('Integration Step $\\tau$')
    ax5.set_ylabel('Bures Velocity $v_\\tau$', color="C4")
    ax5b.set_ylabel('Acceleration $a_\\tau = \\Delta v$', color="C8")
    ax5.set_title('(e)  Transport Velocity & Acceleration')
    ax5.grid(True, ls='--')

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(steps, efficiency_ratio, 'D-', color="C9", lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='$\\eta_\\tau$')
    ax6.axhline(1.0, color="C4", ls='--', lw=2, label='Perfect Geodesic ($\\eta=1$)')
    ax6.set_ylim(0, 1.15)
    ax6.set_xlabel('Integration Step $\\tau$')
    ax6.set_ylabel('Path Efficiency Ratio $\\eta$')
    ax6.set_title('(f)  Instantaneous Path Efficiency')
    ax6.grid(True, ls='--')
    ax6.legend(fontsize=9, loc='lower left')

    fig.suptitle("Bures Geodesic Deviation Analysis", y=0.99, color="C0")
    path = os.path.join(output_dir, 'qfm_geodesic_deviation.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--ensemble", type=int, default=10)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    _, rhos, rhos_pm, target_rhos, _, _, nq, T = \
        train_qfm_collect_all(n_qubits=args.qubits, T_steps=args.steps, M=args.ensemble)

    fidelity_dashboard(rhos, rhos_pm, target_rhos, T, args.out)
    geodesic_deviation(rhos, T, args.out)

if __name__ == "__main__":
    main()
