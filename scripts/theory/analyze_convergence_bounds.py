import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.convergence_bounds import (
    lipschitz_trajectory, discretization_error_vs_steps,
    fit_convergence_rate, action_optimality_ratio,
    expressivity_lower_bound, fidelity_convergence_curve,
    bures_convergence_curve,
)


def run_convergence_bounds_analysis(n_qubits=2, T_steps=10, M=8, out_dir="results"):
    print("Running Convergence Bounds Analysis")
    os.makedirs(out_dir, exist_ok=True)

    trainer, rhos, losses_all, _, _, g_vals, nq, T = train_qfm_collect_all(
        n_qubits=n_qubits, T_steps=T_steps, M=M
    )

    rho_0  = rhos[0]
    rho_T  = rhos[-1]

    # --- Lipschitz constants across steps
    rhos_seq = [[rho] * M for rho in rhos]  # per-step pseudo-ensemble
    L_taus   = lipschitz_trajectory(rhos_seq)
    L_mean   = float(np.mean(L_taus)) if L_taus else 1.0

    # --- Discretization error vs T
    T_range = [2, 4, 6, 8, 10, 12, 15, 20, 30]
    T_vals, err_bounds = discretization_error_vs_steps(L_mean, total_time=1.0, T_range=T_range)

    # --- Convergence rate fitting (losses_all may contain density matrices, extract scalars)
    flat_losses = []
    for step_l in losses_all:
        for l in step_l:
            if hasattr(l, 'numel') and l.numel() > 1:
                flat_losses.append(float(l.abs().mean().real))
            elif hasattr(l, 'item'):
                flat_losses.append(float(l.item()))
            else:
                try:
                    flat_losses.append(float(l))
                except Exception:
                    pass
    conv_fit = fit_convergence_rate(flat_losses)

    # --- Action optimality ratio
    opt_ratio = action_optimality_ratio(rhos, rho_0, rho_T)

    # --- Expressivity bounds for various n_qubits/n_layers
    express_results = []
    for nq_exp in [1, 2, 3, 4]:
        for nl in [2, 4, 6]:
            r = expressivity_lower_bound(nq_exp, nl)
            express_results.append(r)

    # --- Fidelity convergence curve
    fid_curve   = fidelity_convergence_curve(rhos, rho_T)
    bures_curve = bures_convergence_curve(rhos, rho_T)

    # === FIGURE: 4-panel convergence bounds ===
    fig = plt.figure(figsize=(20, 15))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # (a) Lipschitz constants across steps
    ax1 = fig.add_subplot(gs[0, 0])
    step_x = np.arange(len(L_taus))
    ax1.plot(step_x, L_taus, 'o-', color=COLORS['primary'], lw=2.5, ms=7,
             markerfacecolor='white', markeredgewidth=2, label='$L_\\tau$ (per step)')
    ax1.axhline(1.0, color='gray', ls='--', lw=1.5, label='$L=1$ (isometry)')
    ax1.axhline(L_mean, color=COLORS['secondary'], ls='-.', lw=2,
                label=f'$\\bar{{L}}={L_mean:.3f}$')
    ax1.fill_between(step_x, L_taus, 1.0, where=np.array(L_taus) >= 1.0,
                     color='red', alpha=0.15, label='Expansion zone')
    ax1.fill_between(step_x, L_taus, 1.0, where=np.array(L_taus) < 1.0,
                     color='green', alpha=0.15, label='Contraction zone')
    ax1.set_xlabel('Flow Step $\\tau$')
    ax1.set_ylabel('Lipschitz Constant $L_\\tau$')
    ax1.set_title('(a) Per-Step Lipschitz Constant of QFM Map', pad=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, ls='--', alpha=0.5)

    # (b) Discretization error vs T_steps
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(T_vals, err_bounds, 's-', color=COLORS['secondary'], lw=2.5, ms=7,
                 markerfacecolor='white', markeredgewidth=2, label='$e_{global}$ bound')
    ax2.axvline(T_steps, color=COLORS['accent'], ls='--', lw=2,
                label=f'Current $T={T_steps}$')
    # Annotate
    ax2.annotate(f'$e={err_bounds[T_vals.index(T_steps)]:.1e}$',
                 xy=(T_steps, err_bounds[T_vals.index(T_steps)]),
                 xytext=(T_steps + 3, err_bounds[T_vals.index(T_steps)] * 2),
                 arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5),
                 color=COLORS['accent'])
    ax2.set_xlabel('Number of Discrete Steps $T$')
    ax2.set_ylabel('Discretization Error Bound $e_{global}$')
    ax2.set_title(f'(b) Gronwall Error Bound ($L={L_mean:.3f}$)', pad=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, ls='--', which='both', alpha=0.5)

    # (c) Convergence rate: actual loss + fitted curves
    ax3 = fig.add_subplot(gs[1, 0])
    n_show = min(len(flat_losses), 200)
    t_arr  = np.arange(1, n_show + 1)
    ax3.semilogy(t_arr, flat_losses[:n_show], alpha=0.5, color='gray', lw=1, label='Raw loss')
    if 'L0' in conv_fit and 'gamma' in conv_fit:
        exp_fit = conv_fit['L0'] * np.exp(-conv_fit['gamma'] * t_arr)
        ax3.semilogy(t_arr, exp_fit, '-', color=COLORS['primary'], lw=2.5,
                     label=f'Exp. fit ($\\gamma={conv_fit["gamma"]:.4f}$, $R^2={conv_fit["r2_exp"]:.3f}$)')
    if 'C' in conv_fit and 'alpha' in conv_fit:
        pow_fit = conv_fit['C'] * (t_arr ** (-conv_fit['alpha']))
        ax3.semilogy(t_arr, pow_fit, '--', color=COLORS['secondary'], lw=2.5,
                     label=f'Power fit ($\\alpha={conv_fit["alpha"]:.3f}$, $R^2={conv_fit["r2_power"]:.3f}$)')
    ax3.set_xlabel('Optimization Step')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('(c) Convergence Rate: Exponential vs Power-Law Fit', pad=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, ls='--', which='both', alpha=0.5)

    # (d) Expressivity coverage: heatmap n_qubits × n_layers
    ax4 = fig.add_subplot(gs[1, 1])
    nq_range = [1, 2, 3, 4]
    nl_range = [2, 4, 6]
    cov = np.array([[expressivity_lower_bound(nq, nl)['coverage_ratio']
                     for nl in nl_range] for nq in nq_range])
    im = ax4.imshow(cov, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax4.set_xticks(range(len(nl_range)))
    ax4.set_xticklabels([f'L={l}' for l in nl_range])
    ax4.set_yticks(range(len(nq_range)))
    ax4.set_yticklabels([f'n={n}' for n in nq_range])
    for i in range(len(nq_range)):
        for j in range(len(nl_range)):
            color = 'white' if cov[i, j] < 0.5 else 'black'
            ax4.text(j, i, f'{cov[i,j]:.2f}', ha='center', va='center', color=color, fontsize=11)
    ax4.set_xlabel('Number of Layers $L$')
    ax4.set_ylabel('Number of Qubits $n$')
    plt.colorbar(im, ax=ax4, shrink=0.85).set_label('SU($2^n$) Coverage Ratio $\\eta$')
    ax4.set_title('(d) Expressivity Coverage Heatmap\n(Fraction of $SU(2^n)$ parameter space)', pad=12)

    fig.suptitle('Quantum Flow Matching — Convergence Bounds & Mathematical Guarantees',
                 y=0.98, fontsize=14, color=COLORS['primary'])
    path = os.path.join(out_dir, 'qfm_convergence_bounds.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

    # Print summary
    print(f"\n[convergence_bounds] Summary:")
    print(f"  Mean Lipschitz L = {L_mean:.4f}")
    print(f"  Action optimality η = {opt_ratio['optimality_ratio']:.4f}")
    print(f"  Excess transport cost = {opt_ratio['excess_cost']:.6f}")
    if 'gamma' in conv_fit:
        print(f"  Convergence rate γ = {conv_fit['gamma']:.4f} (exp), R² = {conv_fit['r2_exp']:.4f}")
    if 'alpha' in conv_fit:
        print(f"  Power-law exponent α = {conv_fit['alpha']:.4f}, R² = {conv_fit['r2_power']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_convergence_bounds_analysis(args.qubits, args.steps, args.ensemble, args.out)

if __name__ == "__main__":
    main()
