"""
spectrum_dynamics.py — Spectral Dynamics of Density Matrix Evolution.

Visualizes how the eigenvalue spectrum of ρ_τ changes across QFM steps:
  1. Eigenvalue "spaghetti" plot: λ_i(τ) for each i
  2. Entanglement spectrum (Schmidt values) for bipartite systems
  3. Purity–Entropy phase portrait: (γ(τ), S(τ)) trajectory on the
     purity-entropy plane, bounded by the Gibbs state curve
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_spectrum_dynamics(
    rhos_per_tau: list,        # list (T) of mean (or representative) density matrices
    n_qubits_A: int = 1,
    n_qubits_B: int = 0,
    out_path: str = "results/spectrum_dynamics.png",
):
    """
    Advanced 3-panel spectral analysis figure (Ultra-Q1 Quality):
    (1) Eigenspectrum Log-Scale Evolution (Resolution of minor eigenvalues).
    (2) Information Velocity (Entropy Rate dS/dτ & Purity Rate dγ/dτ).
    (3) Phase-Space Trajectory on the Purity-Entropy plane with Gibbs bounds.
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass
        
    T   = len(rhos_per_tau)
    taus = np.arange(T)
    d   = rhos_per_tau[0].shape[0]

    # Collect eigenspectra
    ev_matrix = np.zeros((T, d))
    vn_vals   = []
    pur_vals  = []

    for t, rho in enumerate(rhos_per_tau):
        # Handle both tensor and numpy inputs seamlessly
        if isinstance(rho, torch.Tensor):
            ev = torch.linalg.eigvalsh(rho).real.clamp(min=0)
            ev_sorted = torch.sort(ev, descending=True).values
            ev_np = ev_sorted.detach().cpu().numpy()
        else:
            ev = np.linalg.eigvalsh(rho)
            ev_sorted = np.sort(np.maximum(ev, 0))[::-1]
            ev_np = ev_sorted
            
        ev_matrix[t] = ev_np

        # Von Neumann entropy
        ev_clamped = np.maximum(ev_np, 1e-12)
        S = -float(np.sum(ev_clamped * np.log2(ev_clamped)))
        vn_vals.append(S)
        pur_vals.append(float(np.sum(ev_np ** 2)))

    # Calculate Information Rates (Numerical Derivatives)
    dS_dt = np.gradient(vn_vals)
    dPur_dt = np.gradient(pur_vals)

    # Gibbs bound: purity vs entropy for a d-dim system
    # For a d×d diagonal ρ with eigenvalues λ: S and γ = Σλ²
    lambdas_grid = np.linspace(1e-4, 1.0 - 1e-4, 500)
    gibbs_S, gibbs_gamma = [], []
    for lam in lambdas_grid:
        rest = (1.0 - lam) / (d - 1) if d > 1 else 0.0
        evs = np.array([lam] + [rest] * (d - 1))
        evs = evs / evs.sum()
        evs = np.clip(evs, 1e-12, 1.0)
        S_g = -np.sum(evs * np.log2(evs))
        g   = np.sum(evs ** 2)
        gibbs_S.append(S_g)
        gibbs_gamma.append(g)

    # Figure Setup
    fig = plt.figure(figsize=(19, 5.5))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.35)

    # --- Panel 1: Eigenspectrum Log-Scale Spaghetti
    ax1 = fig.add_subplot(gs[0])
    cmap1 = plt.cm.viridis
    
    num_to_plot = min(16, d)
    for i in range(num_to_plot):
        color = cmap1(i / max(num_to_plot - 1, 1))
        lw = 2.5 if i == 0 else 1.2
        alpha = 1.0 if i < 3 else 0.5
        label = f"$\\lambda_{{{i+1}}}$" if i < 3 else None
        # Add epsilon for log plot
        ax1.plot(taus, np.maximum(ev_matrix[:, i], 1e-10), "o-", color=color,
                 lw=lw, ms=5, alpha=alpha, label=label)

    ax1.set_yscale('log')
    ax1.set_ylim(1e-6, 1.5)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Integration Step $\\tau$", fontweight="bold")
    ax1.set_ylabel("Eigenvalue Magnitude $\\lambda_i$ (Log Scale)", fontweight="bold")
    ax1.set_title("(a) Spectral Condensation", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower left", framealpha=0.9)
    ax1.grid(True, alpha=0.3, ls="--")

    # --- Panel 2: Information Velocity
    ax2 = fig.add_subplot(gs[1])
    
    color_s = '#D32F2F'
    color_p = '#1976D2'
    
    # Plot rates of change
    ax2.plot(taus, np.abs(dS_dt), 's-', color=color_s, lw=2.5, ms=7, label='Entropy Rate $|dS/d\\tau|$')
    ax2.plot(taus, np.abs(dPur_dt), '^-', color=color_p, lw=2.5, ms=7, label='Purity Velocity $|d\\gamma/d\\tau|$')
    
    # Shade the "turbulent" mixing phase where velocities peak
    max_rate_idx = np.argmax(np.abs(dS_dt))
    ax2.axvspan(max(0, max_rate_idx-2), min(T-1, max_rate_idx+2), color='gray', alpha=0.15, label='Maximal Information Mixing')
    
    ax2.set_xlabel("Integration Step $\\tau$", fontweight="bold")
    ax2.set_ylabel("Information Velocity", fontweight="bold")
    ax2.set_title("(b) Thermodynamic Flow Rate", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, alpha=0.3, ls="--")

    # --- Panel 3: Purity-Entropy phase portrait
    ax3 = fig.add_subplot(gs[2])

    # Gibbs state bound (theoretical boundary)
    ax3.plot(gibbs_gamma, gibbs_S, "-", color="#B0BEC5", lw=3.0,
             label="Theoretical Gibbs Bound", zorder=1)
    ax3.fill_betweenx(gibbs_S, 0, gibbs_gamma, alpha=0.1, color="#78909C")

    # QFM trajectory
    sc = ax3.scatter(pur_vals, vn_vals, c=taus, cmap="plasma",
                     s=120, zorder=5, edgecolors="black", lw=1.0)
    ax3.plot(pur_vals, vn_vals, "-", color="#37474F", lw=2.0, zorder=4, alpha=0.8)
    
    # Directional Arrows
    for t in range(0, T - 1, max(T // 6, 1)):
        ax3.annotate("",
                     xy=(pur_vals[t + 1], vn_vals[t + 1]),
                     xytext=(pur_vals[t], vn_vals[t]),
                     arrowprops=dict(arrowstyle="-|>", color="#D84315",
                                     lw=2.0, mutation_scale=15))

    ax3.set_xlabel("State Purity $\\gamma(\\rho) = \\mathrm{Tr}(\\rho^2)$", fontweight="bold")
    ax3.set_ylabel("Von Neumann Entropy $S(\\rho)$", fontweight="bold")
    ax3.set_title("(c) Information Geometry Phase Space", fontsize=13, fontweight="bold")
    
    # Reference Points
    ax3.scatter([1.0], [0.0], marker="*", s=300, color="gold",
                edgecolors="#F57F17", lw=1.5, zorder=8, label="Target Pure State")
    if d > 1:
        ax3.scatter([1.0 / d], [np.log2(d)], marker="D", s=150, color="#E57373",
                    edgecolors="#B71C1C", lw=1.5, zorder=8, label=f"Max Mixed ($d={d}$)")
                    
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, alpha=0.3, ls="--")

    cbar = fig.colorbar(sc, ax=ax3, pad=0.02)
    cbar.set_label("Integration Step $\\tau$", fontweight="bold")

    fig.suptitle("Thermodynamics of Quantum Flow Matching: Spectral Condensation & Information Geometry",
                 fontsize=17, fontweight="bold", y=1.05)
                 
    # Finalize
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectrum] Saved Advanced Q1 Graphic → {out_path}")
