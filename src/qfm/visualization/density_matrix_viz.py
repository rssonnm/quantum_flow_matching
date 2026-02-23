"""
density_matrix_viz.py — Density matrix and eigenspectrum visualizations.
"""
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

logger = logging.getLogger(__name__)

def plot_density_matrix(
    rho: torch.Tensor,
    tau: int,
    out_path: str = None,
    title_prefix: str = "",
):
    """
    Ultra-Q1 6-Panel Density Matrix Analysis Dashboard.

    (a) Re(ρ) heatmap with element annotations and TwoSlopeNorm.
    (b) Im(ρ) heatmap with coherence highlighting.
    (c) |ρ| absolute value matrix — coherence structure.
    (d) Eigenvalue spectrum with purity/entropy metrics.
    (e) 3D bar visualization of |ρ_ij|.
    (f) Quantum state metrics summary panel.
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.mplot3d import Axes3D

    rho_np = rho.detach().cpu().numpy()
    d = rho_np.shape[0]
    n_qubits = int(np.log2(d))
    ev = np.sort(np.linalg.eigvalsh(rho_np))[::-1]

    # Computational basis labels
    basis_labels = [f'$|{format(i, f"0{n_qubits}b")}\\rangle$' for i in range(d)]

    # Metrics
    purity = float(np.sum(ev**2))
    ev_pos = ev[ev > 1e-15]
    entropy = float(-np.sum(ev_pos * np.log2(ev_pos + 1e-30)))
    pr = 1.0 / purity if purity > 0 else d  # Participation ratio
    trace_val = float(np.real(np.trace(rho_np)))
    coherence_l1 = float(np.sum(np.abs(rho_np)) - np.sum(np.abs(np.diag(rho_np))))
    max_coherence = float(d * (d - 1)) if d > 1 else 1.0
    relative_coherence = coherence_l1 / max_coherence

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ========== Panel (a): Re(ρ) ==========
    ax0 = fig.add_subplot(gs[0, 0])
    re_data = rho_np.real
    vmax_re = max(abs(re_data.min()), abs(re_data.max()), 0.01)
    norm_re = TwoSlopeNorm(vmin=-vmax_re, vcenter=0, vmax=vmax_re)
    im0 = ax0.imshow(re_data, cmap='RdBu_r', norm=norm_re, aspect='equal')

    # Annotate elements
    for i in range(d):
        for j in range(d):
            val = re_data[i, j]
            color = 'white' if abs(val) > vmax_re * 0.6 else 'black'
            ax0.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=max(7, 10 - d), color=color, fontweight='bold')

    ax0.set_xticks(range(d))
    ax0.set_yticks(range(d))
    ax0.set_xticklabels(basis_labels, fontsize=9)
    ax0.set_yticklabels(basis_labels, fontsize=9)
    cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label('Amplitude', fontweight='bold')
    ax0.set_title('(a)  $\\mathrm{Re}(\\rho)$', fontsize=14, fontweight='bold')

    # ========== Panel (b): Im(ρ) ==========
    ax1 = fig.add_subplot(gs[0, 1])
    im_data = rho_np.imag
    vmax_im = max(abs(im_data.min()), abs(im_data.max()), 0.01)
    norm_im = TwoSlopeNorm(vmin=-vmax_im, vcenter=0, vmax=vmax_im)
    im1 = ax1.imshow(im_data, cmap='PiYG', norm=norm_im, aspect='equal')

    for i in range(d):
        for j in range(d):
            val = im_data[i, j]
            color = 'white' if abs(val) > vmax_im * 0.6 else 'black'
            ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=max(7, 10 - d), color=color, fontweight='bold')

    ax1.set_xticks(range(d))
    ax1.set_yticks(range(d))
    ax1.set_xticklabels(basis_labels, fontsize=9)
    ax1.set_yticklabels(basis_labels, fontsize=9)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Amplitude', fontweight='bold')
    ax1.set_title('(b)  $\\mathrm{Im}(\\rho)$', fontsize=14, fontweight='bold')

    # ========== Panel (c): |ρ| Coherence Matrix ==========
    ax2 = fig.add_subplot(gs[0, 2])
    abs_data = np.abs(rho_np)
    im2 = ax2.imshow(abs_data, cmap='magma', vmin=0, aspect='equal')

    for i in range(d):
        for j in range(d):
            val = abs_data[i, j]
            color = 'white' if val > abs_data.max() * 0.5 else 'black'
            ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=max(7, 10 - d), color=color, fontweight='bold')

    ax2.set_xticks(range(d))
    ax2.set_yticks(range(d))
    ax2.set_xticklabels(basis_labels, fontsize=9)
    ax2.set_yticklabels(basis_labels, fontsize=9)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('$|\\rho_{ij}|$', fontweight='bold')
    ax2.set_title('(c)  Coherence Matrix $|\\rho|$', fontsize=14, fontweight='bold')

    # ========== Panel (d): Eigenvalue Spectrum ==========
    ax3 = fig.add_subplot(gs[1, 0])
    colors_ev = [plt.cm.viridis(i / max(d - 1, 1)) for i in range(d)]
    bars = ax3.bar(range(d), ev, color=colors_ev, edgecolor='black',
                  linewidth=0.8, alpha=0.85)
    ax3.axhline(1.0 / d, color='#E53935', ls=':', lw=2, alpha=0.7,
               label=f'Max. mixed ($1/d = {1/d:.3f}$)')

    # Annotate each eigenvalue
    for i, val in enumerate(ev):
        ax3.text(i, val + 0.02, f'{val:.4f}', ha='center', fontsize=9,
                fontweight='bold', color=colors_ev[i])

    ax3.set_xticks(range(d))
    ax3.set_xticklabels([f'$\\lambda_{{{i+1}}}$' for i in range(d)], fontsize=10)
    ax3.set_ylabel('Eigenvalue $\\lambda_i$', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, max(ev) * 1.2 + 0.05)
    ax3.grid(True, alpha=0.15, ls='--', axis='y')
    ax3.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax3.set_title('(d)  Eigenvalue Spectrum', fontsize=14, fontweight='bold')

    # ========== Panel (e): 3D Bar Plot ==========
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    xpos, ypos = np.meshgrid(range(d), range(d))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.6
    dz = abs_data.flatten()

    # Color by magnitude
    colors_3d = plt.cm.magma(dz / (dz.max() + 1e-9))
    ax4.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_3d, alpha=0.85,
             edgecolor='black', linewidth=0.3)

    ax4.set_xticks(range(d))
    ax4.set_yticks(range(d))
    if d <= 4:
        ax4.set_xticklabels([f'|{format(i, f"0{n_qubits}b")}⟩' for i in range(d)], fontsize=8)
        ax4.set_yticklabels([f'⟨{format(i, f"0{n_qubits}b")}|' for i in range(d)], fontsize=8)
    ax4.set_zlabel('$|\\rho_{ij}|$', fontsize=10, fontweight='bold')
    ax4.view_init(elev=30, azim=-50)
    ax4.set_title('(e)  3D Density Matrix Structure',
                  fontsize=14, fontweight='bold', pad=12)

    # ========== Panel (f): Metrics Summary ==========
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    metrics = [
        ('Trace $\\mathrm{tr}(\\rho)$',       f'{trace_val:.6f}',        '#1565C0'),
        ('Purity $\\gamma$',                   f'{purity:.6f}',           '#7B1FA2'),
        ('Von Neumann $S$ (bits)',             f'{entropy:.4f}',          '#00838F'),
        ('Max Entropy $\\log_2 d$',            f'{np.log2(d):.4f}',       '#E53935'),
        ('Participation Ratio',               f'{pr:.4f}',               '#FF6F00'),
        ('$l_1$-Coherence $C_{l_1}$',         f'{coherence_l1:.6f}',     '#43A047'),
        ('Relative Coherence',                f'{relative_coherence:.4f}', '#795548'),
        ('Dominant $\\lambda_1$',              f'{ev[0]:.6f}',            '#1565C0'),
        ('Rank (ε=1e-6)',                     f'{np.sum(ev > 1e-6)}',     '#E53935'),
    ]

    y_start = 0.92
    dy_text = 0.095
    for idx, (name, val, color) in enumerate(metrics):
        y = y_start - idx * dy_text
        ax5.text(0.05, y, name, fontsize=12, fontweight='bold',
                transform=ax5.transAxes, color='#333333')
        ax5.text(0.75, y, val, fontsize=12, fontweight='bold',
                transform=ax5.transAxes, color=color,
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                         ec=color, alpha=0.8, lw=1.5))

    # State classification
    if purity > 0.99:
        state_type = "Pure State ✓"
        state_color = '#43A047'
    elif purity > 1.0 / d + 0.01:
        state_type = "Mixed State"
        state_color = '#FF6F00'
    else:
        state_type = "Maximally Mixed"
        state_color = '#E53935'

    ax5.text(0.5, 0.02, state_type, fontsize=16, fontweight='bold',
            transform=ax5.transAxes, ha='center',
            color='white',
            bbox=dict(boxstyle='round,pad=0.5', fc=state_color, alpha=0.9))

    ax5.set_title('(f)  Quantum State Metrics', fontsize=14, fontweight='bold')

    fig.suptitle(f"{title_prefix}Density Matrix Analysis — $\\tau = {tau}$  "
                 f"({n_qubits}-qubit, $d = {d}$)",
                 fontsize=19, fontweight='bold', y=0.99, color='#2D3436')

    if out_path:
        fig.savefig(out_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved Q1 density matrix dashboard to {out_path}")
        print(f"\n[density_matrix] Saved Q1 Dashboard → {out_path}")
    else:
        return fig

def plot_eigenspectrum_sequence(
    rhos_per_tau: list,
    out_path: str = "results/eigenspectrum_sequence.png",
):
    """
    Advanced Eigenspectrum Evolution Analysis (Ultra-Q1 Quality).
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass

    T = len(rhos_per_tau)
    d = rhos_per_tau[0].shape[0]
    ev_matrix = np.zeros((T, d))
    
    # Metrics
    purities = []
    participation_ratios = []
    spectral_gaps = []
    
    for t, rho_t in enumerate(rhos_per_tau):
        # Using numpy to calculate eigenvalues
        if isinstance(rho_t, torch.Tensor):
            rho_np = rho_t.detach().cpu().numpy()
        else:
            rho_np = rho_t
            
        ev = np.linalg.eigvalsh(rho_np)
        ev_sorted = np.sort(ev)[::-1] # Descending order
        ev_matrix[t] = ev_sorted
        
        # Calculate Purity and PR
        purity = np.sum(ev_sorted ** 2)
        purities.append(purity)
        pr = 1.0 / purity if purity > 0 else d
        participation_ratios.append(pr)
        
        # Calculate Spectral Gap (between 1st and 2nd largest eigenvalues)
        gap = ev_sorted[0] - ev_sorted[1] if d > 1 else 0
        spectral_gaps.append(gap)

    steps = np.arange(T)
    fig = plt.figure(figsize=(19, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)
    
    # --- Panel (a): Log-Scale Eigenspectrum Evolution ---
    ax1 = fig.add_subplot(gs[0])
    cmap = plt.cm.viridis
    
    # Plot top few eigenvalues explicitly, others as a density or just lines
    num_to_plot = min(16, d) # Plot up to top 16 eigenvalues
    
    for i in range(num_to_plot):
        color = cmap(i / num_to_plot)
        lw = 2.5 if i < 2 else 1.0 # Highlight top 2
        alpha = 0.9 if i < 5 else 0.4
        label = f"$\\lambda_{{{i+1}}}$" if i < 3 else None
        
        # Add small espilon for log scale safety
        ax1.plot(steps, np.maximum(ev_matrix[:, i], 1e-10), 'o-', color=color, lw=lw, alpha=alpha, markersize=4, label=label)

    ax1.set_yscale('log')
    ax1.set_ylim(1e-6, 1.2)
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Integration Step $\\tau$', fontweight='bold')
    ax1.set_ylabel('Eigenvalue Magnitude $\\lambda_i$ (Log Scale)', fontweight='bold')
    ax1.set_title('(a) Eigenspectrum Flow Dynamics', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, ls='--')
    ax1.legend(loc='lower left')

    # --- Panel (b): Spectral Gap Tracking ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, spectral_gaps, 's-', color='#D32F2F', lw=3, markersize=8, label='Spectral Gap $\\Delta = \\lambda_1 - \\lambda_2$')
    
    # Shade region where gap is close to 1 (pure state)
    ax2.fill_between(steps, 0, spectral_gaps, color='#FFCDD2', alpha=0.3)
    
    ax2.set_xlabel('Integration Step $\\tau$', fontweight='bold')
    ax2.set_ylabel('Spectral Gap Limit $\\Delta$', fontweight='bold')
    ax2.set_title('(b) Target State Isolation (Gap)', fontsize=13, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, ls='--')
    ax2.legend(loc='lower right')
    
    # Annotate max gap
    max_gap_idx = np.argmax(spectral_gaps)
    ax2.annotate(f'Max Gap:\n{spectral_gaps[max_gap_idx]:.3f}', 
                 xy=(steps[max_gap_idx], spectral_gaps[max_gap_idx]),
                 xytext=(steps[max_gap_idx]-2, spectral_gaps[max_gap_idx]-0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # --- Panel (c): Purity & Participation Ratio ---
    ax3 = fig.add_subplot(gs[2])
    
    c_purity = '#388E3C'
    c_pr = '#512DA8'
    
    line1 = ax3.plot(steps, purities, 'D-', color=c_purity, lw=3, markersize=8, label='State Purity $\\gamma = \\mathrm{Tr}(\\rho^2)$')
    ax3.set_xlabel('Integration Step $\\tau$', fontweight='bold')
    ax3.set_ylabel('Purity $\\gamma$', fontweight='bold', color=c_purity)
    ax3.tick_params(axis='y', labelcolor=c_purity)
    ax3.set_ylim(0, 1.05)
    
    # Second y-axis for Participation Ratio
    ax4 = ax3.twinx()
    line2 = ax4.plot(steps, participation_ratios, 'X--', color=c_pr, lw=2.5, markersize=8, label='Participation Ratio $PR$')
    ax4.set_ylabel('Effective Rank (PR)', fontweight='bold', color=c_pr)
    ax4.tick_params(axis='y', labelcolor=c_pr)
    # PR goes from 1 (pure) to d (completely mixed)
    ax4.set_ylim(0.5, max(participation_ratios)*1.1)
    
    ax3.set_title('(c) Decoherence & Rank Evolution', fontsize=13, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='center right')
    ax3.grid(True, alpha=0.3, ls='--')

    fig.suptitle('Quantum Flow Spectral Analysis: Eigenspectrum, Gap Isolation, and Purity Dynamics', 
                 fontsize=17, fontweight='bold', y=1.05)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved highly detailed Q1 eigenspectrum analysis to {out_path}")
