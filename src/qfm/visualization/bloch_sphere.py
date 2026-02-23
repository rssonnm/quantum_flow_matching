"""
bloch_sphere.py — Advanced Bloch Sphere Trajectory Visualization (Q1 Quality).

Multi-panel dashboard for tracking quantum state evolution on the Bloch sphere,
with proper partial trace support for multi-qubit systems.
"""
import os
import io
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D


# ---- Pauli matrices ----
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _partial_trace_qubit(rho_np, n_qubits, keep_qubit=0):
    """Partial trace over all qubits except `keep_qubit`, returning 2×2 reduced rho."""
    d = 2 ** n_qubits
    rho = rho_np.reshape([2] * (2 * n_qubits))
    # Trace over all qubits except keep_qubit
    axes_to_trace = [q for q in range(n_qubits) if q != keep_qubit]
    # We need to trace pairs: qubit q (ket index) with qubit q + n_qubits (bra index)
    # Process in reverse order to keep indices stable
    for q in sorted(axes_to_trace, reverse=True):
        rho = np.trace(rho, axis1=q, axis2=q + n_qubits - (n_qubits - len(axes_to_trace)))
        n_qubits -= 1
    return rho.reshape(2, 2)


def _bloch_vector(rho_2x2):
    """Extract Bloch vector (x, y, z) from a 2×2 density matrix."""
    x = float(np.real(np.trace(rho_2x2 @ _SX)))
    y = float(np.real(np.trace(rho_2x2 @ _SY)))
    z = float(np.real(np.trace(rho_2x2 @ _SZ)))
    return x, y, z


def _draw_bloch_wireframe(ax):
    """Draw a transparent Bloch sphere wireframe with axis labels."""
    # Sphere surface
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_s = np.cos(u) * np.sin(v)
    y_s = np.sin(u) * np.sin(v)
    z_s = np.cos(v)
    ax.plot_wireframe(x_s, y_s, z_s, color='gray', alpha=0.08, linewidth=0.3)

    # Great circles (XY, XZ, YZ planes)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
            color='#B0BEC5', lw=0.8, alpha=0.4)
    ax.plot(np.cos(theta), np.zeros_like(theta), np.sin(theta),
            color='#B0BEC5', lw=0.8, alpha=0.4)
    ax.plot(np.zeros_like(theta), np.cos(theta), np.sin(theta),
            color='#B0BEC5', lw=0.8, alpha=0.4)

    # Axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color='black', alpha=0.2, ls='--', lw=0.5)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='black', alpha=0.2, ls='--', lw=0.5)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color='black', alpha=0.2, ls='--', lw=0.5)

    # Pole labels
    ax.text(0, 0, 1.3, '$|0\\rangle$', fontsize=12, fontweight='bold', ha='center')
    ax.text(0, 0, -1.35, '$|1\\rangle$', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.35, 0, 0, '$X$', fontsize=11, fontweight='bold', ha='center', color='#455A64')
    ax.text(0, 1.35, 0, '$Y$', fontsize=11, fontweight='bold', ha='center', color='#455A64')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_axis_off()


def plot_bloch_trajectory(ensemble_snapshots, out_path="results/bloch_trajectory.png"):
    """
    Advanced 6-Panel Bloch Sphere Dashboard (Q1 Quality).

    Shows BOTH individual ensemble member trajectories AND the ensemble mean,
    revealing quantum dynamics hidden by naive averaging.

    (a) 3D Bloch sphere: individual members + mean trajectory.
    (b) Bloch components with ensemble spread ribbons.
    (c) Individual vs ensemble-averaged purity.
    (d) Angular dispersion (σ_θ, σ_φ) on the Bloch sphere.
    (e) 2D projection with ensemble scatter cloud.
    (f) Von Neumann entropy & l1-norm of coherence.
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass

    T_steps = len(ensemble_snapshots) - 1
    M = len(ensemble_snapshots[0])

    # ---- Compute per-member Bloch vectors ----
    # member_bloch[tau][m] = (x, y, z) for member m at step tau
    member_xs = np.zeros((T_steps + 1, M))
    member_ys = np.zeros((T_steps + 1, M))
    member_zs = np.zeros((T_steps + 1, M))
    member_purities = np.zeros((T_steps + 1, M))

    # Ensemble-averaged metrics
    mean_xs, mean_ys, mean_zs = [], [], []
    ens_purities = []
    ens_entropies = []
    ens_coherences = []

    for tau in range(T_steps + 1):
        snapshot = ensemble_snapshots[tau]
        rho_list = []

        for m, s in enumerate(snapshot):
            if s.dim() == 1:
                rho_m = torch.outer(s, s.conj())
            else:
                rho_m = s
            rho_m_np = rho_m.detach().cpu().numpy()

            d = rho_m_np.shape[0]
            n_qubits = int(np.log2(d))

            if n_qubits == 1:
                rho_1q = rho_m_np
            else:
                rho_1q = _partial_trace_qubit(rho_m_np, n_qubits, keep_qubit=0)

            x, y, z = _bloch_vector(rho_1q)
            member_xs[tau, m] = x
            member_ys[tau, m] = y
            member_zs[tau, m] = z
            member_purities[tau, m] = float(np.real(np.trace(rho_1q @ rho_1q)))
            rho_list.append(rho_m)

        # Ensemble-averaged density matrix
        mean_rho = torch.mean(torch.stack(rho_list), dim=0).detach().cpu().numpy()
        d = mean_rho.shape[0]
        n_qubits = int(np.log2(d))
        if n_qubits == 1:
            rho_ens = mean_rho
        else:
            rho_ens = _partial_trace_qubit(mean_rho, n_qubits, keep_qubit=0)

        mx, my, mz = _bloch_vector(rho_ens)
        mean_xs.append(mx)
        mean_ys.append(my)
        mean_zs.append(mz)
        ens_purities.append(float(np.real(np.trace(rho_ens @ rho_ens))))

        # Von Neumann entropy of ensemble-averaged state
        eigvals = np.linalg.eigvalsh(rho_ens)
        eigvals = eigvals[eigvals > 1e-15]
        S = -np.sum(eigvals * np.log(eigvals + 1e-30))
        ens_entropies.append(float(S))

        # l1-norm of coherence: sum of off-diagonal |rho_ij|
        coh = np.sum(np.abs(rho_ens)) - np.sum(np.abs(np.diag(rho_ens)))
        ens_coherences.append(float(coh))

    mean_xs = np.array(mean_xs)
    mean_ys = np.array(mean_ys)
    mean_zs = np.array(mean_zs)
    ens_purities = np.array(ens_purities)
    ens_entropies = np.array(ens_entropies)
    ens_coherences = np.array(ens_coherences)
    steps = np.arange(T_steps + 1)

    # Component statistics
    x_mean_per_step = np.mean(member_xs, axis=1)
    y_mean_per_step = np.mean(member_ys, axis=1)
    z_mean_per_step = np.mean(member_zs, axis=1)
    x_std = np.std(member_xs, axis=1)
    y_std = np.std(member_ys, axis=1)
    z_std = np.std(member_zs, axis=1)

    # Angular dispersion
    radii_per_member = np.sqrt(member_xs**2 + member_ys**2 + member_zs**2)
    mean_radius = np.mean(radii_per_member, axis=1)
    std_radius = np.std(radii_per_member, axis=1)

    # ---- 6-Panel Dashboard ----
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30)

    n_show = min(M, 20)  # Show up to 20 individual members

    # =============== Panel (a): 3D Bloch Sphere ===============
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    _draw_bloch_wireframe(ax1)

    # Individual member trajectories (translucent)
    for m in range(n_show):
        alpha = 0.15
        ax1.plot(member_xs[:, m], member_ys[:, m], member_zs[:, m],
                '-', color='#90CAF9', lw=0.5, alpha=alpha)

    # Individual member positions at final step (scatter cloud)
    ax1.scatter(member_xs[-1, :n_show], member_ys[-1, :n_show],
               member_zs[-1, :n_show], color='#42A5F5', s=15,
               alpha=0.4, edgecolors='none', label=f'Members ($M$={M})')

    # Individual member positions at initial step
    ax1.scatter(member_xs[0, :n_show], member_ys[0, :n_show],
               member_zs[0, :n_show], color='#A5D6A7', s=15,
               alpha=0.4, edgecolors='none')

    # Ensemble mean trajectory (bold)
    for t in range(T_steps):
        frac = t / max(T_steps - 1, 1)
        color = plt.cm.magma(frac)
        ax1.plot(mean_xs[t:t+2], mean_ys[t:t+2], mean_zs[t:t+2],
                color=color, lw=3.5, alpha=0.95)

    ax1.scatter(mean_xs[0], mean_ys[0], mean_zs[0], color='#2E7D32', s=200,
               zorder=6, edgecolors='black', linewidths=1.5, marker='o',
               label='Ensemble Start')
    ax1.scatter(mean_xs[-1], mean_ys[-1], mean_zs[-1], color='#C62828', s=250,
               zorder=6, edgecolors='black', linewidths=1.5, marker='*',
               label='Ensemble End')

    ax1.view_init(elev=25, azim=-60)
    ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
    ax1.set_title('(a)  Individual Member Trajectories on Bloch Sphere',
                  fontsize=13, fontweight='bold', pad=12)

    # =============== Panel (b): Components with Spread Ribbons ===============
    ax2 = fig.add_subplot(gs[0, 1])

    for vals, std, color, label in [
        (x_mean_per_step, x_std, '#E53935', '$\\langle X \\rangle$'),
        (y_mean_per_step, y_std, '#43A047', '$\\langle Y \\rangle$'),
        (z_mean_per_step, z_std, '#1565C0', '$\\langle Z \\rangle$'),
    ]:
        ax2.plot(steps, vals, 'o-', color=color, lw=2.0, ms=4,
                markerfacecolor='white', markeredgewidth=1.5, label=label)
        ax2.fill_between(steps, vals - std, vals + std, alpha=0.12, color=color)

    ax2.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
    ax2.set_xlabel('Integration Step $\\tau$', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bloch Component', fontsize=12, fontweight='bold')
    ax2.set_ylim(-1.15, 1.15)
    ax2.grid(True, alpha=0.15, ls='--')
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9, ncol=3)
    ax2.set_title('(b)  Bloch Components ± Ensemble Spread',
                  fontsize=13, fontweight='bold')

    # Annotate spread magnitude
    total_spread = np.mean(x_std + y_std + z_std)
    ax2.text(0.02, 0.05, f"Mean $\\sigma$ = {total_spread:.4f}",
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0', alpha=0.9))

    # =============== Panel (c): Member vs Ensemble Purity ===============
    ax3 = fig.add_subplot(gs[1, 0])

    # Individual member purities (all should be ~1 for pure states)
    member_pur_mean = np.mean(member_purities, axis=1)
    member_pur_std = np.std(member_purities, axis=1)

    ax3.plot(steps, member_pur_mean, 'D-', color='#43A047', lw=2.5, ms=6,
            markerfacecolor='white', markeredgewidth=2,
            label='Individual Member $\\gamma_m$ (mean)')
    ax3.fill_between(steps, member_pur_mean - member_pur_std,
                     member_pur_mean + member_pur_std,
                     alpha=0.15, color='#43A047')

    ax3.plot(steps, ens_purities, 's-', color='#7B1FA2', lw=2.5, ms=6,
            markerfacecolor='white', markeredgewidth=2,
            label='Ensemble Average $\\gamma_{ens}$')

    ax3.axhline(1.0, color='#43A047', ls='--', lw=1.5, alpha=0.4, label='Pure ($\\gamma=1$)')
    ax3.axhline(0.5, color='#E53935', ls=':', lw=1.5, alpha=0.4, label='Max. mixed ($1/d$)')

    # Annotate the "purity gap" — key insight!
    gap = member_pur_mean[-1] - ens_purities[-1]
    ax3.annotate(f"Purity Gap\n$\\Delta\\gamma$ = {gap:.3f}",
                xy=(steps[-1], (member_pur_mean[-1] + ens_purities[-1]) / 2),
                xytext=(steps[-1] - 5, 0.7),
                fontsize=10, fontweight='bold', color='#FF6F00',
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9))

    ax3.set_xlabel('Integration Step $\\tau$', fontsize=12, fontweight='bold')
    ax3.set_ylabel('State Purity $\\gamma = \\mathrm{tr}(\\rho^2)$',
                  fontsize=12, fontweight='bold')
    ax3.set_ylim(0.35, 1.1)
    ax3.grid(True, alpha=0.15, ls='--')
    ax3.legend(fontsize=9, loc='center right', framealpha=0.9)
    ax3.set_title('(c)  Individual vs Ensemble Purity — Diversity Gap',
                  fontsize=13, fontweight='bold')

    # =============== Panel (d): Bloch Radius Distribution ===============
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.plot(steps, mean_radius, 'o-', color='#FF6F00', lw=2.5, ms=6,
            markerfacecolor='white', markeredgewidth=2,
            label='Mean $|\\vec{r}_m|$')
    ax4.fill_between(steps, mean_radius - std_radius,
                     mean_radius + std_radius,
                     alpha=0.15, color='#FF6F00', label='$\\pm 1\\sigma$')

    # Ensemble mean radius
    ens_radii = np.sqrt(mean_xs**2 + mean_ys**2 + mean_zs**2)
    ax4.plot(steps, ens_radii, 's--', color='#7B1FA2', lw=2.0, ms=5,
            markerfacecolor='white', markeredgewidth=1.5,
            label='Ensemble Mean $|\\vec{r}_{ens}|$')

    ax4.axhline(1.0, color='#43A047', ls='--', lw=1.5, alpha=0.4,
               label='Pure state boundary')

    # Key insight annotation
    ax4.text(0.02, 0.95,
            f"Members on surface ($|r|$≈{mean_radius[-1]:.3f})\n"
            f"Ensemble at center ($|r|$≈{ens_radii[-1]:.3f})\n"
            f"→ High ensemble diversity",
            transform=ax4.transAxes, fontsize=9, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD', alpha=0.9))

    ax4.set_xlabel('Integration Step $\\tau$', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Bloch Radius $|\\vec{r}|$', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.15)
    ax4.grid(True, alpha=0.15, ls='--')
    ax4.legend(fontsize=9, loc='center right', framealpha=0.9)
    ax4.set_title('(d)  Bloch Radius: Members vs Ensemble',
                  fontsize=13, fontweight='bold')

    # =============== Panel (e): 2D Scatter Cloud ===============
    ax5 = fig.add_subplot(gs[2, 0])

    # Unit circle
    theta_c = np.linspace(0, 2*np.pi, 100)
    ax5.plot(np.cos(theta_c), np.sin(theta_c), color='#E0E0E0', lw=1.5, ls='--')

    # Show member scatter at initial and final steps
    ax5.scatter(member_xs[0, :], member_zs[0, :], color='#A5D6A7', s=25,
               alpha=0.5, edgecolors='#2E7D32', linewidths=0.3,
               label=f'$\\tau=0$ ($M$={M})')
    ax5.scatter(member_xs[-1, :], member_zs[-1, :], color='#EF9A9A', s=25,
               alpha=0.5, edgecolors='#C62828', linewidths=0.3,
               label=f'$\\tau={T_steps}$ ($M$={M})')

    # Mean trajectory
    sc = ax5.scatter(mean_xs, mean_zs, c=steps, cmap='magma', s=80,
                    edgecolors='black', linewidths=0.8, zorder=5)
    ax5.plot(mean_xs, mean_zs, '-', color='#333333', lw=1.5, alpha=0.5)

    ax5.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.3)
    ax5.axvline(0, color='gray', ls=':', lw=0.5, alpha=0.3)
    ax5.set_xlabel('$\\langle X \\rangle$', fontsize=12, fontweight='bold')
    ax5.set_ylabel('$\\langle Z \\rangle$', fontsize=12, fontweight='bold')
    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.15, ls='--')
    ax5.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax5.set_title('(e)  XZ Projection — Ensemble Cloud Evolution',
                  fontsize=13, fontweight='bold')

    cbar = fig.colorbar(sc, ax=ax5, shrink=0.8, pad=0.03)
    cbar.set_label('Step $\\tau$', fontweight='bold')

    # =============== Panel (f): Entropy & Coherence ===============
    ax6 = fig.add_subplot(gs[2, 1])

    color_ent = '#00838F'
    ax6.plot(steps, ens_entropies, 'o-', color=color_ent, lw=2.5, ms=6,
            markerfacecolor='white', markeredgewidth=2,
            label='$S(\\rho_{ens}) = -\\mathrm{tr}(\\rho \\ln \\rho)$')
    max_S = np.log(2)
    ax6.axhline(max_S, color='#E53935', ls=':', lw=1.5, alpha=0.6,
               label=f'$S_{{max}} = \\ln 2 = {max_S:.3f}$')
    ax6.fill_between(steps, ens_entropies, alpha=0.1, color=color_ent)

    # Coherence on right axis
    ax6b = ax6.twinx()
    color_coh = '#FF6F00'
    ax6b.plot(steps, ens_coherences, 'D--', color=color_coh, lw=2.0, ms=5,
             markerfacecolor='white', markeredgewidth=1.5,
             label='$C_{l_1}(\\rho_{ens})$')
    ax6b.set_ylabel('$l_1$-Coherence $C_{l_1}$', fontsize=11,
                    fontweight='bold', color=color_coh)
    ax6b.tick_params(axis='y', labelcolor=color_coh)

    ax6.set_xlabel('Integration Step $\\tau$', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Von Neumann Entropy $S$', fontsize=12,
                  fontweight='bold', color=color_ent)
    ax6.tick_params(axis='y', labelcolor=color_ent)
    ax6.grid(True, alpha=0.15, ls='--')

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              loc='center right', framealpha=0.9)
    ax6.set_title('(f)  Ensemble Entropy & Quantum Coherence',
                  fontsize=13, fontweight='bold')

    fig.suptitle("Bloch Sphere Analysis — Ensemble Quantum State Dynamics\n"
                 f"($M$={M} members, $T$={T_steps} steps, Qubit 0 reduced state)",
                 fontsize=19, fontweight='bold', y=1.00, color='#2D3436')

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[bloch] Saved Q1 6-Panel Bloch Dashboard → {out_path}")


def animate_bloch_sphere(ensemble_snapshots, out_path="results/bloch_animation.gif"):
    """Generate animated GIF of Bloch sphere trajectory."""
    import qutip as qt
    from PIL import Image

    T_steps = len(ensemble_snapshots) - 1
    images = []

    xs, ys, zs = [], [], []
    for tau in range(T_steps + 1):
        snapshot = ensemble_snapshots[tau]
        mean_rho = torch.mean(torch.stack([
            torch.outer(s, s.conj()) if s.dim() == 1 else s
            for s in snapshot
        ]), dim=0).numpy()

        d = mean_rho.shape[0]
        n_qubits = int(np.log2(d))
        if n_qubits == 1:
            rho_1q = mean_rho
        else:
            rho_1q = _partial_trace_qubit(mean_rho, n_qubits, keep_qubit=0)

        x, y, z = _bloch_vector(rho_1q)
        xs.append(x); ys.append(y); zs.append(z)

        b = qt.Bloch()
        b.point_color = plt.cm.plasma(np.linspace(0, 1, tau + 1)).tolist()
        b.add_points([xs, ys, zs], meth='l')
        b.add_points([xs, ys, zs], meth='m')
        b.make_sphere()

        buf = io.BytesIO()
        b.fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(b.fig)
        buf.seek(0)
        img = Image.open(buf)
        images.append(img.copy())
        buf.close()

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0
    )

