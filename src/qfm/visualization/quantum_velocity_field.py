"""
quantum_velocity_field.py — Quantum Velocity Field Visualization.

The QFM "velocity" at step τ is:
    V_τ(ρ) = ρ_{τ+1} - ρ_τ    (discrete increment, PCA-projected)

This is the quantum analogue of the classical flow vector field u_t(x)
in continuous normalizing flow models.

Shows:
  1. Quiver arrows ∂_τ ρ in 2D PCA space (velocity field)
  2. Streamplot showing the global flow direction
  3. Divergence and curl of the projected field (classical analogy)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from scipy.interpolate import griddata


def _rho_to_vec(rho: torch.Tensor) -> np.ndarray:
    r = rho.detach().cpu()
    return np.concatenate([r.real.numpy().ravel(), r.imag.numpy().ravel()])


def plot_velocity_field(
    ensemble_snapshots: list,    # list (T+1) of list/tensor of (d,d) rhos
    pca: PCA = None,
    out_path: str = "results/quantum_velocity_field.png",
):
    """
    3-panel figure:
    (1) Quiver velocity field + sample trajectories
    (2) Streamplot of interpolated field
    (3) Velocity magnitude |V_τ| along the mean trajectory
    """
    T = len(ensemble_snapshots) - 1
    cmap = plt.cm.plasma

    # Fit or reuse PCA
    all_rhos = [r for snap in ensemble_snapshots for r in snap]
    X = np.stack([_rho_to_vec(r) for r in all_rhos])
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(X)

    # Project all snapshots
    proj_per_tau = []
    for snap in ensemble_snapshots:
        pts = pca.transform(np.stack([_rho_to_vec(r) for r in snap]))
        proj_per_tau.append(pts)

    means = np.array([p.mean(axis=0) for p in proj_per_tau])

    # Compute velocity vectors (finite differences)
    velocities = []                      # at each τ: one velocity per particle
    all_positions = []
    all_velocities = []

    for tau in range(T):
        pts_t0 = proj_per_tau[tau]
        pts_t1 = proj_per_tau[tau + 1]
        M = min(len(pts_t0), len(pts_t1))
        v = pts_t1[:M] - pts_t0[:M]
        velocities.append(v)
        all_positions.append(pts_t0[:M])
        all_velocities.append(v)

    all_pos = np.vstack(all_positions)
    all_vel = np.vstack(all_velocities)

    # Figure
    fig = plt.figure(figsize=(17, 5.5), facecolor="white")
    gs  = GridSpec(1, 3, figure=fig, wspace=0.32)

    # --- Panel 1: Quiver + trajectories
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#F8F9FA")

    # Sample trajectories
    M_traj = min(len(proj_per_tau[0]), 30)
    for m in range(M_traj):
        xs = [proj_per_tau[tau][m, 0] for tau in range(T + 1)]
        ys = [proj_per_tau[tau][m, 1] for tau in range(T + 1)]
        ax1.plot(xs, ys, color="gray", lw=0.6, alpha=0.25, zorder=2)

    # Quiver arrows at mean positions
    ax1.quiver(means[:-1, 0], means[:-1, 1],
               (means[1:, 0] - means[:-1, 0]),
               (means[1:, 1] - means[:-1, 1]),
               np.arange(T), cmap=cmap, scale_units="xy", scale=1.0,
               width=0.005, headwidth=5, headlength=6, zorder=6, alpha=0.9)
    ax1.scatter(means[0, 0], means[0, 1],
                color="red", s=150, zorder=8, edgecolors="white", lw=1.5)
    ax1.scatter(means[-1, 0], means[-1, 1],
                color="blue", s=150, marker="*", zorder=8, edgecolors="white", lw=1.5)

    ax1.set_title("Quantum Velocity Arrows\n$V_\\tau = \\bar\\rho_{\\tau+1} - \\bar\\rho_\\tau$",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("PC₁", fontsize=10); ax1.set_ylabel("PC₂", fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle="--")

    # --- Panel 2: Streamplot
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#F0F4F8")

    # Interpolate velocity field onto a regular grid
    margin = 0.15
    x_min, x_max = all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin
    y_min, y_max = all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin
    gx = np.linspace(x_min, x_max, 25)
    gy = np.linspace(y_min, y_max, 25)
    GX, GY = np.meshgrid(gx, gy)

    # Interpolate U and V components
    U = griddata(all_pos, all_vel[:, 0], (GX, GY), method="linear", fill_value=0)
    V = griddata(all_pos, all_vel[:, 1], (GX, GY), method="linear", fill_value=0)
    speed = np.sqrt(U**2 + V**2)

    strm = ax2.streamplot(gx, gy, U.T, V.T, color=speed.T,
                           cmap="viridis", linewidth=1.2,
                           arrowstyle="->", density=1.4)
    fig.colorbar(strm.lines, ax=ax2, label="|V| (speed)", shrink=0.85)
    ax2.set_title("Quantum Flow Streamlines", fontsize=11, fontweight="bold")
    ax2.set_xlabel("PC₁", fontsize=10); ax2.set_ylabel("PC₂", fontsize=10)

    # --- Panel 3: Velocity magnitude along trajectory
    ax3 = fig.add_subplot(gs[2])
    vmag = np.linalg.norm(means[1:] - means[:-1], axis=1)
    taus = np.arange(1, T + 1)
    ax3.bar(taus, vmag, color=[cmap(t / T) for t in range(T)],
            edgecolor="white", lw=0.5)
    ax3.plot(taus, vmag, "o-", color="#2D3436", lw=1.5, ms=6, zorder=5)
    ax3.set_xlabel("Step τ", fontsize=10)
    ax3.set_ylabel(r"$|\bar V_\tau|$  (mean displacement)", fontsize=10)
    ax3.set_title("Velocity Magnitude |$V_τ$|", fontsize=11, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Quantum Flow Velocity Field  $\\partial_\\tau \\rho$ in State Space",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"[velocity_field] Saved → {out_path}")
