"""
state_manifold.py — Quantum State Manifold Projection.

Projects density matrices onto a low-dimensional (2D / 3D) manifold using PCA
on the vectorized form vec(ρ) ∈ R^{2d²}.

Shows:
  - Each state in the ensemble as a colored point
  - Mean trajectory as a thick geodesic
  - Confidence ellipses at each step
  - 3D version with the "quantum manifold" curvature surface
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def _rho_to_vec(rho: torch.Tensor) -> np.ndarray:
    r = rho.detach().cpu()
    return np.concatenate([r.real.numpy().ravel(), r.imag.numpy().ravel()])


def _confidence_ellipse(ax, pts, n_std=1.5, color="gray", alpha=0.15, lw=1.2, zorder=2):
    """Draw a covariance confidence ellipse for a 2D point cloud."""
    if len(pts) < 3:
        return
    mean = pts.mean(axis=0)
    cov  = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                  facecolor=color, alpha=alpha,
                  edgecolor=color, linewidth=lw, zorder=zorder)
    ax.add_patch(ell)


def plot_state_manifold_2d(
    ensemble_snapshots: list,    # list (T+1) of tensors (M, d, d)
    out_path: str = "results/state_manifold_2d.png",
    title: str = "Quantum State Manifold Information Flow (PCA 2D)",
):
    """
    Advanced 2D PCA projection of the structural flow (Ultra-Q1 Quality).
    Shows:
      - Contour density representing the quantum state distribution.
      - Mean geodesic trajectory across tau.
      - Optimization momentum arrows.
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass
        
    T = len(ensemble_snapshots) - 1
    cmap = plt.cm.inferno

    # Collect all rhos and fit PCA
    all_rhos = [r for snap in ensemble_snapshots for r in snap]
    X = np.stack([_rho_to_vec(r) for r in all_rhos])
    pca = PCA(n_components=2)
    pca.fit(X)

    fig, ax = plt.subplots(figsize=(11, 8))
    
    # 1. Background Density / KDE
    from scipy.stats import gaussian_kde
    pts_all = pca.transform(X)
    xmin, xmax = pts_all[:, 0].min() - 0.2, pts_all[:, 0].max() + 0.2
    ymin, ymax = pts_all[:, 1].min() - 0.2, pts_all[:, 1].max() + 0.2
    
    X_grid, Y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    kernel = gaussian_kde(pts_all.T)
    Z = np.reshape(kernel(positions).T, X_grid.shape)
    
    # Shade background density
    ax.contourf(X_grid, Y_grid, Z, levels=15, cmap='Blues', alpha=0.3, zorder=1)
    ax.contour(X_grid, Y_grid, Z, levels=15, colors='steelblue', alpha=0.2, linewidths=0.5, zorder=2)

    means = []
    # Plot Scatter Cloud and Ellipses
    for tau, snap in enumerate(ensemble_snapshots):
        pts = pca.transform(np.stack([_rho_to_vec(r) for r in snap]))
        c   = cmap(tau / T)
        
        # Plot point cloud
        alpha_pt = 0.8 if tau in [0, T] else 0.3
        ax.scatter(pts[:, 0], pts[:, 1], color=c, s=15, alpha=alpha_pt, zorder=4, edgecolor='none')
        
        # Plot 2-sigma confidence ellipse to show ensemble variance
        _confidence_ellipse(ax, pts, n_std=2.0, color=c, alpha=0.08, lw=1.5, zorder=3)
        
        mean = pts.mean(axis=0)
        means.append(mean)

    means = np.array(means)

    # 2. Draw Vector Flow / Geodesic Path
    # Fit a smooth B-spline to the mean path for fluid dynamics look
    from scipy.interpolate import splprep, splev
    tck, u = splprep([means[:, 0], means[:, 1]], s=0.0)
    unew = np.linspace(0, 1.0, 300)
    out = splev(unew, tck)
    
    # Plot smooth geodesic shadow
    ax.plot(out[0], out[1], color='gray', lw=8, alpha=0.3, solid_capstyle='round', zorder=5)
    
    # Plot stepped dynamic path with colored segments
    for t in range(T):
        color = cmap((t + 0.5) / T)
        ax.plot([means[t, 0], means[t+1, 0]], [means[t, 1], means[t+1, 1]], 
                color=color, lw=3.0, zorder=6, solid_capstyle='round')

        # Momentum Arrows
        if t % max(1, T // 8) == 0:
            dx = means[t+1, 0] - means[t, 0]
            dy = means[t+1, 1] - means[t, 1]
            ax.arrow(means[t, 0], means[t, 1], dx*0.5, dy*0.5, 
                     head_width=0.03, head_length=0.045, fc='black', ec='black', 
                     zorder=7, alpha=0.8, length_includes_head=True)

    # Mark start and end states
    ax.scatter(*means[0],  color=cmap(0), s=250, zorder=8, edgecolors="black", linewidths=1.5, marker="o", label="Source Distribution")
    ax.scatter(*means[-1], color=cmap(1.0), s=350, zorder=8, edgecolors="black", linewidths=1.5, marker="*", label="Target Ground State")

    # Add descriptive annotations
    ax.annotate("Initial Thermal/Mixed State", means[0], fontsize=11, fontweight="bold", color=cmap(0), xytext=(-10, 15), textcoords="offset points")
    ax.annotate("Optimal Flow Condensation", means[-1], fontsize=11, fontweight="bold", color=cmap(0.9), xytext=(10, -15), textcoords="offset points")

    # Legend and Colorbar
    ax.legend(loc='upper right', framealpha=0.9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Integration Time $\\tau$", shrink=0.7, pad=0.02)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"Principal Component $x_1$ ({100*var[0]:.1f}% Variance)", fontweight="bold")
    ax.set_ylabel(f"Principal Component $x_2$ ({100*var[1]:.1f}% Variance)", fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[manifold] Saved Advanced Q1 Graphic → {out_path}")
    return pca


def plot_state_manifold_3d(
    ensemble_snapshots: list,
    pca_3d: PCA = None,
    out_path: str = "results/state_manifold_3d.png",
):
    """
    Advanced 3D PCA projection with generated manifold surface mapping (Ultra-Q1 Quality).
    Shows:
      - Continuous Triangulated 3D Energy/Density topology surface.
      - Mean geodesic path following the topology gradients.
      - Elevation markers matching integration step tau.
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass
        
    T = len(ensemble_snapshots) - 1
    cmap = plt.cm.turbo

    all_rhos = [r for snap in ensemble_snapshots for r in snap]
    X = np.stack([_rho_to_vec(r) for r in all_rhos])
    if pca_3d is None:
        pca_3d = PCA(n_components=3)
        pca_3d.fit(X)

    fig = plt.figure(figsize=(13, 10))
    ax  = fig.add_subplot(111, projection="3d")
    
    pts_all = pca_3d.transform(X)
    
    # 1. Fit an implied manifold surface using RBF or Tricontour
    import matplotlib.tri as mtri
    # Create triangulation on the x-y plane
    triang = mtri.Triangulation(pts_all[:, 0], pts_all[:, 1])
    
    # Render the Manifold Curvature Surface
    surf = ax.plot_trisurf(triang, pts_all[:, 2], cmap='viridis', edgecolor='none', alpha=0.25, zorder=1)
    
    means = []
    # 2. Scatter Cloud Dynamics
    for tau, snap in enumerate(ensemble_snapshots):
        pts = pca_3d.transform(np.stack([_rho_to_vec(r) for r in snap]))
        c   = cmap(tau / T)
        
        # Scatter actual points
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=c, s=15, alpha=0.4, zorder=4, edgecolor='none')
        
        # Cast shadows on the generic z-min plane for better depth perception
        zmin = pts_all[:, 2].min() - 0.1
        ax.scatter(pts[:, 0], pts[:, 1], zmin, color=c, s=5, alpha=0.1, zorder=1)
        
        means.append(pts.mean(axis=0))

    means = np.array(means)

    # 3. Dynamic Mean Trajectory Geodesic Tube
    for t in range(T):
        ax.plot([means[t, 0], means[t+1, 0]], 
                [means[t, 1], means[t+1, 1]], 
                [means[t, 2], means[t+1, 2]], 
                color=cmap((t+0.5) / T), lw=4.5, zorder=7, solid_capstyle='round')

    # Drop lines from trajectory to the ground plane
    for t in range(0, T+1, max(1, T//5)):
        ax.plot([means[t, 0], means[t, 0]], [means[t, 1], means[t, 1]], [zmin, means[t, 2]], 
                color=cmap(t/T), linestyle=':', alpha=0.6, lw=1.5, zorder=5)

    # Mark Start and End
    ax.scatter(*means[0],  color=cmap(0), s=350, zorder=9, edgecolors="white", linewidths=1.5, label="Initial State Space $\\rho(0)$")
    ax.scatter(*means[-1], color=cmap(1.0), s=450, marker="*", zorder=9, edgecolors="white", linewidths=1.5, label="Target Attractor $\\rho(1)$")

    var = pca_3d.explained_variance_ratio_
    ax.set_xlabel(f"Vector Space PC₁ ({100*var[0]:.1f}%)", fontweight='bold', labelpad=10)
    ax.set_ylabel(f"Vector Space PC₂ ({100*var[1]:.1f}%)", fontweight='bold', labelpad=10)
    ax.set_zlabel(f"Topological Elevation PC₃ ({100*var[2]:.1f}%)", fontweight='bold', labelpad=10)
    ax.set_title("Information Flow Topography: 3D Quantum State Manifold Embedding", fontsize=15, fontweight="bold", y=0.98)
    ax.legend(loc='upper left', framealpha=0.9)

    # Optimize Viewing Angle for paper display
    ax.view_init(elev=25, azim=45)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Normalized Flow Step $t/T$", shrink=0.5, pad=0.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[manifold] Saved Advanced Q1 3D Graphic → {out_path}")
