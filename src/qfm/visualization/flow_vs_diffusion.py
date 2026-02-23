"""
flow_vs_diffusion.py — Hero figure directly inspired by the Flow vs Diffusion diagram.

Left panel  (a) — Quantum Flow (QFM):
    Smooth deterministic trajectories ρ_0 → ρ_T in 2D PCA state space.
    KDE "blobs" show the ensemble distribution at τ=0, T/2, T.

Right panel (b) — Quantum Diffusion (Lindblad + quantum jump unraveling):
    Stochastic, jagged trajectories from quantum trajectory / Monte-Carlo
    wave function (MCWF) method simulating the same Lindblad dynamics.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


# Custom colormap (red → blue, like the image)
_FLOW_CMAP = LinearSegmentedColormap.from_list(
    "flow", ["#C62828", "#EF9A9A", "#90CAF9", "#1565C0"]
)


# Helpers

def _rho_to_vec(rho: torch.Tensor) -> np.ndarray:
    """Flatten real and imaginary parts of ρ into a real vector."""
    r = rho.detach().cpu()
    return np.concatenate([r.real.numpy().ravel(), r.imag.numpy().ravel()])


def _fit_pca(all_rhos: list) -> PCA:
    """Fit PCA on all density matrices collected across all steps."""
    X = np.stack([_rho_to_vec(r) for r in all_rhos])
    pca = PCA(n_components=2)
    pca.fit(X)
    return pca


def _project(pca: PCA, rhos: list) -> np.ndarray:
    X = np.stack([_rho_to_vec(r) for r in rhos])
    return pca.transform(X)


def _kde_contour(ax, pts, color, levels=5, alpha_fill=0.18, alpha_line=0.6, zorder=1):
    """Draw a Gaussian KDE contour blob around a point cloud."""
    if len(pts) < 4:
        return
    try:
        kde  = gaussian_kde(pts.T, bw_method=0.4)
        xmin, xmax = pts[:, 0].min() - 0.2, pts[:, 0].max() + 0.2
        ymin, ymax = pts[:, 1].min() - 0.2, pts[:, 1].max() + 0.2
        xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
        Z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=levels, alpha=alpha_fill,
                    colors=[color] * (levels + 1), zorder=zorder)
        ax.contour(xx, yy, Z, levels=levels, alpha=alpha_line,
                   colors=[color], linewidths=0.8, zorder=zorder + 1)
    except Exception:
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=8, alpha=0.4)


# Monte-Carlo Wave Function (quantum jump unraveling of Lindblad)

def _mcwf_trajectory(rho0_pure: np.ndarray, H: np.ndarray,
                     jump_ops: list, gammas: list,
                     n_steps: int, dt: float = 0.05) -> list:
    """
    Single quantum trajectory via MCWF (Dalibard, Castin, Mølmer 1992).
    Returns a list of density matrices (pure) at each step.
    """
    psi = rho0_pure.copy().astype(complex)
    psi /= np.linalg.norm(psi)
    rhos = [np.outer(psi, psi.conj())]

    # Effective non-Hermitian Hamiltonian
    H_eff = H.copy().astype(complex)
    for L, g in zip(jump_ops, gammas):
        H_eff -= 0.5j * g * (L.conj().T @ L)

    for _ in range(n_steps - 1):
        # Deterministic evolution under H_eff
        psi_tilde = psi - 1j * dt * (H_eff @ psi)
        dp = dt * sum(g * np.real(psi.conj() @ (L.conj().T @ L) @ psi)
                      for L, g in zip(jump_ops, gammas))
        eps = np.random.rand()
        if eps < dp and dp > 0:
            # Quantum jump: choose which operator fires
            probs = [g * np.real(psi.conj() @ (L.conj().T @ L) @ psi)
                     for L, g in zip(jump_ops, gammas)]
            idx = np.random.choice(len(probs), p=np.array(probs) / sum(probs))
            psi_new = jump_ops[idx] @ psi
            norm = np.linalg.norm(psi_new)
            psi = psi_new / norm if norm > 1e-12 else psi_tilde / np.linalg.norm(psi_tilde)
        else:
            norm = np.linalg.norm(psi_tilde)
            psi = psi_tilde / norm if norm > 1e-12 else psi

        rhos.append(np.outer(psi, psi.conj()))
    return rhos


# Helpers for quantitative analysis

def _compute_purity(rho):
    """Tr(ρ²) — 1 for pure states, 1/d for maximally mixed."""
    if isinstance(rho, torch.Tensor):
        rho_np = rho.detach().cpu().numpy()
    else:
        rho_np = np.array(rho)
    return np.real(np.trace(rho_np @ rho_np))


def _confidence_ellipse(ax, pts, color, n_std=2.0, **kwargs):
    """Draw 2-sigma confidence ellipse around a point cloud."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    if len(pts) < 3:
        return
    mean = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.12,
                      linestyle="--", linewidth=1.5, **kwargs)
    ax.add_patch(ellipse)


# Main plot function

def plot_flow_vs_diffusion(
    qfm_trajectories: list,       # list of M trajectories; each = list of T+1 density matrices
    H_np: np.ndarray,             # Hamiltonian (numpy)
    jump_ops_np: list,            # Kraus/jump operators (numpy)
    gammas: list,
    pca: PCA = None,
    out_path: str = "results/flow_vs_diffusion.png",
    n_diff_trajectories: int = 12,
):
    """
    Advanced 3-panel hero figure (Ultra-Q1 Quality):
    (a)  Quantum Flow [QFM]: Smooth deterministic trajectories with KDE blobs.
    (b)  Quantum Diffusion [Lindblad]: Stochastic MCWF trajectories with
         confidence ellipses.
    (c)  Quantitative Comparison: Transport Cost & Purity evolution.
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass

    from matplotlib.gridspec import GridSpec
    from scipy.ndimage import uniform_filter1d

    # Collect all rhos to fit PCA
    all_rhos = [r for traj in qfm_trajectories for r in traj]
    if pca is None:
        pca = _fit_pca(all_rhos)

    T = len(qfm_trajectories[0]) - 1
    tau_colors = [_FLOW_CMAP(t / T) for t in range(T + 1)]

    # ---- Figure layout: 2 rows ----
    # Row 1: (a) Flow + (b) Diffusion   |  Row 2: (c) Transport Analysis (full width)
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 0.7],
                  hspace=0.35, wspace=0.30)

    blob_taus = [0, T // 2, T]
    blob_colors = ["#B71C1C", "#F9A825", "#1565C0"]

    # ---- Generate Diffusion trajectories once (shared) ----
    d = H_np.shape[0]
    rho0_pure = np.eye(d, dtype=complex)[0]
    diff_trajectories = []
    for _ in range(n_diff_trajectories):
        rhos_np = _mcwf_trajectory(rho0_pure, H_np, jump_ops_np,
                                    gammas, n_steps=T + 1, dt=0.08)
        diff_trajectories.append([torch.tensor(r, dtype=torch.complex128) for r in rhos_np])

    # ================ Panel (a): Quantum Flow (top-left) ================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("(a)  Quantum Flow  [QFM]", fontsize=16, fontweight="bold",
                  pad=14, color="#2D3436")
    ax1.set_xlabel("State space PC$_1$", fontsize=12, fontweight="bold")
    ax1.set_ylabel("State space PC$_2$", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.15, ls="--")

    # KDE blobs at key time-steps
    for bt, bc in zip(blob_taus, blob_colors):
        snapshot = [traj[bt] for traj in qfm_trajectories]
        pts = _project(pca, snapshot)
        _kde_contour(ax1, pts, bc, levels=8, alpha_fill=0.18, alpha_line=0.5)
        _confidence_ellipse(ax1, pts, bc, n_std=1.5)
        mu = pts.mean(axis=0)
        ax1.scatter(*mu, color=bc, s=100, zorder=10,
                   edgecolors="white", linewidths=1.5)
        label = f"$\\rho_{{{bt}}}$" if bt == 0 else \
                (f"$\\rho_{{T/2}}$" if bt == T // 2 else f"$\\rho_T$")
        ax1.annotate(label, mu, fontsize=13, fontweight="bold", color=bc,
                    xytext=(8, 8), textcoords="offset points")

    # Smooth trajectories
    for traj in qfm_trajectories:
        pts = _project(pca, traj)
        pts_sm = np.stack([
            uniform_filter1d(pts[:, 0], size=3),
            uniform_filter1d(pts[:, 1], size=3),
        ], axis=1)
        for t in range(len(pts_sm) - 1):
            ax1.plot(pts_sm[t:t+2, 0], pts_sm[t:t+2, 1],
                    color=tau_colors[t], lw=1.5, alpha=0.6, zorder=5)
        # Start & End markers
        ax1.scatter(pts_sm[0, 0], pts_sm[0, 1], marker="o",
                   color=tau_colors[0], s=20, zorder=6, alpha=0.8)
        ax1.scatter(pts_sm[-1, 0], pts_sm[-1, 1], marker="*",
                   color=tau_colors[-1], s=40, zorder=6, alpha=0.8)

    # ================ Panel (b): Quantum Diffusion (top-right) ================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("(b)  Quantum Diffusion  [Lindblad MCWF]", fontsize=16,
                  fontweight="bold", pad=14, color="#2D3436")
    ax2.set_xlabel("State space PC$_1$", fontsize=12, fontweight="bold")
    ax2.set_ylabel("State space PC$_2$", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.15, ls="--")

    # KDE blobs (same positions for comparison)
    for bt, bc in zip(blob_taus, blob_colors):
        snapshot = [traj[bt] for traj in qfm_trajectories]
        pts = _project(pca, snapshot)
        _kde_contour(ax2, pts, bc, levels=6, alpha_fill=0.15, alpha_line=0.4)
        mu = pts.mean(axis=0)
        ax2.scatter(*mu, color=bc, s=100, zorder=10,
                   edgecolors="white", linewidths=1.5)
        label = f"$\\rho_{{{bt}}}$" if bt == 0 else \
                (f"$\\rho_{{T/2}}$" if bt == T // 2 else f"$\\rho_T$")
        ax2.annotate(label, mu, fontsize=13, fontweight="bold", color=bc,
                    xytext=(8, 8), textcoords="offset points")

    # Stochastic trajectories with confidence ellipses
    diff_end_pts = []
    for dtraj in diff_trajectories:
        pts = _project(pca, dtraj)
        for t in range(len(pts) - 1):
            ax2.plot(pts[t:t+2, 0], pts[t:t+2, 1],
                    color=tau_colors[t], lw=0.9, alpha=0.4, zorder=5,
                    solid_joinstyle="miter")
        ax2.scatter(pts[0, 0], pts[0, 1], marker="o",
                   color=tau_colors[0], s=20, zorder=6, alpha=0.7)
        ax2.scatter(pts[-1, 0], pts[-1, 1], marker="x",
                   color=tau_colors[-1], s=25, zorder=6, alpha=0.7)
        diff_end_pts.append(pts[-1])

    # Confidence ellipse around diffusion endpoints
    if len(diff_end_pts) >= 3:
        diff_end_pts_arr = np.array(diff_end_pts)
        _confidence_ellipse(ax2, diff_end_pts_arr, "#E53935", n_std=2.0)

    # ---- Shared colorbar between (a) and (b) ----
    sm = plt.cm.ScalarMappable(cmap=_FLOW_CMAP, norm=plt.Normalize(0, T))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.4, pad=0.06,
                        location="bottom", aspect=35)
    cbar.set_label("Flow Step $\\tau$", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # ================ Panel (c): Transport Analysis (bottom, full width) ================
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title("(c)  Quantitative Transport Analysis", fontsize=16,
                  fontweight="bold", pad=14, color="#2D3436")

    # Compute purity at each τ for QFM and Diffusion
    qfm_purity_per_tau = []
    diff_purity_per_tau = []
    qfm_spread_per_tau = []
    diff_spread_per_tau = []

    for t in range(T + 1):
        # QFM purity
        qfm_purities_t = [_compute_purity(traj[t]) for traj in qfm_trajectories]
        qfm_purity_per_tau.append(np.mean(qfm_purities_t))

        # Diffusion purity
        diff_purities_t = [_compute_purity(dtraj[min(t, len(dtraj)-1)]) for dtraj in diff_trajectories]
        diff_purity_per_tau.append(np.mean(diff_purities_t))

        # Spread (variance in PCA space) as transport cost proxy
        qfm_pts = _project(pca, [traj[t] for traj in qfm_trajectories])
        qfm_spread_per_tau.append(np.mean(np.var(qfm_pts, axis=0)))

        diff_pts_t = _project(pca, [dtraj[min(t, len(dtraj)-1)] for dtraj in diff_trajectories])
        diff_spread_per_tau.append(np.mean(np.var(diff_pts_t, axis=0)))

    taus = np.arange(T + 1)

    # Left y-axis: Purity
    color_pur = "#1565C0"
    ax3.plot(taus, qfm_purity_per_tau, "o-", color=color_pur, lw=2.5, ms=7,
             markerfacecolor="white", markeredgewidth=2, label="QFM Purity $\\gamma$")
    ax3.plot(taus, diff_purity_per_tau, "s--", color="#E53935", lw=2.0, ms=6,
             markerfacecolor="white", markeredgewidth=1.5, label="Diffusion Purity $\\gamma$")
    ax3.set_xlabel("Flow Step $\\tau$", fontsize=13, fontweight="bold")
    ax3.set_ylabel("State Purity $\\gamma = \\mathrm{tr}(\\rho^2)$",
                   fontsize=12, fontweight="bold", color=color_pur)
    ax3.tick_params(axis="y", labelcolor=color_pur)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.2, ls="--")

    # Right y-axis: Spread (Transport Cost Proxy)
    ax3b = ax3.twinx()
    color_spr = "#FF6F00"
    ax3b.fill_between(taus, qfm_spread_per_tau, alpha=0.15, color=color_spr, label="QFM Spread")
    ax3b.fill_between(taus, diff_spread_per_tau, alpha=0.10, color="#9E9E9E", label="Diffusion Spread")
    ax3b.plot(taus, qfm_spread_per_tau, "-", color=color_spr, lw=2.0)
    ax3b.plot(taus, diff_spread_per_tau, "--", color="#757575", lw=1.5)
    ax3b.set_ylabel("Ensemble Spread (Transport Cost Proxy)",
                    fontsize=12, fontweight="bold", color=color_spr)
    ax3b.tick_params(axis="y", labelcolor=color_spr)

    # Merged legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
              fontsize=11, framealpha=0.9, ncol=2)

    fig.suptitle("Quantum Flow Matching vs Lindblad Diffusion — State Space Transport Analysis",
                 fontsize=20, fontweight="bold", color="#2D3436", y=0.98)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[flow_vs_diffusion] Saved Advanced Q1 Graphic → {out_path}")
    return pca
