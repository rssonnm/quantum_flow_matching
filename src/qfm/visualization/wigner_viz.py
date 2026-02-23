"""
wigner_viz.py — Wigner and Husimi quasi-probability functions for 1-qubit states.

For a single qubit, the Wigner function on the Bloch sphere (Stratonovich–Weyl):
    W_ρ(θ, φ) = Σ_{j,m} ρ_{jm} Y_{jm}(θ, φ)  (spin-j Wigner function)
For j=1/2 this simplifies to:
    W_ρ(n̂) = (1/4π) Tr[ρ (I + √3 n̂·σ)]
where n̂ = (sin θ cos φ, sin θ sin φ, cos θ) is the Bloch vector direction.

Negative regions W < 0 are a signature of genuine quantum non-classicality.

Husimi Q-function: Q_ρ(θ, φ) = ⟨n̂|ρ|n̂⟩ ≥ 0 always (no negativity).
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa
from matplotlib.animation import FuncAnimation, PillowWriter


# Grid setup

def _sphere_grid(n_theta=60, n_phi=100):
    theta = np.linspace(0, np.pi, n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    return TH, PH, X, Y, Z


# Wigner function for a single qubit

def wigner_1qubit(rho: torch.Tensor, n_theta: int = 60, n_phi: int = 100) -> np.ndarray:
    """
    W_ρ(θ, φ) = (1/4π) Tr[ρ (I + √3 n̂·σ)]
              = (1/4π) (1 + √3 ⟨σ_x⟩ sin θ cos φ
                           + √3 ⟨σ_y⟩ sin θ sin φ
                           + √3 ⟨σ_z⟩ cos θ)
    Shape: (n_theta, n_phi).
    """
    rho_np = rho.detach().cpu().numpy()
    # Bloch vector
    sx = 2 * rho_np[0, 1].real   # 2 Re(ρ_{01})
    sy = 2 * rho_np[0, 1].imag   # 2 Im(ρ_{01})  — watch sign convention
    sy = -2 * rho_np[1, 0].imag  # = Tr(ρ σ_y)
    sz = rho_np[0, 0].real - rho_np[1, 1].real

    theta = np.linspace(0, np.pi, n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    nx = np.sin(TH) * np.cos(PH)
    ny = np.sin(TH) * np.sin(PH)
    nz = np.cos(TH)
    W = (1 / (4 * np.pi)) * (1 + np.sqrt(3) * (sx * nx + sy * ny + sz * nz))
    return W


def husimi_1qubit(rho: torch.Tensor, n_theta: int = 60, n_phi: int = 100) -> np.ndarray:
    """
    Q_ρ(θ, φ) = ⟨n̂(θ,φ)|ρ|n̂(θ,φ)⟩
    where |n̂⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.
    Q ≥ 0 always (no negativity).
    """
    rho_np = rho.detach().cpu().numpy()
    theta = np.linspace(0, np.pi, n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    # Coherent state components
    c0 = np.cos(TH / 2)
    c1 = np.exp(1j * PH) * np.sin(TH / 2)
    Q = (rho_np[0, 0] * c0 * c0.conj()
         + rho_np[0, 1] * c0 * c1.conj()
         + rho_np[1, 0] * c1 * c0.conj()
         + rho_np[1, 1] * c1 * c1.conj()).real
    return Q


# Plotting helpers

def _plot_wigner_panel(ax3d, W, X, Y, Z, vmax=None):
    """Draw W on the Bloch sphere surface, coloring by W value."""
    vmax = vmax or max(abs(W.max()), abs(W.min())) + 1e-9
    norm  = plt.Normalize(vmin=-vmax, vmax=vmax)
    colors = plt.cm.RdBu_r(norm(W))
    ax3d.plot_surface(X, Y, Z, facecolors=colors,
                      rstride=1, cstride=1, linewidth=0,
                      antialiased=False, alpha=0.95)
    ax3d.set_axis_off()
    ax3d.set_box_aspect([1, 1, 1])


def _draw_axes(ax, labels=True):
    """Draw Bloch sphere axes."""
    ax.quiver(0, 0, 0, 0, 0, 1.4, arrow_length_ratio=0.08,
              color="gray", linewidth=0.8, alpha=0.7)
    ax.quiver(0, 0, 0, 1.4, 0, 0, arrow_length_ratio=0.08,
              color="gray", linewidth=0.8, alpha=0.7)
    if labels:
        ax.text(0, 0, 1.55, r"$|0\rangle$", fontsize=9, ha="center", color="#455A64")
        ax.text(1.55, 0, 0,  r"$|+\rangle$", fontsize=9, ha="center", color="#455A64")


# Main figure

def plot_wigner_sequence(
    rhos_per_tau: list,    # list of (2,2) density matrices
    out_path: str = "results/wigner_sequence.png",
    n_panels: int = 5,
):
    """
    Advanced Q1-level array of Wigner W (top row) and Husimi Q (bottom row) 
    quasi-probability distributions.
    Key enhancements:
     - High-fidelity 3D surface shading.
     - 2D contour projections on the z=-1 plane to highlight negativity (W < 0).
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass
        
    TH, PH, X, Y, Z = _sphere_grid(100, 150)  # Hi-Res grid
    T = len(rhos_per_tau)
    idxs = [int(i * (T - 1) / (n_panels - 1)) for i in range(n_panels)]

    fig = plt.figure(figsize=(4.5 * n_panels, 9.5))
    vmax_w = max(abs(wigner_1qubit(rhos_per_tau[i], 100, 150).max()) for i in idxs) * 1.05
    vmax_q = 1.0 / (4 * np.pi)

    from matplotlib.colors import TwoSlopeNorm
    # Wigner norm: centered exactly at zero to emphasize non-classicality
    norm_w = TwoSlopeNorm(vmin=-vmax_w, vcenter=0.0, vmax=vmax_w)

    for col, tau_idx in enumerate(idxs):
        rho = rhos_per_tau[tau_idx]
        W = wigner_1qubit(rho, 100, 150)
        Q = husimi_1qubit(rho, 100, 150)

        # --- Top Row: Wigner W(θ, φ) ---
        ax = fig.add_subplot(2, n_panels, col + 1, projection="3d")
        
        # 3D Surface
        colors = plt.cm.RdBu_r(norm_w(W))
        ax.plot_surface(X, Y, Z, facecolors=colors,
                        rstride=2, cstride=2, linewidth=0.1,
                        antialiased=True, alpha=0.9, zorder=3)
                        
        # Z-Plane Contour Projection (-1.5)
        offset = -1.6
        ax.contourf(X, Y, W, zdir='z', offset=offset, cmap="RdBu_r", norm=norm_w, alpha=0.7, zorder=1)
        ax.contour(X, Y, W, zdir='z', offset=offset, colors='black', linewidths=0.5, alpha=0.5, zorder=2)
        
        _draw_axes(ax, labels=(col == 0))
        ax.set_zlim(offset - 0.1, 1.2)
        ax.view_init(elev=25, azim=45)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1.4])
        
        # Title per panel
        ax.set_title(f"Step $\\tau = {tau_idx}$", fontsize=13, fontweight="bold", pad=2)
        
        if col == 0:
            ax.text2D(-0.15, 0.5, "Wigner Distribution $W(\\theta,\\varphi)$",
                      transform=ax.transAxes, fontsize=14, fontweight="bold",
                      ha="center", va="center", rotation=90, color="#C62828")

        # --- Bottom Row: Husimi Q(θ, φ) ---
        ax2 = fig.add_subplot(2, n_panels, n_panels + col + 1, projection="3d")
        norm_q = plt.Normalize(0, vmax_q)
        colors_q = plt.cm.magma(norm_q(Q))
        
        # 3D Surface
        ax2.plot_surface(X, Y, Z, facecolors=colors_q,
                         rstride=2, cstride=2, linewidth=0.1,
                         antialiased=True, alpha=0.9, zorder=3)
                         
        # Z-Plane Contour Projection
        ax2.contourf(X, Y, Q, zdir='z', offset=offset, cmap="magma", norm=norm_q, levels=12, alpha=0.7, zorder=1)
        ax2.contour(X, Y, Q, zdir='z', offset=offset, colors='black', linewidths=0.5, alpha=0.5, zorder=2)
        
        _draw_axes(ax2, labels=(col == 0))
        ax2.set_zlim(offset - 0.1, 1.2)
        ax2.view_init(elev=25, azim=45)
        ax2.set_axis_off()
        ax2.set_box_aspect([1, 1, 1.4])
        
        if col == 0:
            ax2.text2D(-0.15, 0.5, "Husimi Distribution $Q(\\theta,\\varphi)$",
                       transform=ax2.transAxes, fontsize=14, fontweight="bold",
                       ha="center", va="center", rotation=90, color="#283593")

    # Global Colorbars with careful spacing
    # Wigner
    sm_w = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_w)
    sm_w.set_array([])
    cbar_w = fig.colorbar(sm_w, ax=fig.axes[:n_panels], location="right",
                          shrink=0.55, pad=0.02, aspect=15)
    cbar_w.set_label("Wigner Probability Density\n(Blue = Non-Classical $W<0$)", fontweight="bold")

    # Husimi
    sm_q = plt.cm.ScalarMappable(cmap="magma", norm=norm_q)
    sm_q.set_array([])
    cbar_q = fig.colorbar(sm_q, ax=fig.axes[n_panels:], location="right",
                          shrink=0.55, pad=0.02, aspect=15)
    cbar_q.set_label("Husimi Magnitude $Q$", fontweight="bold")

    fig.suptitle("Quantum Phase Space Quasi-Probability Distributions: Spherical Condensation Flow",
                 fontsize=18, fontweight="bold", y=1.03)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[wigner_viz] Saved Advanced Q1 Graphic → {out_path}")


def animate_wigner(
    rhos_per_tau: list,
    out_path: str = "results/wigner_animation.gif",
):
    """Create an animated GIF of the Wigner function evolving."""
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("[wigner] Cannot create animation")
        return

    TH, PH, X, Y, Z = _sphere_grid(40, 70)
    vmax_w = max(abs(wigner_1qubit(r).max()) for r in rhos_per_tau) * 1.05

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")

    def _frame(t):
        ax.cla()
        W = wigner_1qubit(rhos_per_tau[t])
        _plot_wigner_panel(ax, W, X, Y, Z, vmax=vmax_w)
        _draw_axes(ax, labels=True)
        ax.set_title(f"Wigner  W(θ,φ)   τ = {t}", fontsize=12, fontweight="bold")

    anim = FuncAnimation(fig, _frame, frames=len(rhos_per_tau), interval=200)
    anim.save(out_path, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"[wigner_viz] Animated GIF → {out_path}")
