import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.qfm.ansatz import EHA_Circuit
from src.qfm.utils import state_vector_to_density_matrix
from src.qfm.metrics import uhlmann_fidelity

_BG = "C2"
_CFM_CMAP = LinearSegmentedColormap.from_list("cfm_t", ["C4", "C0", "C1"])
_QFM_CMAP = LinearSegmentedColormap.from_list("qfm_t", ["C7", "C2", "C3"])
_DENSITY_CFM = "cividis"
_DENSITY_QFM = "magma"

class _VectorFieldMLP(nn.Module):
    def __init__(self, dim=2, hidden=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))

def _sample_two_moons(n, noise=0.06):
    n1, n2 = n // 2, n - n // 2
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n2)
    x = np.vstack([
        np.column_stack([np.cos(t1), np.sin(t1)]),
        np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5])
    ]) + noise * np.random.randn(n, 2)
    return torch.from_numpy(x).float()

def _train_cfm(target_pts, n_iter=2500, batch=512):
    model = _VectorFieldMLP()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for step in range(n_iter):
        x0 = torch.randn(batch, 2)
        x1 = target_pts[torch.randint(0, len(target_pts), (batch,))]
        t = torch.rand(batch)
        x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
        u_t = x1 - x0
        v = model(x_t, t)
        loss = ((v - u_t) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 500 == 0:
            print(f"  CFM step {step}/{n_iter}  loss={loss.item():.4f}")
    return model, losses

@torch.no_grad()
def _sample_cfm(model, n=800, n_steps=50):
    x = torch.randn(n, 2)
    dt = 1.0 / n_steps
    traj = [x.clone()]
    for i in range(n_steps):
        t = torch.ones(n) * (i / n_steps)
        x = x + dt * model(x, t)
        traj.append(x.clone())
    return traj

def _bloch(psi):
    
    psi = psi.detach().cpu().to(torch.complex128)
    a, b = psi[0], psi[1]
    bx = float(2 * (a * b.conj()).real)
    by = float(2 * (a * b.conj()).imag)
    bz = float(abs(a)**2 - abs(b)**2)
    return bx, by, bz

def _slerp(psi0: torch.Tensor, psi1: torch.Tensor, t: float) -> torch.Tensor:
    
    overlap = torch.vdot(psi0, psi1)
    phase = overlap / (torch.abs(overlap) + 1e-12)
    psi1_rot = psi1 * phase.conj()
    
    omega = torch.acos(torch.clamp(torch.abs(overlap), -1.0, 1.0))
    if omega < 1e-6:
        return psi0
        
    sin_omega = torch.sin(omega)
    c0 = torch.sin((1 - t) * omega) / sin_omega
    c1 = torch.sin(t * omega) / sin_omega
    return c0 * psi0 + c1 * psi1_rot

def _train_qfm(n_steps=20, n_ensemble=80, n_layers=5, lr=0.08, epochs=40):
    
    target = torch.tensor([1.0, 1.0], dtype=torch.complex128) / np.sqrt(2)
    init_states = []
    for _ in range(n_ensemble):
        theta = 0.3 * np.random.randn()
        phi = 2 * np.pi * np.random.rand()
        psi = torch.tensor([np.cos(theta/2),
                            np.sin(theta/2) * np.exp(1j * phi)],
                           dtype=torch.complex128)
        psi = psi / psi.norm()
        init_states.append(psi)

    snapshots = [torch.stack(init_states)]
    current = init_states
    losses = []

    for tau in range(1, n_steps + 1):
        frac = tau / n_steps
        target_rhos = []
        for psi0 in init_states:
            psi_t = _slerp(psi0, target, frac)
            target_rhos.append(state_vector_to_density_matrix(psi_t))
        target_rhos = torch.stack(target_rhos)

        model = EHA_Circuit(n_data=1, n_layers=n_layers, n_ancilla=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
        inputs = torch.stack(current)

        for ep in range(epochs):
            optimizer.zero_grad()
            out = model(inputs)
            rhos = torch.stack([state_vector_to_density_matrix(p) for p in out])
            fids = torch.stack([uhlmann_fidelity(r, tr) for r, tr in zip(rhos, target_rhos)])
            loss = 1.0 - fids.mean()
            loss.backward(); optimizer.step(); scheduler.step()

        losses.append(loss.item())
        with torch.no_grad():
            out = model(inputs)
            current = [s.detach() for s in out]
        snapshots.append(torch.stack(current))
        print(f"step {tau} loss={loss.item():.5f}")

    return snapshots, losses

def _draw_cfm_trajectories(ax, traj, n_show=120):
    
    T = len(traj) - 1
    n = min(n_show, traj[0].shape[0])

    for m in range(n):
        xs = [traj[t][m, 0].item() for t in range(T + 1)]
        ys = [traj[t][m, 1].item() for t in range(T + 1)]
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = [_CFM_CMAP(t / T) for t in range(T)]
        lc = LineCollection(segments, colors=colors, linewidth=0.6)
        ax.add_collection(lc)
    p0 = traj[0][:n].numpy()
    ax.scatter(p0[:, 0], p0[:, 1], s=6, color="C4",
               zorder=5, label="$t=0$ (noise)")
    p1 = traj[-1][:n].numpy()
    ax.scatter(p1[:, 0], p1[:, 1], s=8, color="C1",
               zorder=6, edgecolors="white", linewidths=0.2,
               label="$t=1$ (data)")

    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left",
              facecolor="C9", labelcolor="white")

def _draw_qfm_bloch(ax3d, snapshots, n_show=50):
    
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 30)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax3d.plot_surface(sx, sy, sz, color="C5", linewidth=0)
    ax3d.plot_wireframe(sx, sy, sz, color="C6", linewidth=0.15)

    T = len(snapshots) - 1
    n = min(n_show, snapshots[0].shape[0])

    for m in range(n):
        bxs, bys, bzs = [], [], []
        for t in range(T + 1):
            bx, by, bz = _bloch(snapshots[t][m])
            bxs.append(bx); bys.append(by); bzs.append(bz)

        points = np.array([bxs, bys, bzs]).T
        for t in range(T):
            ax3d.plot(points[t:t+2, 0], points[t:t+2, 1], points[t:t+2, 2],
                      color=_QFM_CMAP(t / T), lw=0.7)
    for m in range(n):
        b0 = _bloch(snapshots[0][m])
        ax3d.scatter(*b0, s=8, color="C7", zorder=5)
        b1 = _bloch(snapshots[-1][m])
        ax3d.scatter(*b1, s=12, color="C3", zorder=6,
                     edgecolors="white", linewidths=0.3)
    ax3d.quiver(0, 0, -1.3, 0, 0, 2.6, arrow_length_ratio=0.03,
                color="gray", lw=0.6)
    ax3d.text(0, 0, 1.5, "$|0\\rangle$", ha="center", color="C4")
    ax3d.text(0, 0, -1.6, "$|1\\rangle$", ha="center", color="C4")
    ax3d.text(1.4, 0, 0, "$X$", color="C4")
    ax3d.text(0, 1.4, 0, "$Y$", color="C4")

    ax3d.set_axis_off()
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.view_init(elev=20, azim=35)

def _draw_density_strip(ax, traj_or_snaps, is_quantum, n_cols=6):
    
    T = len(traj_or_snaps) - 1
    indices = [int(f * T) for f in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    cmap = _DENSITY_QFM if is_quantum else _DENSITY_CFM

    for col, idx in enumerate(indices):
        sub_ax = ax.inset_axes([col / n_cols, 0, 1.0 / n_cols - 0.01, 1.0])
        sub_ax.set_facecolor(_BG)

        if is_quantum:
            states = traj_or_snaps[idx]
            pts = np.array([_bloch(s)[:2] for s in states])
        else:
            pts = traj_or_snaps[idx].numpy()

        if len(pts) > 5:
            try:
                kde = gaussian_kde(pts.T, bw_method=0.3)
                if is_quantum:
                    ext = [-1.2, 1.2, -1.2, 1.2]
                else:
                    ext = [pts[:, 0].min()-0.5, pts[:, 0].max()+0.5,
                           pts[:, 1].min()-0.5, pts[:, 1].max()+0.5]
                xx, yy = np.mgrid[ext[0]:ext[1]:60j, ext[2]:ext[3]:60j]
                Z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                sub_ax.imshow(np.rot90(Z), cmap=cmap,
                              extent=ext, aspect="auto")
            except Exception:
                pass

        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        for sp in sub_ax.spines.values():
            sp.set_visible(False)

        pct = int(idx / T * 100) if T > 0 else 0
        sub_ax.set_title(f"{pct}%", color="white", pad=2)

    ax.axis("off")

def _draw_loss(ax, cfm_losses, qfm_losses, academic=False):
    
    def smooth(arr, w=30):
        if len(arr) < w: return arr
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    def norm(arr):
        mn, mx = min(arr), max(arr)
        return [(v - mn) / (mx - mn + 1e-12) for v in arr]

    cfm_s = smooth(cfm_losses)
    x_cfm = np.linspace(0, 100, len(cfm_s))
    x_qfm = np.linspace(0, 100, len(qfm_losses))
    cfm_color = "C3" if academic else "C0"
    qfm_color = "C0" if academic else "C2"
    text_color = "black" if academic else "white"
    legend_bg = "white" if academic else "C9"
    ax.fill_between(x_cfm, norm(cfm_s), color=cfm_color)
    ax.plot(x_cfm, norm(cfm_s), color=cfm_color, lw=2.2, label="CFM loss")
    qfm_max = max(qfm_losses)
    norm_factor = qfm_max if qfm_max > 0.1 else 1.0
    
    qfm_norm_y = [y / (norm_factor + 1e-12) for y in qfm_losses]
    ax.fill_between(x_qfm, qfm_norm_y, color=qfm_color)
    ax.plot(x_qfm, qfm_norm_y, color=qfm_color, lw=2.2, ls="--", label="QFM infidelity")

    ax.set_xlabel("Training Progress (%)", color=text_color)
    ax.set_ylabel("Normalized Loss", color=text_color)
    ax.legend(fontsize=10, facecolor=legend_bg, labelcolor=text_color)
    ax.set_ylim(-0.05, 1.05)

def generate_figure(args):
    
    print("Training Classical FM (Two Moons)...")
    target = _sample_two_moons(3000)
    cfm_model, cfm_losses = _train_cfm(target, n_iter=args.cfm_steps)
    cfm_traj = _sample_cfm(cfm_model, n=800, n_steps=50)

    print("Training Quantum FM (|ψ₀⟩ → |+⟩)...")
    qfm_snaps, qfm_losses = _train_qfm(
        n_steps=args.flow_steps, n_ensemble=80,
        n_layers=args.qfm_layers, epochs=args.qfm_epochs
    )
    qfm_losses = np.array(qfm_losses)

    print("Rendering transport comparison figure...")
    fig = plt.figure(figsize=(20, 16), facecolor=_BG)
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[3, 1, 1.2],
                           hspace=0.25, wspace=0.15)
    ax_cfm = fig.add_subplot(gs[0, 0])
    ax_cfm.set_facecolor(_BG)
    _draw_cfm_trajectories(ax_cfm, cfm_traj, n_show=150)
    ax_cfm.set_title("Classical Flow Matching\n"
                     "$\\mathcal{N}(0, I) \\;\\to\\;$ Two Moons", color="C0", pad=12)
    ax_cfm.tick_params(colors="gray", labelsize=8)
    for sp in ax_cfm.spines.values():
        sp.set_color("#333")

    ax_qfm = fig.add_subplot(gs[0, 1], projection="3d",
                              facecolor=_BG)
    ax_qfm.set_facecolor(_BG)
    _draw_qfm_bloch(ax_qfm, qfm_snaps, n_show=60)
    ax_qfm.set_title("Quantum Flow Matching\n"
                     "$|\\psi_0\\rangle \\;\\to\\;$ $|+\\rangle$", color="C2", pad=12)
    ax_ds_cfm = fig.add_subplot(gs[1, 0])
    ax_ds_cfm.set_facecolor(_BG)
    _draw_density_strip(ax_ds_cfm, cfm_traj, is_quantum=False)
    ax_ds_cfm.set_title("CFM Density Evolution", color="C0", pad=4)

    ax_ds_qfm = fig.add_subplot(gs[1, 1])
    ax_ds_qfm.set_facecolor(_BG)
    _draw_density_strip(ax_ds_qfm, qfm_snaps, is_quantum=True)
    ax_ds_qfm.set_title("QFM Density Evolution  (Bloch projection)", color="C2", pad=4)
    ax_loss = fig.add_subplot(gs[2, :])
    ax_loss.set_facecolor("C8")
    _draw_loss(ax_loss, cfm_losses, qfm_losses)
    ax_loss.set_title("Training Convergence Comparison", color="white", pad=8)
    ax_loss.tick_params(colors="gray", labelsize=9)
    ax_loss.grid(True, color="gray", ls="--")
    for sp in ax_loss.spines.values():
        sp.set_color("#333")
    fig.suptitle("Noise → Data Transport:  Classical FM  vs  Quantum FM", color="white", y=0.98,
                 fontfamily="serif")
    fig.text(0.25, 0.63, "← Gaussian noise flows toward moon clusters →", color="C1", ha="center", style="italic")
    fig.text(0.75, 0.63, "← Random |ψ⟩ converge to target |+⟩ on Bloch sphere →", color="C1", ha="center", style="italic")
    out_dir = os.path.dirname(args.out) or "results"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=_BG,
                pad_inches=0.4)
    plt.close(fig)
    print(f"Transport comparison saved to {args.out}")
    panels_dir = os.path.join(out_dir, "transport_panels")
    os.makedirs(panels_dir, exist_ok=True)
    fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor=_BG)
    ax1.set_facecolor(_BG)
    _draw_cfm_trajectories(ax1, cfm_traj, n_show=200)
    ax1.set_title("Classical FM: Noise → Two Moons", color="C0", pad=10)
    ax1.tick_params(colors="gray"); [s.set_color("#333") for s in ax1.spines.values()]
    fig1.savefig(os.path.join(panels_dir, "cfm_trajectories.png"),
                 dpi=200, bbox_inches="tight", facecolor=_BG)
    plt.close(fig1)
    fig2 = plt.figure(figsize=(8, 8), facecolor=_BG)
    ax2 = fig2.add_subplot(111, projection="3d", facecolor=_BG)
    _draw_qfm_bloch(ax2, qfm_snaps, n_show=80)
    ax2.set_title("Quantum FM: |ψ₀⟩ → |+⟩ on Bloch Sphere", color="C2", pad=10)
    fig2.savefig(os.path.join(panels_dir, "qfm_bloch_trajectories.png"),
                 dpi=200, bbox_inches="tight", facecolor=_BG)
    plt.close(fig2)
    fig3, ax3 = plt.subplots(figsize=(10, 4), facecolor=_BG)
    ax3.set_facecolor("C8")
    _draw_loss(ax3, cfm_losses, qfm_losses, academic=False)
    ax3.set_title("Loss Convergence",
                  color="white", pad=8)
    ax3.tick_params(colors="gray"); ax3.grid(True, color="gray", ls="--")
    [s.set_color("#333") for s in ax3.spines.values()]
    fig3.savefig(os.path.join(panels_dir, "loss_comparison.png"),
                 dpi=200, bbox_inches="tight", facecolor=_BG)
    plt.close(fig3)
            
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    _draw_loss(ax4, cfm_losses, qfm_losses, academic=True)
    ax4.set_title("Loss Convergence",
                  color="black", pad=8)
    ax4.grid(True, color="gray", ls="--")
    
    fig4.savefig(os.path.join(panels_dir, "loss_comparison_academic.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig4)

    print(f"Individual panels saved to {panels_dir}/")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize noise→data transport for CFM and QFM."
    )
    parser.add_argument("--cfm-steps", type=int, default=2500,
                        help="CFM training iterations.")
    parser.add_argument("--flow-steps", type=int, default=15,
                        help="Number of QFM flow steps (T).")
    parser.add_argument("--qfm-layers", type=int, default=3,
                        help="EHA circuit layers.")
    parser.add_argument("--qfm-epochs", type=int, default=30,
                        help="Epochs per QFM step.")
    parser.add_argument("--out", type=str,
                        default="results/transport_comparison.png",
                        help="Output path.")
    args = parser.parse_args()
    generate_figure(args)

if __name__ == "__main__":
    main()
