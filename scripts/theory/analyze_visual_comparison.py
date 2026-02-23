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
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.qfm.ansatz import EHA_Circuit
from src.qfm.utils import state_vector_to_density_matrix
from src.qfm.metrics import uhlmann_fidelity

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
    
    n1 = n // 2
    n2 = n - n1
    theta1 = np.linspace(0, np.pi, n1)
    theta2 = np.linspace(0, np.pi, n2)
    x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    x2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])
    pts = np.vstack([x1, x2]) + noise * np.random.randn(n, 2)
    return torch.from_numpy(pts).float()

def _train_cfm(target_pts, n_iter=2000, batch=512):
    
    model = _VectorFieldMLP()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for step in range(n_iter):
        x0 = torch.randn(batch, 2)
        idx = torch.randint(0, len(target_pts), (batch,))
        x1 = target_pts[idx]
        t = torch.rand(batch)
        x_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
        u_t = x1 - x0
        v = model(x_t, t)
        loss = ((v - u_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step} loss={loss.item():.5f}")

    return model

@torch.no_grad()
def _sample_cfm_trajectory(model, n=1000, n_steps=20):
    
    x = torch.randn(n, 2)
    dt = 1.0 / n_steps
    snapshots = [x.clone()]
    for i in range(n_steps):
        t = torch.ones(n) * (i / n_steps)
        v = model(x, t)
        x = x + dt * v
        snapshots.append(x.clone())
    return snapshots

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

def _train_qfm_1qubit(n_steps=10, n_ensemble=60, n_layers=5, lr=0.08, epochs=40):
    
    target_state = torch.tensor([1.0, 1.0], dtype=torch.complex128) / np.sqrt(2)

    init_states = []
    for _ in range(n_ensemble):
        psi = torch.tensor([1.0, 0.0], dtype=torch.complex128)
        noise = 0.05 * torch.randn(2, dtype=torch.float64)
        psi = psi + noise.to(torch.complex128)
        psi = psi / psi.norm()
        init_states.append(psi)

    snapshots = [torch.stack(init_states)]
    current = init_states

    for tau in range(1, n_steps + 1):
        frac = tau / n_steps
        
        target_rhos = []
        for psi0 in init_states:
            psi_t = _slerp(psi0, target_state, frac)
            target_rhos.append(state_vector_to_density_matrix(psi_t))
        target_rhos = torch.stack(target_rhos)

        model = EHA_Circuit(n_data=1, n_layers=n_layers, n_ancilla=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

        inputs = torch.stack(current)

        for ep in range(epochs):
            optimizer.zero_grad()
            out_states = model(inputs)
            rhos_out = torch.stack([
                state_vector_to_density_matrix(psi) for psi in out_states
            ])
            fids = torch.stack([
                uhlmann_fidelity(r, tr) for r, tr in zip(rhos_out, target_rhos)
            ])
            loss = 1.0 - fids.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            out_states = model(inputs)
            current = [s.detach() for s in out_states]

        snapshots.append(torch.stack(current))
        print(f"step {tau} loss={loss.item():.5f}")

    return snapshots

_CMAP_CFM = "cividis"
_CMAP_QFM = "magma"

def _render_kde_panel(ax, pts_np, cmap, vmin=None, vmax=None,
                      fixed_extent=None):
    
    if len(pts_np) < 10:
        ax.axis("off")
        return
    try:
        kde = gaussian_kde(pts_np.T, bw_method=0.25)
        if fixed_extent is not None:
            xr, yr = fixed_extent[:2], fixed_extent[2:]
        else:
            pad = 0.5
            xr = [pts_np[:, 0].min() - pad, pts_np[:, 0].max() + pad]
            yr = [pts_np[:, 1].min() - pad, pts_np[:, 1].max() + pad]
        xx, yy = np.mgrid[xr[0]:xr[1]:80j, yr[0]:yr[1]:80j]
        Z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.imshow(np.rot90(Z), cmap=cmap,
                  extent=[xr[0], xr[1], yr[0], yr[1]],
                  aspect="auto", vmin=vmin, vmax=vmax)
    except Exception:
        pass
    ax.set_xticks([]); ax.set_yticks([])

def _bloch_coords(psi):
    
    psi = psi.detach().cpu().to(torch.complex128)
    a, b = psi[0], psi[1]
    bx = float(2 * (a * b.conj()).real)
    by = float(abs(a) ** 2 - abs(b) ** 2)
    return bx, by

def _save_individual_panel(pts_data, panel_type, pct_label, out_dir,
                            row_name, col_idx):
    
    fig_s, ax_s = plt.subplots(1, 1, figsize=(4, 4), facecolor="C0")
    ax_s.set_facecolor("C0")

    if panel_type == "cfm_density":
        _render_kde_panel(ax_s, pts_data, _CMAP_CFM)
    elif panel_type == "cfm_samples":
        ax_s.scatter(pts_data[:, 0], pts_data[:, 1], s=1.5,
                     color="C2")
        ax_s.set_xlim(-2.5, 3.0); ax_s.set_ylim(-1.5, 2.0)
        ax_s.set_xticks([]); ax_s.set_yticks([])
    elif panel_type == "qfm_density":
        _render_kde_panel(ax_s, pts_data, _CMAP_QFM,
                          fixed_extent=[-1.2, 1.2, -1.2, 1.2])
    elif panel_type == "qfm_states":
        ax_s.scatter(pts_data[:, 0], pts_data[:, 1], s=18,
                     color="C1", edgecolors="white", linewidths=0.3)
        ax_s.set_xlim(-1.2, 1.2); ax_s.set_ylim(-1.2, 1.2)
        ax_s.set_xticks([]); ax_s.set_yticks([])

    for spine in ax_s.spines.values():
        spine.set_visible(False)

    ax_s.set_title(f"{row_name}  {pct_label}", color="white", pad=8)

    fname = f"{row_name}_{pct_label.replace('%','pct')}.png"
    path = os.path.join(out_dir, fname)
    fig_s.savefig(path, dpi=200, bbox_inches="tight", facecolor="C0",
                  pad_inches=0.15)
    plt.close(fig_s)
    return path

def generate_comparison_figure(args):
    
    print("Training CFM")
    target_moons = _sample_two_moons(3000)
    cfm_model = _train_cfm(target_moons, n_iter=args.cfm_steps)
    cfm_traj = _sample_cfm_trajectory(cfm_model, n=1500, n_steps=args.flow_steps)

    print("Training QFM")
    qfm_traj = _train_qfm_1qubit(n_steps=args.flow_steps, n_ensemble=80,
                                   n_layers=args.qfm_layers, epochs=args.qfm_epochs)
    T = args.flow_steps
    snap_idx = [0,
                max(1, int(0.2 * T)),
                max(1, int(0.4 * T)),
                max(1, int(0.6 * T)),
                max(1, int(0.8 * T)),
                T]
    pct_labels = ["0%", "20%", "40%", "60%", "80%", "100%"]
    out_base = os.path.dirname(args.out) if os.path.dirname(args.out) else "results"
    panels_dir = os.path.join(out_base, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    row_configs = [
        ("cfm_density",  "CFM_Density"),
        ("cfm_samples",  "CFM_Samples"),
        ("qfm_density",  "QFM_Density"),
        ("qfm_states",   "QFM_States"),
    ]

    print(f"Saving individual panels to {panels_dir}/...")
    for col_i, (si, pct) in enumerate(zip(snap_idx, pct_labels)):
        for panel_type, row_name in row_configs:
            if panel_type.startswith("cfm"):
                pts = cfm_traj[si].numpy()
            else:
                states = qfm_traj[si]
                pts = np.array([_bloch_coords(s) for s in states])

            path = _save_individual_panel(pts, panel_type, pct, panels_dir,
                                           row_name, col_i)
            print(f"  Saved {path}")
    fig = plt.figure(figsize=(21, 12), facecolor="C0")
    gs = gridspec.GridSpec(4, 7, figure=fig,
                           width_ratios=[1] * 6 + [1],
                           hspace=0.08, wspace=0.08)

    row_labels = ["CFM\nDensity", "CFM\nSamples", "QFM\nDensity", "QFM\nStates"]
    row_colors = ["C2", "C2", "C1", "C1"]

    for row in range(4):
        for col_i, (si, pct) in enumerate(zip(snap_idx, pct_labels)):
            ax = fig.add_subplot(gs[row, col_i])
            ax.set_facecolor("C0")

            if row == 0:
                pts = cfm_traj[si].numpy()
                _render_kde_panel(ax, pts, _CMAP_CFM)
                ax.set_title(pct,
                             color="white", pad=6)
            elif row == 1:
                pts = cfm_traj[si].numpy()
                ax.scatter(pts[:, 0], pts[:, 1], s=0.6,
                           color="C2")
                ax.set_xlim(-2.5, 3.0); ax.set_ylim(-1.5, 2.0)
                ax.set_xticks([]); ax.set_yticks([])
            elif row == 2:
                states = qfm_traj[si]
                bloch_pts = np.array([_bloch_coords(s) for s in states])
                _render_kde_panel(ax, bloch_pts, _CMAP_QFM,
                                  fixed_extent=[-1.2, 1.2, -1.2, 1.2])
            else:
                states = qfm_traj[si]
                bloch_pts = np.array([_bloch_coords(s) for s in states])
                ax.scatter(bloch_pts[:, 0], bloch_pts[:, 1], s=10,
                           color="C1", edgecolors="white", linewidths=0.2)
                ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
                ax.set_xticks([]); ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

        ax_lbl = fig.add_subplot(gs[row, 6])
        ax_lbl.axis("off")
        ax_lbl.text(0.5, 0.5, row_labels[row],
                    transform=ax_lbl.transAxes, color=row_colors[row],
                    ha="center", va="center", rotation=0)

    fig.suptitle("Flow Matching Dynamics: Classical (Two Moons)  vs  Quantum (Bloch Sphere)", color="white", y=0.97,
                 fontfamily="serif")
    fig.text(0.30, 0.93, "Classical CFM  ·  $\\mathbb{R}^2$: $\\mathcal{N}(0,I) \\to$ Two Moons", color="C2", ha="center")
    fig.text(0.30, 0.47, "Quantum QFM  ·  Bloch Sphere: $|0\\rangle \\to |+\\rangle$", color="C1", ha="center")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight", facecolor="C0",
                pad_inches=0.3)
    plt.close(fig)
    print(f"Combined grid saved to {args.out}")
    print(f"Individual panels saved to {panels_dir}/ (24 files)")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a publication-quality QFM vs CFM snapshot comparison."
    )
    parser.add_argument("--cfm-steps", type=int, default=2000,
                        help="Number of CFM training iterations.")
    parser.add_argument("--flow-steps", type=int, default=10,
                        help="Number of discrete flow steps (T).")
    parser.add_argument("--qfm-layers", type=int, default=5,
                        help="Number of EHA layers for QFM.")
    parser.add_argument("--qfm-epochs", type=int, default=40,
                        help="Training epochs per QFM step.")
    parser.add_argument("--out", type=str,
                        default="results/visual_comparison_qfm_vs_cfm.png",
                        help="Output path for the generated figure.")
    args = parser.parse_args()


    generate_comparison_figure(args)

if __name__ == "__main__":
    main()
