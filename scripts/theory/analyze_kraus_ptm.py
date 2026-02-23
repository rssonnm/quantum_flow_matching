import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

from src.qfm.ansatz import EHA_Circuit
from src.qfm.channels import (kraus_from_unitary, kraus_completeness_error,
                                choi_matrix, chi_matrix, pauli_transfer_matrix)

def run_kraus_analysis(n_data=1, n_ancilla=1, n_layers=3, out_dir="results"):
    print(f"Kraus/PTM Analysis (data={n_data}, ancilla={n_ancilla})")
    circuit = EHA_Circuit(n_data=n_data, n_layers=n_layers, n_ancilla=n_ancilla)
    params  = circuit.theta.detach()
    d_total = 2**(n_data + n_ancilla)

    U_cols = []
    for k in range(d_total):
        basis = torch.zeros(d_total, dtype=torch.complex128); basis[k] = 1.0
        col   = circuit.qnode(basis, params); U_cols.append(col.detach())
    U = torch.stack(U_cols, dim=1)

    kraus_ops = kraus_from_unitary(U, n_data, n_ancilla)
    err = kraus_completeness_error(kraus_ops)
    print(f"Completeness Error: {err:.2e}")
    R      = pauli_transfer_matrix(kraus_ops)
    R_np   = R.numpy()
    C      = choi_matrix(kraus_ops)
    C_np   = C.detach().cpu().numpy()
    chi    = chi_matrix(kraus_ops)
    chi_np = chi.detach().cpu().numpy()
    kraus_svs = []
    for i, K in enumerate(kraus_ops):
        K_np = K.detach().cpu().numpy()
        svs  = np.linalg.svd(K_np, compute_uv=False)
        kraus_svs.append(svs)
    d_data = 2 ** n_data
    R_sub   = R_np[1:, 1:]
    unitarity = np.trace(R_sub.T @ R_sub) / 3.0
    ptm_trace = np.trace(R_np)
    avg_fidelity = (ptm_trace + d_data) / (d_data * (d_data + 1))
    choi_eigs = np.real(np.linalg.eigvalsh(C_np))

    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(20, 16))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    pauli_labels = ["$I$", "$X$", "$Y$", "$Z$"]
    ax1 = fig.add_subplot(gs[0, 0])
    norm_ptm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cax = ax1.imshow(R_np, cmap="RdBu", norm=norm_ptm, aspect="equal")
    ax1.set_xticks(range(4)); ax1.set_xticklabels(pauli_labels)
    ax1.set_yticks(range(4)); ax1.set_yticklabels(pauli_labels)
    ax1.set_xlabel("Input Pauli Basis")
    ax1.set_ylabel("Output Pauli Basis")
    for i in range(4):
        for j in range(4):
            val = R_np[i, j]
            color = "white" if abs(val) > 0.35 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color)
    cbar1 = fig.colorbar(cax, ax=ax1, shrink=0.8, pad=0.03)
    cbar1.set_label("Transfer Amplitude")
    ax1.set_title(f"(a)  Pauli Transfer Matrix\nCompleteness: {err:.1e}", pad=12)
    ax2 = fig.add_subplot(gs[0, 1])
    C_real = np.real(C_np)
    vmax_c = max(abs(C_real.max()), abs(C_real.min())) + 1e-9
    norm_choi = TwoSlopeNorm(vmin=-vmax_c, vcenter=0.0, vmax=vmax_c)
    cax2 = ax2.imshow(C_real, cmap="coolwarm", norm=norm_choi, aspect="equal")
    ax2.set_xlabel("Column Index")
    ax2.set_ylabel("Row Index")
    d2 = C_real.shape[0]
    for i in range(d2):
        for j in range(d2):
            val = C_real[i, j]
            color = "white" if abs(val) > 0.3 * vmax_c else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color)
    cbar2 = fig.colorbar(cax2, ax=ax2, shrink=0.8, pad=0.03)
    cbar2.set_label("Re(Choi) Amplitude")
    cp_status = "CP ✓" if np.all(choi_eigs >= -1e-8) else "NOT CP ✗"
    ax2.set_title(f"(b)  Choi–Jamiołkowski Matrix\n{cp_status}", pad=12)
    ax3 = fig.add_subplot(gs[1, 0])
    n_kraus = len(kraus_svs)
    x_positions = np.arange(n_kraus)
    bar_width = 0.35
    colors_sv = ["C2", "C1", "C3", "C0"]

    for i, svs in enumerate(kraus_svs):
        for j, sv in enumerate(svs):
            offset = (j - (len(svs)-1)/2) * bar_width
            ax3.bar(x_positions[i] + offset, sv, bar_width * 0.9,
                   color=colors_sv[j % len(colors_sv)],
                   edgecolor="black", linewidth=0.5,
                   label=f"$\\sigma_{j+1}$" if i == 0 else "")

    ax3.set_xticks(x_positions)
    ax3.set_xticklabels([f"$K_{{{i}}}$" for i in range(n_kraus)])
    ax3.set_xlabel("Kraus Operator")
    ax3.set_ylabel("Singular Value $\\sigma_i$")
    ax3.set_ylim(0, 1.15)
    ax3.axhline(1.0, color="gray", ls="--", lw=1, label="Unitary bound")
    ax3.legend(fontsize=10, loc="upper right")
    ax3.grid(True, axis="y", ls="--")
    ax3.set_title("(c)  Kraus Operator Spectrum (SVD)", pad=12)
    ax4 = fig.add_subplot(gs[1, 1])
    metric_names  = ["Avg Fidelity\n$\\bar{F}$",
                     "Unitarity\n$u(\\mathcal{E})$",
                     "Trace Pres.\n$1-\\epsilon$",
                     "Min Choi\nEigenvalue"]
    metric_values = [float(avg_fidelity),
                     float(unitarity),
                     float(1.0 - err),
                     float(choi_eigs.min())]
    bar_colors    = ["C2", "C3", "C0", "C1"]

    bars = ax4.bar(metric_names, metric_values, color=bar_colors,
                   edgecolor="black", linewidth=0.8, width=0.6)
    for bar, val in zip(bars, metric_values):
        ypos = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, ypos + 0.02,
                f"{val:.4f}", ha="center", va="bottom", color="C4")

    ax4.set_ylim(min(0, min(metric_values) - 0.1), 1.15)
    ax4.axhline(1.0, color="C5", ls="--", lw=1.5, label="Ideal = 1.0")
    ax4.axhline(0.0, color="C5", ls="-", lw=0.8)
    ax4.set_ylabel("Metric Value")
    ax4.legend(fontsize=10, loc="lower right")
    ax4.grid(True, axis="y", ls="--")
    ax4.set_title("(d)  Channel Quality Dashboard", pad=12)

    fig.suptitle("Quantum Channel Analysis — Kraus Decomposition & Process Tomography", y=0.98, color="C4")

    out_path = os.path.join(out_dir, "ptm_heatmap.png")
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved advanced PTM Dashboard to {out_path}")
    print(f"\n[ptm] Saved Q1 Channel Analysis Dashboard → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_kraus_analysis(out_dir=args.out)
