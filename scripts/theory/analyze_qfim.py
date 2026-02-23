import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.qfm.ansatz import EHA_Circuit
from src.qfm.qfim import compute_qfim_fd, barren_plateau_report

def run_qfim_analysis(n_data=2, depth_list=[1, 2, 4, 8], out_dir="results"):
    
    print(f"QFIM Analysis & Barren Plateau Diagnosis ({n_data}-qubit EHA)")
    psi0 = torch.zeros(2**n_data, dtype=torch.complex128)
    psi0[0] = 1.0
    
    dims = []
    condition_numbers = []
    spectra = []
    qfim_matrices = []

    for L in depth_list:
        circuit = EHA_Circuit(n_data=n_data, n_layers=L, n_ancilla=0)
        def _wrapped_qnode(state, theta):
            return circuit.qnode(state, theta.reshape(L, -1))
            
        F = compute_qfim_fd(_wrapped_qnode, circuit.theta.detach(), psi0)
        rep = barren_plateau_report(F)
        
        dims.append(rep["effective_dimension"])
        condition_numbers.append(rep["condition_number"])
        spectra.append(rep["eigenvalues"])
        qfim_matrices.append(F.numpy())
        
        print(f"L={L}: eff_dim={rep['effective_dimension']:.3f}, Cond={rep['condition_number']:.1e}")
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(18, 5))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    deepest_idx = -1
    mat = qfim_matrices[deepest_idx]
    mat_log = np.sign(mat) * np.log1p(np.abs(mat) * 1e4)
    im = ax1.imshow(mat_log, cmap="coolwarm", interpolation="none")
    ax1.set_title(f"(a) QFIM Structure at $L={depth_list[deepest_idx]}$")
    ax1.set_xlabel("Parameter Index $i$")
    ax1.set_ylabel("Parameter Index $j$")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Pseudo-Log Curvature", rotation=270, labelpad=15)
    ax2 = fig.add_subplot(gs[1])
    colors = ["C5", "C0", "C2", "C3"]
    for i, L in enumerate(depth_list):
        evals = np.array(spectra[i])
        evals = evals[evals > 1e-12] 
        x_idx = np.arange(1, len(evals) + 1) / len(evals)
        ax2.plot(x_idx, evals, "o-", color=colors[i % len(colors)], label=f"$L={L}$", lw=2)
    
    ax2.set_yscale("log")
    ax2.set_title("(b) QFIM Eigenvalue Spectrum")
    ax2.set_xlabel("Normalized Parameter Rank $k / P$")
    ax2.set_ylabel("Eigenvalue $\\lambda_k$ (Log Scale)")
    ax2.legend(facecolor="white")
    ax2.grid(True, ls="--")
    ax3 = fig.add_subplot(gs[2])
    ax3_twin = ax3.twinx()
    
    l1 = ax3.plot(depth_list, dims, "s-", color="C4", lw=2.5, label="Effective Dimension $d_{\\mathrm{eff}}$")
    l2 = ax3_twin.plot(depth_list, condition_numbers, "^-", color="C1", lw=2.5, label="Condition Number $\\kappa$")
    
    ax3_twin.set_yscale("log")
    ax3.set_title("(c) Manifold Capacity vs Circuit Depth")
    ax3.set_xlabel("Ansatz Layers $L$")
    ax3.set_ylabel("Effective Dimension Fraction", color="C4")
    ax3_twin.set_ylabel("Condition Number (Log Scale)", color="C1")
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc="center right")
    ax3.grid(True, ls="--")
    ax3.set_xticks(depth_list)

    fig.suptitle("Quantum Information Geometry: QFIM Landscape & Barren Plateau Diagnosis", y=1.05)
                 
    fig.tight_layout()
    out_path = os.path.join(out_dir, "qfim_analysis.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved advanced QFIM analysis plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    run_qfim_analysis(out_dir=args.out)
