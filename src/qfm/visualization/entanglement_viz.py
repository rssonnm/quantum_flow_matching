"""
entanglement_viz.py — Visualizations for entanglement entropy and purity.
"""
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from ..metrics import von_neumann_entropy, renyi_entropy, negativity

logger = logging.getLogger(__name__)

def plot_entropy_growth(
    rhos_per_tau: list,
    target_entropies: list,
    out_path: str = "results/entanglement_entropy_growth.png",
    n_qubits_A: int = 1,
    n_qubits_B: int = 1,
):
    """
    Plots the growth of entanglement entropy and purity across time steps.
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass

    T = len(rhos_per_tau)
    taus = list(range(1, T + 1))
    dA = 2 ** n_qubits_A
    dB = 2 ** n_qubits_B

    def _partial_trace_A(rho):
        rho_r = rho.reshape(dA, dB, dA, dB)
        return torch.einsum('ibjb->ij', rho_r)

    vn_mean, vn_std = [], []
    r2_mean, r2_std = [], []
    pur_mean, pur_std = [], []
    neg_mean, neg_std = [], []

    for rhos_at_tau in rhos_per_tau:
        vn_list, r2_list, pur_list, neg_list = [], [], [], []
        for r in rhos_at_tau:
            neg_list.append(float(negativity(r, n_qubits_A, n_qubits_B)))
            
            rho_A = _partial_trace_A(r)
            rho_A = rho_A / (torch.real(torch.trace(rho_A)) + 1e-12)
            
            vn_list.append(float(von_neumann_entropy(rho_A)))
            r2_list.append(float(renyi_entropy(rho_A, 2.0)))
            pur_list.append(float(torch.real(torch.trace(rho_A @ rho_A))))
            
        vn_mean.append(np.mean(vn_list)); vn_std.append(np.std(vn_list))
        r2_mean.append(np.mean(r2_list)); r2_std.append(np.std(r2_list))
        pur_mean.append(np.mean(pur_list)); pur_std.append(np.std(pur_list))
        neg_mean.append(np.mean(neg_list)); neg_std.append(np.std(neg_list))

    vn_mean = np.array(vn_mean); vn_std = np.array(vn_std)
    r2_mean = np.array(r2_mean); r2_std = np.array(r2_std)
    pur_mean = np.array(pur_mean); pur_std = np.array(pur_std)
    neg_mean = np.array(neg_mean); neg_std = np.array(neg_std)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    def plot_with_std(ax, x, mean, std, color, label):
        ax.plot(x, mean, "o-", color=color, lw=2.5, markersize=6, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    # 1. Von Neumann Entropy
    ax = axes[0]
    plot_with_std(ax, taus, vn_mean, vn_std, "#E91E63", "QFM Ensemble Mean $\pm$ 1$\sigma$")
    ax.plot(taus, target_entropies, "--", color="#37474F", lw=2, label="Exact Target Dynamics")
    ax.set_title("Von Neumann Entropy $S(\\rho_A)$", fontsize=14, fontweight="bold")
    ax.set_ylabel("Entropy (bits)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, ls="--")

    # 2. Renyi-2 Entropy
    ax = axes[1]
    plot_with_std(ax, taus, r2_mean, r2_std, "#9C27B0", "QFM $S_2(\\rho_A)$")
    ax.plot(taus, target_entropies, "--", color="#37474F", lw=2)
    ax.set_title("Rényi-2 Entropy $S_2(\\rho_A)$", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")

    # 3. Purity
    ax = axes[2]
    plot_with_std(ax, taus, pur_mean, pur_std, "#00BCD4", "QFM Subsystem A Purity")
    target_purities = [2**(-S) for S in target_entropies]
    ax.plot(taus, target_purities, "--", color="#37474F", lw=2, label="Ideal Purity")
    ax.set_title("Subsystem A Purity $\\mathrm{Tr}(\\rho_A^2)$", fontsize=14, fontweight="bold")
    ax.set_xlabel("Flow Step ($\\tau$)", fontweight="bold")
    ax.set_ylabel("Purity", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, ls="--")

    # 4. Negativity
    ax = axes[3]
    plot_with_std(ax, taus, neg_mean, neg_std, "#FF9800", "QFM Negativity $\\mathcal{N}(\\rho)$")
    ax.set_title("Log-Negativity (Full State)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Flow Step ($\\tau$)", fontweight="bold")
    ax.grid(True, alpha=0.3, ls="--")

    fig.suptitle(f"Quantum Flow Matching: Driven Entanglement Dynamics\n($n_A={n_qubits_A}, n_B={n_qubits_B}$)", 
                 fontsize=18, fontweight="bold", y=0.96)
                 
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved advanced entanglement growth plot to {out_path}")
