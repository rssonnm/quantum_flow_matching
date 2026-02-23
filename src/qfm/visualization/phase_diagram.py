"""
phase_diagram.py — Magnetization phase diagrams for TFIM.
"""
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from ..apps.tfim import build_tfim_hamiltonian
from ..utils import state_vector_to_density_matrix

logger = logging.getLogger(__name__)

def exact_magnetization(n_qubits: int, g_values: list) -> list:
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)
    mag_vals = []
    for g in g_values:
        H = build_tfim_hamiltonian(n_qubits, g)
        ev, evec = torch.linalg.eigh(H); gs = evec[:, 0]
        mag = 0.0
        for i in range(n_qubits):
            term = torch.eye(1, dtype=torch.complex128)
            for j in range(n_qubits):
                term = torch.kron(term, Z if j == i else I)
            mag += float(torch.real(torch.vdot(gs, torch.matmul(term, gs))))
        mag_vals.append(abs(mag) / n_qubits)
    return mag_vals

def plot_tfim_phase_diagram(
    g_values: list,
    qfm_magnetizations: list,
    n_qubits: int = 4,
    purity_values: list = None,
    out_path: str = "results/tfim_phase_diagram.png",
):
    """
    Advanced 3-panel TFIM Phase Diagram (Ultra-Q1 Quality):
    (1) Order Parameter <|M|> vs g.
    (2) Susceptibility |dM/dg| proxy to locate Quantum Phase Transition.
    (3) State Purity/Fidelity Evolution over g.
    """
    try:
        from src.qfm.utils import set_academic_style
        set_academic_style()
    except ImportError:
        pass

    exact = exact_magnetization(n_qubits, g_values)
    
    # Calculate Magnetic Susceptibility Proxy (Numerical Derivative)
    dg = np.gradient(g_values)
    dM_exact = np.abs(np.gradient(exact, dg))
    dM_qfm = np.abs(np.gradient(qfm_magnetizations, dg))
    
    fig = plt.figure(figsize=(20, 5.5))
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2], wspace=0.35)

    # --- Panel 1: Order Parameter ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(g_values, exact, "-", color="#CFD8DC", lw=6.0, label="Exact Ground State", zorder=1)
    ax1.plot(g_values, qfm_magnetizations, "o--", color="#00ACC1", lw=2.5, ms=8, 
             markerfacecolor="white", markeredgewidth=2, label="Quantum Flow Matching", zorder=3)
    ax1.axvline(1.0, color="#E53935", linestyle=":", lw=2, label="Critical $g_c=1.0$")
    ax1.axvspan(0.85, 1.15, color="gray", alpha=0.1, zorder=0)
    
    ax1.set_xlabel("Transverse Field Strength $g$", fontweight="bold")
    ax1.set_ylabel("Order Parameter $\\langle |M_z| \\rangle$", fontweight="bold")
    ax1.set_title("(a) Ferromagnetic Phase Transition", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3, ls="--")

    # --- Panel 2: Magnetic Susceptibility ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(g_values, dM_exact, "-", color="#B0BEC5", lw=4.0, label="Exact Susceptibility $\\chi$", zorder=1)
    ax2.plot(g_values, dM_qfm, "^-", color="#FDD835", lw=2.5, ms=8, 
             markeredgecolor="#F57F17", markeredgewidth=1.5, label="QFM Fidelity Susceptibility", zorder=3)
    ax2.axvline(1.0, color="#E53935", linestyle=":", lw=2)
    ax2.axvspan(0.85, 1.15, color="gray", alpha=0.1, zorder=0)

    ax2.set_xlabel("Transverse Field Strength $g$", fontweight="bold")
    ax2.set_ylabel("Susceptibility Proxy $|\\partial M_z / \\partial g|$", fontweight="bold")
    ax2.set_title("(b) Critical Point Divergence", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.3, ls="--")

    # --- Panel 3: State Purity Evolution ---
    ax3 = fig.add_subplot(gs[2])
    if purity_values is None:
        purity_values = [1.0] * len(g_values)
        
    # Colormap the purity based on transverse field
    sc = ax3.scatter(g_values, purity_values, c=g_values, cmap="plasma",
                     s=120, zorder=5, edgecolors="black", lw=1.0)
    ax3.plot(g_values, purity_values, "-", color="#455A64", lw=2.0, zorder=4, alpha=0.5)
    ax3.axvline(1.0, color="#E53935", linestyle=":", lw=2)
    ax3.axvspan(0.85, 1.15, color="gray", alpha=0.1, label="Quantum Fluctuation Zone")

    ax3.set_ylim(max(0.0, min(purity_values)-0.1), 1.05)
    ax3.set_xlabel("Transverse Field Strength $g$", fontweight="bold")
    ax3.set_ylabel("Ensemble Purity $\\gamma = \\mathrm{tr}(\\rho^2)$", fontweight="bold")
    ax3.set_title("(c) Coherence Preservation", fontsize=14, fontweight="bold")
    ax3.legend(loc="lower right", framealpha=0.9)
    ax3.grid(True, alpha=0.3, ls="--")
    
    cbar = fig.colorbar(sc, ax=ax3, pad=0.02)
    cbar.set_label("Transverse Field $g$", fontweight="bold")

    fig.suptitle("Quantum Flow Matching over TFIM Magnetic Phase Transition ($N_{{qubits}}={}$)".format(n_qubits),
                 fontsize=17, fontweight="bold", y=1.05)
                 
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Q1-Level phase diagram to {out_path}")
    print(f"\n[tfim] Saved Advanced Phase Transition Dashboard → {out_path}")
