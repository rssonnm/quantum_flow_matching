import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.apps.tfim import build_tfim_hamiltonian

def energy_manifold_3d(rhos, g_values, n_qubits, output_dir):
    print("Generating Advanced 4-Panel 3D Energy Manifold Dashboard")
    energies, magnetizations_raw, thetas = [], [], []
    ground_energies, spectral_gaps, purities = [], [], []

    Z_op_1q = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    for t, g in enumerate(g_values):
        H_t = build_tfim_hamiltonian(n_qubits, g)
        energies.append(float(torch.real(torch.trace(rhos[t] @ H_t))))
        
        eigs = np.linalg.eigvalsh(H_t.detach().cpu().numpy())
        ground_energies.append(float(eigs[0]))
        spectral_gaps.append(float(eigs[1] - eigs[0]))
        
        magnetization = 0
        for i in range(n_qubits):
            Z_i = Z_op_1q if i == 0 else torch.eye(2, dtype=torch.complex128)
            for j in range(1, n_qubits):
                Z_i = torch.kron(Z_i, Z_op_1q if j == i else torch.eye(2, dtype=torch.complex128))
            magnetization += float(torch.real(torch.trace(rhos[t] @ Z_i)))
            
        magnetizations_raw.append(magnetization / n_qubits)
        M_norm = np.clip(-magnetization / n_qubits, -1, 1)
        thetas.append(np.arccos(M_norm))
        purities.append(float(torch.real(torch.trace(rhos[t] @ rhos[t]))))

    energies, ground_energies = np.array(energies), np.array(ground_energies)
    thetas, spectral_gaps = np.array(thetas), np.array(spectral_gaps)
    purities, magnetizations_raw = np.array(purities), np.array(magnetizations_raw)
    g_arr = np.array(g_values)
    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.30)
    
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    g_grid, theta_grid = np.linspace(0, 1, 40), np.linspace(0, np.pi, 40)
    G, TH = np.meshgrid(g_grid, theta_grid)
    E_surf = -n_qubits * (np.cos(TH) + G * np.sin(TH))
    
    ax1.plot_surface(G, TH, E_surf, cmap='coolwarm', antialiased=True, edgecolor='none', alpha=0.8)
    for t in range(len(g_arr) - 1):
        ax1.plot(g_arr[t:t+2], thetas[t:t+2], energies[t:t+2], color=plt.cm.viridis(t / max(len(g_arr) - 2, 1)), lw=3.0)
        
    ax1.scatter(g_arr, thetas, energies, c=np.arange(len(g_arr)), cmap='viridis', s=50, edgecolors='black', linewidths=0.5, zorder=5)
    ax1.scatter(g_arr[0], thetas[0], energies[0], color=COLORS['accent'], s=200, zorder=6, edgecolors='black', linewidths=1.5)
    ax1.scatter(g_arr[-1], thetas[-1], energies[-1], color="C3", marker='*', s=350, zorder=6, edgecolors='black', linewidths=1)
    ax1.text(g_arr[0], thetas[0], energies[0] + 0.6, f"$E_0$={energies[0]:.2f}", color=COLORS['accent'])
    ax1.text(g_arr[-1], thetas[-1], energies[-1] + 0.6, f"$E_T$={energies[-1]:.2f}", color="C3")
    ax1.plot(g_arr, thetas, ground_energies, '--', color='gray', lw=1.5, label='$E_{gs}$ (exact)')

    ax1.set_xlabel('Driving $g(t)$', labelpad=8)
    ax1.set_ylabel('$\\theta = \\arccos(\\langle Z \\rangle)$', labelpad=8)
    ax1.set_zlabel('$\\langle H \\rangle$', labelpad=8)
    ax1.view_init(elev=22, azim=-55)
    ax1.set_title("(a)  3D Energy Manifold Trajectory", pad=15)
    ax1.legend(fontsize=9, loc='upper left')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(g_arr, energies, 'o-', color=COLORS['primary'], lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='QFM $\\langle H \\rangle$')
    ax2.plot(g_arr, ground_energies, 's--', color=COLORS['dark'], lw=2.0, ms=5, markerfacecolor='white', markeredgewidth=1.5, label='Exact $E_{gs}$')
    ax2.fill_between(g_arr, energies, ground_energies, color=COLORS['secondary'], alpha=0.3, label='Energy Gap $\\Delta E$')
    ax2b = ax2.twinx()
    ax2b.plot(g_arr, np.abs(energies - ground_energies), '^-', color="C8", lw=1.5, ms=5, label='$|\\Delta E|$')
    ax2b.set_ylabel('$|E_{QFM} - E_{gs}|$', color="C8")
    ax2.set_xlabel('Driving Field $g$')
    ax2.set_ylabel('Energy $\\langle H(g) \\rangle$', color=COLORS['primary'])
    ax2.grid(True, ls='--')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.set_title("(b)  Energy Convergence vs Ground State", pad=12)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(g_arr, magnetizations_raw, 'D-', color="C9", lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='$\\langle M_z \\rangle / N$')
    ax3.axhline(0, color='gray', ls=':', lw=1)
    ax3.axvline(1.0, color="C3", ls='--', lw=2, label='$g_c = 1.0$ (QPT)')
    ax3.fill_betweenx([-1, 1], 0.9, 1.1, color='red', alpha=0.1)
    ax3.set_xlabel('Driving Field $g$')
    ax3.set_ylabel('Magnetization $\\langle M_z \\rangle / N$', color="C9")
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, ls='--')
    ax3b = ax3.twinx()
    ax3b.plot(g_arr, purities, 's--', color=COLORS['tertiary'], lw=2.0, ms=5, markerfacecolor='white', markeredgewidth=1.5, label='Purity $\\gamma$')
    ax3b.set_ylabel('$\\gamma = \\mathrm{tr}(\\rho^2)$', color=COLORS['tertiary'])
    ax3b.set_ylim(0, 1.15)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    ax3.set_title("(c)  Order Parameter & State Purity", pad=12)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(g_arr, spectral_gaps, color=COLORS['primary'], alpha=0.2)
    ax4.plot(g_arr, spectral_gaps, 'o-', color=COLORS['primary'], lw=2.5, ms=7, markerfacecolor='white', markeredgewidth=2, label='$\\Delta(g) = E_1 - E_0$')
    min_idx = np.argmin(spectral_gaps)
    ax4.scatter(g_arr[min_idx], spectral_gaps[min_idx], color="C3", s=200, zorder=6, edgecolors='black', linewidths=1.5, marker='v')
    ax4.annotate(f"Min Gap\n$\\Delta$={spectral_gaps[min_idx]:.3f}\n$g$={g_arr[min_idx]:.2f}",
                 xy=(g_arr[min_idx], spectral_gaps[min_idx]), xytext=(g_arr[min_idx] + 0.1, spectral_gaps[min_idx] + 0.3), color="C3",
                 arrowprops=dict(arrowstyle='->', color="C3", lw=1.5))
    ax4.axvline(1.0, color="C3", ls='--', lw=2, label='$g_c = 1.0$')
    ax4.set_xlabel('Driving Field $g$')
    ax4.set_ylabel('Spectral Gap $\\Delta(g)$')
    ax4.grid(True, ls='--')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.set_title("(d)  Spectral Gap — Quantum Phase Transition", pad=12)

    fig.suptitle(f"Quantum Flow Matching — {n_qubits}-Qubit TFIM Energy Manifold Analysis", y=0.98, color="C0")
    path = os.path.join(output_dir, 'qfm_3d_energy_manifold.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

def quantum_phase_portrait(rhos, g_values, n_qubits, output_dir):
    print("Generating Advanced Quantum Phase Portrait")
    X_op = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    Y_op = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    Z_op = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    
    for _ in range(n_qubits - 1):
        X_op = torch.kron(X_op, torch.eye(2, dtype=torch.complex128))
        Y_op = torch.kron(Y_op, torch.eye(2, dtype=torch.complex128))
        Z_op = torch.kron(Z_op, torch.eye(2, dtype=torch.complex128))
        
    pts_3d = np.array([[float(torch.real(torch.trace(r @ op))) for op in (X_op, Y_op, Z_op)] for r in rhos])
    c_start, c_end, c_path = "C6", "C9", "C0"

    fig = plt.figure(figsize=(20, 6.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0], projection='3d')
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    ax1.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), color='gray', alpha=0.3)
    ax1.plot([0,0], [0,0], [-1,1], color='black', ls='--', alpha=0.5)
    ax1.plot([-1,1], [0,0], [0,0], color='black', ls='--', alpha=0.5)
    ax1.plot([0,0], [-1,1], [0,0], color='black', ls='--', alpha=0.5)
    
    ax1.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2], c=np.linspace(0, 1, len(pts_3d)), cmap='cool', s=80, edgecolors='black')
    ax1.plot(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2], color=c_path, lw=2)
    ax1.scatter(*pts_3d[0], color=c_start, s=200, marker='o', edgecolors='white', label='Init $\\rho_0$')
    ax1.scatter(*pts_3d[-1], color=c_end, s=400, marker='*', edgecolors='white', label='Target $\\rho_T$')

    ax1.set_title('(a) Bloch State Trajectory $\\mathbf{r}(\\tau)$')
    ax1.set_xlim([-1.1, 1.1]); ax1.set_ylim([-1.1, 1.1]); ax1.set_zlim([-1.1, 1.1])
    ax1.legend(loc='lower right')
    ax1.view_init(elev=30, azim=45)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(g_values, pts_3d[:,0], 's-', color="C3", lw=2, ms=5, label='$\\langle X \\rangle(\\tau)$')
    ax2.plot(g_values, pts_3d[:,1], 'o-', color="C8", lw=2, ms=5, label='$\\langle Y \\rangle(\\tau)$')
    ax2.plot(g_values, pts_3d[:,2], 'D-', color="C9", lw=2, ms=5, label='$\\langle Z \\rangle(\\tau)$')
    ax2.axvline(1.0, color="red", ls='--', lw=2, label='$g_c = 1.0$')
    
    ax2.set_xlabel('Driving Field $g_\\tau$')
    ax2.set_ylabel('Expectation Value $\\langle \\sigma_i \\rangle$')
    ax2.set_title('(b) Order Parameter Components')
    ax2.legend(loc='best')
    ax2.grid(True, ls='--')

    ax3 = fig.add_subplot(gs[2])
    V_3d = np.diff(pts_3d, axis=0)
    velocities = np.linalg.norm(V_3d, axis=1)
    acceleration = np.diff(velocities)
    
    ax3.plot(g_values[1:], velocities, '*-', color=c_start, lw=2.5, ms=8, label='Speed $|\\mathbf{v}(\\tau)|$')
    if len(acceleration) > 0:
        ax3b = ax3.twinx()
        ax3b.bar(g_values[2:], acceleration, color=c_end, alpha=0.5, width=0.04, label='Accel $a_\\tau$')
        ax3b.set_ylabel('Bloch Acceleration', color=c_end)
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax3.legend(loc='upper right')
        
    ax3.set_xlabel('Driving Field $g_\\tau$')
    ax3.set_ylabel('Bloch Velocity', color=c_start)
    ax3.set_title('(c) Trajectory Kinematics (Speed & Accel)')
    ax3.grid(True, ls='--')

    fig.suptitle("Quantum Phase Portrait & Bloch Manifold Dynamics", y=1.02, color="C0")
    path = os.path.join(output_dir, 'qfm_phase_portrait.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--ensemble", type=int, default=10)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    _, rhos, _, _, _, g_vals, nq, T = train_qfm_collect_all(n_qubits=args.qubits, T_steps=args.steps, M=args.ensemble)

    energy_manifold_3d(rhos, g_vals, nq, args.out)
    quantum_phase_portrait(rhos, g_vals, nq, args.out)

if __name__ == "__main__":
    main()
