import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from simulate_qfm import COLORS
from src.qfm.apps.tfim import build_tfim_hamiltonian
from src.qfm.ansatz import EHA_Circuit
from src.qfm.utils import state_vector_to_density_matrix

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

def gradient_landscape(n_qubit, output_dir):
    print("Generating Advanced Gradient Loss Landscape Analysis (High-Res)")
    H_t = build_tfim_hamiltonian(n_qubit, 1.0)
    model = EHA_Circuit(n_data=n_qubit, n_layers=4, n_ancilla=0)
    
    torch.manual_seed(42)
    for param in model.parameters():
        torch.nn.init.uniform_(param, -1.0, 1.0)
        
    theta_init = model.theta.detach().clone()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    zero_state = torch.zeros(2**n_qubit, dtype=torch.complex128)
    zero_state[0] = 1.0
    input_states = torch.stack([zero_state for _ in range(3)]) 
    
    trajectory_theta = [theta_init.clone()]
    
    print("Simulating High-Dimensional Gradient Optimization...")
    for _ in range(60):
        optimizer.zero_grad()
        out_states = model(input_states)
        loss = 0
        for s in out_states:
            rho = state_vector_to_density_matrix(s)
            loss += torch.real(torch.trace(rho @ H_t))
        loss /= 3
        loss.backward()
        optimizer.step()
        trajectory_theta.append(model.theta.detach().clone())
        
    theta_opt = model.theta.detach().clone()
    dir1 = theta_init - theta_opt
    dist = torch.norm(dir1).item()
    if dist < 1e-5:
        dir1 = torch.randn_like(theta_opt)
        dir1 = dir1 / torch.norm(dir1)
        dist = 1.0
    else:
        dir1 = dir1 / dist
        
    torch.manual_seed(100)
    dir2 = torch.randn_like(theta_opt)
    dir2 = dir2 - torch.sum(dir1 * dir2) * dir1
    dir2 = dir2 / torch.norm(dir2)
    
    trajectory_alpha = []
    trajectory_beta = []
    for th in trajectory_theta:
        diff = th - theta_opt
        a = torch.sum(diff * dir1).item()
        b = torch.sum(diff * dir2).item()
        trajectory_alpha.append(a)
        trajectory_beta.append(b)
        
    grid_size = 40
    span = max(dist * 1.5, 2.5)
    alpha_range = np.linspace(-span, span, grid_size)
    beta_range = np.linspace(-span, span, grid_size)
    A, B = np.meshgrid(alpha_range, beta_range)
    Z = np.zeros((grid_size, grid_size))
    
    print(f"Scanning high-res landscape ({grid_size}x{grid_size}) around Minimum...")
    for i in range(grid_size):
        for j in range(grid_size):
            theta_scan = theta_opt + A[i, j] * dir1 + B[i, j] * dir2
            model.theta.data.copy_(theta_scan)
            out = model(input_states)
            l_val = sum(torch.real(torch.trace(state_vector_to_density_matrix(s) @ H_t)).item() for s in out)
            Z[i, j] = l_val / 3
            
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.35)
    
    ax1 = fig.add_subplot(gs[0])
    contour = ax1.contourf(A, B, Z, levels=50, cmap='magma')
    ax1.contour(A, B, Z, levels=15, colors='black', linewidths=0.5, alpha=0.5)
    ax1.plot(trajectory_alpha, trajectory_beta, 'wo-', lw=2.5, markeredgecolor='black', zorder=10, label='Gradient Trajectory')
    ax1.plot(trajectory_alpha[0], trajectory_beta[0], 'go', markeredgecolor='white', zorder=11, label='Initialization', ms=8)
    ax1.plot(trajectory_alpha[-1], trajectory_beta[-1], 'r*', markeredgecolor='white', zorder=11, label='Converged Minimum', ms=12)
    
    fig.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04).set_label('Energy Expectation $\\langle H \\rangle_{tfim}$')
    ax1.set_title('(a) Energy Landscape Slice (2D Topographic)')
    ax1.set_xlabel('Orthogonal Direction $\\mathbf{v}_1$')
    ax1.set_ylabel('Orthogonal Direction $\\mathbf{v}_2$')
    ax1.legend(loc='upper right')
    
    ax2 = fig.add_subplot(gs[1], projection='3d')
    surf = ax2.plot_surface(A, B, Z, cmap='magma', edgecolor='none', rstride=1, cstride=1, antialiased=True, shade=True)
    offset = np.min(Z) - (np.max(Z)-np.min(Z))*0.2
    ax2.contourf(A, B, Z, zdir='z', offset=offset, cmap='magma', alpha=0.5)
    ax2.scatter(trajectory_alpha[-1], trajectory_beta[-1], np.min(Z), color='red', marker='*', s=300, edgecolors='white', zorder=10, label='Global Minimum')
    
    ax2.set_title('(b) 3D Loss Surface Topology')
    ax2.set_xlabel('Direction $\\mathbf{v}_1$', labelpad=12)
    ax2.set_ylabel('Direction $\\mathbf{v}_2$', labelpad=12)
    ax2.set_zlabel('Energy $\\langle H \\rangle$', labelpad=12)
    ax2.set_zlim(offset, np.max(Z))
    ax2.view_init(elev=40, azim=45)
    
    fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=15, pad=0.1).set_label('Energy Basin Profile')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95))
    
    fig.suptitle(f"Quantum Parameter Space Geometry: Entanglement-Induced Non-Convexity\n({n_qubit}-Qubit TFIM, {model.n_layers}-Layer Hardware Efficient Ansatz)", y=1.05)
    path = os.path.join(output_dir, 'qfm_gradient_landscape.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gradient_landscape(args.qubits, args.out)

if __name__ == "__main__":
    main()
