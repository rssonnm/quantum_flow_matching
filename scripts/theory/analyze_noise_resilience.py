import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.metrics import uhlmann_fidelity, bures_distance, von_neumann_entropy
from src.qfm.noise_models import apply_noise

def noise_resilience(n_qubit, T_step, rhos, output_dir):
    print("Generating Advanced Noise Resilience Dashboard")
    noise_rates = np.linspace(0.0, 0.3, 15)
    steps = np.arange(T_step + 1)
    
    fid_dep = np.zeros((len(noise_rates), T_step + 1))
    fid_amp = np.zeros((len(noise_rates), T_step + 1))
    for i, p in enumerate(noise_rates):
        for t in range(T_step + 1):
            noisy_dep = apply_noise(rhos[t].unsqueeze(0), "depolarizing", float(p), n_qubit)[0]
            fid_dep[i, t] = float(uhlmann_fidelity(noisy_dep, rhos[t]))
            noisy_amp = apply_noise(rhos[t].unsqueeze(0), "amplitude_damping", float(p), n_qubit)[0]
            fid_amp[i, t] = float(uhlmann_fidelity(noisy_amp, rhos[t]))
            
    entropies = [float(von_neumann_entropy(r)) for r in rhos]
    fixed_p_idx = len(noise_rates) // 2
    fixed_p = noise_rates[fixed_p_idx]
    vulnerability_dep = [float(bures_distance(apply_noise(rhos[t].unsqueeze(0), "depolarizing", float(fixed_p), n_qubit)[0], rhos[t])) for t in range(T_step+1)]
    
    fig = plt.figure(figsize=(19, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.38)
    cmap = 'turbo'
    
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.pcolormesh(steps, noise_rates, fid_dep, cmap=cmap, shading='gouraud', vmin=0.4, vmax=1.0)
    ax1.contour(steps, noise_rates, fid_dep, levels=[0.8, 0.9, 0.95, 0.99], colors='white', linewidths=1.5, linestyles='dashed')
    ax1.set_title('(a) Depolarizing Channel $\\mathcal{E}_{dep}(\\rho; p)$')
    ax1.set_xlabel('Integration Step $\\tau$')
    ax1.set_ylabel('Error Rate $p$')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label('State Fidelity $\\mathcal{F}$')
    
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.pcolormesh(steps, noise_rates, fid_amp, cmap=cmap, shading='gouraud', vmin=0.4, vmax=1.0)
    ax2.contour(steps, noise_rates, fid_amp, levels=[0.8, 0.9, 0.95, 0.99], colors='white', linewidths=1.5, linestyles='dashed')
    ax2.set_title('(b) Amplitude Damping $\\mathcal{E}_{amp}(\\rho; \\gamma)$')
    ax2.set_xlabel('Integration Step $\\tau$')
    ax2.set_ylabel('Damping Rate $\\gamma$')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label('State Fidelity $\\mathcal{F}$')
    
    ax3 = fig.add_subplot(gs[2])
    color1 = COLORS['secondary']
    ax3.plot(steps, vulnerability_dep, 's-', color=color1, lw=3, label=f'Bures Error $D_B$ (at $p={fixed_p:.2f}$)')
    ax3.set_xlabel('Integration Step $\\tau$')
    ax3.set_ylabel('Noise-Induced Bures Dist $D_B$', color=color1)
    ax3.fill_between(steps, 0, vulnerability_dep, color=color1, alpha=0.3)
    
    ax4 = ax3.twinx()
    color2 = COLORS['primary']
    ax4.plot(steps, entropies, 'o-', color=color2, lw=3, label='Von Neumann Entropy $S(\\rho)$')
    ax4.set_ylabel('Entanglement $S(\\rho)$', color=color2)
    
    corr_start = T_step // 2
    ax3.axvspan(corr_start, T_step, color='gray', alpha=0.2)
    
    lines_1, labels_1 = ax3.get_legend_handles_labels()
    lines_2, labels_2 = ax4.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    ax3.set_title('(c) Physical Origin of Noise Vulnerability')
    ax3.grid(True, linestyle='--')
    
    fig.suptitle('Hardware Resilience of Quantum Flow Matching: Decoherence Impact Analysis', y=1.05)
    path = os.path.join(output_dir, 'qfm_noise_resilience.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
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
    
    _, rhos, _, _, _, _, nq, T = train_qfm_collect_all(n_qubits=args.qubits, T_steps=args.steps, M=args.ensemble)
    noise_resilience(nq, T, rhos, args.out)

if __name__ == "__main__":
    main()
