import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from simulate_qfm import train_qfm_collect_all, COLORS
from src.qfm.metrics import von_neumann_entropy

def information_flow(rhos, n_qubit, T_step, output_dir):
    print("Generating Advanced Information Flow Diagram")
    def get_reduced_rho(rho_full, n):
        dim_A, dim_B = 2, 2**(n-1)
        rho_res = rho_full.view(dim_A, dim_B, dim_A, dim_B)
        rho_A = torch.einsum('i j k j -> i k', rho_res)
        rho_B = torch.einsum('i j i l -> j l', rho_res)
        return rho_A, rho_B

    entropies = []
    purities = []
    entanglement_A = []
    mutual_info = []

    for r in rhos:
        S_AB = float(von_neumann_entropy(r))
        gamma = float(torch.real(torch.trace(r @ r)))
        entropies.append(S_AB)
        purities.append(gamma)
        
        rho_A, rho_B = get_reduced_rho(r, n_qubit)
        S_A = float(von_neumann_entropy(rho_A))
        S_B = float(von_neumann_entropy(rho_B))
        
        entanglement_A.append(S_A)
        mutual_info.append(S_A + S_B - S_AB)

    steps = list(range(T_step + 1))
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, entropies, 'o-', color=COLORS['primary'], lw=3, ms=8, label='Global Entropy $S(\\rho_{AB})$')
    ax1.axhline(n_qubit * np.log(2), color='gray', linestyle='--', label=f'Max Mix ($n\\ln2={n_qubit * np.log(2):.2f}$)')
    ax1.set_title('(a) Global von Neumann Entropy')
    ax1.set_xlabel('Optimal Transport Time $\\tau$')
    ax1.set_ylabel('Entropy $S$')
    ax1.grid(True, ls='--')
    ax1.legend(loc='lower left')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, purities, 's-', color=COLORS['secondary'], lw=3, ms=8, label='State Purity $\\gamma$')
    ax2.axhline(1.0, color='gray', linestyle='--', label='Pure State $\\gamma=1$')
    ax2.axhline(1.0/(2**n_qubit), color='black', linestyle=':', label='Max Mixed State')
    ax2.set_title('(b) Manifold Purity Collapse & Recovery')
    ax2.set_xlabel('Optimal Transport Time $\\tau$')
    ax2.set_ylabel('Purity $\\mathrm{Tr}(\\rho^2)$')
    ax2.grid(True, ls='--')
    ax2.legend(loc='lower right')
    
    ax3 = fig.add_subplot(gs[1, 0])
    rate_S_A = np.gradient(entanglement_A)
    ax3.plot(steps, entanglement_A, 'D-', color=COLORS['tertiary'], lw=3, ms=8, label='Subsystem Entropy $S(\\rho_A)$')
    ax3_twin = ax3.twinx()
    ax3_twin.bar(steps, rate_S_A, color=COLORS['tertiary'], alpha=0.4, label='Entanglement Rate $\\partial S_A/\\partial \\tau$')
    ax3.axhline(np.log(2), color='gray', linestyle='--', label='Max Bipartite Entanglement')
    ax3.set_title('(c) Local Bipartite Entanglement Growth')
    ax3.set_xlabel('Optimal Transport Time $\\tau$')
    ax3.set_ylabel('Entanglement $S(\\rho_A)$', color=COLORS['tertiary'])
    ax3_twin.set_ylabel('Generation Rate', color=COLORS['tertiary'])
    ax3.grid(True, ls='--')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps, mutual_info, 'p-', color=COLORS['accent'], lw=3, ms=9, label='Mutual Information $I(A:B)$')
    ax4.fill_between(steps, mutual_info, color=COLORS['accent'], alpha=0.3)
    ax4.set_title('(d) Quantum Mutual Information Dynamics')
    ax4.set_xlabel('Optimal Transport Time $\\tau$')
    ax4.set_ylabel('Correlations $I(A:B)$')
    ax4.grid(True, ls='--')
    ax4.legend(loc='upper right')

    fig.suptitle("Quantum Flow Matching: Information-Theoretic and Entanglement Dynamics", y=1.02)
    path = os.path.join(output_dir, 'qfm_information_flow.png')
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
    information_flow(rhos, nq, T, args.out)

if __name__ == "__main__":
    main()
