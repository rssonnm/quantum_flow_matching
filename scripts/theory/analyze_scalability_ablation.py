import argparse
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.qfm.trainer import QFMTrainer
from src.qfm.apps.tfim import build_tfim_hamiltonian, loss_fn_energy
from src.qfm.utils import state_vector_to_density_matrix, partial_trace_pure_to_mixed
from src.qfm.metrics import uhlmann_fidelity, bures_distance, von_neumann_entropy
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 13, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

COLORS = {'blue': "C0", 'red': "C9", 'green': "C8",
          'amber': "C5", 'purple': "C4", 'gray': "C0"}

def build_heisenberg_xxz(n_qubits, delta=1.0, Jxy=1.0):
    
    d = 2**n_qubits
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)
    Y = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex128)
    Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
    I = torch.eye(2, dtype=torch.complex128)

    H = torch.zeros((d, d), dtype=torch.complex128)
    for i in range(n_qubits - 1):
        ops_X = [X if k in (i, i+1) else I for k in range(n_qubits)]
        ops_Y = [Y if k in (i, i+1) else I for k in range(n_qubits)]
        ops_Z = [Z if k in (i, i+1) else I for k in range(n_qubits)]

        XX = ops_X[0]; YY = ops_Y[0]; ZZ = ops_Z[0]
        for k in range(1, n_qubits):
            XX = torch.kron(XX, ops_X[k])
            YY = torch.kron(YY, ops_Y[k])
            ZZ = torch.kron(ZZ, ops_Z[k])
        H += Jxy * (XX + YY) + delta * ZZ
    return H

def build_random_hamiltonian(n_qubits, seed=42):
    
    torch.manual_seed(seed)
    d = 2**n_qubits
    A = torch.randn((d, d), dtype=torch.complex128)
    H = (A + A.conj().T) / 2.0
    ev = torch.linalg.eigvalsh(H)
    return H * 4.0 / (ev.max() - ev.min())

def train_qfm_quick(n_qubits, T_steps, n_layers, M, H_source, H_target, threshold=0.05):
    
    ev0, evec0 = torch.linalg.eigh(H_source)
    gs0 = evec0[:, 0]
    initial = torch.stack([gs0 for _ in range(M)])

    ev_t, evec_t = torch.linalg.eigh(H_target)
    rho_target = state_vector_to_density_matrix(evec_t[:, 0])

    trainer = QFMTrainer(
        n_data=n_qubits, n_ancilla=0, n_layers=n_layers,
        T_steps=T_steps, threshold=threshold, lr=0.1
    )

    t0 = time.time()
    current = initial
    prev = None
    for tau in range(1, T_steps + 1):
        g_tau = 1.0 * tau / T_steps
        H_tau = (1 - g_tau) * H_source + g_tau * H_target
        current, _, prev = trainer.train_step(
            tau, current, lambda r, t: loss_fn_energy(r, t),
            lambda t: H_tau, max_epochs=40, prev_model=prev
        )
    elapsed = time.time() - t0
    rho_final = torch.stack([state_vector_to_density_matrix(s) for s in current]).mean(dim=0)
    fid = float(uhlmann_fidelity(rho_final, rho_target))
    n_params = sum(p.numel() for m, _ in trainer.models for p in m.parameters())
    return fid, elapsed, n_params, rho_final

def train_qfm_tracked(n_qubits, T_steps, n_layers, M, H_source, H_target, threshold=0.05, max_epochs=40, lr=0.1):
    
    ev0, evec0 = torch.linalg.eigh(H_source)
    gs0 = evec0[:, 0]
    initial = torch.stack([gs0 for _ in range(M)])

    ev_t, evec_t = torch.linalg.eigh(H_target)
    rho_target_final = state_vector_to_density_matrix(evec_t[:, 0])

    trainer = QFMTrainer(
        n_data=n_qubits, n_ancilla=0, n_layers=n_layers,
        T_steps=T_steps, threshold=threshold, lr=lr
    )

    t0 = time.time()
    current = initial
    prev = None
    
    history = {'tau': [], 'loss': [], 'fidelity': [], 'entropy': []}
    
    for tau in range(1, T_steps + 1):
        g_tau = 1.0 * tau / T_steps
        H_tau = (1 - g_tau) * H_source + g_tau * H_target
        current, losses, prev = trainer.train_step(
            tau, current, lambda r, t: loss_fn_energy(r, t),
            lambda t: H_tau, max_epochs=max_epochs, prev_model=prev
        )
        rho_current = torch.stack([state_vector_to_density_matrix(s) for s in current]).mean(dim=0)
        ev_tau, evec_tau = torch.linalg.eigh(H_tau)
        rho_target_tau = state_vector_to_density_matrix(evec_tau[:, 0])
        
        fid = float(uhlmann_fidelity(rho_current, rho_target_tau))
        ent = float(von_neumann_entropy(rho_current))
        
        history['tau'].append(tau)
        history['loss'].append(losses[-1] if losses else 0.0)
        history['fidelity'].append(fid)
        history['entropy'].append(ent)
        
    elapsed = time.time() - t0
    rho_final = torch.stack([state_vector_to_density_matrix(s) for s in current]).mean(dim=0)
    final_fid = float(uhlmann_fidelity(rho_final, rho_target_final))
    n_params = sum(p.numel() for m, _ in trainer.models for p in m.parameters())
    return final_fid, elapsed, n_params, history

def scalability(output_dir):
    
    print("Starting Advanced Scalability Analysis (n=2..6)")
            
    qubit_range = [2, 3, 4, 5, 6]
    fidelities, times, params = [], [], []
    losses = []

    for nq in qubit_range:
        print(f"Benching n={nq}")
        H_s = build_tfim_hamiltonian(nq, 0.0)
        H_t = build_tfim_hamiltonian(nq, 1.0)
        T_opt = 10 + 2 * (nq - 2)
        layers = 4 + (nq - 2)
        epochs = 40 + 15 * (nq - 2)
        fid, t, p, hist = train_qfm_tracked(nq, T_opt, layers, 10, H_s, H_t, max_epochs=epochs, lr=0.05)
        
        fidelities.append(fid)
        times.append(t)
        params.append(p)
        losses.append(hist['loss'][-1] if hist['loss'] else 0)

    fig = plt.figure(figsize=(18, 5.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(qubit_range, fidelities, color=COLORS['blue'], edgecolor='black')
    ax1.axhline(0.99, color='gray', linestyle='--', label='99% Adequacy')
    for bar in bars:
        yval = bar.get_height()
        offset = 0.015 if yval > 0.1 else 0.05
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f'{yval:.3f}', ha='center', va='bottom')
        
    min_f = min(fidelities) if fidelities else 0.5
    ax1.set_ylim(max(0.0, min_f - 0.1), 1.05)
    ax1.set_xlabel('System Size (Qubits $n$)')
    ax1.set_ylabel('Final State Fidelity $\\mathcal{F}$')
    ax1.set_title('(a) Scalability of Target Fidelity')
    ax1.set_xticks(qubit_range)
    ax1.grid(axis='y', ls='--')
    ax1.legend(loc='lower right')
    ax2 = fig.add_subplot(gs[1])
    from scipy.optimize import curve_fit
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    
    popt, _ = curve_fit(exp_func, qubit_range, times, p0=(1, 0.5))
    x_fit = np.linspace(min(qubit_range), max(qubit_range), 50)
    y_fit = exp_func(x_fit, *popt)
    
    ax2.plot(qubit_range, times, 's', color=COLORS['amber'], label='Measured $T_{wall}$')
    ax2.plot(x_fit, y_fit, 'r--', lw=2, label=f'Fit: $\\mathcal{{O}}(e^{{{popt[1]:.2f}n}})$')
    
    ax2.set_xlabel('System Size (Qubits $n$)')
    ax2.set_ylabel('Wall-Clock Training Time (s)')
    ax2.set_title('(b) Empirical Time Complexity')
    ax2.set_xticks(qubit_range)
    ax2.grid(True, ls='--')
    ax2.legend(loc='upper left')
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(qubit_range, losses, 'D-', color=COLORS['red'], lw=3)
    
    ax3.set_xlabel('System Size (Qubits $n$)')
    ax3.set_ylabel('Converged Final Energy $\\langle H \\rangle$')
    ax3.set_title('(c) Ground State Energy Saturation')
    ax3.set_xticks(qubit_range)
    ax3.grid(True, ls='--')
    ax4 = fig.add_subplot(gs[3])
    hilbert_dims = [2**nq for nq in qubit_range]
    classical_params = [d**2 for d in hilbert_dims] # O(d^2) for classical density matrix tracking
    
    ax4.plot(qubit_range, classical_params, 'o--', color='gray', lw=2, label='Classical Memory $\\mathcal{O}(2^{2n})$')
    ax4.plot(qubit_range, params, '^-', color=COLORS['green'], lw=3, label='QFM Parameters $\\mathcal{O}(\\text{poly}(n))$')
    ax4.fill_between(qubit_range, params, classical_params, color='lightgreen', label='Exponential Advantage Zone')
    
    ax4.set_yscale('log')
    ax4.set_xlabel('System Size (Qubits $n$)')
    ax4.set_ylabel('Model Dimensionality (Log Scale)')
    ax4.set_title('(d) Fundamental Parameter Efficiency')
    ax4.set_xticks(qubit_range)
    ax4.grid(True, ls='--')
    ax4.legend(loc='upper left')

    fig.suptitle('Scalability Characteristics of Quantum Flow Matching: Performance vs Complexity bounds', y=1.05)

    path = os.path.join(output_dir, 'qfm_scalability.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved highly detailed Q1 scalability figure to {path}")

def ablation(output_dir):
    
    print("Starting Advanced Annotated Ablation Study")
            
    nq = 3; 
    H_s = build_tfim_hamiltonian(nq, 0.0); 
    H_t = build_tfim_hamiltonian(nq, 1.0)
    T_range = [3, 5, 10, 15]       
    L_range = [1, 2, 4, 6]         
    M_range = [1, 5, 10, 20]       
    fid_T = [train_qfm_quick(nq, T, 4, 10, H_s, H_t)[0] for T in T_range]
    fid_L = [train_qfm_quick(nq, 10, L, 10, H_s, H_t)[0] for L in L_range]
    fid_M = [train_qfm_quick(nq, 10, 4, M, H_s, H_t)[0] for M in M_range]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    def annotate_points(ax, x_data, y_data):
        for x, y in zip(x_data, y_data):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color="black", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray"))

    def setup_ax(ax, title, xlabel, y_min, y_max):
        ax.axhline(0.99, color="gray", ls="--", label="99% Threshold")
        ax.axhspan(0.99, max(1.05, y_max + 0.05), facecolor="C6", label="Success Zone (>99%)")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, ls="--")
        lower_bound = min(y_min - 0.05, 0.45)
        upper_bound = max(y_max + 0.05, 1.05)
        ax.set_ylim(lower_bound, upper_bound)
    all_fids = fid_T + fid_L + fid_M
    y_min, y_max = min(all_fids), max(all_fids)
    ax = axes[0]
    ax.plot(T_range, fid_T, "o-", color="C1", lw=2.5, label="$\\mathcal{F}(\\rho_{\\mathrm{QFM}}, \\rho_{\\mathrm{target}})$")
    setup_ax(ax, "(a) Convergence vs Time Grid Resolution ($T$)", "Integration Steps $T$", min(fid_T), max(fid_T))
    ax.set_ylabel("Final State Fidelity $\\mathcal{F}$")
    annotate_points(ax, T_range, fid_T)
    ax.legend(loc="lower right")
    ax.set_xticks(T_range)
    ax = axes[1]
    ax.plot(L_range, fid_L, "s-", color="C3", lw=2.5, label="$\\mathcal{F}(\\rho_{\\mathrm{QFM}}, \\rho_{\\mathrm{target}})$")
    setup_ax(ax, "(b) Expressibility vs Ansatz Depth ($L$)", "EHA Layers $L$", min(fid_L), max(fid_L))
    annotate_points(ax, L_range, fid_L)
    ax.legend(loc="lower right")
    ax.set_xticks(L_range)
    ax = axes[2]
    ax.plot(M_range, fid_M, "^-", color="C1", lw=2.5, label="$\\mathcal{F}(\\rho_{\\mathrm{QFM}}, \\rho_{\\mathrm{target}})$")
    setup_ax(ax, "(c) Robustness vs Quantum Ensemble Size ($M$)", "Ensemble States $M$", min(fid_M), max(fid_M))
    annotate_points(ax, M_range, fid_M)
    ax.legend(loc="lower right")
    ax.set_xticks(M_range)

    fig.suptitle("Quantum Flow Matching: Ablation Study Results (Exact Final Fidelity Values)", y=1.05)
                 
    fig.tight_layout()
    path = os.path.join(output_dir, 'qfm_ablation.png')
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved advanced annotated ablation figure to {path}")

def multi_hamiltonian(output_dir):
    
    print("Starting Multi-Hamiltonian Advanced Benchmark")
    
    nq = 3; T = 15; M = 10
    names = ['TFIM', 'Heisenberg XXZ', 'Random GUE']
    colors = [COLORS['blue'], COLORS['red'], COLORS['green']]
    
    results = {}
    fids = []
    H_s_tfim = build_tfim_hamiltonian(nq, 0.0)
    H_t_tfim = build_tfim_hamiltonian(nq, 1.0)
    fid1, _, _, hist1 = train_qfm_tracked(nq, T, 4, M, H_s_tfim, H_t_tfim)
    results['TFIM'] = hist1
    fids.append(fid1)
    
    H_s_xxz = build_heisenberg_xxz(nq, 0.0)
    H_t_xxz = build_heisenberg_xxz(nq, 1.0)
    fid2, _, _, hist2 = train_qfm_tracked(nq, T, 4, M, H_s_xxz, H_t_xxz)
    results['Heisenberg XXZ'] = hist2
    fids.append(fid2)
    
    H_s_rand = build_random_hamiltonian(nq, 42)
    H_t_rand = build_random_hamiltonian(nq, 99)
    fid3, _, _, hist3 = train_qfm_tracked(nq, T, 4, M, H_s_rand, H_t_rand)
    results['Random GUE'] = hist3
    fids.append(fid3)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(names, fids, color=colors, edgecolor='black')
    axes[0, 0].set_ylim(max(0.0, min(fids) - 0.1), 1.02)
    axes[0, 0].axhline(1.0, color='gray', linestyle='--')
    axes[0, 0].set_ylabel('Target State Fidelity $\\mathcal{F}$')
    axes[0, 0].set_title('(a) Final State Fidelity by Hamiltonian Class')
    for i, v in enumerate(fids):
        axes[0, 0].text(i, v + 0.01, f"{v:.4f}", ha='center')
    for name, c in zip(names, colors):
        axes[0, 1].plot(results[name]['tau'], results[name]['loss'], 'o-', color=c, lw=2, label=name)
    axes[0, 1].set_xlabel('Flow Step ($\\tau$)')
    axes[0, 1].set_ylabel('Final Epoch Loss')
    axes[0, 1].set_title('(b) Optimization Convergence Tracking')
    axes[0, 1].grid(True, linestyle='--')
    axes[0, 1].legend()
    for name, c in zip(names, colors):
        axes[1, 0].plot(results[name]['tau'], results[name]['fidelity'], 's-', color=c, lw=2, label=name)
    axes[1, 0].set_xlabel('Flow Step ($\\tau$)')
    axes[1, 0].set_ylabel('Instantaneous Fidelity $\\mathcal{F}_\\tau$')
    axes[1, 0].set_title('(c) Target Tracking Fidelity Over Time')
    axes[1, 0].grid(True, linestyle='--')
    for name, c in zip(names, colors):
        axes[1, 1].plot(results[name]['tau'], results[name]['entropy'], '^-', color=c, lw=2, label=name)
    axes[1, 1].set_xlabel('Flow Step ($\\tau$)')
    axes[1, 1].set_ylabel('Von Neumann Entropy $S(\\rho)$')
    axes[1, 1].set_title('(d) Entanglement Generation Dynamics')
    axes[1, 1].grid(True, linestyle='--')

    fig.suptitle('Generalizability of Quantum Flow Matching Across Hamiltonian Classes', y=1.02)
    fig.tight_layout()
    
    path = os.path.join(output_dir, 'qfm_multi_hamiltonian.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved advanced multi-hamiltonian figure to {path}")

def qsl(output_dir):
    
    print("Starting Advanced QSL Analysis")
            
    nq = 3
    H_s = build_tfim_hamiltonian(nq, 0.0)
    H_t = build_tfim_hamiltonian(nq, 1.0)
    T_steps = 15
    trainer = QFMTrainer(n_data=nq, n_ancilla=0, n_layers=4, T_steps=T_steps, lr=0.08)
    ev0, evec0 = torch.linalg.eigh(H_s)
    current = torch.stack([evec0[:, 0] for _ in range(12)]) # M=12
    
    rhos = []
    rho_t_final = state_vector_to_density_matrix(torch.linalg.eigh(H_t)[1][:, 0])
    rhos.append(sum(state_vector_to_density_matrix(c) for c in current) / 12)
    
    prev = None
    for tau in range(1, T_steps + 1):
        g_tau = tau / T_steps
        H_tau = (1 - g_tau) * H_s + g_tau * H_t
        current, _, prev = trainer.train_step(
            tau, current, lambda r, t: loss_fn_energy(r, t), lambda t: H_tau, max_epochs=50, prev_model=prev
        )
        rho_tau = sum(state_vector_to_density_matrix(c) for c in current) / 12
        rhos.append(rho_tau)
    bures_vel = []
    fidelities = []
    energy_variances = []
    entropies = []
    
    for i in range(len(rhos)-1):
        r1 = rhos[i]
        r2 = rhos[i+1]
        dist = float(bures_distance(r1, r2))
        bures_vel.append(dist)
        fidelities.append(float(uhlmann_fidelity(r1, rho_t_final)))
        entropies.append(float(von_neumann_entropy(r1)))
        g_i = (i + 0.5) / T_steps
        H_i = (1 - g_i) * H_s + g_i * H_t
        H_sq = H_i @ H_i
        
        E = float(torch.real(torch.trace(r1 @ H_i)))
        E_sq = float(torch.real(torch.trace(r1 @ H_sq)))
        var = np.sqrt(max(0.0, E_sq - E**2))
        energy_variances.append(var)
        
    entropies.append(float(von_neumann_entropy(rhos[-1])))
    
    bures_vel = np.array(bures_vel)
    energy_variances = np.array(energy_variances)
    entropies = np.array(entropies)[:-1] # match velocity length
    
    bures_length = np.cumsum(bures_vel)
    action_limit = np.cumsum(energy_variances)
    steps = np.arange(1, T_steps + 1)
    
    fig = plt.figure(figsize=(19, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)
    
    c_vel, c_mt = "C2", "C7"
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, energy_variances, 'o-', color=c_mt, lw=3, label='MT Limit (Energy Var $\\Delta E$)')
    ax1.plot(steps, bures_vel, 's-', color=c_vel, lw=3, label='Operation Velocity $v_{Bures}$')
    
    ax1.fill_between(steps, bures_vel, energy_variances, color='gray', label='Kinetic Freedom Gap')
    
    ax1.set_xlabel('Integration Step $\\tau$')
    ax1.set_ylabel('Quantum Information Rate')
    ax1.set_title('(a) Mandelstam-Tamm Speed Envelope')
    ax1.set_ylim(0, max(energy_variances) * 1.3)
    ax1.grid(True, ls='--')
    ax1.legend(loc='lower center')
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, action_limit, '^--', color=c_mt, lw=3, label='Cumulative MT Action $\\Phi(\\tau)$')
    ax2.plot(steps, bures_length, 'D-', color=COLORS['green'], lw=3, label='Actual Path Length $\\mathcal{L}(\\tau)$')
    efficiency = bures_length[-1] / (action_limit[-1] + 1e-9)
    ax2.text(0.05, 0.85, f'Geodesic Efficiency:\n$\\eta = {efficiency * 100:.1f}\\%$', transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white"))
    
    ax2.set_xlabel('Integration Step $\\tau$')
    ax2.set_ylabel('Information Distance / Action')
    ax2.set_title('(b) Transport Geodesic Efficiency')
    ax2.set_ylim(0, max(action_limit) * 1.2)
    ax2.grid(True, ls='--')
    ax2.legend(loc='lower right')
    ax3 = fig.add_subplot(gs[2])
    scatter = ax3.scatter(entropies, bures_vel, c=steps, cmap='plasma', s=120, edgecolor='black', zorder=5)
    z = np.polyfit(entropies, bures_vel, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(entropies), max(entropies), 10)
    ax3.plot(x_trend, p(x_trend), 'k--', lw=2, label='Empirical Trend')
    
    cbar = fig.colorbar(scatter, ax=ax3, pad=0.02)
    cbar.set_label('Time Step $\\tau$')
    
    ax3.set_xlabel('Von Neumann Entropy $S(\\rho)$')
    ax3.set_ylabel('Manifold Velocity $v_{Bures}$')
    ax3.set_title('(c) Phase-Space Acceleration Mechanics')
    ax3.grid(True, ls='--')
    ax3.legend(loc='upper right')

    fig.suptitle('Quantum Speed Limit & Information Geometry of QFM Optimization Pathways', y=1.05)

    path = os.path.join(output_dir, 'qfm_speed_limit.png')
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Advanced QSL figure to {path}")

def main():
    parser = argparse.ArgumentParser(description="Professionalized QFM Scalability Benchmarks.")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scalability(args.out)
    ablation(args.out)
    multi_hamiltonian(args.out)
    qsl(args.out)

if __name__ == "__main__":
    main()
