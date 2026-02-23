"""
classical_baseline.py — Classical Flow Matching baseline for benchmarking against QFM.

Implements a classical conditional flow matching (CFM) model that operates on
vectorized density matrix representations (d² real parameters), providing
a fair apples-to-apples comparison with Quantum Flow Matching.

The classical model:
  - Parameterizes the time-dependent vector field v_θ(t, x) via an MLP
  - Uses the conditional flow matching loss (Lipman et al., 2022):
      L_CFM = E_{t,ρ₁,ρ₀} ||v_θ(t, ρ_t | ρ₁) - (ρ₁ - ρ₀)||²
  - Where ρ_t = (1-t)ρ₀ + t·ρ₁ (linear interpolation path)
  - Evaluated on the same transport task as QFM

Reference:
    Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
    Albergo & Vanden-Eijnden, "Building Normalizing Flows with Stochastic Interpolants", 2022.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, Callable

from .metrics import bures_distance, uhlmann_fidelity, von_neumann_entropy, purity


# ---------------------------------------------------------------------------
# Helper: density matrix ↔ real vector
# ---------------------------------------------------------------------------

def rho_to_vector(rho: torch.Tensor) -> torch.Tensor:
    """
    Flatten a complex density matrix to a real vector of dimension 2*d².
    Layout: [Re(rho.flat), Im(rho.flat)]

    Preserves all information while enabling standard MLP processing.
    """
    re = rho.real.flatten()
    im = rho.imag.flatten()
    return torch.cat([re, im]).float()


def vector_to_rho(v: torch.Tensor, d: int) -> torch.Tensor:
    """
    Reconstruct a Hermitian density matrix from a real vector.

    Enforces: Hermiticity, trace=1, positive semi-definiteness (via Cholesky projection).
    """
    d2 = d * d
    re = v[:d2].double().reshape(d, d)
    im = v[d2:2*d2].double().reshape(d, d)
    rho_raw = torch.complex(re, im)
    # Enforce Hermiticity
    rho_h = 0.5 * (rho_raw + rho_raw.conj().T)
    # Shift to PSD via eigenvalue projection
    ev, V = torch.linalg.eigh(rho_h)
    ev_pos = torch.clamp(ev.real, min=0.0).to(torch.complex128)
    rho_psd = (V * ev_pos.unsqueeze(0)) @ V.conj().T
    # Normalize trace
    tr = torch.real(torch.trace(rho_psd))
    if tr > 1e-12:
        rho_psd = rho_psd / tr
    return rho_psd


# ---------------------------------------------------------------------------
# Classical MLP Vector Field
# ---------------------------------------------------------------------------

class ClassicalVectorField(nn.Module):
    """
    Time-conditioned MLP vector field v_θ(t, x) for Classical Flow Matching.

    Maps: (t, x) → dx/dt
    where x ∈ R^{2d²} is the vectorized density matrix representation.

    Architecture: Input(2d²+1) → [hidden]×n_layers → Output(2d²)
    with GELU activations and LayerNorm for stability.
    """

    def __init__(
        self,
        d: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.d   = d
        self.d2  = d * d
        input_dim = 2 * self.d2 + 1  # vectorized rho + time

        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        layers += [nn.Linear(hidden_dim, 2 * self.d2)]

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time scalar or (B,1) tensor ∈ [0,1].
            x: Vectorized state (B, 2d²) or (2d²,).

        Returns:
            Vector field v_θ(t, x) of shape (B, 2d²) or (2d²,).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if isinstance(t, (float, int)):
            t = torch.tensor([[t]], dtype=torch.float32).expand(x.shape[0], 1)
        elif t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1)
        inp = torch.cat([x, t.float()], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Conditional Flow Matching Loss
# ---------------------------------------------------------------------------

def cfm_loss(
    model: ClassicalVectorField,
    rhos_0: List[torch.Tensor],
    rhos_1: List[torch.Tensor],
    n_time_samples: int = 10,
) -> torch.Tensor:
    """
    Conditional Flow Matching (CFM) loss:
        L = E_{t, (ρ₀,ρ₁)} ||v_θ(t, ρ_t) - (ρ₁ - ρ₀)||²

    Uses straight-line paths: ρ_t = (1-t)ρ₀ + t·ρ₁.

    Args:
        model:         ClassicalVectorField.
        rhos_0:        Source density matrices.
        rhos_1:        Target density matrices.
        n_time_samples: Number of random time samples per pair.

    Returns:
        Scalar loss tensor (differentiable).
    """
    d = rhos_0[0].shape[0]
    total_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    for rho_0, rho_1 in zip(rhos_0, rhos_1):
        x0 = rho_to_vector(rho_0)
        x1 = rho_to_vector(rho_1)
        u_star = x1 - x0  # target vector field direction

        for _ in range(n_time_samples):
            t = torch.rand(1).item()
            x_t = (1.0 - t) * x0 + t * x1
            v_pred = model(t, x_t.unsqueeze(0)).squeeze(0)
            total_loss = total_loss + torch.mean((v_pred - u_star.float()) ** 2)

    return total_loss / (len(rhos_0) * n_time_samples)


# ---------------------------------------------------------------------------
# Classical FM Trainer
# ---------------------------------------------------------------------------

class ClassicalFlowMatchingBaseline:
    """
    End-to-end Classical Flow Matching pipeline on density matrices.

    Trains v_θ to learn the map from source to target density matrix,
    then generates trajectories via Euler integration of the ODE dx/dt = v_θ(t, x).
    """

    def __init__(
        self,
        d: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 3,
        lr: float = 1e-3,
    ):
        self.d     = d
        self.model = ClassicalVectorField(d, hidden_dim, n_hidden_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.train_losses: List[float] = []

    def train(
        self,
        rhos_0: List[torch.Tensor],
        rhos_1: List[torch.Tensor],
        n_epochs: int = 300,
        n_time_samples: int = 8,
        verbose: bool = False,
    ) -> List[float]:
        """
        Train the classical FM model.

        Returns:
            List of per-epoch losses.
        """
        self.model.train()
        losses = []
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            loss = cfm_loss(self.model, rhos_0, rhos_1, n_time_samples)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            losses.append(float(loss.item()))
            if verbose and epoch % 50 == 0:
                print(f"  CFM Epoch {epoch:4d} | Loss: {loss.item():.6f}")
        self.train_losses = losses
        return losses

    @torch.no_grad()
    def generate_trajectory(
        self,
        rho_0: torch.Tensor,
        T_steps: int = 20,
    ) -> List[torch.Tensor]:
        """
        Generate a trajectory from rho_0 to predicted rho_T using Euler integration.

        dx/dt = v_θ(t, x)  →  x_{t+1} = x_t + Δt · v_θ(t, x_t)

        Returns:
            List of T_steps+1 density matrices.
        """
        self.model.eval()
        d = self.d
        x = rho_to_vector(rho_0).float()
        trajectory = [rho_0]
        dt = 1.0 / T_steps

        for step in range(T_steps):
            t = step * dt
            v = self.model(t, x.unsqueeze(0)).squeeze(0)
            x = x + dt * v
            rho_new = vector_to_rho(x.double(), d)
            trajectory.append(rho_new)

        return trajectory


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

def compare_qfm_vs_cfm(
    qfm_trajectory: List[torch.Tensor],
    cfm_trajectory: List[torch.Tensor],
    rho_target: torch.Tensor,
    qfm_n_params: int,
    cfm_n_params: int,
) -> Dict[str, object]:
    """
    Head-to-head benchmark comparison between QFM and Classical FM.

    Metrics:
        - Final Uhlmann fidelity with target
        - Final Bures distance to target
        - Total trajectory action (transport cost)
        - Parameter efficiency (final_fidelity / n_params)
        - Von Neumann entropy evolution (quantum coherence preservation)

    Args:
        qfm_trajectory:  List of density matrices from QFM.
        cfm_trajectory:  List of density matrices from CFM.
        rho_target:      Target density matrix.
        qfm_n_params:    Number of trainable parameters in QFM.
        cfm_n_params:    Number of trainable parameters in CFM.

    Returns:
        Comprehensive comparison dict.
    """
    def _metrics(traj, name):
        final_fid  = float(uhlmann_fidelity(traj[-1], rho_target))
        final_bures = float(bures_distance(traj[-1], rho_target))
        action     = sum(
            float(bures_distance(traj[t], traj[t+1]))**2
            for t in range(len(traj) - 1)
        )
        entropy_curve = [float(von_neumann_entropy(r)) for r in traj]
        purity_curve  = [float(purity(r)) for r in traj]
        return {
            f'{name}_final_fidelity':  final_fid,
            f'{name}_final_bures':     final_bures,
            f'{name}_action':          action,
            f'{name}_entropy_curve':   entropy_curve,
            f'{name}_purity_curve':    purity_curve,
        }

    result = {}
    result.update(_metrics(qfm_trajectory, 'qfm'))
    result.update(_metrics(cfm_trajectory, 'cfm'))

    result['qfm_n_params']     = qfm_n_params
    result['cfm_n_params']     = cfm_n_params
    result['qfm_param_efficiency'] = result['qfm_final_fidelity'] / (qfm_n_params + 1)
    result['cfm_param_efficiency'] = result['cfm_final_fidelity'] / (cfm_n_params + 1)
    result['fidelity_advantage']   = result['qfm_final_fidelity'] - result['cfm_final_fidelity']
    result['action_advantage']     = result['cfm_action'] - result['qfm_action']

    return result
