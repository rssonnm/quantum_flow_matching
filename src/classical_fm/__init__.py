"""
classical_fm — Classical Flow Matching (OT-CFM) baseline implementation.
"""
import logging
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class VectorFieldMLP(nn.Module):
    """Time-conditioned MLP modeling the flow vector field v_θ(x, t) ∈ R²."""

    def __init__(self, dim: int = 2, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t_emb = t.unsqueeze(-1)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)


def sample_gaussian(n: int, dim: int = 2, device="cpu") -> torch.Tensor:
    return torch.randn(n, dim, device=device)


def sample_ring(n: int, radius: float = 3.0, sigma: float = 0.3,
                device="cpu") -> torch.Tensor:
    """Sample n points from a ring (annulus) distribution."""
    angles = torch.rand(n, device=device) * 2 * np.pi
    r = radius + sigma * torch.randn(n, device=device)
    x = r * torch.cos(angles)
    y = r * torch.sin(angles)
    return torch.stack([x, y], dim=1)


def minibatch_ot_coupling(x0: torch.Tensor, x1: torch.Tensor) -> tuple:
    """
    Minibatch OT coupling via the Hungarian algorithm on squared Euclidean cost.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        C = torch.cdist(x0, x1) ** 2
        i_, j_ = linear_sum_assignment(C.detach().cpu().numpy())
        return x0[i_], x1[j_]
    except Exception:
        return x0, x1


def cfm_loss(model: VectorFieldMLP, x0: torch.Tensor, x1: torch.Tensor,
             sigma_min: float = 1e-4) -> torch.Tensor:
    """
    Conditional Flow Matching loss with linear interpolant:
        L = E_{t,x0,x1} || v_θ(x_t, t) - (x1 - x0) ||²
    """
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device)
    t_b = t.unsqueeze(-1)
    x_t = (1.0 - t_b) * x0 + t_b * x1
    target_v = x1 - x0
    pred_v = model(x_t, t)
    return ((pred_v - target_v) ** 2).sum(dim=-1).mean()


class CFMTrainer:
    """Classical OT-CFM trainer."""

    def __init__(self, dim=2, hidden=256, n_layers=4, lr=1e-3, device="cpu"):
        self.model = VectorFieldMLP(dim, hidden, n_layers).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.model.train()

    def train(self, n_iter=3000, batch=512, use_ot=True,
              log_every=200) -> list:
        losses = []
        for step in range(n_iter):
            x0 = sample_gaussian(batch, device=self.device)
            x1 = sample_ring(batch, device=self.device)
            if use_ot:
                x0, x1 = minibatch_ot_coupling(x0, x1)
            self.opt.zero_grad()
            loss = cfm_loss(self.model, x0, x1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            losses.append(float(loss))
            if step % log_every == 0 or step == n_iter - 1:
                logger.info(f"[CFM] step {step:4d}/{n_iter}  loss={loss:.5f}")
        return losses

    @torch.no_grad()
    def sample(self, n: int = 1000, n_steps: int = 50) -> tuple:
        """
        Generate samples by integrating the learned ODE via Euler method.
        Returns (trajectories, final_samples).
        """
        self.model.eval()
        x = sample_gaussian(n, device=self.device)
        ts = torch.linspace(0, 1, n_steps + 1, device=self.device)
        dt = 1.0 / n_steps
        trajectories = [x.clone()]
        for i in range(n_steps):
            t_cur = ts[i].expand(n)
            v = self.model(x, t_cur)
            x = x + dt * v
            trajectories.append(x.clone())
        self.model.train()
        return trajectories, x
