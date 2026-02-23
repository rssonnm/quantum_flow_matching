"""
optimizers.py — Quantum-aware optimizers for Flow Matching.

Implements Quantum Natural Gradient (QNG) and other Riemannian 
optimization techniques for quantum state manifolds.
"""
import torch
import torch.optim as optim
import numpy as np
from .qfim import compute_qfim_ensemble

class QuantumNaturalGradient(optim.Optimizer):
    """
    Quantum Natural Gradient (QNG) optimizer.
    Uses the QFIM to precondition the standard Euclidean gradient.
    """
    def __init__(self, params, lr=0.01, reg=1e-3, qfim_update_freq=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, reg=reg, qfim_update_freq=qfim_update_freq)
        super(QuantumNaturalGradient, self).__init__(params, defaults)
        
        # Internal state to hold the most recently computed QFIM
        self.state['F_inv'] = None
        self.state['step'] = 0

    def step(self, model, inputs, closure=None):
        """
        Performs a single QNG optimization step.
        Requires the `model` and `inputs` to compute the QFIM.
        Since QFIM is expensive, we recompute it every `qfim_update_freq` steps.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            reg = group['reg']
            freq = group['qfim_update_freq']
            
            # 1. Flatten gradients
            grads = []
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad.view(-1))
            
            if not grads:
                continue
                
            grad_vec = torch.cat(grads)
            
            # 2. Update QFIM inverse periodically
            if self.state['step'] % freq == 0 or self.state['F_inv'] is None:
                # We need to compute F. We assume inputs is batched, we compute F for the first input 
                # (or average over batch) for simplicity. Here we just take inputs[0] because 
                # QFIM for EHA ansatz purely depends on theta and the initial state.
                # In QFM, inputs are the pure states |psi_tau>
                state_0 = inputs[0] if inputs.dim() > 1 else inputs
                
                # compute_qfim takes (circuit, state, params)
                # Ensure params are passed correctly to compute_qfim
                try:
                    qnode = model.qnode
                    # Extract params directly from model
                    params_list = list(model.parameters())
                    params_tensor = torch.cat([p.view(-1) for p in params_list])
                    F = compute_qfim_ensemble(qnode, params_tensor, inputs, shift=np.pi/2)
                except Exception as e:
                    # Fallback to Identity if we can't extract qnode properly right now
                    print(f"Warning: QFIM computation failed ({e}), falling back to Euclidean Adam step.")
                    F = torch.eye(sum(p.numel() for p in group['params']), device=inputs.device)
                
                # Regularize and invert
                F_reg = F + reg * torch.eye(F.shape[0], dtype=F.dtype, device=F.device)
                F_inv = torch.linalg.inv(F_reg).real.float() # QFIM is inherently real symmetric
                
                self.state['F_inv'] = F_inv
                
            # 3. Apply QNG update: nat_grad = F^{-1} * grad
            F_inv = self.state['F_inv']
            nat_grad = torch.matmul(F_inv, grad_vec)
            
            # 4. Scatter natural gradients back to parameters
            offset = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                numel = p.numel()
                nat_g = nat_grad[offset:offset+numel].view_as(p)
                p.data.add_(nat_g, alpha=-lr)
                offset += numel
                
        self.state['step'] += 1
        return loss
