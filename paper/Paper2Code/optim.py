"""Optimizers: Muon (with head-wise variant) and Sinkhorn-balanced updates.

Section 2.5 / 4.2.2: Muon for linear weights, AdamW for normalization and other
non-matrix params, and a momentum + Sinkhorn-balanced update (Algorithm 1) for
the Engram tables, token embedding, and prediction head. The Sinkhorn update
replaces Newton-Schulz orthogonalization with row/column RMS equalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize a square matrix (Muon's update projection)."""
    a = g.float()
    a = a / (a.norm() + eps)
    for _ in range(steps):
        a = 1.5 * a - 0.5 * (a @ a.mT @ a)
    return a


class Muon(Optimizer):
    """Minimal Muon: momentum + Nesterov + Newton-Schulz, RMS rescale to 0.18."""

    def __init__(self, params, lr: float = 2.6e-4, momentum: float = 0.95,
                 weight_decay: float = 0.1, rms: float = 0.18):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rms=rms)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu, wd, rms = group["lr"], group["momentum"], group["weight_decay"], group["rms"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd:
                    g = g + wd * p
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                m.mul_(mu).add_(g, alpha=1 - mu)
                g_hat = mu * m + (1 - mu) * g  # Nesterov
                # Muon orthogonalization applies to 2-D weight matrices.
                if g_hat.dim() >= 2:
                    upd = newton_schulz(g_hat, steps=5)
                    upd = upd * rms
                else:
                    upd = g_hat
                p.add_(upd, alpha=-lr)


def sinkhorn_update(g_hat: torch.Tensor, n: int, k: int = 11, tau: float = 1e-3,
                    eps: float = 1e-20, rms: float = 0.18) -> torch.Tensor:
    """Algorithm 1: Sinkhorn-balanced update matrix.

    Mask near-zero rows (rho_i <= tau * mean(rho)), then alternate row/column
    L2 normalization K times, and scale by sqrt(n) for unit row-wise RMS.
    """
    u = g_hat.float().clone()
    rho = u.norm(dim=1)
    mean_rho = rho.mean()
    u[rho <= tau * mean_rho] = 0.0
    for i in range(k):
        if i % 2 == 0:
            u = u / (u.norm(dim=1, keepdim=True) + eps)
        else:
            u = u / (u.norm(dim=0, keepdim=True) + eps)
    return (n ** 0.5) * u * rms
