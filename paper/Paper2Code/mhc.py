"""Single-Pass mHC: residual-stream mixing between Transformer blocks.

mHC (Xie et al., 2026) maintains ``n`` residual streams X_l in R^{n x d} between
adjacent blocks and updates them with token-wise predicted coefficients:

    X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l),      (A_l, B_l, C_l) = H(X_l)

Single-Pass mHC shifts the input-mixing coefficients by one block (using A_{l-1}
instead of A_l), which removes the data dependency that forced the original
multi-kernel implementation to re-read the residual. This is the version used in
the deployment kernel (Mega-mHC) and what we implement here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


def sinkhorn(b: torch.Tensor, iters: int) -> torch.Tensor:
    """Sinkhorn-Knopp normalization of non-negative (..., n, n) matrices."""
    b = F.softplus(b) + 1e-6
    for _ in range(iters):
        b = b / b.sum(dim=-1, keepdim=True)
        b = b / b.sum(dim=-2, keepdim=True)
    return b


class mHCCoefficients(nn.Module):
    """Predicts the token-wise mixing coefficients (A, B, C) from streams X."""

    def __init__(self, n_streams: int, hidden: int, sinkhorn_iters: int = 20):
        super().__init__()
        self.n = n_streams
        self.norm = RMSNorm(n_streams * hidden)
        self.proj = nn.Linear(n_streams * hidden, n_streams * n_streams + 2 * n_streams, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, T, n, d) -> (A, B, C)."""
        b, t, n, d = x.shape
        flat = x.reshape(b, t, n * d)
        coeff = self.proj(self.norm(flat))          # (B, T, n^2 + 2n)
        a, bc = coeff.split([n, n * n + n], dim=-1)
        b_mat, c = bc.split([n * n, n], dim=-1)
        A = F.softmax(a, dim=-1).unsqueeze(-2)      # (B, T, 1, n)
        B = sinkhorn(b_mat.view(b, t, n, n), 20)    # (B, T, n, n)
        C = c.unsqueeze(-1)                          # (B, T, n, 1)
        return A, B, C


class mHCBlock(nn.Module):
    """Wraps a sub-block F and performs the mHC residual-stream update."""

    def __init__(self, n_streams: int, hidden: int, block: nn.Module, sinkhorn_iters: int = 20):
        super().__init__()
        self.n = n_streams
        self.block = block
        self.coeff = mHCCoefficients(n_streams, hidden, sinkhorn_iters)

    def forward(self, x: torch.Tensor, a_prev: torch.Tensor | None = None):
        """x: (B, T, n, d) streams; a_prev: (B, T, 1, n) previous input-mixing coeffs."""
        b, t, n, d = x.shape
        A, B, C = self.coeff(x)
        if a_prev is None:
            a_prev = torch.full((b, t, 1, n), 1.0 / n, device=x.device, dtype=x.dtype)
        mixed = (a_prev @ x).squeeze(-2)            # (B, T, d)
        y = self.block(mixed)                        # (B, T, d) -> F_l output
        x_new = B @ x + C * y.unsqueeze(-2)         # (B, T, n, d)
        return x_new, A
