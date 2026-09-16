"""Normalization and rotary position embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(dtype) * self.weight


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """1-D rotary frequencies (for text attention)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)  # (seq_len, dim/2)
    return torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to a tensor of shape (..., seq, dim)."""
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape_as(x)
    return x * cos + x_rot * sin


def precompute_2d_rope_freqs(dim: int, height: int, width: int, theta: float = 10000.0):
    """2-D rotary frequencies for the vision encoder (x/y axes split the dims)."""
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half))
    ys = torch.arange(height).float()
    xs = torch.arange(width).float()
    freqs_y = torch.outer(ys, inv_freq)
    freqs_x = torch.outer(xs, inv_freq)
    freqs_y = torch.cat([freqs_y, freqs_y], dim=-1)  # (H, half)
    freqs_x = torch.cat([freqs_x, freqs_x], dim=-1)  # (W, half)
    freqs = torch.cat([
        freqs_y[:, None, :].expand(height, width, half),
        freqs_x[None, :, :].expand(height, width, half),
    ], dim=-1).reshape(height, width, dim)  # (H, W, dim)
    return freqs
