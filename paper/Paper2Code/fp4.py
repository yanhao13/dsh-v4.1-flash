"""FP4 (E2M1) KV-cache quantization.

DeepSeek-V4.1-Flash stores the *main* global KV cache in FP4. The format is the
OCP MXFP4 / NVFP4 flavour: data values use E2M1 (1 sign, 2 exponent, 1 mantissa
bits -> max magnitude 6.0) with a per-16-channel E4M3 scale (max 448.0). The
global second-level scale of NVFP4 is omitted (Section 2.4.4), giving a total
representable magnitude of 448 x 6 = 2688.
"""

from __future__ import annotations

import torch

# E2M1 representable positive magnitudes.
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e2m1_grid(device: torch.device) -> torch.Tensor:
    grid = _E2M1_GRID.to(device)
    return torch.cat([-grid.flip(0)[:-1], grid])  # symmetric magnitudes


def quantize_e2m1(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize ``x`` to E2M1 given a per-group scale.

    Args:
        x: (..., C) values to quantize.
        scale: (..., C//block) or (..., 1) per-group scale broadcast to ``x``.
    Returns:
        E2M1-encoded magnitudes (the raw FP4 value, pre-scale).
    """
    grid = _e2m1_grid(x.device)
    q = x / scale
    # Find nearest grid point by expanding the grid along the last dim.
    diff = q.unsqueeze(-1) - grid  # (..., C, G)
    idx = diff.abs().argmin(dim=-1)
    return grid[idx]


def dequantize_e2m1(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize E2M1 values back to float (``value * scale``)."""
    return q * scale


def compute_block_scale(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Per-``block``-channel scale so that |x| <= 6.0 after scaling.

    Args:
        x: (..., C) tensor.
    Returns:
        (..., C//block) scale tensor.
    """
    shape = x.shape
    xb = x.reshape(*shape[:-1], -1, block)
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    scale = amax / 6.0
    return scale.squeeze(-1)


def quantize_kv(x: torch.Tensor, block: int = 16):
    """Quantize a KV tensor to FP4 (E2M1 + per-block E4M3 scale).

    Returns:
        (q_e2m1, scale) tuple. Bytes per element = 0.5 + 1/block.
    """
    scale = compute_block_scale(x, block)
    scale_expanded = scale.repeat_interleave(block, dim=-1)[..., : x.shape[-1]]
    q = quantize_e2m1(x, scale_expanded)
    return q, scale


def quantized_kv_bytes_per_token(dim: int, block: int = 16) -> int:
    """Storage bytes for a single token's KV vector in FP4."""
    data_bits = dim * 4
    scale_bits = (dim // block) * 8
    return (data_bits + scale_bits) // 8
