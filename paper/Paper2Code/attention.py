"""Multi-head latent attention (MLA) projections + attention kernels.

DeepSeek-V4.1-Flash compresses the query to a ``q_lora_rank``-dim latent and
the KV to a ``kv_lora_rank``-dim latent before expanding to heads (the MLA
family). RoPE is applied only to the dedicated "rope" head slice; the "nope"
slice is position-free. This module also provides the sliding-window and
sparse core-attention routines used by CSA2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import precompute_rope_freqs, apply_rotary_emb


class MLAProjections(nn.Module):
    """MLA-style Q/KV projections for a single attention layer."""

    def __init__(self, hidden: int, n_heads: int, q_lora_rank: int, kv_lora_rank: int,
                 qk_nope: int, qk_rope: int, v_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.qk_nope = qk_nope
        self.qk_rope = qk_rope
        self.v_head = v_head
        self.q_head_dim = qk_nope + qk_rope

        self.q_down = nn.Linear(hidden, q_lora_rank, bias=False)
        self.q_up_nope = nn.Linear(q_lora_rank, n_heads * qk_nope, bias=False)
        self.q_up_rope = nn.Linear(q_lora_rank, n_heads * qk_rope, bias=False)
        self.kv_down = nn.Linear(hidden, kv_lora_rank, bias=False)
        self.k_up_nope = nn.Linear(kv_lora_rank, n_heads * qk_nope, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, n_heads * v_head, bias=False)
        self.o = nn.Linear(n_heads * v_head, hidden, bias=False)

    def query(self, h: torch.Tensor):
        """h: (B, T, hidden) -> q (B, T, n_heads, qk_nope + qk_rope)."""
        b, t, _ = h.shape
        ql = self.q_down(h)
        q_nope = self.q_up_nope(ql).view(b, t, self.n_heads, self.qk_nope)
        q_rope = self.q_up_rope(ql).view(b, t, self.n_heads, self.qk_rope)
        return torch.cat([q_nope, q_rope], dim=-1)

    def kv(self, h: torch.Tensor):
        """h -> (k_nope, k_rope, v) each (B, T, n_heads, dim)."""
        b, t, _ = h.shape
        kvl = self.kv_down(h)
        k_nope = self.k_up_nope(kvl).view(b, t, self.n_heads, self.qk_nope)
        v = self.v_up(kvl).view(b, t, self.n_heads, self.v_head)
        # RoPE keys share the rope projection of the query latent.
        q_rope = self.q_up_rope(self.q_down(h)).view(b, t, self.n_heads, self.qk_rope)
        return k_nope, q_rope, v

    def output(self, attn: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = attn.shape
        return self.o(attn.reshape(b, t, -1))


def apply_rope_heads(x: torch.Tensor, rope: torch.Tensor, nope_dim: int) -> torch.Tensor:
    """Apply RoPE to the trailing ``rope`` dims of head tensors.

    Args:
        x: (B, T, n_heads, nope_dim + rope_dim).
        rope: (T, rope_dim) rotary embeddings.
    """
    nope, rope_part = x[..., :nope_dim], x[..., nope_dim:]
    rope_part = apply_rotary_emb(rope_part, rope.unsqueeze(0).unsqueeze(0))
    return torch.cat([nope, rope_part], dim=-1)


def core_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention. q/k: (..., S, dim), v: (..., S, vdim)."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def sliding_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             window: int) -> torch.Tensor:
    """Causal SWA: query i attends to keys in [max(0, i - window + 1), i]."""
    b, t, h, d = q.shape
    scale = d ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, T, h, T)
    idx = torch.arange(t, device=q.device)
    rel = idx[None, :] - idx[:, None]            # rel[i, j] = j - i
    window_mask = (rel <= 0) & (rel > -window)   # causal + bounded
    scores = scores.masked_fill(~window_mask[None, :, None, :], float("-inf"))
    probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)
