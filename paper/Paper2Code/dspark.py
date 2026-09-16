"""DSpark: semi-autoregressive speculative decoding (Cheng et al., 2026a).

DSpark replaces MTP for speculative decoding. A 3-block drafter (SWA window
128) produces base logits for five draft positions in a single pass; a Markov
head models inter-token dependencies; a confidence head predicts per-position
acceptance probabilities used by the scheduler to pick the verification length.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class DSpark(nn.Module):
    def __init__(self, hidden: int, n_heads: int, head_dim: int, vocab: int,
                 n_blocks: int = 3, swa_window: int = 128, n_draft: int = 5):
        super().__init__()
        self.n_draft = n_draft
        self.swa_window = swa_window
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.blocks = nn.ModuleList([_DraftBlock(hidden, n_heads, head_dim, swa_window)
                                     for _ in range(n_blocks)])
        self.norm = RMSNorm(hidden)
        self.base_heads = nn.ModuleList([nn.Linear(hidden, vocab, bias=False)
                                         for _ in range(n_draft)])
        self.markov_head = nn.Linear(vocab, vocab, bias=False)
        self.confidence_head = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor):
        """h: (B, T, hidden) -> (draft_logits, confidence).

        Returns:
            draft_logits: (B, n_draft, vocab) logits for the five draft positions.
            confidence: (B, n_draft) per-position acceptance probabilities.
        """
        b, t, _ = h.shape
        x = h
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, -1])  # (B, hidden), use the last position
        base = [head(x) for head in self.base_heads]           # list of (B, vocab)
        # Markov correction chaining draft-token dependencies.
        logits = []
        prev = base[0]
        for i in range(self.n_draft):
            l = base[i] + (self.markov_head(F.softmax(prev.float(), dim=-1))
                           if i > 0 else 0)
            logits.append(l)
            prev = l
        logits = torch.stack(logits, dim=1)                    # (B, n_draft, vocab)
        conf = torch.sigmoid(self.confidence_head(x)).squeeze(-1)  # (B,)
        # Per-position survival probability (simple product prefix).
        survival = torch.cumprod(conf.clamp_min(1e-6).expand(self.n_draft, b).T, dim=1)
        return logits, survival

    def schedule(self, confidence: torch.Tensor, throughput_curve):
        """Choose verification length given acceptance probs + engine curve."""
        # Reference stub: greedily accept while expected survival is profitable.
        n = confidence.numel()
        return max(1, int((confidence > 0.5).sum().item()))


class _DraftBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, head_dim: int, window: int):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.window = window
        self.norm1 = RMSNorm(hidden)
        self.norm2 = RMSNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 4, bias=False), nn.SiLU(),
                                 nn.Linear(hidden * 4, hidden, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        scale = self.head_dim ** -0.5
        scores = torch.einsum("bthd,bshd->bhts", q, k) * scale
        idx = torch.arange(t, device=x.device)
        rel = idx[None, :] - idx[:, None]               # rel[i, j] = j - i
        mask = (rel <= 0) & (rel > -self.window)        # causal + bounded window
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).to(x.dtype)
        attn = torch.einsum("bhts,bshd->bthd", attn, v).reshape(b, t, h)
        x = x + self.o(attn)
        x = x + self.mlp(self.norm2(x))
        return x
