"""Engram: sparsely-accessed conditional memory (Cheng et al., 2026c).

DeepSeek-V4.1-Flash augments the backbone with two Engram modules (at layers 1
and 14) that decouple memorization from computation. Each module uses N-gram
orders {2, 3, 4}, eight hash heads per order, and a 2048-dim embedding per order.
Each head indexes a table of ~16M entries (distinct primes). Compared with the
original design, the short causal convolution is dropped and the embedding is
optimized with momentum + Sinkhorn balancing (see ``optim.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Distinct primes used as per-head hash bases / table sizes.
_PRIMES = [16777213, 16777259, 16777289, 16777321, 16777333, 16777367,
           16777441, 16777471, 16777477, 16777501]


def _hash_ngram(token_ids: torch.Tensor, order: int, head: int, table_size: int) -> torch.Tensor:
    """Hash the ``order``-gram ending at each position into a table index.

    token_ids: (B, T). Returns (B, T) indices in [0, table_size).
    """
    b, t = token_ids.shape
    base = _PRIMES[head % len(_PRIMES)]
    # Gather the trailing `order` tokens per position (with left padding).
    pad = token_ids.new_zeros(b, order - 1)
    seq = torch.cat([pad, token_ids], dim=1)  # (B, T + order - 1)
    idx = torch.zeros(b, t, dtype=torch.long, device=token_ids.device)
    for j in range(order):
        window = seq[:, j: j + t]
        idx = (idx * base + window) % table_size
    return idx


class Engram(nn.Module):
    def __init__(self, hidden: int, ngrams: tuple[int, ...] = (2, 3, 4),
                 n_heads: int = 8, embed_dim: int = 2048, table_entries: int = 16_777_216):
        super().__init__()
        self.ngrams = ngrams
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        # Per order: n_heads embedding tables (each table_entries x head_dim).
        self.tables = nn.ModuleDict()
        self.table_sizes = {}
        for o in ngrams:
            size = _PRIMES[o % len(_PRIMES)]
            size = min(size, table_entries)
            self.table_sizes[o] = size
            self.tables[str(o)] = nn.ModuleList([
                nn.Embedding(size, self.head_dim) for _ in range(n_heads)
            ])
        # Context-aware gating over the aggregated order embeddings.
        self.gate = nn.Linear(hidden, len(ngrams), bias=False)
        self.proj = nn.Linear(embed_dim, hidden, bias=False)

    def forward(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Return a memory vector to add into the residual stream.

        hidden: (B, T, hidden). token_ids: (B, T).
        """
        b, t, _ = hidden.shape
        order_out = []
        for o in self.ngrams:
            size = self.table_sizes[o]
            head_outs = []
            for h in range(self.n_heads):
                idx = _hash_ngram(token_ids, o, h, size)
                head_outs.append(self.tables[str(o)][h](idx))  # (B, T, head_dim)
            order_out.append(torch.cat(head_outs, dim=-1))     # (B, T, embed_dim)
        mem = torch.stack(order_out, dim=-1)                    # (B, T, embed_dim, O)
        g = torch.sigmoid(self.gate(hidden))                    # (B, T, O)
        mem = (mem * g.unsqueeze(-2)).sum(dim=-1)              # (B, T, embed_dim)
        return self.proj(mem)
