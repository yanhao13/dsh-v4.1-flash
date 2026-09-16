"""Compressed Sparse Attention 2 (CSA2) with cross-layer KV / index reuse.

CSA2 is the paper's core contribution. Each CSA2 layer is statically assigned
one of three modes:

* ``full``    - computes its own main KV and indexer K, runs the indexer over
                the full causally-visible range, and (in the decoder) builds the
                hierarchical candidate pool.
* ``reindex`` - reuses the most recent main KV + indexer K, recomputes Top-K
                indices from its own indexer Q (restricted to the candidate pool).
* ``reuse``   - reuses main KV and the latest Top-K indices; no indexing work.

Every mode still computes its own main Q and SWA KV locally. The main KV is
compressed along the sequence by ratio ``m`` (no overlap, no absolute positional
embedding - the simplification relative to CSA). Consistent with CSA2, the global
(main) KV carries no rotary encoding; RoPE is used only on the layer-local SWA
branch, whose positions are unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import apply_rotary_emb


@dataclass
class CSA2State:
    """Shared state flowing across layers (main KV / indexer K / selections)."""
    main_k: torch.Tensor | None = None       # (B, C, H, D) compressed main keys
    main_v: torch.Tensor | None = None       # (B, C, H, Dv)
    indexer_k: torch.Tensor | None = None    # (B, C, n_ih, ih_dim)
    topk_idx: torch.Tensor | None = None     # (B, T, K) selected compressed positions
    candidate_pool: torch.Tensor | None = None  # (B, C) bool mask


def compress(h: torch.Tensor, m: int, proj: nn.Linear) -> torch.Tensor:
    """Compress the sequence by ``m`` then project to the KV latent.

    For m=1 this is a pure latent projection (no sequence compression), which
    matches the decoder configuration.
    """
    b, t, d = h.shape
    if m == 1:
        return proj(h)
    pad = (-t) % m
    if pad:
        h = F.pad(h, (0, 0, 0, pad))
    c = (t + pad) // m
    return proj(h.reshape(b, c, m * d))


class CSA2Layer(nn.Module):
    def __init__(self, hidden: int, n_heads: int, head_dim: int, v_dim: int,
                 q_lora: int, kv_lora: int, n_indexer_heads: int, indexer_head_dim: int,
                 top_k: int, ratio: int, swa_window: int, mode: str,
                 hierarchical: bool = False, n_blocks: int = 2048, block_size: int = 8):
        super().__init__()
        self.mode = mode
        self.ratio = ratio
        self.top_k = top_k
        self.swa_window = swa_window
        self.hierarchical = hierarchical
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.v_dim = v_dim
        self.n_indexer_heads = n_indexer_heads
        self.indexer_head_dim = indexer_head_dim

        # Main Q (MLA-style query compression).
        self.q_down = nn.Linear(hidden, q_lora, bias=False)
        self.q_up = nn.Linear(q_lora, n_heads * head_dim, bias=False)

        # SWA KV (layer-local).
        self.swa_k = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.swa_v = nn.Linear(hidden, n_heads * v_dim, bias=False)

        # Main KV: sequence compression (ratio) then latent projection.
        in_dim = hidden * ratio
        self.compress_kv = nn.Linear(in_dim, kv_lora, bias=False)
        self.main_k_up = nn.Linear(kv_lora, n_heads * head_dim, bias=False)
        self.main_v_up = nn.Linear(kv_lora, n_heads * v_dim, bias=False)

        # Indexer: indexer Q is local; indexer K is projected from main KV.
        self.indexer_q = nn.Linear(hidden, n_indexer_heads * indexer_head_dim, bias=False)
        self.indexer_k_proj = nn.Linear(kv_lora, n_indexer_heads * indexer_head_dim, bias=False)

        self.out_proj = nn.Linear(n_heads * v_dim, hidden, bias=False)

    # ------------------------------------------------------------------ parts
    def _main_q(self, h: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        b, t, _ = h.shape
        q = self.q_up(self.q_down(h)).view(b, t, self.n_heads, self.head_dim)
        return apply_rotary_emb(q, rope.unsqueeze(0).unsqueeze(2))

    def _swa_kv(self, h: torch.Tensor, rope: torch.Tensor):
        b, t, _ = h.shape
        k = apply_rotary_emb(self.swa_k(h).view(b, t, self.n_heads, self.head_dim),
                             rope.unsqueeze(0).unsqueeze(2))
        v = self.swa_v(h).view(b, t, self.n_heads, self.v_dim)
        return k, v

    def _main_kv(self, h_global: torch.Tensor):
        b, t, _ = h_global.shape
        latent = compress(h_global, self.ratio, self.compress_kv)  # (B, C, kv_lora)
        c = latent.shape[1]
        main_k = self.main_k_up(latent).view(b, c, self.n_heads, self.head_dim)
        main_v = self.main_v_up(latent).view(b, c, self.n_heads, self.v_dim)
        return latent, main_k, main_v

    def _indexer_k(self, latent: torch.Tensor) -> torch.Tensor:
        b, c, _ = latent.shape
        return self.indexer_k_proj(latent).view(b, c, self.n_indexer_heads, self.indexer_head_dim)

    def _indexer_scores(self, h: torch.Tensor, indexer_k: torch.Tensor) -> torch.Tensor:
        """(B, T, C) scores over compressed positions, averaged over indexer heads."""
        b, t, _ = h.shape
        iq = self.indexer_q(h).view(b, t, self.n_indexer_heads, self.indexer_head_dim)
        scores = torch.einsum("bthd,bchd->btch", iq, indexer_k)  # (B, T, C, n_ih)
        return scores.mean(dim=-1) / (self.indexer_head_dim ** 0.5)

    def _candidate_pool(self, scores: torch.Tensor) -> torch.Tensor:
        """Blockwise candidate selection for the Hierarchical Sparse Indexer.

        Each block of ``block_size`` positions gets the max index score among its
        members; the top-``n_blocks`` blocks (per query) contribute their positions
        to the shared candidate pool.
        """
        b, t, c = scores.shape
        nb = min(self.n_blocks, c // self.block_size)
        trimmed = scores[:, :, : nb * self.block_size]
        blk = trimmed.view(b, t, nb, self.block_size)
        blk_score = blk.amax(dim=-1)                            # (B, T, nb)
        top_blocks = blk_score.topk(nb, dim=-1).indices        # (B, T, nb)
        # Map selected blocks back to global compressed positions.
        pos = top_blocks * self.block_size                       # (B, T, nb)
        offs = torch.arange(self.block_size, device=scores.device)
        pos = (pos.unsqueeze(-1) + offs).reshape(b, t, nb * self.block_size)  # (B, T, nb*bs)
        pool = torch.zeros(b, c, dtype=torch.bool, device=scores.device)
        # All queries share the pool: union over the query axis.
        pool.scatter_(1, pos.reshape(b, -1), True)
        return pool

    def forward(self, h: torch.Tensor, rope: torch.Tensor, state: CSA2State,
                h_global: torch.Tensor | None = None,
                make_pool: bool = False) -> tuple[torch.Tensor, CSA2State]:
        b, t, _ = h.shape

        main_q = self._main_q(h, rope)
        swa_k, swa_v = self._swa_kv(h, rope)

        if self.mode == "full":
            src = h_global if h_global is not None else h
            latent, main_k, main_v = self._main_kv(src)
            indexer_k = self._indexer_k(latent)
            scores = self._indexer_scores(h, indexer_k)
            pool = self._candidate_pool(scores) if make_pool else state.candidate_pool
            topk = scores.topk(self.top_k, dim=-1).indices
            state = CSA2State(main_k=main_k, main_v=main_v, indexer_k=indexer_k,
                              topk_idx=topk, candidate_pool=pool)
        elif self.mode == "reindex":
            scores = self._indexer_scores(h, state.indexer_k)
            if state.candidate_pool is not None:
                scores = scores.masked_fill(~state.candidate_pool[:, None, :], float("-inf"))
            topk = scores.topk(self.top_k, dim=-1).indices
            state = CSA2State(main_k=state.main_k, main_v=state.main_v,
                              indexer_k=state.indexer_k, topk_idx=topk,
                              candidate_pool=state.candidate_pool)
        else:  # reuse
            topk = state.topk_idx

        # Sparse attention over the selected main KV entries (advanced indexing).
        b_idx = torch.arange(b, device=h.device).view(b, 1, 1)
        sel_k = state.main_k[b_idx, topk]  # (B, T, K, H, D)
        sel_v = state.main_v[b_idx, topk]  # (B, T, K, H, Dv)
        sparse_scores = torch.einsum("bthd,btkhd->bthk", main_q, sel_k) / (self.head_dim ** 0.5)
        sparse_out = torch.einsum(
            "bthk,btkhv->bthv",
            F.softmax(sparse_scores.float(), dim=-1).to(main_q.dtype), sel_v)

        # SWA attention (causal, bounded window).
        swa_scores = torch.einsum("bthd,bkhd->bthk", main_q, swa_k) / (self.head_dim ** 0.5)
        idx = torch.arange(t, device=h.device)
        rel = idx[None, :] - idx[:, None]               # rel[i, j] = j - i
        swa_mask = (rel <= 0) & (rel > -self.swa_window)   # causal + bounded window
        swa_scores = swa_scores.masked_fill(~swa_mask[None, :, None, :], float("-inf"))
        swa_out = torch.einsum(
            "bthk,bkhv->bthv",
            F.softmax(swa_scores.float(), dim=-1).to(main_q.dtype), swa_v)

        attn_out = sparse_out + swa_out
        return self.out_proj(attn_out.reshape(b, t, -1)), state
