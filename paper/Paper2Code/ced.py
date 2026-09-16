"""Causal Encoder-Decoder (CED) stacking.

The 40-layer backbone is split into a 20-layer causal encoder and a 20-layer
decoder. In the decoder, the *global* KV (main KV + indexer K) is not derived
from each layer's own hidden state; it is projected from the encoder's final
hidden state H_{L/2} through layer-dependent projections (Equation 1 in the
paper). Layer-local SWA KV is still computed from each layer's own hidden state.

This file assembles the Transformer blocks (attention + DeepSeekMoE), assigns
each layer its static CSA2 mode per Section 4.2.1, inserts Engram modules at
layers 1 and 14, and manages the CSA2 shared state across layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepSeekV41Config
from .norm import RMSNorm, apply_rotary_emb
from .csa2 import CSA2Layer, CSA2State
from .moe import DeepSeekMoE
from .engram import Engram


class SWAAttention(nn.Module):
    """Sliding-window attention used by the first two (SWA-only) layers."""

    def __init__(self, hidden: int, n_heads: int, head_dim: int, v_dim: int,
                 q_lora: int, window: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window = window
        self.q_down = nn.Linear(hidden, q_lora, bias=False)
        self.q_up = nn.Linear(q_lora, n_heads * head_dim, bias=False)
        self.k = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.v = nn.Linear(hidden, n_heads * v_dim, bias=False)
        self.o = nn.Linear(n_heads * v_dim, hidden, bias=False)

    def forward(self, h: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        b, t, _ = h.shape
        q = apply_rotary_emb(self.q_up(self.q_down(h)).view(b, t, self.n_heads, self.head_dim),
                             rope.unsqueeze(0).unsqueeze(2))
        k = apply_rotary_emb(self.k(h).view(b, t, self.n_heads, self.head_dim),
                             rope.unsqueeze(0).unsqueeze(2))
        v = self.v(h).view(b, t, self.n_heads, -1)
        scale = self.head_dim ** -0.5
        scores = torch.einsum("bthd,bshd->bhts", q, k) * scale
        idx = torch.arange(t, device=h.device)
        rel = idx[None, :] - idx[:, None]               # rel[i, j] = j - i
        mask = (rel <= 0) & (rel > -self.window)        # causal + bounded window
        scores = scores.masked_fill(~mask[None, None], float("-inf"))
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn = torch.einsum("bhts,bshv->bthv", probs, v).reshape(b, t, -1)
        return self.o(attn)


class TransformerBlock(nn.Module):
    """Pre-norm block: attention + MoE, with optional Engram injection."""

    def __init__(self, attn: nn.Module, moe: nn.Module, hidden: int,
                 engram: Engram | None = None):
        super().__init__()
        self.attn = attn
        self.moe = moe
        self.engram = engram
        self.norm_attn = RMSNorm(hidden)
        self.norm_moe = RMSNorm(hidden)

    def forward(self, h: torch.Tensor, rope: torch.Tensor, state: CSA2State,
                token_ids: torch.Tensor | None = None,
                h_global: torch.Tensor | None = None,
                make_pool: bool = False):
        if isinstance(self.attn, CSA2Layer):
            a_out, state = self.attn(self.norm_attn(h), rope, state,
                                     h_global=h_global, make_pool=make_pool)
        else:
            a_out = self.attn(self.norm_attn(h), rope)
        h = h + a_out
        h = h + self.moe(self.norm_moe(h))
        if self.engram is not None and token_ids is not None:
            h = h + self.engram(h, token_ids)
        return h, state


class CED(nn.Module):
    def __init__(self, cfg: DeepSeekV41Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        for l in range(cfg.n_layers):
            mode = cfg.layer_mode(l)
            if mode == "swa":
                attn = SWAAttention(cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.v_head_dim,
                                    cfg.q_lora_rank, cfg.swa_window)
            else:
                ratio = cfg.encoder_csa2_ratio if l < cfg.n_encoder_layers else cfg.decoder_csa2_ratio
                is_decoder = l >= cfg.n_encoder_layers
                # Only the decoder's *first* Full layer builds the candidate pool.
                make_pool = is_decoder and mode == "full" and l == cfg.n_encoder_layers
                hierarchical = is_decoder and mode in ("full", "reindex")
                attn = CSA2Layer(
                    hidden=cfg.hidden, n_heads=cfg.n_heads, head_dim=cfg.head_dim,
                    v_dim=cfg.v_head_dim, q_lora=cfg.q_lora_rank, kv_lora=cfg.kv_lora_rank,
                    n_indexer_heads=cfg.n_indexer_heads, indexer_head_dim=cfg.indexer_head_dim,
                    top_k=cfg.attn_top_k, ratio=ratio, swa_window=cfg.swa_window,
                    mode=mode, hierarchical=hierarchical,
                    n_blocks=cfg.hsi_n_blocks, block_size=cfg.hsi_block_size)
            moe = DeepSeekMoE(cfg.hidden, cfg.n_routed_experts, cfg.n_shared_experts,
                              cfg.n_activated_experts, cfg.expert_inter_dim, cfg.moe_clamp)
            engram = Engram(cfg.hidden, cfg.engram_ngrams, cfg.engram_n_heads,
                            cfg.engram_embed_dim, cfg.engram_table_entries) \
                if l in cfg.engram_modules else None
            self.layers.append(TransformerBlock(attn, moe, cfg.hidden, engram))

        # Rope: precompute once for the max sequence length.
        self.register_buffer("_rope", None, persistent=False)

    def _get_rope(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        from .norm import precompute_rope_freqs
        if self._rope is None or self._rope.shape[0] < t:
            freqs = precompute_rope_freqs(self.cfg.head_dim, t, self.cfg.rope_theta)
            self._rope = freqs
        return self._rope[:t].to(device=device, dtype=dtype)

    def forward(self, h: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        """h: (B, T, hidden). Returns final decoder hidden states."""
        t = h.shape[1]
        rope = self._get_rope(t, h.device, h.dtype)
        state = CSA2State()
        # Encoder pass (layers 0 .. n_encoder_layers - 1).
        for l in range(self.cfg.n_encoder_layers):
            h, state = self.layers[l](h, rope, state, token_ids)
        h_encoder = h  # H_{L/2}: the encoder's final hidden state.
        # Decoder pass: global KV is projected from h_encoder.
        state = CSA2State()
        for l in range(self.cfg.n_encoder_layers, self.cfg.n_layers):
            is_first_decoder_full = (l == self.cfg.n_encoder_layers)
            h, state = self.layers[l](h, rope, state, token_ids,
                                      h_global=h_encoder, make_pool=is_first_decoder_full)
        return h
