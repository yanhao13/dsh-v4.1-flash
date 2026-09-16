"""Configuration for DeepSeek-V4.1-Flash.

Every value below is transcribed verbatim from Section 4.2.1 ("Model Setups")
of the DeepSeek-V4.1-Flash technical report. A few attention head dimensions
are recorded exactly as printed in the paper even where they do not trivially
reconcile with the hidden size; see the inline notes. The reference model uses
a self-consistent MLA-style layout (see ``attention.py``) so the code remains
runnable, while keeping the paper's reported values here for traceability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DeepSeekV41Config:
    # ---------------------------------------------------------------- backbone
    n_layers: int = 40                      # total Transformer layers
    n_encoder_layers: int = 20              # bottom half -> causal encoder
    n_decoder_layers: int = 20              # top half -> decoder
    hidden: int = 5120                      # hidden dimension d
    vocab_size: int = 129280                # DeepSeek tokenizer vocabulary
    max_seq_len: int = 1_048_576            # 1M context support

    # ------------------------------------------------------------- attention
    n_heads: int = 64                       # number of query heads (paper)
    head_dim: int = 512                     # head dimension (paper, verbatim)
    q_lora_rank: int = 1280                 # "query compression dimension"
    kv_lora_rank: int = 512                 # "512-channel KV latent" (Section 2.4.4)
    # MLA-style head split used by the runnable reference (self-consistent):
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    out_proj_groups: int = 8                # "output projection groups"
    intermediate_attn_out_dim: int = 1024   # "intermediate attention output dim"

    # ------------------------------------------------------------------- SWA
    swa_window: int = 128                   # sliding-window attention size n_win

    # ------------------------------------------------------------------ CSA2
    # Encoder CSA2: 18 layers, compression ratio m=2, three groups of six.
    # Each group: [Full, Reuse, Reuse, Reuse, Reuse, Reuse].
    encoder_csa2_ratio: int = 2
    encoder_csa2_groups: int = 3            # groups of 6 layers each
    # Decoder CSA2: 20 layers, compression ratio m=1, five groups of four.
    # Group 0: [Full, Reuse, Reuse, Reuse];
    # Groups 1..4: [Reindex, Reuse, Reuse, Reuse].
    decoder_csa2_ratio: int = 1
    decoder_csa2_groups: int = 5            # groups of 4 layers each

    n_indexer_heads: int = 32               # indexer query heads
    indexer_head_dim: int = 128             # indexer head dimension
    attn_top_k: int = 512                   # KV entries selected for sparse attention

    # Hierarchical Sparse Indexer (decoder only).
    hsi_n_blocks: int = 2048                # max blocks selected for candidate pool
    hsi_block_size: int = 8                 # positions per block
    # -> up to 16,384 candidate positions

    # -------------------------------------------------------------------- MoE
    n_routed_experts: int = 384             # routed experts
    n_shared_experts: int = 1               # shared experts
    n_activated_experts: int = 6            # routed experts activated per token
    expert_inter_dim: int = 2304            # intermediate dim of each expert
    moe_clamp: float = 10.0                 # SwiGLU clamping threshold
    load_balance_bias_update_speed: float = 0.001
    aux_loss_seq_weight: float = 0.0001     # sequence-level balance loss

    # ------------------------------------------------------------------- mHC
    mhc_expansion: int = 4                  # n residual streams
    mhc_sinkhorn_iters: int = 20            # Sinkhorn-Knopp iterations

    # ---------------------------------------------------------------- Engram
    engram_modules: tuple = (1, 14)         # zero-indexed layers hosting Engram
    engram_ngrams: tuple = (2, 3, 4)        # N-gram orders
    engram_n_heads: int = 8                 # hash heads per order
    engram_embed_dim: int = 2048            # embedding dim per order
    engram_table_entries: int = 16_777_216  # ~16M entries per head (distinct primes)

    # -------------------------------------------------------------- DSpark
    dspark_n_blocks: int = 3                # drafter Transformer blocks
    dspark_swa_window: int = 128
    dspark_n_draft: int = 5                 # parallel draft positions

    # ---------------------------------------------------------------- Vision
    vit_n_layers: int = 32
    vit_hidden: int = 1024
    vit_n_heads: int = 16
    vit_patch: int = 14
    vit_pixel_unshuffle: int = 3            # 3x3 -> 9x token reduction
    projector_n_layers: int = 2
    projector_hidden: int = 5120

    # -------------------------------------------------------------------- FP4
    fp4_block_size: int = 16                # one E4M3 scale per 16 channels (NVFP4-like)

    # ------------------------------------------------------------ Optimizers
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-20
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_update_rms: float = 0.18
    sinkhorn_k: int = 11
    sinkhorn_tau: float = 1e-3
    sinkhorn_eps: float = 1e-20
    sinkhorn_lr_correction: float = 0.18

    # ------------------------------------------------------------------- misc
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dtype: str = "bfloat16"

    # ------------------------------------------------------------------- ctor
    @classmethod
    def smoke(cls) -> "DeepSeekV41Config":
        """A tiny configuration used only for the unit smoke test.

        Structural layout (group sizes / mode patterns / ratios) is preserved
        exactly; only the widths are shrunk so the full forward pass fits on a
        laptop CPU.
        """
        return cls(
            n_layers=16,                 # 8 encoder + 8 decoder
            n_encoder_layers=8,
            n_decoder_layers=8,
            hidden=128,
            vocab_size=1000,
            max_seq_len=512,
            n_heads=4,
            head_dim=32,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=32,
            out_proj_groups=2,
            intermediate_attn_out_dim=64,
            n_indexer_heads=4,
            indexer_head_dim=32,
            attn_top_k=16,
            encoder_csa2_groups=1,       # 6 encoder CSA2 layers: [full, reuse x5]
            decoder_csa2_groups=2,       # 8 decoder CSA2 layers: [full,r,r,r]+[reindex,r,r,r]
            hsi_n_blocks=16,
            hsi_block_size=8,
            n_routed_experts=8,
            n_shared_experts=1,
            n_activated_experts=2,
            expert_inter_dim=128,
            vit_hidden=128,
            vit_n_layers=2,
            vit_n_heads=4,
            vit_patch=8,
            projector_hidden=128,
            engram_table_entries=1024,
        )

    # ----------------------------------------------------------------- helpers
    @property
    def n_csa2_encoder_layers(self) -> int:
        return self.n_encoder_layers - 2  # first two layers are SWA-only

    def encoder_mode_pattern(self) -> list[str]:
        """Per-encoder-CSA2-layer mode: [Full] + 5x[Reuse] per group."""
        group = ["full"] + ["reuse"] * 5
        return group * self.encoder_csa2_groups

    def decoder_mode_pattern(self) -> list[str]:
        """Per-decoder-CSA2-layer mode: group0=[Full,3x Reuse], rest=[Reindex,3x Reuse]."""
        first = ["full"] + ["reuse"] * 3
        rest = ["reindex"] + ["reuse"] * 3
        return first + rest * (self.decoder_csa2_groups - 1)

    def layer_mode(self, layer_idx: int) -> str:
        """Return 'swa' | 'full' | 'reindex' | 'reuse' for an absolute layer index."""
        if layer_idx < 2:
            return "swa"
        if layer_idx < self.n_encoder_layers:
            local = layer_idx - 2
            return self.encoder_mode_pattern()[local]
        local = layer_idx - self.n_encoder_layers
        return self.decoder_mode_pattern()[local]

    def global_kv_bytes_per_token(self) -> int:
        """Paper reports 890 bytes/token for the FP4 global KV cache."""
        return 890

    def estimated_params(self) -> int:
        """Rough parameter count for the backbone (order-of-magnitude)."""
        # Dominant term: MoE experts + attention + embeddings.
        per_layer = (
            self.hidden * self.hidden * 4                          # attn projections
            + (self.n_routed_experts + self.n_shared_experts) * self.hidden * self.expert_inter_dim * 3
        )
        return int(self.n_layers * per_layer + self.vocab_size * self.hidden)
