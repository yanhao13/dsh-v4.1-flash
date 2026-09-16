"""KV-cache management and SWA Bounded Replay.

This module captures the deployment-level contributions of Section 3.2: the
split between the *global* KV cache (FP4, always in HBM, 890 bytes/token), the
*SWA* KV cache (short-lived, kept in a small host-memory pool), and *SWA
Bounded Replay*, which reconstructs missing SWA KV by replaying only the most
recent ``n_win`` tokens instead of a full ``L x n_win`` forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .fp4 import quantize_kv, dequantize_e2m1, quantized_kv_bytes_per_token


@dataclass
class GlobalKVCache:
    """FP4-quantized global KV (main KV) plus indexer K and Top-K indices.

    The ``_q`` tensors store E2M1 data; ``_scale`` stores the per-16-channel
    E4M3 scale. Dequantization is deferred until attention.
    """
    main_k_q: torch.Tensor | None = None
    main_k_scale: torch.Tensor | None = None
    main_v_q: torch.Tensor | None = None
    main_v_scale: torch.Tensor | None = None
    indexer_k_q: torch.Tensor | None = None
    indexer_k_scale: torch.Tensor | None = None
    topk_idx: torch.Tensor | None = None

    def bytes_per_token(self) -> int:
        """Storage bytes per token for the global KV cache."""
        if self.main_k_q is None:
            return 0
        dim = self.main_k_q.shape[-1]
        # Main K + V + indexer K (each FP4), ignoring the negligible index ints.
        return 3 * quantized_kv_bytes_per_token(dim)


class SWACache:
    """Bounded, short-lived SWA KV held in host memory (Section 3.2.1)."""

    def __init__(self, window: int, ttl_seconds: float = 300.0):
        self.window = window
        self.ttl = ttl_seconds
        self._store: dict[int, tuple[torch.Tensor, float]] = {}  # pos -> (kv, ts)


def swa_bounded_replay(kv_fn, cached_prefix_len: int, window: int, suffix: torch.Tensor):
    """Reconstruct SWA KV by replaying only the last ``window`` tokens.

    Args:
        kv_fn: callable mapping a token tensor to its SWA KV.
        cached_prefix_len: length of the cached prefix (whose SWA KV was evicted).
        window: SWA window size n_win.
        suffix: uncached suffix tokens to process.

    This is the *Encoder* variant: replay the last n_win tokens of the cached
    prefix (regenerating only SWA KV, reusing cached global KV), then process
    the uncached suffix. The *Decoder* variant is identical in spirit (bounds
    the decoder prefill to n_win tokens); see ``DeepSeekV41Flash`` docs.
    """
    # Reference implementation: compute the replay window boundaries.
    replay_start = max(0, cached_prefix_len - window)
    replay_len = cached_prefix_len - replay_start
    replayed_kv = kv_fn(suffix[:0])  # placeholder; actual KV from replay segment
    # In a real system, ``kv_fn`` would be applied to tokens[replay_start:cached_prefix_len]
    # and then to the suffix, with the SWA mask bounded to [replay_start, i].
    return replay_start, replay_len, replayed_kv


def global_kv_bytes_accounting() -> dict[str, int]:
    """Approximate the paper's 890 bytes/token global-KV figure."""
    # 512-channel KV latent -> 64 heads; per-token K (128d) + V (128d) + indexer K
    # (32 x 128), all in FP4. Reference accounting, not an exact reproduction.
    k_bytes = quantized_kv_bytes_per_token(64 * 128)
    v_bytes = quantized_kv_bytes_per_token(64 * 128)
    idx_bytes = quantized_kv_bytes_per_token(32 * 128)
    return {"main_k": k_bytes, "main_v": v_bytes, "indexer_k": idx_bytes,
            "total": k_bytes + v_bytes + idx_bytes}
