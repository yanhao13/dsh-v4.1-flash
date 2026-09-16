"""Smoke tests for the DeepSeek-V4.1-Flash reference implementation.

Run:  python -m tests.test_smoke
"""

from __future__ import annotations

import torch

from deepseek_v41_flash import (DeepSeekV41Config, DeepSeekV41Flash, CSA2Layer,
                                CSA2State, DeepSeekMoE, mHCBlock, Engram, DSpark)
from deepseek_v41_flash.fp4 import quantize_kv, dequantize_e2m1, quantized_kv_bytes_per_token


def _make_cfg() -> DeepSeekV41Config:
    return DeepSeekV41Config.smoke()


def test_config_modes():
    cfg = _make_cfg()
    # First two layers are SWA-only.
    assert cfg.layer_mode(0) == "swa"
    assert cfg.layer_mode(1) == "swa"
    # Encoder CSA2: [full, reuse x5].
    assert cfg.layer_mode(2) == "full"
    assert cfg.layer_mode(3) == "reuse"
    # Decoder group 0: [full, reuse x3].
    assert cfg.layer_mode(8) == "full"
    assert cfg.layer_mode(9) == "reuse"
    # Decoder group 1: [reindex, reuse x3].
    assert cfg.layer_mode(12) == "reindex"
    assert cfg.layer_mode(13) == "reuse"
    print("[ok] layer-mode assignment matches Section 4.2.1")


def test_fp4_roundtrip():
    x = torch.randn(4, 64) * 3
    q, scale = quantize_kv(x, block=16)
    xr = dequantize_e2m1(q, scale.repeat_interleave(16, dim=-1)[..., : x.shape[-1]])
    err = (x - xr).abs().max().item()
    assert err < 6.0 * 0.25 + 0.5  # E2M1 step is coarse; just sanity bound
    print(f"[ok] FP4 quant/dequant max err = {err:.3f}")
    per_vec = quantized_kv_bytes_per_token(512)
    print(f"[ok] FP4 bytes/token (512-ch vector) = {per_vec}; "
          f"K+V+indexerK ~= {3 * per_vec} (paper reports 890 global)")


def test_full_forward():
    cfg = _make_cfg()
    model = DeepSeekV41Flash(cfg)
    model.eval()
    tok = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        logits = model(tok)
    assert logits.shape == (2, 64, cfg.vocab_size), logits.shape
    assert torch.isfinite(logits).all()
    print(f"[ok] text forward -> logits {tuple(logits.shape)}")


def test_multimodal_forward():
    cfg = _make_cfg()
    model = DeepSeekV41Flash(cfg)
    model.eval()
    tok = torch.randint(0, cfg.vocab_size, (1, 32))
    img = torch.randn(1, 3, 56, 56)  # (56/8=7) x 7 grid, divisible by 3 after unshuffle? 7 not /3
    img = torch.randn(1, 3, 48, 48)  # 48/8=6 -> 6x6 grid -> /3 = 2x2
    with torch.no_grad():
        logits = model(tok, images=[img], image_positions=[3])
    assert logits.shape[0] == 1 and logits.shape[-1] == cfg.vocab_size
    print(f"[ok] multimodal forward -> logits {tuple(logits.shape)}")


def test_csa2_three_modes():
    cfg = _make_cfg()
    b, t = 2, 64
    rope = torch.randn(t, cfg.head_dim)
    h = torch.randn(b, t, cfg.hidden)

    full = CSA2Layer(cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.v_head_dim,
                     cfg.q_lora_rank, cfg.kv_lora_rank, cfg.n_indexer_heads,
                     cfg.indexer_head_dim, cfg.attn_top_k, ratio=2,
                     swa_window=cfg.swa_window, mode="full", hierarchical=True,
                     n_blocks=cfg.hsi_n_blocks, block_size=cfg.hsi_block_size)
    out, state = full(h, rope, CSA2State())
    assert out.shape == h.shape and state.main_k is not None
    assert state.main_k.shape[1] == t // 2  # ratio 2 -> halved

    reindex = CSA2Layer(cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.v_head_dim,
                        cfg.q_lora_rank, cfg.kv_lora_rank, cfg.n_indexer_heads,
                        cfg.indexer_head_dim, cfg.attn_top_k, ratio=2,
                        swa_window=cfg.swa_window, mode="reindex", hierarchical=True)
    out2, state2 = reindex(h, rope, state)
    assert out2.shape == h.shape and state2.topk_idx is not None

    reuse = CSA2Layer(cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.v_head_dim,
                      cfg.q_lora_rank, cfg.kv_lora_rank, cfg.n_indexer_heads,
                      cfg.indexer_head_dim, cfg.attn_top_k, ratio=2,
                      swa_window=cfg.swa_window, mode="reuse")
    out3, state3 = reuse(h, rope, state2)
    assert out3.shape == h.shape
    print("[ok] CSA2 full/reindex/reuse + hierarchical pool")


def test_mhc():
    cfg = _make_cfg()
    b, t, n = 2, 16, cfg.mhc_expansion
    block = torch.nn.Linear(cfg.hidden, cfg.hidden)
    mhc = mHCBlock(n, cfg.hidden, block, cfg.mhc_sinkhorn_iters)
    x = torch.randn(b, t, n, cfg.hidden)
    x_new, a = mhc(x)
    assert x_new.shape == x.shape and a.shape == (b, t, 1, n)
    print("[ok] Single-Pass mHC stream update")


def test_engram():
    cfg = _make_cfg()
    eng = Engram(cfg.hidden, cfg.engram_ngrams, cfg.engram_n_heads,
                 cfg.engram_embed_dim, cfg.engram_table_entries)
    tok = torch.randint(0, cfg.vocab_size, (2, 16))
    h = torch.randn(2, 16, cfg.hidden)
    mem = eng(h, tok)
    assert mem.shape == h.shape
    print("[ok] Engram conditional-memory lookup")


def test_dspark():
    cfg = _make_cfg()
    d = DSpark(cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.vocab_size,
               n_blocks=cfg.dspark_n_blocks, swa_window=cfg.dspark_swa_window,
               n_draft=cfg.dspark_n_draft)
    h = torch.randn(2, 8, cfg.hidden)
    logits, conf = d(h)
    assert logits.shape == (2, cfg.dspark_n_draft, cfg.vocab_size)
    assert conf.shape == (2, cfg.dspark_n_draft)
    print("[ok] DSpark speculative drafter")


def test_moe():
    cfg = _make_cfg()
    moe = DeepSeekMoE(cfg.hidden, cfg.n_routed_experts, cfg.n_shared_experts,
                      cfg.n_activated_experts, cfg.expert_inter_dim, cfg.moe_clamp)
    x = torch.randn(2, 16, cfg.hidden)
    modality = torch.zeros(2, 16, dtype=torch.long)
    modality[:, 8:] = 1
    y = moe(x, modality)
    assert y.shape == x.shape
    print("[ok] DeepSeekMoE routing (modality-aware)")


if __name__ == "__main__":
    test_config_modes()
    test_fp4_roundtrip()
    test_moe()
    test_csa2_three_modes()
    test_mhc()
    test_engram()
    test_dspark()
    test_full_forward()
    test_multimodal_forward()
    print("\nALL SMOKE TESTS PASSED")
