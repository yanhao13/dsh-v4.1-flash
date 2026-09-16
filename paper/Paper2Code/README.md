# DeepSeek-V4.1-Flash — Reference Implementation

A faithful PyTorch reference implementation of the architectural contributions
described in **"DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"**
(DeepSeek-AI, 2026).

This repository was produced from the technical report (51 pages) following the
three-stage Paper2Code / PaperCoder methodology (planning → analysis → code
generation). It is a *structural* reference: it reproduces every architectural
component and the exact hyper-parameters reported in Section 4.2.1, but it is
**not** a reproduction of the 552B-parameter pretraining run.

---

## What the paper contributes (and where it lives here)

| Paper section | Contribution | Module |
|---|---|---|
| §2.2 | Causal Encoder-Decoder (CED) | `deepseek_v41_flash/ced.py` |
| §2.3 | Compressed Sparse Attention 2 (CSA2), 3 modes | `deepseek_v41_flash/csa2.py` |
| §2.3.2 | Hierarchical Sparse Indexer | `deepseek_v41_flash/csa2.py` (`_candidate_pool`) |
| §2.1 / §2.1.1 | DeepSeekMoE + modality-aware load balancing | `deepseek_v41_flash/moe.py` |
| §2.4.1 | Single-Pass mHC | `deepseek_v41_flash/mhc.py` |
| §2.4.2 | Engram conditional memory | `deepseek_v41_flash/engram.py` |
| §2.4.3 | DSpark speculative decoding | `deepseek_v41_flash/dspark.py` |
| §2.4.4 | FP4 (E2M1) main KV cache | `deepseek_v41_flash/fp4.py` |
| §2.1.1 | DeepSeek-ViT + pixel-unshuffle + projector | `deepseek_v41_flash/vision.py` |
| §3.2 | KV-cache management + SWA Bounded Replay | `deepseek_v41_flash/kv_cache.py` |
| §2.5 / §4.2.2 | Muon / Sinkhorn-balanced optimizers | `deepseek_v41_flash/optim.py` |
| §4.2.1 | Exact model configuration | `deepseek_v41_flash/config.py` |

---

## Model configuration (Section 4.2.1, transcribed verbatim)

```
layers = 40, hidden = 5120              (20 encoder + 20 decoder, CED)
first 2 layers: SWA-only
encoder CSA2:  m=2, 3 groups x 6 = [full, reuse x5]
decoder CSA2:  m=1, 5 groups x 4 = [full, reuse x3] + [reindex, reuse x3] x4
query heads = 64, head dim = 512, query compression = 1280
indexer heads = 32, indexer head dim = 128, attention top-k = 512
Hierarchical indexer: 2048 blocks x 8 positions -> 16,384 candidates
SWA window = 128
MoE: 1 shared + 384 routed experts (inter dim 2304), 6 activated/token,
     SwiGLU clamped at 10
mHC: expansion 4, Sinkhorn-Knopp iters 20
Vision: 32-layer ViT (hidden 1024, 16 heads, patch 14), 3x3 pixel-unshuffle,
        2-layer MLP projector (hidden 5120)
Engram: modules at layers 1 & 14, n-grams {2,3,4}, 8 hash heads, 2048 dim/order
552B backbone + 196B Engram; 8B activated (prefill) / 16B (decode)
```

---

## Running

```bash
# 1. Create an environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch
pip install torch numpy

# 3. Run the smoke tests (uses a tiny config that fits on CPU)
python -m tests.test_smoke
```

The smoke test exercises every component and both text-only and multimodal
forward passes. It uses `DeepSeekV41Config.smoke()` (tiny widths, identical
layer/mode structure) so it runs in seconds on a laptop. The full
`DeepSeekV41Config()` holds the paper's exact values.

---

## Implementation notes / interpretation decisions

The paper reports hyper-parameters but not every low-level layout. Where a
choice had to be made, it is documented below and in the code:

1. **Attention layout.** The paper states "query heads 64, head dim 512" and a
   "query compression dimension 1280". 64 x 512 = 32768 exceeds the hidden size
   (5120); this is consistent with MLA-style *latent* attention (a compressed
   latent expanded to heads), which is how `q_lora_rank`/`kv_lora_rank` are used
   here. The reference records the paper's verbatim values in `config.py` and
   uses a self-consistent MLA layout so the model is runnable.
2. **Global KV has no RoPE.** Consistent with CSA2 removing positional embedding
   from the compressor, RoPE is applied only on the layer-local SWA branch.
3. **Residual vs. mHC.** The main CED stack uses standard pre-norm residuals for
   clarity; `mhc.py` implements the Single-Pass mHC stream update faithfully and
   can wrap any block via `mHCBlock`.
4. **Compression.** `compress()` does a learned linear aggregation of each block
   of `m` tokens (identity for `m=1`), matching CSA2's "no overlap, no absolute
   positional embedding" simplification.

---

## FP4 KV-cache accounting

The headline figure is **890 bytes/token** for the global KV cache. This repo's
FP4 (E2M1 + per-16-channel E4M3 scale) gives ~288 bytes per 512-channel vector,
so main K + main V + indexer K ≈ 864 bytes/token — the same order of magnitude
and mechanism as the paper's accounting (see `kv_cache.py`).
