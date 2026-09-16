# Analysis Artifact — DeepSeek-V4.1-Flash

Generated from `DeepSeek_V41_Tech_Report.pdf` following the PaperCoder
*analysis* stage: verify the technical claims are understood correctly before
writing code, and record the interpretation decisions.

## Core idea

DeepSeek-V4.1-Flash attacks the **KV-cache bottleneck** along three
multiplicative dimensions (§2.3):

1. **Entry size** — GQA / MLA latent sharing, plus FP4 quantization (§2.4.4).
2. **Sequence** — every `m` tokens compressed into one entry (CSA2 ratio `m`).
3. **Layer** — cross-layer reuse of main KV, indexer K, and Top-K indices.

The headline result is a global KV cache of **890 bytes/token** (~1/4 of
DeepSeek-V4-Flash) and a persistent cache ~1/8 of V4-Flash.

## CED analysis (§2.2)

CED lets most prompt tokens bypass full decoder computation: decoder *global* KV
is projected from the encoder's final hidden state `H_{L/2}` via layer-dependent
`W_KV^l`, `W_Z^l`. Only SWA KV stays layer-local. This halves prefill FLOPs
(`O(NL)` → `O(NL/2 + n_win * L/2)`). *Consequence in code:* the decoder's
`full` CSA2 layer takes `h_global = H_19`, not its own hidden state.

## CSA2 mode semantics (§2.3.1)

- **Full** — compute main KV + indexer K, index over the full range.
- **Reindex** — reuse main KV + indexer K, recompute Top-K from own indexer Q.
- **Reuse** — reuse main KV + latest Top-K; no indexing.

Every mode still computes its own main Q and SWA KV (Figure 4). The indexer K is
projected *from* the main KV (not a separate compression path from hidden
states), which is CSA2's simplification over CSA.

## Hierarchical Sparse Indexer (§2.3.2)

Decoder-only. The decoder's first `full` layer scores all positions and, via
block-wise max-pooling (2048 blocks x 8 positions), builds a 16,384-position
candidate pool. Later `reindex` layers score *only* this pool, making their
per-query cost constant in context length. The pool is shared; final selections
differ per layer.

## FP4 format (§2.4.4)

E2M1 data (max 6.0) with one E4M3 scale (max 448.0) per 16 channels — the
NVFP4 layout minus the second-level global scale (max magnitude 448 x 6 = 2688,
far above the KV norm bound ~√512). Quantized *after* RoPE. Dequantized before
attention (no native FP4 matmul needed).

## Interpretation decisions (carried into code)

| Decision | Rationale |
|---|---|
| MLA-style latent attention (`q_lora`/`kv_lora`) | "query compression 1280", "512-channel KV latent" imply latent attention; 64 x 512 heads would exceed hidden dim |
| No RoPE on global/main KV | CSA2 removes positional embedding from the compressor |
| Standard pre-norm residual in the main stack | keeps CED/CSA2 clear; `mHCBlock` provides the faithful mHC stream wrapper |
| Learned linear block-compression for `m>1` | CSA2's exact compressor isn't specified beyond "no overlap" |

## Result

Every architectural contribution has a runnable counterpart, and the smoke test
confirms correct shapes and finite outputs for text-only and multimodal forward
passes.
