# Planning Artifact — DeepSeek-V4.1-Flash

Generated from `DeepSeek_V41_Tech_Report.pdf` (51 pages) following the
PaperCoder *planning* stage: decompose the paper into implementable components,
map each to a file, and identify the key equations.

## Paper scope

The paper's *algorithmic* content (what is implementable) is concentrated in
**Section 2 (Architecture)** and **Section 3 (Infrastructure)**. Sections 4–5
(pre-/post-training) describe data pipelines and RL recipes with "no algorithmic
innovation" (Section 5's own words), so they are summarized rather than coded.
Appendix A is the author list; B/C are evaluation details.

## Component decomposition

| # | Component | Paper ref | Key equation / spec | File |
|---|---|---|---|---|
| 1 | CED encoder/decoder split | §2.2 | `C_l = H_{L/2} W_KV^l`, `Z_l = H_{L/2} W_Z^l` (Eq. 1) | `ced.py` |
| 2 | CSA2 three modes | §2.3 | Full / Reindex / Reuse state rules | `csa2.py` |
| 3 | CSA2 compressor | §2.3 | ratio `m`, no overlap, no abs-pos | `csa2.py::compress` |
| 4 | Hierarchical Sparse Indexer | §2.3.2 | 2048 blocks x 8 → 16384 candidates | `csa2.py::_candidate_pool` |
| 5 | SWA + Bounded Replay | §2.2, §3.2.2 | replay last `n_win` tokens | `kv_cache.py` |
| 6 | Single-Pass mHC | §2.4.1 | `X_{l+1}=B X + C F(A_{l-1}X)` (Eq. 6) | `mhc.py` |
| 7 | Engram | §2.4.2 | n-grams {2,3,4}, 8 heads, 2048 dim | `engram.py` |
| 8 | DSpark | §2.4.3 | 5 draft positions, Markov + confidence heads | `dspark.py` |
| 9 | FP4 KV cache | §2.4.4 | E2M1 + E4M3 scale per 16 ch | `fp4.py` |
| 10 | DeepSeekMoE + balance | §2.1.1 | aux-loss-free, modality-specific bias | `moe.py` |
| 11 | DeepSeek-ViT + projector | §2.1.1 | 32 layers, patch 14, 3x3 unshuffle | `vision.py` |
| 12 | Optimizers | §2.5, §4.2.2 | Muon, Sinkhorn-balanced (Alg. 1) | `optim.py` |
| 13 | Config | §4.2.1 | full hyper-parameter table | `config.py` |

## Layer-mode schedule (Section 4.2.1)

```
Layer 0..1      : SWA-only
Layer 2..19     : encoder CSA2, m=2, 3 x [full, reuse x5]
Layer 20..39    : decoder CSA2, m=1, [full, reuse x3] + 4 x [reindex, reuse x3]
                  (decoder global KV projected from H_19)
```

## Test plan

`tests/test_smoke.py` verifies (a) the mode schedule, (b) FP4 round-trip,
(c) MoE routing, (d) all three CSA2 modes + the candidate pool, (e) mHC, Engram,
DSpark, and (f) text + multimodal forward passes.
