# Ablation study: round4_ablation

- Code: git b77ba27fcb1934795621cc48eabd1c4fd1f3cabc (uncommitted changes: False)
- Hardware: NVIDIA GeForce RTX 5070 (sm_120), torch 2.11.0+cu128
- Recipe: config `configs\experiment_3class.json` (locked) (lr 0.001, warmup 3, clip 1.0, BN recalibration True, 25 epochs). Only the model differs.
- Seeds: [42, 43, 44, 45, 46]. Metrics are the checkpoint with the lowest validation loss (selection criterion).
- **Validation only (322 clips). The held-out test set was not used.**
- Deltas are paired by seed (variant minus M0); 95% CIs use the t-distribution with n-1 df.

## Summary

| ID | Variant | Params | Val loss | Val acc (%) | Val macro-F1 (%) | ΔF1 vs M0, pp (95% CI) | ΔLoss vs M0 (95% CI) | Seeds F1 better/worse | Verdict (val loss) |
|---|---|---:|---|---|---|---|---|---|---|
| M0 | Full TF-FaultNet (proposed) | 1,361,379 | 0.1893 ± 0.0021 | 99.32 ± 0.34 | 99.30 ± 0.68 | — | — | — | reference |
| A1 | A1: No SE channel attention | 1,339,875 | 0.2017 ± 0.0082 | 98.88 ± 0.68 | 99.08 ± 0.70 | -0.22 (-1.70, +1.26) | +0.0124 (+0.0009, +0.0240) | 1/4 | removing it HURTS (val loss higher, CI excludes 0) |
| A2 | A2: Average pooling only | 1,295,843 | 0.1980 ± 0.0102 | 98.82 ± 0.46 | 99.03 ± 0.42 | -0.27 (-0.90, +0.36) | +0.0087 (-0.0054, +0.0227) | 1/4 | no clear effect (CI includes 0) |
| A3 | A3: Max pooling only | 1,295,843 | 0.1916 ± 0.0109 | 99.32 ± 0.40 | 99.50 ± 0.30 | +0.20 (-0.72, +1.12) | +0.0022 (-0.0119, +0.0164) | 2/2 | no clear effect (CI includes 0) |
| A4 | A4: No per-spectrogram normalisation | 1,361,379 | 0.1877 ± 0.0111 | 99.32 ± 0.60 | 99.30 ± 0.50 | -0.01 (-1.07, +1.06) | -0.0016 (-0.0166, +0.0134) | 2/3 | no clear effect (CI includes 0) |
| A5 | A5: No residual connections | 1,317,475 | 0.2092 ± 0.0080 | 98.76 ± 0.38 | 98.79 ± 0.57 | -0.51 (-1.18, +0.15) | +0.0198 (+0.0081, +0.0316) | 1/3 | removing it HURTS (val loss higher, CI excludes 0) |

## Reproducibility check

M0 best validation loss vs the Round 2 sweep `bnrecal_lr1e-3` (same recipe, same seeds):

- seed 42: 0.1875 vs reference 0.1875 (|Δ| = 0.00002)
- seed 43: 0.1893 vs reference 0.1893 (|Δ| = 0.00002)
- seed 44: 0.1928 vs reference 0.1928 (|Δ| = 0.00002)
- seed 45: 0.1894 vs reference 0.1894 (|Δ| = 0.00001)
- seed 46: 0.1877 vs reference 0.1877 (|Δ| = 0.00004)

## How to read this

- With 5 seeds and 322 validation clips, a difference whose CI includes 0 is **not** evidence that a component matters.
- Do not claim a component is responsible for an improvement unless its row says "removing it HURTS" (AGENTS.md §28).
- These are validation results. They do not change the locked test result (98.76%).
