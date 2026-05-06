# 5-fold whale-gpt-style redesign — R0 (B4 single-channel control) vs R2 (multi-channel multi-task) — `fast` training mode

Sequence-level KFold (5 folds, no within-sequence leakage). Target = `Coda` (V = 131 including PAD). Context K = 25 past codas. Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. Perplexity = 2^(bits/token).

**Apples-to-apples caveat.** The prior `~/Code/whale-gpt @ claude/whale-language-research-tEudI` benchmark used a rhythm·tempo·orn·rubato compound `Token` with V≈207 on the Sharma-only corpus (~4,800 codas). This run targets the rhythm-class integer `Coda1` directly on the unified corpus (~38k codas, V=131), so absolute bits/token are not directly comparable to the 4.63 figure.

| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |
|---|-------|-------:|---------------:|---------------:|---------:|
| R0 | R0 control (B4 mimic): single-channel Coda + DT | 47,588 | 3.215 ± 0.238 | 9.29 | 0.458 ± 0.058 |
| R1 | R1: multi-channel inputs, Coda CE only | 50,496 | 3.244 ± 0.225 | 9.47 | 0.448 ± 0.060 |

## Per-fold bits/token

| fold | R0 control (B4 mimic): single-channel Coda + DT | R1: multi-channel inputs, Coda CE only |
|------|---:|---:|
| 0 | 2.803 | 2.833 |
| 1 | 3.341 | 3.363 |
| 2 | 3.362 | 3.358 |
| 3 | 3.465 | 3.477 |
| 4 | 3.105 | 3.189 |

## Takeaway

- The best held-out model is **R0 control (B4 mimic): single-channel Coda + DT** at 3.215 bits/token (perplexity ≈ 9.29).
- R0 (single-channel control) = **3.215 ± 0.238** bpt
- R1 (multi-channel input, Coda CE only) = **3.244 ± 0.225** bpt; Δ(R0 → R1) = **+0.029**
- Compare each Δ to the fold-variance σ. The R0 → R1 gap isolates *whether the extra input channels help*; the R1 → R2 gap isolates *whether multi-task supervision adds anything on top of multi-channel inputs*.
