# 5-fold whale-gpt-style redesign — R0 (B4 single-channel control) vs R2 (multi-channel multi-task) — `fast` training mode

Sequence-level KFold (5 folds, no within-sequence leakage). Target = `Coda` (V = 131 including PAD). Context K = 25 past codas. Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. Perplexity = 2^(bits/token).

**Apples-to-apples caveat.** The prior `~/Code/whale-gpt @ claude/whale-language-research-tEudI` benchmark used a rhythm·tempo·orn·rubato compound `Token` with V≈207 on the Sharma-only corpus (~4,800 codas). This run targets the rhythm-class integer `Coda1` directly on the unified corpus (~38k codas, V=131), so absolute bits/token are not directly comparable to the 4.63 figure.

| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |
|---|-------|-------:|---------------:|---------------:|---------:|
| R0 | R0 control (B4 mimic): single-channel Coda + DT | 47,588 | 3.203 ± 0.229 | 9.21 | 0.457 ± 0.053 |
| R2 | R2 redesign: multi-channel + multi-task | 50,496 | 3.319 ± 0.222 | 9.98 | 0.440 ± 0.050 |

## Per-fold bits/token

| fold | R0 control (B4 mimic): single-channel Coda + DT | R2 redesign: multi-channel + multi-task |
|------|---:|---:|
| 0 | 2.796 | 2.893 |
| 1 | 3.341 | 3.391 |
| 2 | 3.333 | 3.349 |
| 3 | 3.432 | 3.430 |
| 4 | 3.110 | 3.534 |

## Takeaway

- The best held-out model is **R0 control (B4 mimic): single-channel Coda + DT** at 3.203 bits/token (perplexity ≈ 9.21).
- R0 control (B4 mimic) = **3.203 ± 0.229** bpt; R2 multi-channel multi-task = **3.319 ± 0.222** bpt. Δ(R0 → R2) = **+0.117 bits/token**.
- Compare Δ to fold-variance σ in the rightmost column. If |Δ| < σ, the redesign is within noise; if |Δ| ≥ σ the new input channels + multi-task supervision earned their complexity.
