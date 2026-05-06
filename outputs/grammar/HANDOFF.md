# Transformer benchmark — handoff for next session

What's been done, where we are, and what to run next. Read this end-to-end before touching the code; everything you need is linked from here.

## TL;DR

We ported the prior `~/Code/whale-gpt @ tEudI` transformer benchmark to the unified corpus (38 k codas across Sharma-DSWP + Sharma-birth + Hersh-Pacific), then walked through three experiment generations on the same fold split:

| generation | architecture | best variant | 5-fold bpt | accuracy |
|---|---|---|---:|---:|
| 1. initial M0–M7 | 8 baselines, K=8, single-target `Coda1` | MiniTransformer M7 (2L, 4h, d=64) | 3.186 ± 0.237 | 0.464 |
| 2. ablation B0–B4 | depth × DeltaTime × whale-gpt-shape, K=8 or 25 | B4 (3L, 8h, d=32, K=25, +DT) | 3.170 ± 0.246 | 0.468 |
| 3. redesign R0 vs R2 | whale-gpt-style 6-col schema + multi-channel multi-task | **R0 control (B4 mimic)** | **3.202 ± 0.255** | **0.457** |
| | | R2 multi-channel multi-task | 3.319 ± 0.227 | 0.440 |

**Net finding so far**: the schema redesign (Whale, Synchrony, has_timestamps, drop sub-split, log-DT) was a corpus-side win — useful for analysis — but the **multi-channel + multi-task transformer (R2) is +0.117 bpt worse than the single-channel B4 mimic (R0)** in 5-fold CV. R2 trails R0 on 4/5 folds.

User's hypothesis: multi-task aux heads were the active poison, not multi-channel inputs. The next experiment isolates that. **R1 = multi-channel input + Coda CE only (no aux heads)** is wired and ready to run.

## What to run next

```bash
python -u -m src.grammar.predict_kfold --redesign --variants R0,R1 \
  2>&1 | tee reproducibility/logs/predict_kfold_redesign_r0r1.log
```

Wall-time ≈ 35–40 min on a single CPU (5 folds × ~7 min/fold for R0 + R1; R2 dropped to halve compute, since we already have its 5-fold result).

What this answers:

- **R0 → R1 gap**: do the extra input channels (Whale, Synchrony, Ornamentation, Duration, log-DT, has_timestamps) help when supervised by Coda CE alone?
- **R1 → R2 gap (using R2's 5-fold from the previous run)**: does multi-task supervision add anything on top of multi-channel inputs?

Decision rule for what to ship:

| outcome | recommendation |
|---|---|
| R1 ≤ R0 (within σ) | Drop the redesign model entirely — channels and aux heads both unhelpful. Ship R0 as the production model. Schema stays for non-modeling uses. |
| R1 < R0 by ≥ 0.05 bpt | Multi-channel inputs help; multi-task heads were the poison. Ship R1 as production. Document that aux heads should NOT be added back. |
| R1 ≈ R0 but R2 (existing) >> R1 | Confirms multi-task heads are net-negative. Ship R0 (simpler). Same takeaway as the first row but cleaner attribution. |
| R1 < R0 *and* R2 < R1 | Surprise: revisit the multi-task loss weighting; current weights `(coda=1, whale=1, orn/sync=0.5, dur/td=0.2)` may be wrong. |

## Result files written so far

| file | what it contains |
|---|---|
| `outputs/grammar/predict_results_unified.{md,json}` | M0–M7 5-fold (generation 1) |
| `outputs/grammar/predict_results_ablation.{md,json}` | B0–B4 5-fold (generation 2) |
| `outputs/grammar/predict_results_redesign.{md,json}` | R0 + R2 5-fold (generation 3, current) |
| `outputs/grammar/predict_results_quick.{md,json}` | smoke-test artifact (50-seq subsample) |
| `outputs/grammar/REPRODUCING_TRANSFORMER.md` | how to reproduce the generation-1 numbers from scratch |
| `reproducibility/logs/predict_kfold*.log` | fold-by-fold print logs for each run |

The next R1 run will write `outputs/grammar/predict_results_redesign.{md,json}` — **note this overwrites the current R0+R2 file**. Move/rename it first if you want to preserve the R0+R2 result for archival comparison:

```bash
mv outputs/grammar/predict_results_redesign.json outputs/grammar/predict_results_redesign_R0R2.json
mv outputs/grammar/predict_results_redesign.md   outputs/grammar/predict_results_redesign_R0R2.md
```

## What's in the codebase that's load-bearing

| path | role |
|---|---|
| `src/pipeline/E_render_csv.py` | renders the new 6-col `whale_dialogues.csv` + `whale_id_index.csv` |
| `src/grammar/predict_kfold.py` | benchmark runner (`--quick`, `--ablation`, `--redesign --variants R0,R1,R2`, `--slow`) |
| `data/classified/whale_dialogues.csv` | 38,840 rows, 488 sequences, 9-column whale-gpt-style schema |
| `data/classified/whale_id_index.csv` | `Whale` string → int (151 distinct ids incl. UNK) |
| `data/classified/rhythm_class_index.csv` | `Coda` int → cluster name (V=131) |
| `tests/test_render_csv.py` | 9 invariants on the rendered CSV; all currently green |
| `docs/transformer_corpus_design.md` | full schema doc + comparison vs whale-gpt's `train-dialogue-script.yaml` |
| `reproducibility/whale_grammar_transformer_plan.md` | the active forward plan (predates the redesign experiments) |

## The R0/R1/R2 model spec

All three at architecture **3L, 8h, d=32, K=25** (whale-gpt-main shape), **fast training mode** (AdamW + lr=1e-3 + bs=128 + 80 epochs + ES patience 10).

| variant | inputs | loss |
|---|---|---|
| **R0 control** | single channel: `Coda` token + `(log_dt, has_timestamps)` projection | CE on Coda only |
| **R1 multi-channel single-task** | per-column embeddings concat to d=32: `Whale`(6) + `Coda`(8) + `Orn`(3) + `Sync`(3) + `Duration`(4) + `TimeDelta_log`(4) + `has_timestamps`(4) | CE on Coda only |
| **R2 multi-channel multi-task** | same inputs as R1 | CE on Whale + CE on Coda + 0.5·CE on Orn + 0.5·CE on Sync + 0.2·L1 on Duration + 0.2·L1 on TimeDelta_log |

Param count: ~48 k for R0, ~50 k for R1 / R2.

DT encoding: `log(0.1 + max(dt, 0))` for present rows; `log(0.1)` (≈ -2.3) sentinel for missing rows; the `has_timestamps` channel separately tells the model the value is uninformative on Hersh.

## Per-source data populations (re-stated for the next session)

| feature | hersh2022_pacific (24,237) | sharma2024_dswp (8,872) | sharma2025_birth (5,731) |
|---|---:|---:|---:|
| `has_timestamps = 1` | 0 % | 42 % | 100 % |
| `Whale != ::UNK` | 0 % | 62 % | 100 % |
| `Synchrony == 1` | 0 % | 9 % | 26 % |
| `Ornamentation == 1` | 0 % | 3 % | 11 % |

Hersh is **62 % of the corpus** and is structurally NA on every time-derived or speaker-derived column. The `has_timestamps` flag is the model's gate for those channels; on Hersh it's always 0 and the model should learn to ignore them. Whether it actually does is part of what R1 vs R0 is testing.

## Why we expect the R0 → R1 gap to be small (the bear case)

With Hersh dominating the corpus, the multi-channel embeddings spend most of their capacity on majority-class entries:
- `Whale = ::UNK` for 62 % of rows (Hersh) + ~3 % of rows (DSWP NA) → the UNK embedding is the "default Whale" the model sees most of the time.
- `Synchrony = 0` for ~92 % of rows → the Sync embedding is mostly the "no chorus" default.
- `Ornamentation = 0` for ~98 % of rows → similar story.
- `has_timestamps = 0` for 62 % of rows.

So the genuinely-informative channels (Whale on the 38 % of rows where it's known; Sync on the 8 % where it fires) are a small fraction of the windows the model sees during training. The strong prior — and what generation-3's R2 result already shows — is that the marginal information is below the σ ≈ 0.25 fold-variance floor.

## What would change my mind on shipping R0

If R1 beats R0 by ≥ 0.05 bpt on the 5-fold mean *and* on at least 4/5 folds individually, that's enough signal to consider R1 as production. Anything less and R0 wins on simplicity.

## Commits

- **`2ec8e51`** ship classifier pipeline + transformer port (Stage 1+2) — pre-redesign revert point
- **`08dc5a6`** schema redesign: whale-gpt-style 6-col input + multi-task transformer

The R1 wiring is uncommitted at handoff time — see `git diff src/grammar/predict_kfold.py` for the changes (added `single_task` flag to `_train_multitask`, `_eval_multichannel`, R1 branch in `evaluate_redesign_fold`, `--variants` CLI flag, R1 markdown labels).

## Outstanding questions for the next session

1. **If R1 also fails**, is it worth one expensive `--slow` run (R0 + R2 in slow mode, ~15 hours) to test whether the FAST_MODE training recipe is hiding gains the multi-task setup would otherwise show? Probably no — the corpus is small enough that ES at epoch 30–50 is already finding the loss floor.
2. **Vocabulary axis** — we never tested whether the V=131 OPTICS-discovered vocab is better/worse than the prior compound `Token` (V≈207, rhythm × tempo × orn × rubato fused). Different axis from architecture. Worth a separate experiment.
3. **K-sweep on the strongest model** — once we settle on R0 vs R1, sweep K ∈ {4, 8, 16, 24, 32} to verify K=25 is actually the best context length and not noise.
4. **Per-source bpt breakdown** — does R0 do uniformly well, or is it carried by DSWP/birth and weak on Hersh? Currently we report only aggregate. A per-source breakdown might surface "Hersh is genuinely a different distribution" as a structural finding.
