---
tags:
  - interp
  - continuations
summary: Sampled completions from the locked MiniTransformer-DT — held-out seeds with reference continuations
created: 2026-05-06
updated: 2026-05-06
---

# Continuations

Held-out generation samples from the persisted MiniTransformer-DT (2L, 4h, d=64, K=8 + DT). For each corpus we hold out 3 sequences (random, fixed seed), feed the first K=8 tokens, and autoregressively sample 3 continuations of length ≈ mean training-set sequence length.

Sampling: T=0.9, top-k=40, multinomial. DT during generation: each new position gets the *mean* `(log_dt, has_ts)` from training, so the DT channel contributes a constant bias and the continuation is governed by the token-id stream.

Code: `src/grammar/continuations.py`. Output: `outputs/grammar/continuations.md`.

## Whale (V=467 compound, mean seq length = 80, continuations = 72 tokens)

Whale tokens are compound `rhythm|tempo|orn|sync` strings. `?` is OOV; `·` is PAD.

The qualitative observation across all 9 sampled whale continuations:

- **Mode-collapse to the corpus mode.** The dominant tokens in nearly every continuation are `1+1+3|t1|*` and `5R|t1|*` — the two highest-frequency rhythm classes in the unified corpus. The model has learned a strong unigram prior plus "stay in the same rhythm class as the last token within reason".
- **Realistic switches between t1, t2, and occasionally t3.** Tempo bin doesn't get stuck the way rhythm class does — the model alternates `t1 ↔ t2` at roughly the right empirical rate.
- **Ornament-1 fires occasionally** (e.g. `1+1+3|t2|o1|r0`) at low rate, matching the ~3 % corpus base rate on DSWP/birth.
- **Rare classes show up.** The continuations include `7R|t4`, `6R|t3`, `8R|t4`, `2+3|t1`, `4D|t1`, `3+1+1+1+1|t2` — not just the modal class. The model isn't a unigram with extra steps.
- **Long-run repetition like the actual corpus.** Whale recordings have stretches of dozens of identical-or-near-identical codas. The continuations reproduce that texture (and that's why the [[transformer/smoothed-baselines|recency-cache]] model is so close to the transformer in bpt — both are picking up the same long-run repetition).

For the actual seed text + reference continuation + sampled continuations, see `outputs/grammar/continuations.md`. Those tables are large — there are three seeds × three samples × ~70 tokens each, so embedding them here would mostly be wallpaper.

## CHILDES UK (V=1605 lemma, mean seq length = 97, continuations = 89 tokens)

CHILDES tokens are lemmas via the `%mor` tier (`go` covers `going`/`went`/`gone`).

Qualitative read on the 9 sampled CHILDES continuations:

- **Locally fluent, semantically random.** Continuations like `*MOT:	oh that be a little spider` parse as English, conjugate noun phrases roughly correctly, and respect speaker-turn structure (`*MOT:` then `*CHI:` then `*MOT:` etc.) — but topic doesn't persist.
- **The DT head's `switch` bucket drives the speaker rotation.** When the DT head fires the switch bucket the renderer ends the utterance and changes speaker; when it fires `period` it ends the utterance with `.` and stays on the same speaker. So `*SPEAKER:` lines come out at roughly the right rate — that's a generation feature, not a model claim.
- **No long-range coherence.** A 89-token transformer trained on 39k tokens isn't going to track topic. The headline qualitative use is "is the model outputting plausible English at all" — yes — and "do tokens within ~K=8 of each other agree on POS / number / agreement" — mostly.

## What this evaluates

These samples don't have a numeric metric attached — held-out bpt is in [[transformer/results]] and the persisted-checkpoint validation numbers are in [[interp/interp]]. The continuations are the *qualitative* artifact: read them and decide whether the model is doing what the bpt numbers say it's doing.

For whale: the answer is "the model is doing rhythm-class autocorrelation with realistic tempo switching and rare-class spice". For CHILDES: "the model is doing locally fluent English with no topic structure". Neither claim was strongly contested by the bpt numbers — but seeing the actual output makes the claim concrete.

## Reproduce

```bash
python -m src.grammar.continuations
```

Wall time ≈ 10 min on CPU (trains a fresh checkpoint on the held-out-3-seqs subset). The persisted checkpoints under `outputs/grammar/checkpoints/{whale,childes_v1605}/` are *separate* — they're trained on the full corpus and used by the interp scripts. The continuations script trains its own one-off model so the seeds are honestly held out.

## Related

- [[interp/interp]] — what's happening internally during generation.
- [[transformer/transformer]] — the architecture being sampled from.
- The full sample tables: `outputs/grammar/continuations.md`.
- Per-source seeds also rendered alongside attention HTMLs at `outputs/grammar/interp/{whale,childes}/continuations.md`.
