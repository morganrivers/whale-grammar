---
tags:
  - index
summary: Entry point for the whale-grammar vault — four islands of documentation
created: 2026-05-06
updated: 2026-05-06
---

# whale-grammar vault

Four self-contained islands. Each one has its own index page; cross-links exist where they help.

## [[overview/overview|Overview]]

What this repo does, the corpora it reads, the pipeline it runs, the glossary of terms, and how to reproduce everything.

- [[overview/overview]] — pipeline diagram + repo layout
- [[overview/corpora]] — Hersh Pacific + Sharma DSWP + Sharma birth: who has what
- [[overview/whale-clans-pacific-vs-caribbean]] — why V=218 (Caribbean compound + rubato, no chorus folded in) is the right reference, not V=467 (unified, multi-clan-pooled)
- [[overview/glossary]] — rhythm / tempo / rubato / ornament / synchrony defined once
- [[overview/reproduce]] — end-to-end commands
- [[Welcome]] — vault entry point

## [[classifier/classifier|Classifier]]

The hybrid OPTICSXi + kNN pipeline that labels each coda with a `coda_type_gero21` and an integer `rhythm_class`.

- [[classifier/classifier]] — overview + four-numbers regression baseline
- [[classifier/gero-2016-replication]] — Phase 1: reverse-engineering `minpts=10`
- [[classifier/pacific-extension]] — Phase 1b: kNN-on-DSWP-anchors for Pacific
- [[classifier/locked-parameters]] — frozen constants + ELKI/JRE vendoring URLs
- [[classifier/papers]] — paper-by-paper provenance for every fact

## [[transformer/transformer|Transformer]]

Next-coda prediction on `whale_dialogues.csv`. Baselines + MiniTransformer; schema redesign experiment; smoothed-classical models; English (CHILDES) cross-corpus comparison.

- [[transformer/transformer]] — roadmap + which models are best now
- [[transformer/corpus-design]] — the 9-column CSV schema and what each column means
- [[transformer/results]] — all bits-per-token tables in one place
- [[transformer/schema-redesign]] — the R0/R1/R2 multi-channel multi-task experiment
- [[transformer/smoothed-baselines]] — Modified Kneser-Ney 5-gram + cache + PPM-D
- [[transformer/childes-comparison]] — same architecture on Eng-UK lemma stream
- [[transformer/vocab-cap-experiment]] — cap CHILDES at V=467 (matches whale); −0.87 bpt, no overfit
- [[transformer/hersh-mirror]] — synthetic 62% missingness on CHILDES (Hersh-equivalent NA)
- [[transformer/cross-language-units]] — sub-lexical units across English / Mandarin / Japanese vs whale Dominica compound
- [[transformer/multilang-corpus-plan]] — proposed 3-language CHILDES counterpart at V≈218

## [[interp/interp|Interpretability]]

What the trained MiniTransformer is doing internally and how the whale model differs from the CHILDES model trained on the same architecture.

- [[interp/interp]] — per-layer accuracy decomposition + induction-head check
- [[interp/continuations]] — sampled completions with held-out reference

## Conventions

- One frontmatter block per file (`tags`, `summary`, `created`, `updated`).
- Wikilinks `[[file]]` for cross-vault references; relative paths for code (`src/grammar/...`).
- Auto-generated result files (`outputs/grammar/predict_results_*.md`, `outputs/grammar/interp/SUMMARY.md`) live outside the vault and are linked from [[transformer/results]] and [[interp/interp]].
- Reference PDFs (Gero 2016 supp, Sharma 2024 supp, Hersh 2022 supp, Sharma methods extract) are at `docs/`.
