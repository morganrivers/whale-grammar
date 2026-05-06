---
tags:
  - index
summary: Entry point for the whale-grammar Obsidian vault — start here
created: 2026-05-06
updated: 2026-05-06
---

# whale-grammar — Obsidian vault

Scientific and code documentation for the [whale-grammar](https://github.com/morganrivers/whale-grammar) repository. Four self-contained islands of notes; each has its own index page and they cross-link where it helps. Open the [[README|vault README]] for the full table of contents.

## Start here

- New to the repo → [[overview/overview]] for the pipeline diagram.
- Want to know what a token in `whale_dialogues.txt` means → [[overview/glossary]].
- Want to reproduce a result → [[overview/reproduce]].
- Want to understand the classifier — *how* a coda becomes a `coda_type_gero21` → [[classifier/classifier]].
- Want to understand the transformer — what predicts the next coda and why → [[transformer/transformer]].
- Want to understand what the trained model is doing inside → [[interp/interp]].

## The four islands

| island | what it covers |
|---|---|
| [[overview/overview\|Overview]] | repo layout, pipeline diagram, per-source corpora, glossary, end-to-end commands |
| [[classifier/classifier\|Classifier]] | hybrid OPTICSXi + kNN+τ + discovery; reproduces Gero 2016 at 96 % on DSWP |
| [[transformer/transformer\|Transformer]] | next-coda baselines, ablation, schema-redesign, smoothed classical, CHILDES comparison |
| [[interp/interp\|Interpretability]] | per-layer accuracy decomposition, induction-head check, sampled continuations |

## Conventions

- One frontmatter block per file (`tags`, `summary`, `created`, `updated`).
- Wikilinks `[[file]]` for cross-vault references; relative paths for code (`src/grammar/...`).
- Auto-generated result files (`outputs/grammar/predict_results_*.md`, `outputs/grammar/interp/SUMMARY.md`) live outside the vault and are linked from inside.
- Reference PDFs live at `docs/`. Notes that *cite* a paper provenance live at [[classifier/papers]].
