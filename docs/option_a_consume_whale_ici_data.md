# Plan: refactor whale-grammar to consume whale-ici-data's unified corpus

**Status:** not started. Plan written 2026-05-03.

**Goal:** stop maintaining a duplicate ICI-corpus build inside whale-grammar.
whale-ici-data (https://github.com/morganrivers/whale-ici-data) is the
canonical corpus build. whale-grammar should be a downstream renderer that
loads `codas_unified.csv`, runs the rhythm/ornament classifier, and produces
the human-readable dialogue script.

After this refactor, fixing or extending the corpus happens in one place.

---

## Why this refactor exists

whale-grammar and whale-ici-data both started from the same commit
(`76b4c60 Initial whale-ici-data corpus`). They diverged: whale-ici-data
became a pure ICI-corpus build (3 sources, 38,840 codas, no Gero, includes
Hersh, lat/lon columns). whale-grammar kept the older A/B/C loaders + Gero
and grew an `E_classify` and `F_render_readable` on top.

Audit on 2026-05-03 found that whale-grammar's local state is:
- still on the original `76b4c60` commit, no remote, nothing pushed
- still has `C_load_gero_atlantic.py`, `gero2016_*.xlsx` in `data/raw/`
- missing `hersh2022_pacific_codas.csv`
- `schema.py` lacks `latitude`/`longitude` columns
- `data/raw/sw_combinatoriality_rhythms.p` and `..._ornaments.p` are missing,
  which means `E_classify` cannot currently run

Mirroring all the whale-ici-data fixes into whale-grammar would touch ~10
files plus a long README rewrite, and create an ongoing maintenance burden
every time whale-ici-data changes. Consuming `codas_unified.csv` directly
eliminates the duplication.

---

## Architectural decisions baked into this plan

### 1. How whale-grammar finds `codas_unified.csv`

**Decision: curl from GitHub raw on first use, cache to `data/cache/`,
re-use the cache on subsequent runs.**

Default branch is `main`. URL template:
`https://raw.githubusercontent.com/morganrivers/whale-ici-data/main/<path>`

Files fetched:
- `data/unified/codas_unified.csv` — the corpus (~30 MB)
- `data/raw/sw_combinatoriality_dialogues.csv` — labelled training subset
- `data/raw/sw_combinatoriality_rhythms.p` — rhythm labels
- `data/raw/sw_combinatoriality_ornaments.p` — ornament labels

Cache layout (mirrors upstream paths so the file URL ↔ cache path mapping
is obvious):

```
data/cache/
├── data/unified/codas_unified.csv
└── data/raw/
    ├── sw_combinatoriality_dialogues.csv
    ├── sw_combinatoriality_rhythms.p
    └── sw_combinatoriality_ornaments.p
```

Add `data/cache/` to `.gitignore` — these are derived artifacts.

CLI: `--refresh` flag on the entry point forces a re-download of every
file before running the pipeline. Without it, the cache is reused
indefinitely.

Rejected alternatives:
- **sibling-repo path** — requires the user to clone whale-ici-data
  separately and keep it in a specific location.
- **git submodule** — friction every clone/pull, overkill for one CSV.
- **Pin to a specific SHA** — would freeze the corpus version. We want the
  renderer to see corpus updates by default; `--refresh` is the explicit
  knob.

If the network is unreachable on first run (no cache yet), fail loudly with
the URL and the cache path so the user can manually `curl` the file in.

### 2. Hersh codas in the renderer

**Decision: render all sources by default; add a `--sources` CLI flag for
filtering.**

Hersh has `recording_id` (`grpvar` = repertoire-day) but no
`time_in_recording_s`. The current renderer would create ~191 sections
(one per Hersh repertoire-day), each ordered by `source_coda_id` since no
timestamp is available. Galápagos repertoires can have 10,000+ codas in one
section.

Rendering all of them produces a useful but massive dialogue file.
Filtering is a usability concern, not a correctness one — handle it with a
flag, not a default exclusion. Sensible filter values:
`--sources sharma2024_dswp,sharma2025_birth` for Caribbean-only output.

### 3. Training labels for `E_classify`

**Decision: training files live in whale-ici-data (already there), and
whale-grammar fetches them via the same curl-and-cache mechanism as the
unified CSV.**

`sw_combinatoriality_rhythms.p` and `sw_combinatoriality_ornaments.p` are
already committed in `whale-ici-data/data/raw/`. They're DSWP-derived
training labels — conceptually corpus assets, not whale-grammar's own data.
Keep them upstream.

`B_classify` resolves them by calling the same fetch helper, with one
URL/cache entry per file. `--refresh` on the entry point invalidates these
along with the unified CSV.

### 4. Schema reuse

**Decision: drop whale-grammar's `schema.py`. Read column expectations from
the unified CSV header at run-time.**

`F_render_readable` only needs a fixed subset of columns
(`source`, `recording_id`, `date`, `time_in_recording_s`, `n_clicks`,
`coda_duration_s`, `whale_photo_id`, `local_speaker_id`, `clan`,
`location`, `rhythm`, `extra_click`). It already reads them positionally
via `pd.read_csv`. No schema module needed; if the upstream schema gains
columns, the renderer is unaffected.

`E_classify` only needs `ICI1`..`ICIN` and the same identifier columns.
Same story.

### 5. Pipeline letter scheme

After the refactor, the surviving stages are: **load (E)**, **classify (F)**,
**render (G)**. Renumber for clarity:

- `A_load_unified.py` — fetch `codas_unified.csv` from whale-ici-data
- `B_classify.py` — port from current `E_classify.py`, drop loader-specific
  bits
- `C_render_readable.py` — port from current `F_render_readable.py`
- Entry point: `D_run.py` (or rename to a clearer `run.py`)

Rejected: keeping E/F/G to preserve old letters. The whale-grammar pipeline
is now four steps end-to-end; restarting at A is honest about that.

---

## Step-by-step execution

Run these in order. Each step ends with a clean checkpoint that can be
committed independently if needed.

### Step 0 — pre-flight

Verify the upstream repo is reachable and the file paths still exist:

```bash
curl -sfI https://raw.githubusercontent.com/morganrivers/whale-ici-data/main/data/unified/codas_unified.csv | head -1
# expect: HTTP/2 200

curl -sfI https://raw.githubusercontent.com/morganrivers/whale-ici-data/main/data/raw/sw_combinatoriality_rhythms.p | head -1
curl -sfI https://raw.githubusercontent.com/morganrivers/whale-ici-data/main/data/raw/sw_combinatoriality_ornaments.p | head -1
curl -sfI https://raw.githubusercontent.com/morganrivers/whale-ici-data/main/data/raw/sw_combinatoriality_dialogues.csv | head -1
```

Spot-check corpus contents:

```bash
curl -s https://raw.githubusercontent.com/morganrivers/whale-ici-data/main/data/unified/codas_unified.csv | head -1
# expect: header row starting with "source,source_doi,..."
```

Expect (as of 2026-05-03): 24,237 hersh + 8,872 dswp + 5,731 birth = 38,840.
If row counts have shifted, that's fine — the renderer handles whatever
`source` values appear.

### Step 1 — strip dead code from whale-grammar

Delete these files entirely:

```
src/pipeline/A_load_dswp.py
src/pipeline/B_load_birth.py
src/pipeline/C_load_gero_atlantic.py
src/pipeline/schema.py
data/raw/dswp_dominica_codas.csv
data/raw/sharma2025_birth.csv
data/raw/gero2016_atlantic_clans.xlsx
data/raw/gero2016_individual_clan_id.xlsx
data/raw/sw_combinatoriality_dialogues.csv
data/intermediate/A_dswp.csv
data/intermediate/B_birth.csv
data/intermediate/C_gero_atlantic.csv
data/unified/codas_unified.csv
```

Both `data/intermediate/` and `data/raw/` should be removed entirely —
whale-grammar no longer holds any input data. The cache lives at
`data/cache/` (gitignored) and is populated at run-time.

Add to `.gitignore`:
```
data/cache/
```

Commit the deletion separately so the diff is reviewable:
```
git commit -m "remove duplicated ICI-corpus build; whale-grammar now consumes whale-ici-data"
```

### Step 2 — write `src/pipeline/A_load_unified.py`

New module. Single responsibility: fetch upstream files via curl-from-raw,
cache locally, and load the unified CSV.

```python
"""Fetch the unified ICI corpus and training labels from whale-ici-data on GitHub.

Files are downloaded once into data/cache/ (gitignored), keyed by their
upstream path, and re-used on subsequent runs. Pass refresh=True to any
fetch helper, or set --refresh on the entry point, to re-download.
"""
from pathlib import Path
import urllib.request
import shutil
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "cache"
RAW_BASE = "https://raw.githubusercontent.com/morganrivers/whale-ici-data/main"


def _fetch(upstream_path: str, *, refresh: bool = False) -> Path:
    """Ensure the file at <RAW_BASE>/<upstream_path> exists in CACHE; return its local path."""
    local = CACHE / upstream_path
    if local.exists() and not refresh:
        return local
    url = f"{RAW_BASE}/{upstream_path}"
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"A_load_unified: downloading {url}")
    try:
        with urllib.request.urlopen(url) as resp, open(local, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch {url}.\n"
            f"  Cache target: {local}\n"
            f"  If your network is offline, manually place the file at the cache target.\n"
            f"  Underlying error: {e}"
        )
    return local


def unified_csv(refresh: bool = False) -> Path:
    return _fetch("data/unified/codas_unified.csv", refresh=refresh)


def training_file(name: str, refresh: bool = False) -> Path:
    """For B_classify: rhythms.p, ornaments.p, dialogues.csv."""
    return _fetch(f"data/raw/{name}", refresh=refresh)


def load(refresh: bool = False) -> pd.DataFrame:
    return pd.read_csv(unified_csv(refresh=refresh), low_memory=False)
```

Note: do not duplicate the schema. If a column is missing the renderer
will fail naturally with a clear `KeyError`.

### Step 3 — port `E_classify` to `B_classify`

Rename `src/pipeline/E_classify.py` → `src/pipeline/B_classify.py`.

Replace its file-path constants with lazy fetches inside `build_centroids`
and `classify` (don't call them at import time — that triggers a download
when the user only wants `--help`):

```python
from . import A_load_unified

def build_centroids(refresh=False):
    labeled_csv     = A_load_unified.training_file("sw_combinatoriality_dialogues.csv", refresh=refresh)
    labeled_rhythms = A_load_unified.training_file("sw_combinatoriality_rhythms.p",     refresh=refresh)
    ...
```

Pass `refresh` through `classify(df, refresh=False)` so the entry point can
plumb the CLI flag down.

Drop the `UNIFIED_CSV` constant — `B_classify.classify(df)` is called with
the DataFrame from `A_load_unified.load()`, never reads its own CSV.

The `main()` block can stay for standalone debugging but should call
`A_load_unified.load()` instead of reading `data/unified/codas_unified.csv`.

### Step 4 — port `F_render_readable` to `C_render_readable`

Rename `src/pipeline/F_render_readable.py` → `src/pipeline/C_render_readable.py`.

Update the docstring that says "DSWP/Gero rows lacking timing" → "DSWP/Hersh
rows lacking timing".

CLI handling moves to `D_run.py` (Step 5) — the renderer's `main()` becomes
a thin wrapper that loads, classifies, and renders without flags.

The `render()` function itself needs no changes — it already groups by
`source` and `recording_id` and handles missing timestamps.

**Sanity-check the output size on the first run.** 191 Hersh repertoire-day
sections is a lot. If the file is unwieldy, add a per-section truncation
(e.g. first 200 codas) before deciding on a permanent design.

### Step 5 — write the new entry point

Create `src/pipeline/D_run.py` (or `run.py` at the repo root if you prefer
no package prefix):

```python
"""whale-grammar pipeline: load corpus -> classify -> render dialogue script."""
import argparse
from . import A_load_unified, B_classify, C_render_readable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=None,
                        help="comma-separated source keys to render "
                             "(e.g. sharma2024_dswp,sharma2025_birth). Default: all.")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download all whale-ici-data files before running.")
    args = parser.parse_args()

    df = A_load_unified.load(refresh=args.refresh)
    df = B_classify.classify(df, refresh=args.refresh)
    if args.sources:
        keep = set(args.sources.split(","))
        df = df[df["source"].isin(keep)].copy()
    C_render_readable.render(df, C_render_readable.OUT)
    print(f"D_run: rendered {len(df):,} codas -> {C_render_readable.OUT}")

if __name__ == "__main__":
    main()
```

Delete `src/pipeline/D_merge_unified.py` once `D_run.py` is wired up.

### Step 6 — update `requirements.txt`

The Hersh-loader-only dependency `openpyxl` (used by the deleted Gero
loader for `.xlsx`) can come out. Verify by grepping the surviving modules:

```bash
grep -rn "openpyxl\|xlrd" src/pipeline/
```

Expected: no hits.

### Step 7 — update whale-grammar README

Rewrite top-to-bottom. Key sections:

```markdown
# whale-grammar

A renderer that turns the unified sperm whale ICI corpus into a readable
dialogue script. Reads `codas_unified.csv` from the sibling repo
[whale-ici-data](https://github.com/morganrivers/whale-ici-data), assigns a
rhythm + extra-click label per coda (Sharma 2024 reconstruction), and emits
`data/readable/whale_dialogues.txt` with codas tokenised as `<letter><digit>`.

## Sibling repos

- **[whale-ici-data](https://github.com/morganrivers/whale-ici-data)** —
  upstream corpus. whale-grammar reads its `codas_unified.csv` and the
  `sw_combinatoriality_*` training labels from this repo. Clone it as a
  sibling: `git clone https://github.com/morganrivers/whale-ici-data` from
  the parent directory of this repo, or set `$WHALE_ICI_DATA_DIR`.

## Reproduce

  pip install -r requirements.txt
  python -m src.pipeline.D_run
  # or filter:
  python -m src.pipeline.D_run --sources sharma2024_dswp,sharma2025_birth

## Pipeline

  whale-ici-data:codas_unified.csv  -->  B_classify  -->  C_render_readable
                                                                      |
                                                                      v
                                              data/readable/whale_dialogues.txt

## Classification

(keep the existing rhythm + extra_click reconstruction sections from the
current README — they are still accurate)

## Output format

(keep the existing token-format explanation: lowercase = unornamented,
uppercase = extra_click, digit = tempo bucket, ? = unclassified)
```

Drop entirely:
- the Sources table (whale-ici-data owns that)
- the schema table (whale-ici-data owns that)
- the "Important caveats" section (move corpus-level caveats to
  whale-ici-data; keep only renderer-specific ones, e.g. "rendered output is
  large; use --sources for a Caribbean-only view")
- the "Adding another source" section (whale-ici-data owns that)

### Step 8 — update whale-ici-data README to point at whale-grammar

In whale-ici-data's README, add a new section near the top, after Contents:

```markdown
## Downstream consumers

- **[whale-grammar](https://github.com/morganrivers/whale-grammar)** —
  reads `data/unified/codas_unified.csv` from this repo and renders a
  human-readable dialogue script. Also re-uses
  `data/raw/sw_combinatoriality_*` for its rhythm/ornament classifier
  training.
```

(NB: whale-grammar's GitHub URL is the future URL — confirm the user has
created/will create that repo before linking.)

### Step 9 — initial whale-grammar commit + push

```bash
cd ~/Code/whale-grammar
git status            # verify what's tracked
git add -A
git commit -m "consume whale-ici-data corpus directly; drop duplicated loader stack"
gh repo create whale-grammar --public --source=. --push
```

(If whale-grammar already exists on GitHub, `git remote add` + `git push -u
origin main` instead.)

---

## Verification checklist

After Step 9, in a fresh shell:

- [ ] `python -m src.pipeline.D_run` runs without error
- [ ] Output `data/readable/whale_dialogues.txt` exists and is non-empty
- [ ] File contains a section per source (3 expected, or fewer with `--sources`)
- [ ] Hersh sections render with `?` placeholders for `extra_click` (since
      Hersh has no timing) and rhythm tokens where applicable
- [ ] DSWP sections still render with proper `extra_click` labels (uppercase
      tokens) for the timed subset
- [ ] No references to Gero anywhere in the rendered output, the README, or
      the code
- [ ] `WHALE_ICI_DATA_DIR=/some/other/path python -m src.pipeline.D_run`
      respects the override
- [ ] Running with whale-ici-data not present at all gives a clear error
      message pointing at the GitHub URL

---

2. **Is the `--sources` flag the right UX?** Or does the user want Hersh
   excluded by default with `--include-hersh` to opt in? (Argument either
   way; current plan says default = all.)

ANSWER: don't include messy flags. do include hersh.

3. **Should the classifier output (rhythm + extra_click) be cached?** Right
   now `D_run` recomputes it on every render. For 38,840 rows it's ~1
   minute. A cache file in `data/intermediate/` might be worth the
   complexity if iterations on the renderer become frequent.

have a separate script that runs on the codas, rythms, ornament, tempo and whale ids (or whatever similar to this is reasonable) to generate the dialogue quickly.

4. **Tempo classification location.** `F_render_readable` currently computes
   tempo on demand from `coda_duration_s`. Should this move into the
   classifier and become a column on the unified CSV (in whale-ici-data)?
   Argument for: makes tempo available to other downstream consumers.
   Argument against: it's a deterministic function of `coda_duration_s`, so
   storing it is redundant.

do store it, see above

---

## Things NOT to do (mistakes-prevention notes for future-me)

- **Don't recreate `schema.py`** in whale-grammar. The whole point is to
  not duplicate the schema. If a column is missing, fail loudly at
  consumption time, then go fix whale-ici-data.

- **Don't pin a specific `whale-ici-data` SHA.** This refactor is meant to
  let the corpus evolve freely. Pinning would re-introduce the lockstep
  maintenance problem in a different form.

- **Don't bundle `codas_unified.csv` into whale-grammar.** Same reason. If
  you want offline reproducibility for a paper, take a snapshot in your
  paper repo, not here.

- **Don't auto-clone whale-ici-data from `D_run`** to "be helpful". Network
  side-effects in build scripts are a footgun. The error message tells the
  user what to do; let them do it.

- **Don't move `E_classify`'s training files into whale-grammar** to make
  whale-grammar self-contained. The training labels are derived from the
  Sharma 2024 DSWP release, which is a corpus asset, not a renderer asset.
  They belong in whale-ici-data.
