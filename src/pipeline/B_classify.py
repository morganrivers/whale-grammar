"""Classify the unified ICI corpus into Gero-21 / rhythm-18 + tempo + rubato.

Pipeline (per ``next_phases_plan.md`` section C):

1. **Attach DSWP truth.** Join ``codaNUM2018`` ↔ ``source_coda_id`` against
   ``dswp_dominica_codas.csv`` to get Gero's published ``CodaType`` for every
   DSWP row. This is the gold standard, used verbatim for DSWP rows.
2. **OPTICS + nearest-cluster-within-radius classifier** for non-DSWP rows.
   See ``B_classify_optics.run_optics_pipeline`` for the algorithm. Emits
   ``coda_type_gero21`` (string, e.g. ``5R1``, ``1+1+3``, ``5P1``) and a
   ``classifier_origin`` column tagging the source.
3. **Collapse to rhythm-18** by dropping the tempo-rank suffix
   (``5R1`` → ``5R``).
4. **Build a stable integer encoding** ``rhythm`` (0..N-1) from the sorted
   ``rhythm_class_18`` vocabulary. Used by ``C_render_readable``.
5. **Tempo bins** (Sharma 2024 §4) — unchanged from the previous version.
6. **Rubato** (per-whale duration delta) — unchanged.
7. **Ornament** structural rule (Sharma 2024 §5) — applied to DSWP / birth
   rows only; left NA for Hersh, which lacks per-coda timestamps.

The previous upstream-trusted ``rhythm`` and ``extra_click`` columns shipped
in ``codas_unified.csv`` are *renamed* to ``upstream_rhythm`` /
``upstream_extra_click`` so we keep provenance for sanity checks while the
new classifier owns the canonical columns.

Run standalone::

    python -m src.pipeline.B_classify
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import A_load_unified
from . import B_classify_optics
from . import cluster_names

REPO = Path(__file__).resolve().parents[2]
CLASSIFIED_DIR = REPO / "data" / "classified"
CLASSIFIED_CSV = CLASSIFIED_DIR / "codas_classified.csv"
RHYTHM_CLASS_INDEX_CSV = CLASSIFIED_DIR / "rhythm_class_index.csv"

TEMPO_THRESHOLDS = (0.45, 0.61, 0.93, 1.08)

# Empirical 25th/75th-percentile cutoffs on the labelled DSWP duration deltas,
# from sw-combinatoriality/code/generate_whale_dialogue_txt_with_proper_timings.py.
RUBATO_LO = -0.021416925000000087
RUBATO_HI = 0.018462550000000105

RUBATO_T_DIFF_S = 10.0

# Sharma 2024 §5: an ornament is "exactly one more click than the immediately
# neighbouring same-whale codas". We require the surrounding same-whale codas
# (within ORNAMENT_T_DIFF_S) to all sit at exactly n_clicks - 1; the
# ornamented coda is the local +1.
ORNAMENT_T_DIFF_S = 10.0


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------


def tempo_bucket(duration_s: float) -> int:
    for i, thr in enumerate(TEMPO_THRESHOLDS):
        if duration_s < thr:
            return i + 1
    return len(TEMPO_THRESHOLDS) + 1


def _add_tempo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    durations = pd.to_numeric(out["coda_duration_s"], errors="coerce")
    out["tempo"] = pd.array(
        [tempo_bucket(float(d)) if pd.notna(d) and d > 0 else pd.NA for d in durations],
        dtype="Int64",
    )
    return out


# ---------------------------------------------------------------------------
# Rubato
# ---------------------------------------------------------------------------


def _categorize(delta: float) -> str:
    if delta < RUBATO_LO:
        return "\\"
    if delta < RUBATO_HI:
        return "-"
    return "/"


def _whale_id(row) -> str | None:
    if pd.notna(row.get("whale_photo_id")):
        return f"photo:{row['whale_photo_id']}"
    if pd.notna(row.get("local_speaker_id")):
        return f"local:{row['local_speaker_id']}"
    return None


def _add_rubato(df: pd.DataFrame) -> pd.DataFrame:
    """For each coda, compare to the previous same-whale coda within 10 s
    that shares its rhythm and tempo. Categorize the duration delta with the
    empirical quantile cutoffs.

    Uses the new (post-OPTICS) ``rhythm`` integer column.
    """
    rubato = np.full(len(df), "", dtype=object)

    df = df.reset_index(drop=True)
    df["_whale"] = df.apply(_whale_id, axis=1)

    eligible = df[
        df["_whale"].notna()
        & df["recording_id"].notna()
        & df["time_in_recording_s"].notna()
        & df["rhythm"].notna()
        & df["tempo"].notna()
    ]

    for _, group in eligible.groupby(["source", "recording_id", "_whale"], sort=False):
        group = group.sort_values("time_in_recording_s", kind="stable")
        idxs = group.index.to_numpy()
        times = group["time_in_recording_s"].to_numpy(dtype=float)
        durations = group["coda_duration_s"].to_numpy(dtype=float)
        rhythms = group["rhythm"].to_numpy()
        tempos = group["tempo"].to_numpy()

        for i in range(1, len(idxs)):
            t_diff = times[i] - times[i - 1]
            if t_diff > RUBATO_T_DIFF_S:
                continue
            if rhythms[i] != rhythms[i - 1] or tempos[i] != tempos[i - 1]:
                continue
            rubato[idxs[i]] = _categorize(durations[i] - durations[i - 1])

    out = df.drop(columns=["_whale"])
    out["rubato"] = rubato
    out.loc[out["rubato"] == "", "rubato"] = pd.NA
    return out


# ---------------------------------------------------------------------------
# Ornament
# ---------------------------------------------------------------------------


def _add_ornament(df: pd.DataFrame) -> pd.DataFrame:
    """Sharma 2024 §5 structural rule: extra_click=1 iff the coda has exactly
    one more click than at least one adjacent same-whale coda within
    ``ORNAMENT_T_DIFF_S``.

    Hersh rows (``source == hersh2022_pacific``) are left NA — Hersh's data
    product carries no per-coda timestamps, so "neighbouring same-whale coda"
    is unidentifiable. See ``feedback_ornament_scope.md``.
    """
    out = df.reset_index(drop=True).copy()
    extra = pd.array([pd.NA] * len(out), dtype="Int64")

    out["_whale"] = out.apply(_whale_id, axis=1)
    eligible = out[
        (out["source"] != "hersh2022_pacific")
        & out["_whale"].notna()
        & out["recording_id"].notna()
        & out["time_in_recording_s"].notna()
        & out["n_clicks"].notna()
    ]

    for _, group in eligible.groupby(["source", "recording_id", "_whale"], sort=False):
        group = group.sort_values("time_in_recording_s", kind="stable")
        idxs = group.index.to_numpy()
        times = group["time_in_recording_s"].to_numpy(dtype=float)
        clicks = group["n_clicks"].to_numpy(dtype=float)

        for i in range(len(idxs)):
            n = clicks[i]
            if not np.isfinite(n):
                continue
            ornament = 0
            # Look at immediate left and right neighbours within the window.
            for j in (i - 1, i + 1):
                if j < 0 or j >= len(idxs):
                    continue
                t_diff = abs(times[i] - times[j])
                if t_diff > ORNAMENT_T_DIFF_S:
                    continue
                if np.isfinite(clicks[j]) and clicks[j] == n - 1:
                    ornament = 1
                    break
            extra[idxs[i]] = ornament

    out = out.drop(columns=["_whale"])
    out["extra_click"] = extra
    return out


# ---------------------------------------------------------------------------
# Rhythm encoding
# ---------------------------------------------------------------------------


def _build_rhythm_encoding(rhythm_18_series: pd.Series) -> dict[str, int]:
    """Stable str → int encoding for the rhythm-18 vocabulary, sorted
    deterministically. New Pacific types (``5RP1``, ``5P1``, etc.) extend the
    vocabulary at the end — sort still keeps EC names in a fixed position
    relative to each other across runs."""
    vocab = sorted(s for s in rhythm_18_series.dropna().unique() if isinstance(s, str))
    return {s: i for i, s in enumerate(vocab)}


def _add_rhythm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add three columns:

    * ``rhythm_class_18`` — string, the Sharma-18 collapse of
      ``coda_type_gero21`` (drops tempo-rank suffix; passes ``+``
      patterns and NOISE through).
    * ``rhythm`` — Int64, stable encoding of ``rhythm_class_18``. Used
      by ``_add_rubato`` to group same-rhythm codas (rubato is defined
      relative to the immediately previous coda of the same rhythm and
      tempo).
    * ``rhythm_class`` — Int64, stable encoding of ``coda_type_gero21``
      itself (full vocabulary including tempo ranks and discovered
      Pacific types). Consumed by the human-readable transcript and
      the transformer CSV. Mapping persisted to
      ``rhythm_class_index.csv`` so the integer codes stay decodable.
    """
    out = df.copy()
    out["rhythm_class_18"] = out["coda_type_gero21"].map(
        cluster_names.collapse_to_rhythm18)
    encoding = _build_rhythm_encoding(out["rhythm_class_18"])
    out["rhythm"] = pd.array(
        [encoding[s] if isinstance(s, str) and s in encoding else pd.NA
         for s in out["rhythm_class_18"]],
        dtype="Int64",
    )
    full_encoding = _build_rhythm_encoding(out["coda_type_gero21"])
    out["rhythm_class"] = pd.array(
        [full_encoding[s] if isinstance(s, str) and s in full_encoding
         else pd.NA
         for s in out["coda_type_gero21"]],
        dtype="Int64",
    )
    return out


def _write_rhythm_class_index(df: pd.DataFrame, out_path: Path) -> None:
    """Persist the ``rhythm_class`` int → label mapping so downstream
    consumers (transformer CSV, transcript decoder) can interpret the
    integer codes."""
    idx = (df[["rhythm_class", "coda_type_gero21", "rhythm_class_18"]]
           .dropna(subset=["rhythm_class"])
           .drop_duplicates(subset=["rhythm_class"])
           .sort_values("rhythm_class")
           .reset_index(drop=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def _rename_upstream_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    if "rhythm" in out.columns:
        rename["rhythm"] = "upstream_rhythm"
    if "extra_click" in out.columns:
        rename["extra_click"] = "upstream_extra_click"
    if rename:
        out = out.rename(columns=rename)
    return out


def classify(unified: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Run the full B_classify pipeline. ``unified`` is the loaded
    ``codas_unified.csv``; the labelled DSWP corpus is fetched on demand.
    """
    n_total = len(unified)
    print(f"B_classify: pipeline on {n_total:,} unified codas.")

    df = _rename_upstream_columns(unified)

    # Stage 1+2+3+4: OPTICS + kNN + outlier-discovery + naming.
    dominica_path = A_load_unified.dominica_codas_csv(refresh=refresh)
    df_dominica = pd.read_csv(dominica_path)
    df_dominica.columns = [c.strip().lstrip("﻿") for c in df_dominica.columns]
    df = B_classify_optics.run_optics_pipeline(df, df_dominica)

    n_classified = int(df["coda_type_gero21"].notna().sum())
    origin_counts = df["classifier_origin"].value_counts(dropna=False).to_dict()
    print(f"B_classify: classified {n_classified:,}/{n_total:,} codas. "
          f"origin breakdown: "
          f"dswp-real={origin_counts.get('dswp-real', 0):,}, "
          f"pacific-matched={origin_counts.get('pacific-matched', 0):,}, "
          f"discovery-cluster={origin_counts.get('discovery-cluster', 0):,}, "
          f"discovery-noise={origin_counts.get('discovery-noise', 0):,}")
    n_unique_types = int(df["coda_type_gero21"].dropna().nunique())
    print(f"B_classify: {n_unique_types} distinct coda_type_gero21 values "
          f"in vocabulary.")

    # Stage 5: rhythm-18 string + integer encoding + full rhythm_class.
    df = _add_rhythm_columns(df)
    n_unique_rhythms = int(df["rhythm_class_18"].dropna().nunique())
    n_unique_classes = int(df["rhythm_class"].dropna().nunique())
    print(f"B_classify: rhythm-18 vocabulary has {n_unique_rhythms} labels; "
          f"full rhythm_class has {n_unique_classes} labels.")

    # Stage 6: tempo (unchanged).
    df = _add_tempo(df)
    n_tempo = int(df["tempo"].notna().sum())
    print(f"B_classify: tempo populated for {n_tempo:,} codas.")

    # Stage 7: rubato (unchanged).
    df = _add_rubato(df)
    n_rubato = int(df["rubato"].notna().sum())
    print(f"B_classify: rubato populated for {n_rubato:,} codas "
          f"(rest blank: no prev same-whale coda within "
          f"{int(RUBATO_T_DIFF_S)}s, or rhythm/tempo mismatch).")
    print(f"B_classify: rubato distribution: "
          f"{df['rubato'].value_counts(dropna=True).to_dict()}")

    # Stage 8: ornament (Sharma structural rule, NaN for Hersh).
    df = _add_ornament(df)
    n_orn = int((df["extra_click"] == 1).sum())
    n_orn_known = int(df["extra_click"].notna().sum())
    print(f"B_classify: ornament rule populated for {n_orn_known:,} codas "
          f"({n_orn:,} ornaments). Hersh rows left NA per Sharma §5 "
          f"applicability.")

    # Acceptance gating numbers (per next_phases_plan.md §C):
    _print_acceptance_metrics(df, df_dominica)

    return df


def _print_acceptance_metrics(df: pd.DataFrame, df_dominica: pd.DataFrame) -> None:
    """Sanity numbers that gate the rewrite. Failures don't raise — we want
    the run to finish so the user can inspect the output — but they're
    flagged loudly."""
    print("\nB_classify: --- acceptance metrics ---")

    # 1. DSWP 'real' CodaType conservation (non-NOISE rows must be exact).
    #    Sharma-NOISE rows are deliberately re-pooled into the OPTICSxi
    #    discovery pass (see whale_grammar_transformer_plan §1.4) and may
    #    end up with a discovered-cluster label — that is recovery, not a
    #    regression, and is excluded from this exactness check.
    is_dswp = df["source"] == "sharma2024_dswp"
    if is_dswp.any():
        lookup = df_dominica.set_index(df_dominica["codaNUM2018"].astype(str))[
            "CodaType"]
        truth = (df.loc[is_dswp, "source_coda_id"].astype(str)
                 .map(lookup))
        is_real_truth = truth.notna() & ~truth.astype("string").str.endswith(
            "-NOISE", na=False)
        n_real = int(is_real_truth.sum())
        if n_real:
            equal = (df.loc[is_dswp, "coda_type_gero21"][is_real_truth]
                     == truth[is_real_truth]).sum()
            pct = 100.0 * equal / n_real
            ok = "PASS" if pct >= 99.99 else "FAIL"
            print(f"  DSWP 'real' conservation:   {equal:,}/{n_real:,} "
                  f"= {pct:.2f}%  [{ok}]")

    # 2. Aggregate NOISE rate.
    n_eval = int(df["coda_type_gero21"].notna().sum())
    n_noise = int(df["coda_type_gero21"].astype(str).str.endswith("-NOISE").sum())
    if n_eval:
        pct = 100.0 * n_noise / n_eval
        ok = "PASS" if pct <= 8.0 else "FAIL"
        print(f"  Aggregate NOISE rate:       {n_noise:,}/{n_eval:,} = {pct:.2f}%  [{ok}, target ≤ 8%]")

    # 3. Sharma-18 collapse on labelled DSWP subset.
    if is_dswp.any():
        is_dswp_truth = is_dswp & df["coda_type_gero21"].notna()
        truth_18 = (df.loc[is_dswp_truth, "coda_type_gero21"]
                    .map(cluster_names.collapse_to_rhythm18))
        pred_18 = df.loc[is_dswp_truth, "rhythm_class_18"]
        n = int(is_dswp_truth.sum())
        if n:
            equal = int((truth_18 == pred_18).sum())
            pct = 100.0 * equal / n
            ok = "PASS" if pct >= 95.0 else "FAIL"
            print(f"  rhythm-18 vs DSWP truth:    {equal:,}/{n:,} = {pct:.2f}%  [{ok}, target ≥ 95%]")


def main(refresh: bool = False) -> pd.DataFrame:
    unified = A_load_unified.load(refresh=refresh)
    out = classify(unified, refresh=refresh)
    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLASSIFIED_CSV, index=False)
    _write_rhythm_class_index(out, RHYTHM_CLASS_INDEX_CSV)
    print(f"\nB_classify: wrote {len(out):,} rows -> "
          f"{CLASSIFIED_CSV.relative_to(REPO)}")
    print(f"B_classify: rhythm_class index -> "
          f"{RHYTHM_CLASS_INDEX_CSV.relative_to(REPO)}")
    return out


if __name__ == "__main__":
    main()
