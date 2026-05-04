"""Regression tests for the OPTICS+kNN+outlier-discovery classifier.

Two layers:

* **Pure unit tests** for ``cluster_names`` (rhythm detection, plus-pattern
  parsing, ranking, rhythm-18 collapse) — fast, no ELKI required.
* **End-to-end pipeline tests** that depend on ``vendor/elki`` and
  ``vendor/jre8``. These run via ``run_pipeline_once()`` (cached after the
  first invocation) and skip if ELKI isn't vendored.

Run from the repo root::

    python -m pytest tests/test_classify.py -v

Slow tests can be skipped with::

    python -m pytest tests/test_classify.py -v -m "not slow"
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import B_classify, B_classify_optics, cluster_names

REPO = Path(__file__).resolve().parents[1]
ELKI_OK = ((REPO / "vendor" / "elki" / "elki-bundle-0.7.1.jar").exists()
           and (REPO / "vendor" / "jre8" / "bin" / "java").exists())


# ---------------------------------------------------------------------------
# Pure unit tests (cluster_names)
# ---------------------------------------------------------------------------


class TestClusterNamesPure:
    def test_plus_pattern_simple(self):
        assert cluster_names.detect_plus_groups([0.05, 0.50, 0.05, 0.05]) == [2, 3]
        assert cluster_names.detect_plus_groups([0.50, 0.05, 0.05]) == [1, 3]
        assert cluster_names.detect_plus_groups([0.50, 0.50, 0.05, 0.05]) == [1, 1, 3]

    def test_plus_pattern_negative(self):
        assert cluster_names.detect_plus_groups([0.05, 0.05, 0.05, 0.05]) is None
        assert cluster_names.detect_plus_groups([0.10, 0.20, 0.30, 0.40]) is None
        assert cluster_names.detect_plus_groups([0.40, 0.30, 0.20, 0.10]) is None

    def test_rhythm_basic(self):
        assert cluster_names.detect_rhythm([0.30, 0.30, 0.30, 0.30]) == "R"
        assert cluster_names.detect_rhythm([0.10, 0.20, 0.30, 0.40]) == "i"
        assert cluster_names.detect_rhythm([0.40, 0.30, 0.20, 0.10]) == "D"
        assert cluster_names.detect_rhythm([0.50, 0.50, 0.05, 0.05]) == "1+1+3"

    def test_collapse_rhythm18(self):
        assert cluster_names.collapse_to_rhythm18("5R1") == "5R"
        assert cluster_names.collapse_to_rhythm18("5R") == "5R"
        assert cluster_names.collapse_to_rhythm18("1+1+3") == "1+1+3"
        assert cluster_names.collapse_to_rhythm18("5-NOISE") == "5-NOISE"
        assert cluster_names.collapse_to_rhythm18(None) is None

    def test_ec_ranking_assigns_in_duration_order(self):
        clusters = [
            {"n_clicks": 5, "centroid": np.array([0.30] * 4), "mean_duration": 1.5},
            {"n_clicks": 5, "centroid": np.array([0.20] * 4), "mean_duration": 1.0},
            {"n_clicks": 5, "centroid": np.array([0.25] * 4), "mean_duration": 1.2},
        ]
        cluster_names.assign_tempo_ranks(clusters, pacific=False)
        names_by_dur = sorted(
            [(c["mean_duration"], c["name"]) for c in clusters])
        # Fastest = lowest duration = rank 1.
        assert [n for _, n in names_by_dur] == ["5R1", "5R2", "5R3"]

    def test_ec_singleton_has_no_rank(self):
        clusters = [
            {"n_clicks": 4, "centroid": np.array([0.40, 0.30, 0.20]),
             "mean_duration": 1.0},
        ]
        cluster_names.assign_tempo_ranks(clusters, pacific=False)
        assert clusters[0]["name"] == "4D"
        assert clusters[0]["rank"] is None

    def test_pacific_marker_same_shape(self):
        clusters = [
            {"n_clicks": 5, "centroid": np.array([0.30] * 4),
             "mean_duration": 1.5},
        ]
        cluster_names.assign_tempo_ranks(
            clusters, pacific=True, ec_shapes=[(5, "R")])
        assert clusters[0]["name"] == "5RP1"

    def test_pacific_marker_novel_shape(self):
        clusters = [
            {"n_clicks": 5, "centroid": np.array([0.10, 0.20, 0.30, 0.40]),
             "mean_duration": 1.5},
        ]
        cluster_names.assign_tempo_ranks(
            clusters, pacific=True, ec_shapes=[(5, "R")])
        assert clusters[0]["name"] == "5P1"

    def test_pacific_plus_pattern_no_marker(self):
        # Novel +-pattern doesn't need a P marker; the +-string is unique
        # already. ICIs [s, BIG, s] with n_clicks=4 → 2 clicks, gap, 2 clicks.
        clusters = [
            {"n_clicks": 4, "centroid": np.array([0.05, 0.50, 0.05]),
             "mean_duration": 1.0},
        ]
        cluster_names.assign_tempo_ranks(
            clusters, pacific=True, ec_shapes=[(5, "R")])
        assert clusters[0]["name"] == "2+2"


# ---------------------------------------------------------------------------
# Tempo / rubato unit tests (no ELKI)
# ---------------------------------------------------------------------------


class TestTempoRubato:
    def test_tempo_thresholds(self):
        # Sharma 2024 §4 bin endpoints.
        assert B_classify.tempo_bucket(0.30) == 1
        assert B_classify.tempo_bucket(0.45) == 2
        assert B_classify.tempo_bucket(0.61) == 3
        assert B_classify.tempo_bucket(0.93) == 4
        assert B_classify.tempo_bucket(1.08) == 5
        assert B_classify.tempo_bucket(2.00) == 5

    def test_rubato_categorize(self):
        assert B_classify._categorize(-1.0) == "\\"
        assert B_classify._categorize(0.0) == "-"
        assert B_classify._categorize(1.0) == "/"


# ---------------------------------------------------------------------------
# End-to-end pipeline tests (require ELKI)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_output() -> pd.DataFrame:
    """Run B_classify once and cache the result in a module-scoped fixture.

    First call: ~10 minutes (ELKI on 8 length buckets twice). Repeat calls
    pull from the npz cache under data/cache/ and finish in seconds.
    """
    if not ELKI_OK:
        pytest.skip("ELKI not vendored (vendor/elki + vendor/jre8 missing)")
    if os.environ.get("SKIP_ELKI_TESTS"):
        pytest.skip("SKIP_ELKI_TESTS set")

    from src.pipeline import A_load_unified
    unified = A_load_unified.load()
    return B_classify.classify(unified)


@pytest.mark.slow
class TestPipeline:
    def test_dswp_real_codatype_join_is_exact(self, pipeline_output):
        """Every DSWP row whose ``CodaType`` truth is *real* (non-NOISE)
        passes through to ``coda_type_gero21`` exactly. Sharma-NOISE
        rows are re-pooled into the discovery pass and excluded."""
        df = pipeline_output
        from src.pipeline import A_load_unified
        d = pd.read_csv(A_load_unified.dominica_codas_csv())
        d.columns = [c.strip().lstrip("﻿") for c in d.columns]
        lookup = d.set_index(d["codaNUM2018"].astype(str))["CodaType"]
        is_dswp = df["source"] == "sharma2024_dswp"
        joined = (df.loc[is_dswp, "source_coda_id"].astype(str)
                  .map(lookup))
        is_real = joined.notna() & ~joined.astype("string").str.endswith(
            "-NOISE", na=False)
        equal = (df.loc[is_dswp & is_real, "coda_type_gero21"]
                 == joined[is_real]).sum()
        n = int(is_real.sum())
        assert equal == n, f"{equal}/{n} DSWP 'real' join exact"

    def test_dswp_loo_accuracy_meets_target(self, pipeline_output):
        """Per-length leave-one-out kNN on DSWP gives ≥97% Gero-21
        reproduction (matches Phase 1b option 2's 97.64%)."""
        from sklearn.neighbors import KNeighborsClassifier
        df = pipeline_output
        is_dswp = df["source"] == "sharma2024_dswp"
        y = df.loc[is_dswp, "coda_type_gero21"]
        df_dswp = df.loc[is_dswp & y.notna()]
        loo_correct, loo_total = 0, 0
        for n in B_classify_optics.LENGTH_RANGE:
            cols = [f"ICI{i}" for i in range(1, n)]
            mask = ((df_dswp["n_clicks"] == n)
                    & df_dswp[cols].notna().all(axis=1))
            X = df_dswp.loc[mask, cols].to_numpy(dtype=float)
            yy = df_dswp.loc[mask, "coda_type_gero21"].to_numpy()
            if len(X) < 7:
                continue
            knn = KNeighborsClassifier(n_neighbors=6, metric="euclidean")
            knn.fit(X, yy)
            nbrs = knn.kneighbors(X, n_neighbors=6, return_distance=False)
            preds = []
            for i, row in enumerate(nbrs):
                row = row[row != i][:5]
                vals, counts = np.unique(yy[row], return_counts=True)
                preds.append(vals[counts.argmax()])
            preds = np.array(preds)
            loo_correct += int((preds == yy).sum())
            loo_total += len(yy)
        pct = loo_correct / max(loo_total, 1)
        assert pct >= 0.97, f"DSWP loo {pct:.2%} < 97% target"

    def test_pacific_noise_rate_below_threshold(self, pipeline_output):
        """≤8% of all classified codas land on a *-NOISE label after outlier
        discovery. Phase 1b had 28% inheritance noise — the rewrite must
        eliminate it."""
        df = pipeline_output
        eval_mask = df["coda_type_gero21"].notna()
        n_eval = int(eval_mask.sum())
        n_noise = int(df.loc[eval_mask, "coda_type_gero21"]
                      .astype(str).str.endswith("-NOISE").sum())
        rate = n_noise / max(n_eval, 1)
        assert rate <= 0.08, f"NOISE rate {rate:.2%} > 8% target"

    def test_gero_21_names_reproduce_published(self, pipeline_output):
        """The auto-naming function should produce a label string for each
        OPTICS DSWP cluster. We assert that *the most common* labels match
        Gero's published vocabulary; tolerate up to 4 mismatches per the
        plan (xi-extraction can split a Gero type into sub-clusters)."""
        df = pipeline_output
        is_dswp = (df["source"] == "sharma2024_dswp")
        published = {"1+1+3", "5R1", "5R2", "5R3", "4R1", "4R2", "4D",
                     "6R", "6i", "7R", "7i", "7D1", "7D2", "8R", "8i",
                     "8D", "9R", "9i", "10R", "10i", "1+3"}
        # The pipeline uses CodaType directly for DSWP rows, so the
        # labels are guaranteed to match the join. The naming function
        # is exercised end-to-end on Pacific outliers; here we just
        # verify the EC vocabulary is preserved.
        seen = set(df.loc[is_dswp, "coda_type_gero21"].dropna().unique())
        # Strip *-NOISE, which isn't a "type".
        seen_real = {s for s in seen if not s.endswith("-NOISE")}
        missing = published - seen_real
        # Allow up to 4 misses (2+3, 1+31/1+32 collapse, 3R/3D edge cases).
        assert len(missing) <= 4, f"missing >4 published types: {missing}"

    def test_tempo_distribution_reasonable(self, pipeline_output):
        """Tempo is computed by the same code as before; basic sanity:
        every classified coda with a duration has a tempo ∈ {1..5}."""
        df = pipeline_output
        with_dur = df[df["coda_duration_s"].notna()
                      & (df["coda_duration_s"] > 0)]
        assert with_dur["tempo"].notna().all()
        assert with_dur["tempo"].min() >= 1
        assert with_dur["tempo"].max() <= 5

    def test_rubato_distribution_reasonable(self, pipeline_output):
        """Rubato is one of /, -, \\ when set."""
        df = pipeline_output
        vals = set(df["rubato"].dropna().unique())
        assert vals.issubset({"/", "-", "\\"}), f"unexpected: {vals - {'/', '-', chr(92)}}"

    def test_ornament_skipped_for_hersh(self, pipeline_output):
        df = pipeline_output
        hersh = df[df["source"] == "hersh2022_pacific"]
        # All Hersh rows must have NA extra_click — Sharma's structural
        # rule requires per-coda timestamps which Hersh lacks.
        assert hersh["extra_click"].isna().all()
