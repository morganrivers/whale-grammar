"""Re-segment every ICI sequence in the unified corpus against the Sharma 2024
rhythm templates and emit a classified DataFrame with `rhythm`, `extra_click`
and `tempo` columns filled in.

  Stage 1 - Build coda templates.
  -------------------------------
  Templates come from the labelled dialogue subset shipped by whale-ici-data:
  `sw_combinatoriality_dialogues.csv` (the ICI sequences) plus the aligned
  `sw_combinatoriality_rhythms.p` and `sw_combinatoriality_ornaments.p`
  (per-row rhythm class id and ornament flag). Ornamented rows are excluded
  from centroid building so the extra click does not smear templates. For each
  class we min-trim the rows to the class's shortest length, normalise every
  coda to its cumulative-fraction-of-total profile, and average.

  Stage 2 - Tree-search segmentation.
  -----------------------------------
  For each input row's nonzero ICI sequence, recursively try every template
  whose length exactly equals the current evaluation window. Manhattan
  distance between the normalised cumulative profile and the template mean
  must be <= THRESHOLD (0.1). When no template fits, the segmenter is allowed
  to emit a single class-100 marker (an "extra click") and continue. The path
  with the lowest cumulative score wins (class-100 markers carry a small
  penalty so they don't dominate).

  Stage 3 - Materialise rows.
  ---------------------------
  Each non-marker segment becomes one output row, inheriting all source
  metadata. `rhythm` is the detected class id; `extra_click` is 1 iff the
  immediately following segment in the same input row is a class-100 marker;
  `tempo` is the Sharma 2024 tempo bucket (1..5) computed from the segment's
  total duration.

Run standalone:
    python -m src.pipeline.B_classify
"""
from pathlib import Path
import math
import pickle

import numpy as np
import pandas as pd

from . import A_load_unified

REPO = Path(__file__).resolve().parents[2]
CLASSIFIED_DIR = REPO / "data" / "classified"
CLASSIFIED_CSV = CLASSIFIED_DIR / "codas_classified.csv"

# Tree-search parameters (whale-gpt defaults).
THRESHOLD = 0.1            # Manhattan-distance cutoff for accepting a candidate
SEQUENCE_EVAL_INDEX = 9    # max template length tried per recursion step
CANDIDATE_LIMIT = 100      # max candidates retained per node
EXTRA_CLICK_PENALTY = 0.05 # score penalty for inserting a class-100 marker
ORNAMENT_CLASS = 100       # special class id used by the segmenter for an extra click

# Sharma 2024 tempo thresholds (seconds). Bucket index 0..4 -> digit 1..5.
TEMPO_THRESHOLDS = (0.45, 0.61, 0.93, 1.08)


def tempo_bucket(duration_s: float) -> int:
    """Return tempo digit 1..5 from coda duration."""
    for i, thr in enumerate(TEMPO_THRESHOLDS):
        if duration_s < thr:
            return i + 1
    return len(TEMPO_THRESHOLDS) + 1


def _ici_columns(df: pd.DataFrame) -> list[str]:
    """Sort ICI columns by their numeric suffix so ICI10 doesn't sort before ICI2."""
    cols = [c for c in df.columns if c.startswith("ICI") and c[3:].isdigit()]
    return sorted(cols, key=lambda c: int(c[3:]))


def _pad_icis(values, n_clicks: int, n_cols: int) -> list[float]:
    """Pad/truncate ICIs to n_cols columns; positions past n_clicks-1 are NaN."""
    n_real = max(0, int(n_clicks) - 1) if n_clicks else 0
    out = []
    for i in range(n_cols):
        if i < n_real and i < len(values):
            v = values[i]
            out.append(float(v) if v not in (None, "") else math.nan)
        else:
            out.append(math.nan)
    return out


# ---------- Stage 1: build coda template means ----------

def _standardize(rows: np.ndarray, length: int) -> np.ndarray:
    """For each row, take the first `length` ICIs, normalise by total, then
    cumsum. Returns shape (n_rows, length+1) with a leading zero column."""
    head = rows[:, :length]
    sums = head.sum(axis=1, keepdims=True)
    cumsum = np.cumsum(head / sums, axis=1)
    zeros = np.zeros((cumsum.shape[0], 1))
    return np.concatenate([zeros, cumsum], axis=1)


def build_centroids(refresh: bool = False):
    """Return ({class_id: mean_vec_no_leading_zero}, {class_id: n_icis}).

    Built from the labelled dialogue subset; ornamented rows excluded so
    extra clicks do not smear the templates.
    """
    csv_path      = A_load_unified.training_file("sw_combinatoriality_dialogues.csv", refresh=refresh)
    rhythms_path  = A_load_unified.training_file("sw_combinatoriality_rhythms.p",     refresh=refresh)
    ornaments_path = A_load_unified.training_file("sw_combinatoriality_ornaments.p",   refresh=refresh)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    with open(rhythms_path, "rb") as f:
        rhythms = pickle.load(f)
    with open(ornaments_path, "rb") as f:
        ornaments = pickle.load(f)
    if not (len(df) == len(rhythms) == len(ornaments)):
        raise ValueError(
            f"Training files are misaligned: dialogues={len(df)}, "
            f"rhythms={len(rhythms)}, ornaments={len(ornaments)}"
        )
    df = df.assign(_class=rhythms, _ornament=ornaments)
    df = df[df["_ornament"] == 0]

    ici_cols = _ici_columns(df)
    means = {}
    for class_id, group in df.groupby("_class"):
        if class_id < 0:
            continue
        ici_block = group[ici_cols].to_numpy(dtype=float)
        ici_block = np.where(np.isnan(ici_block), 0.0, ici_block)
        nonzero_per_row = (ici_block != 0).sum(axis=1)
        min_len = int(nonzero_per_row.min())
        if min_len < 2:
            continue
        std = _standardize(ici_block, min_len)
        mean_with_zero = std.mean(axis=0)
        means[int(class_id)] = mean_with_zero[1:]
    coda_lengths = {k: len(v) for k, v in means.items()}
    return means, coda_lengths


# ---------- Stage 2: tree-search segmentation ----------

class TreeNode:
    __slots__ = ("val", "children")

    def __init__(self, val):
        self.val = val
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def best_path(self, _root=True, _path=None, _score=0.0):
        path = (_path or []) + [(self.val[0], self.val[2], self.val[3])]
        if self.val[0] == ORNAMENT_CLASS:
            score = _score + EXTRA_CLICK_PENALTY
        else:
            score = _score + (self.val[1] if isinstance(self.val[1], (int, float)) else 0.0)
        if not self.children:
            return [(path, score)] if not _root else (path[1:], score)
        results = []
        for child in self.children:
            r = child.best_path(_root=False, _path=path, _score=score)
            results.extend(r)
        results.sort(key=lambda x: x[1])
        if _root:
            return results[0][0][1:], results[0][1]
        return results


def _coda_distances(seq, means, only_equal=True):
    seq_arr = np.asarray(seq, dtype=float)
    total = seq_arr.sum()
    if total <= 0:
        return {}
    norm_full = np.cumsum(seq_arr / total)
    out = {}
    for cls, mean in means.items():
        m_len = len(mean)
        if len(seq_arr) < m_len:
            continue
        slice_ = norm_full[:m_len]
        n_equal_one = int(np.sum(np.abs(slice_ - 1.0) < 1e-10))
        if (n_equal_one <= 1 and not only_equal) or n_equal_one == 1:
            out[cls] = float(np.sum(np.abs(slice_ - mean)))
    return out


def _candidates(seq, means, threshold, only_equal=True):
    dists = _coda_distances(seq, means, only_equal)
    return sorted(dists.items(), key=lambda kv: kv[1])


def _expand(tree, candidates, sequence, eval_index, sequence_start,
            means, coda_lengths, only_equal):
    for cls, score in candidates[:CANDIDATE_LIMIT]:
        clen = min(len(sequence), coda_lengths[cls])
        seg_end = sequence_start + clen
        remainder = sequence[clen:]
        child = TreeNode((cls, score, sequence_start, seg_end))
        if len(remainder) > 1:
            _grow(child, remainder, eval_index, seg_end,
                  means, coda_lengths, only_equal)
        elif len(remainder) == 1:
            child.add_child(TreeNode((ORNAMENT_CLASS, 1.0, seg_end, seg_end + 1)))
        tree.add_child(child)


def _grow(tree, sequence, eval_index, sequence_start,
          means, coda_lengths, only_equal=True):
    cands = _candidates(sequence[:eval_index], means, THRESHOLD, only_equal)
    if cands:
        _expand(tree, cands, sequence, eval_index, sequence_start,
                means, coda_lengths, only_equal)
        return tree

    has_orn_branch = any(c.val[0] == ORNAMENT_CLASS for c in tree.children)
    if not has_orn_branch:
        cands_skip = _candidates(
            sequence[1: eval_index + 1], means, THRESHOLD, only_equal)
        if cands_skip:
            orn = TreeNode((ORNAMENT_CLASS, 1.0, sequence_start, sequence_start + 1))
            _expand(orn, cands_skip, sequence[1:], eval_index, sequence_start + 1,
                    means, coda_lengths, only_equal)
            tree.add_child(orn)

    if eval_index > 1:
        _grow(tree, sequence, eval_index - 1, sequence_start,
              means, coda_lengths, only_equal)
    return tree


def segment(sequence, means, coda_lengths):
    if len(sequence) < 2:
        return []
    root = TreeNode((None, 0.0, 0, 0))
    _grow(root, list(sequence), SEQUENCE_EVAL_INDEX, 0, means, coda_lengths)
    if not root.children:
        return []
    path, _score = root.best_path()
    return path


# ---------- Stage 3: materialise output rows ----------

def _nonzero_icis(row, ici_cols):
    vals = pd.to_numeric(row[ici_cols], errors="coerce").to_numpy()
    return vals[np.isfinite(vals) & (vals > 0)]


def _emit_row(source_row, *, class_id, ornament, icis_slice, time_offset,
              seg_idx, n_segments, ici_cols):
    out = source_row.to_dict()
    n_clicks = len(icis_slice) + 1
    out["n_clicks"] = int(n_clicks)
    duration = float(np.sum(icis_slice)) if len(icis_slice) else 0.0
    out["coda_duration_s"] = duration
    if pd.notna(out.get("time_in_recording_s")) and time_offset > 0:
        out["time_in_recording_s"] = float(out["time_in_recording_s"]) + float(time_offset)
    if n_segments > 1 and pd.notna(source_row.get("source_coda_id")):
        out["source_coda_id"] = f"{source_row['source_coda_id']}_seg{seg_idx}"
    out["rhythm"] = int(class_id) if class_id is not None and class_id >= 0 else pd.NA
    out["extra_click"] = pd.NA if ornament is None else int(ornament)
    out["tempo"] = tempo_bucket(duration) if duration > 0 else pd.NA
    for col, v in zip(ici_cols, _pad_icis(list(icis_slice), n_clicks, len(ici_cols))):
        out[col] = v
    return out


def classify(unified: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Re-segment every row of `unified`. Returns a NEW DataFrame; an input
    row produces 1+ output rows, or 1 row with rhythm=<NA> if the segmenter
    couldn't match it."""
    print("B_classify: building centroids from labelled dialogue subset...")
    means, coda_lengths = build_centroids(refresh=refresh)
    print(f"  templates: {len(means)} rhythm classes "
          f"(class_ids {sorted(means.keys())}); "
          f"lengths {sorted(set(coda_lengths.values()))} ICIs.")

    ici_cols = _ici_columns(unified)
    new_rows = []
    n_input = len(unified)
    n_unmatched = 0
    n_orn = 0
    n_segments_total = 0

    for _, row in unified.iterrows():
        icis = _nonzero_icis(row, ici_cols)
        path = segment(icis, means, coda_lengths) if len(icis) >= 2 else []

        if not path:
            new_rows.append(_emit_row(
                row, class_id=None, ornament=None,
                icis_slice=icis, time_offset=0.0,
                seg_idx=0, n_segments=1, ici_cols=ici_cols,
            ))
            n_unmatched += 1
            continue

        non_marker = [(j, c, s, e) for j, (c, s, e) in enumerate(path) if c != ORNAMENT_CLASS]
        n_segments = len(non_marker)
        for k, (j, cls, s, e) in enumerate(non_marker):
            ornament = (j + 1 < len(path) and path[j + 1][0] == ORNAMENT_CLASS)
            new_rows.append(_emit_row(
                row, class_id=cls, ornament=int(ornament),
                icis_slice=icis[s:e], time_offset=float(np.sum(icis[:s])),
                seg_idx=k, n_segments=n_segments, ici_cols=ici_cols,
            ))
            n_orn += int(ornament)
            n_segments_total += 1

    out = pd.DataFrame(new_rows)
    out["rhythm"] = out["rhythm"].astype("Int64")
    out["extra_click"] = out["extra_click"].astype("Int64")
    out["tempo"] = out["tempo"].astype("Int64")

    print(f"B_classify: {n_input:,} input rows -> {len(out):,} segmented codas "
          f"({n_segments_total:,} matched, {n_unmatched:,} kept as <NA>).")
    print(f"  ornaments: {n_orn:,} / {n_segments_total:,}.")
    return out


def main(refresh: bool = False) -> pd.DataFrame:
    unified = A_load_unified.load(refresh=refresh)
    out = classify(unified, refresh=refresh)
    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLASSIFIED_CSV, index=False)
    print(f"B_classify: wrote {len(out):,} rows -> {CLASSIFIED_CSV.relative_to(REPO)}")
    return out


if __name__ == "__main__":
    main()
