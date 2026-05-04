"""Subprocess wrapper around ELKI 0.7.1's OPTICSXi.

ELKI's MiniGUI is the default entrypoint and won't run headless under JRE 8;
we drive it via the KDDCLIApplication subcommand and ClusteringVectorDumper
result handler, which writes one space-separated cluster id per input row
followed by the cluster name (which we strip)."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
JRE = REPO / "vendor" / "jre8" / "bin" / "java"
ELKI_JAR = REPO / "vendor" / "elki" / "elki-bundle-0.7.1.jar"


def run(X: np.ndarray, *, xi: float, minpts: int,
        epsilon: float | None = None) -> np.ndarray:
    """Run ELKI OPTICSXi on X (n_samples x n_dim) and return per-row leaf
    cluster ids. -1 marks rows ELKI dropped (duplicates, etc.); we leave the
    leafmost cluster id otherwise (the largest leaf is typically the
    'rest' bucket — the caller decides what to treat as noise)."""
    if not JRE.exists() or not ELKI_JAR.exists():
        raise RuntimeError(
            f"ELKI not vendored. Expected {JRE} and {ELKI_JAR}.\n"
            f"See vendor/README or run setup script.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_path = tmp / "in.txt"
        out_path = tmp / "labels.txt"
        np.savetxt(in_path, X, fmt="%.6f")

        cmd = [
            str(JRE), "-jar", str(ELKI_JAR), "KDDCLIApplication",
            "-dbc.in", str(in_path),
            "-algorithm", "clustering.optics.OPTICSXi",
            "-opticsxi.xi", str(xi),
            "-optics.minpts", str(minpts),
            "-resulthandler", "ClusteringVectorDumper",
            "-clustering.output", str(out_path),
        ]
        if epsilon is not None:
            cmd += ["-optics.epsilon", str(epsilon)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ELKI failed (rc={proc.returncode}):\n"
                f"stderr tail:\n{proc.stderr[-1500:]}\n"
                f"stdout tail:\n{proc.stdout[-1500:]}")
        if not out_path.exists():
            raise RuntimeError(
                f"ELKI produced no output file. stderr:\n{proc.stderr[-1500:]}")

        raw = out_path.read_text().split()
        ids: list[int] = []
        for tok in raw:
            try:
                ids.append(int(tok))
            except ValueError:
                # trailing cluster name like "OPTICS Xi-Clusters"
                break
        if len(ids) != len(X):
            raise RuntimeError(
                f"ELKI returned {len(ids)} labels for {len(X)} rows. "
                f"Tail of file:\n{out_path.read_text()[-200:]}")
        return np.asarray(ids, dtype=np.int64)
