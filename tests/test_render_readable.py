"""Stage 4.1 acceptance gate — transcript rendering.

Smoke-tests on the on-disk ``data/readable/whale_dialogues.txt``:

* Numeric ``Δt<value>`` tokens appear for timestamped sources
  (Sharma DSWP and Sharma birth).
* ``Δt?`` sentinel appears for Hersh recordings (no timestamps).
* Uppercase letters in coda tokens (signalling ``extra_click == 1``)
  appear at least once.
* Rubato prefixes (``/``, ``-``, ``\\``) appear at least once.
* Each source shows up as its own ``=== source: ... ===`` header.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRANSCRIPT = REPO / "data" / "readable" / "whale_dialogues.txt"


def _load() -> str:
    if not TRANSCRIPT.exists():
        pytest.skip(f"{TRANSCRIPT.relative_to(REPO)} not found; "
                    f"run `python -m src.pipeline.C_render_readable`")
    return TRANSCRIPT.read_text()


def _section(text: str, source: str) -> str:
    """Slice the transcript to a single source section."""
    start = text.find(f"=== source: {source} ===")
    if start == -1:
        return ""
    end = text.find("=== source:", start + 1)
    return text[start: end if end != -1 else None]


def test_all_three_sources_appear():
    text = _load()
    for src in ("sharma2024_dswp", "sharma2025_birth", "hersh2022_pacific"):
        assert f"=== source: {src} ===" in text, f"missing {src} section"


def test_numeric_dt_tokens_in_timed_sources():
    text = _load()
    pattern = re.compile(r"Δt\d+\.\d{2}")
    for src in ("sharma2024_dswp", "sharma2025_birth"):
        section = _section(text, src)
        n = len(pattern.findall(section))
        assert n > 100, (
            f"{src}: only {n} numeric Δt tokens found; expected many "
            f"(timestamps populate this column)"
        )


def test_dt_sentinel_in_hersh_section():
    text = _load()
    section = _section(text, "hersh2022_pacific")
    assert "Δt?" in section, "Δt? sentinel missing from Hersh section"


def test_no_numeric_dt_in_hersh():
    text = _load()
    section = _section(text, "hersh2022_pacific")
    assert not re.search(r"Δt\d", section), (
        "numeric Δt tokens leaked into Hersh transcript "
        "(Hersh has no timestamps)"
    )


def test_uppercase_ornaments_render():
    text = _load()
    # Look for uppercase letters in the coda-letter slot (between optional
    # rubato prefix and final tempo digit).
    matches = re.findall(r"[/\-\\]?[A-Z]+\d", text)
    assert len(matches) >= 100, (
        f"only {len(matches)} uppercase coda tokens "
        f"(expected hundreds — DSWP+birth ornaments)"
    )


def test_rubato_prefixes_render():
    text = _load()
    for prefix in ("/a", "-a", "\\a"):
        assert prefix in text, f"rubato prefix {prefix!r} not found"
