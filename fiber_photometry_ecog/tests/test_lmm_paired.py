"""Tests for the R-backed LMM paired comparison."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from fiber_photometry_ecog.statistics import r_bridge
from fiber_photometry_ecog.statistics.lmm_paired import compare_paired_lmm
from fiber_photometry_ecog.statistics.runner import _write_csv


GROUP_A = "Pre"
GROUP_B = "Post"


# Skip the R-dependent tests when R / lmerTest aren't installed locally.
needs_r = pytest.mark.skipif(
    not r_bridge.is_r_available(),
    reason="R or lmerTest not installed; LMM path skipped",
)


def _make_long_df(seed: int = 0) -> pd.DataFrame:
    """Six mice: four with one pair, two with two pairs, and a Post effect of -4.

    Session-level baseline variation supports variance-component estimation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mid, base in zip(range(1, 5), [10.0, 12.0, 11.0, 13.0]):
        day = rng.normal(0, 0.6)
        rows.append((f"m{mid}", 1, GROUP_A, base + day + rng.normal(0, 0.3)))
        rows.append((f"m{mid}", 1, GROUP_B, base + day - 4 + rng.normal(0, 0.3)))
    for mid in (5, 6):
        base = 20.0
        for hsess in (1, 2):
            day = rng.normal(0, 0.6)
            rows.append((f"m{mid}", hsess, GROUP_A, base + day + rng.normal(0, 0.3)))
            rows.append((f"m{mid}", hsess, GROUP_B, base + day - 4 + rng.normal(0, 0.3)))
    return pd.DataFrame(rows, columns=["mouse_id", "heating_session", "cohort", "value"])


@needs_r
def test_lmm_detects_real_effect_on_unbalanced_design():
    df = _make_long_df()
    result = compare_paired_lmm(
        "y", GROUP_A, GROUP_B, df, group_a=GROUP_A, group_b=GROUP_B,
    )
    assert result.test_name == "LMM (Satterthwaite)"
    # 8 pairs: 4 singletons + 2 mice × 2 pairs each.
    assert result.n_pairs == 8
    # Descriptive `n` should reflect SESSIONS, not pair count or mouse count.
    desc_a = next(d for d in result.descriptives if d.name == GROUP_A)
    desc_b = next(d for d in result.descriptives if d.name == GROUP_B)
    assert desc_a.n == 8
    assert desc_b.n == 8
    # 4-unit shift on noise SD 0.3 -> p should be tiny.
    assert result.p_value < 1e-3
    assert result.statistic < 0  # Post < Pre, contrast direction is Post - Pre


@needs_r
def test_lmm_no_effect_yields_nonsig():
    rng = np.random.default_rng(42)
    rows = []
    for mid in range(1, 7):
        base = 10.0 + mid
        n_sess = 2 if mid in (5, 6) else 1
        for hsess in range(1, n_sess + 1):
            rows.append((f"m{mid}", hsess, GROUP_A, base + rng.normal(0, 0.5)))
            rows.append((f"m{mid}", hsess, GROUP_B, base + rng.normal(0, 0.5)))
    df = pd.DataFrame(rows, columns=["mouse_id", "heating_session", "cohort", "value"])
    result = compare_paired_lmm(
        "y", GROUP_A, GROUP_B, df, group_a=GROUP_A, group_b=GROUP_B,
    )
    assert result.test_name == "LMM (Satterthwaite)"
    assert result.p_value > 0.05


@needs_r
def test_all_identical_falls_back_to_paired_t(caplog):
    """Constant paired values produce no test, without an independent fallback."""
    rows = []
    for mid in range(1, 7):
        rows.append((f"m{mid}", 1, GROUP_A, 5.0))
        rows.append((f"m{mid}", 1, GROUP_B, 5.0))
    df = pd.DataFrame(rows, columns=["mouse_id", "heating_session", "cohort", "value"])
    with caplog.at_level(logging.WARNING,
                         logger="fiber_photometry_ecog.statistics.lmm_paired"):
        result = compare_paired_lmm(
            "y", GROUP_A, GROUP_B, df, group_a=GROUP_A, group_b=GROUP_B,
        )
    # Constant response: LMM short-circuits -> paired-t code path -> "None"
    # because std(differences) == 0.
    assert result.test_name == "None"
    assert "identical" in " ".join(r.message.lower() for r in caplog.records)


def test_pairs_missing_from_one_cohort_are_dropped():
    """A (mouse, session) pair without both cohorts shouldn't contribute."""
    rng = np.random.default_rng(0)
    rows = []
    # Six (mouse, session) pairs in Pre, only four in Post.
    for mid, base in zip(range(1, 7), [10.0, 12.0, 11.0, 13.0, 9.0, 14.0]):
        rows.append((f"m{mid}", 1, GROUP_A, base + rng.normal(0, 0.3)))
    for mid, base in zip(range(1, 5), [10.0, 12.0, 11.0, 13.0]):
        rows.append((f"m{mid}", 1, GROUP_B, base - 4 + rng.normal(0, 0.3)))
    df = pd.DataFrame(rows, columns=["mouse_id", "heating_session", "cohort", "value"])
    result = compare_paired_lmm(
        "y", GROUP_A, GROUP_B, df, group_a=GROUP_A, group_b=GROUP_B,
    )
    assert result.n_pairs == 4


def test_too_few_pairs_returns_none_test():
    rows = [
        ("m1", 1, GROUP_A, 1.0),
        ("m1", 1, GROUP_B, 2.0),
    ]
    df = pd.DataFrame(rows, columns=["mouse_id", "heating_session", "cohort", "value"])
    result = compare_paired_lmm(
        "y", GROUP_A, GROUP_B, df, group_a=GROUP_A, group_b=GROUP_B,
    )
    assert result.test_name == "None"
    assert result.n_pairs == 1


def test_all_untestable_rows_write_parseable_header_only_csv(tmp_path):
    path = _write_csv(
        [
            {
                "Module": "Baseline transients",
                "Folder": "transients",
                "Measurement": "frequency_hz",
                "Test": "None",
                "Statistic": np.nan,
                "p_value": np.nan,
                "Pre_mean": 0.0,
                "Pre_sem": 0.0,
                "Pre_n": 8,
                "Post_mean": 0.0,
                "Post_sem": 0.0,
                "Post_n": 8,
                "n_pairs": 8,
            },
        ],
        tmp_path / "paired_stats.csv",
    )

    written = pd.read_csv(path)
    assert written.empty
    assert "Folder" not in written.columns
    assert {
        "Module", "Measurement", "Test", "Statistic", "p_value",
        "Pre_mean", "Pre_sem", "Pre_n",
        "Post_mean", "Post_sem", "Post_n", "n_pairs",
    } == set(written.columns)


def test_separate_repeats_become_separate_pairs():
    """Separate sessions from the same mouse retain distinct pair IDs."""
    rng = np.random.default_rng(0)
    rows = []
    base = 10.0
    for hsess in (1, 2):
        rows.append(("5008", hsess, GROUP_A, base + rng.normal(0, 0.3)))
        rows.append(("5008", hsess, GROUP_B, base - 2 + rng.normal(0, 0.3)))
    # Need another mouse so we have multiple groups for the LMM/OLS path.
    rows.append(("5009", 1, GROUP_A, base + rng.normal(0, 0.3)))
    rows.append(("5009", 1, GROUP_B, base - 2 + rng.normal(0, 0.3)))
    df = pd.DataFrame(rows, columns=["mouse_id", "heating_session", "cohort", "value"])
    result = compare_paired_lmm(
        "y", GROUP_A, GROUP_B, df, group_a=GROUP_A, group_b=GROUP_B,
    )
    # 5008_s1, 5008_s2, 5009_s1 = 3 pairs
    assert result.n_pairs == 3
