"""
LMM-enhanced paired comparison via R's ``lmerTest`` (Satterthwaite df).

Model
-----
``value ~ cohort + (1|mouse_id) + (1|pair_id)``

where ``pair_id = mouse_id + "_s" + heating_session``. Each ``pair_id``
contributes one Pre and one Post observation (the actual paired
structure of the design). ``mouse_id`` captures the additional
within-mouse correlation when a mouse contributes multiple
(mouse, session) pairs — without it we'd be treating 5008's s1 and s2
pairs as independent, which they are not.

This mirrors the EphysAutomatedAnalysis `(1|Mouse_ID) + (1|Cell_ID)`
specification, with ``pair_id`` filling the same role ``Cell_ID`` does
there: the inner-most clustering level that the two paired
observations live inside.

Fallback chain
--------------
1. **Primary**: ``lmer(value ~ cohort + (1|mouse_id) + (1|pair_id))``.
   Satterthwaite-corrected denominator df (lmerTest), Wald t from emmeans
   pairwise contrast.

2. **When the full LMM fails or is singular** (or R is unavailable):
   paired t-test on ``(mouse_id, heating_session)`` differences with a
   loud warning. The pair-only intermediate LMM was removed because
   with 2 obs per ``pair_id`` it collapses algebraically to the paired
   t-test and was failing the singular check for the same reason the
   full LMM did — providing no extra rescue path in practice. Install
   R + lmerTest to enable the primary LMM.

The PairedTestResult is the unified return type so the runner CSV
code stays uniform; the ``test_name`` field tells you which path ran:
``"LMM (Satterthwaite)"``, ``"Paired t-test"``, or ``"None"``.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from . import r_bridge
from .paired import PairedTestResult, _describe, compare_paired

logger = logging.getLogger(__name__)


def _paired_t_on_sessions_fallback(
    measurement: str,
    group_a_name: str, group_b_name: str,
    df_long: pd.DataFrame, group_a: str, group_b: str,
    reason: str = "R/lmerTest unavailable",
) -> PairedTestResult:
    """Paired t-test on (mouse_id, heating_session) pairs.

    Used both when R is unavailable AND when both LMM fits (full and
    pair-only) degenerate. Pairs are formed on
    (mouse_id, heating_session), so 5008_s1 and 5008_s2 each contribute
    one pair. This does NOT account for within-mouse correlation across
    sessions — when reached via singular LMMs, that correlation simply
    cannot be identified from the data.
    """
    logger.warning(
        "LMM paired test [%s]: %s. Falling back to paired t-test on "
        "(mouse, session) pairs — mouse-level clustering across "
        "sessions is unmodeled.",
        measurement, reason,
    )
    # df_long uses long-format with one row per (mouse_id, heating_session,
    # cohort). Build pair_id from columns we know are present.
    a_dict, b_dict = {}, {}
    for _, row in df_long.iterrows():
        if "pair_id" in df_long.columns:
            key = str(row["pair_id"])
        else:
            key = f"{row['mouse_id']}_s{int(row['heating_session'])}"
        if row["cohort"] == group_a:
            a_dict[key] = float(row["value"])
        elif row["cohort"] == group_b:
            b_dict[key] = float(row["value"])
    return compare_paired(
        measurement=measurement,
        group_a_name=group_a_name, values_a_by_subject=a_dict,
        group_b_name=group_b_name, values_b_by_subject=b_dict,
    )


def compare_paired_lmm(
    measurement: str,
    group_a_name: str,
    group_b_name: str,
    df_long: pd.DataFrame,
    group_a: str,
    group_b: str,
) -> PairedTestResult:
    """Run one LMM-based paired comparison on a long-format frame.

    Parameters
    ----------
    measurement : measurement label (used only for the result record).
    group_a_name, group_b_name : display labels for the descriptive output.
    df_long : DataFrame with columns ``mouse_id``, ``heating_session``,
        ``cohort``, ``value``. One row per session. NaN values are
        dropped before fitting.
    group_a, group_b : cohort keys defining the contrast direction
        (effect is ``group_b - group_a``).

    Returns
    -------
    PairedTestResult. ``test_name`` indicates which path ran:
    ``"LMM (Satterthwaite)"`` on success, ``"Paired t-test"`` when the
    LMM fails / is singular or R is unavailable, ``"None"`` for
    zero-variance differences or too few pairs.
    """
    df = df_long[df_long["cohort"].isin([group_a, group_b])].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    # Keep only mice present in both cohorts (paired design).
    by_mouse_cohorts = df.groupby("mouse_id")["cohort"].nunique()
    complete_mice = by_mouse_cohorts[by_mouse_cohorts == 2].index
    df = df[df["mouse_id"].isin(complete_mice)]

    # pair_id = (mouse_id, heating_session) — the actual paired unit.
    df["heating_session"] = df["heating_session"].astype(int)
    df["pair_id"] = (df["mouse_id"].astype(str)
                     + "_s" + df["heating_session"].astype(str))

    # Keep only pair_ids present in both cohorts (the proper paired design).
    by_pair_cohorts = df.groupby("pair_id")["cohort"].nunique()
    complete_pairs = by_pair_cohorts[by_pair_cohorts == 2].index
    df = df[df["pair_id"].isin(complete_pairs)]

    a_vals = df.loc[df["cohort"] == group_a, "value"].to_numpy(dtype=float)
    b_vals = df.loc[df["cohort"] == group_b, "value"].to_numpy(dtype=float)
    descriptives = [
        _describe(group_a_name, a_vals),
        _describe(group_b_name, b_vals),
    ]
    n_pairs = int(len(complete_pairs))
    n_mice = int(df["mouse_id"].nunique())

    if n_pairs < 2:
        return PairedTestResult(
            measurement=measurement,
            test_name="None",
            statistic=float("nan"), p_value=float("nan"),
            n_pairs=n_pairs, descriptives=descriptives,
        )

    # If R isn't available, paired t-test fallback (ignores mouse clustering).
    if not r_bridge.is_r_available():
        return _paired_t_on_sessions_fallback(
            measurement, group_a_name, group_b_name,
            df, group_a, group_b,
        )

    # Primary path: lmerTest with nested random effects.
    r_data = df[["value", "cohort", "mouse_id", "pair_id"]].copy()
    # All-or-none variation check — lmer crashes when the response is constant.
    if r_data["value"].nunique() < 2:
        return _paired_t_on_sessions_fallback(
            measurement, group_a_name, group_b_name,
            r_data, group_a, group_b,
            reason="all observations identical",
        )

    result = r_bridge.fit_lmm_pairwise(
        r_data,
        formula="value ~ cohort",
        random_effects="(1|mouse_id) + (1|pair_id)",
        factor="cohort",
        ddf="Satterthwaite",
    )

    if result is None or not result.pairwise:
        # Bridge returns None for FIT_FAILED, SINGULAR, or CONVERGENCE_WARNING.
        return _paired_t_on_sessions_fallback(
            measurement, group_a_name, group_b_name,
            r_data, group_a, group_b,
            reason="LMM failed or singular",
        )

    # emmeans contrast is `<level_a> - <level_b>`. For factor levels alpha
    # ordered (A=Post, B=Pre or vice-versa) we want the Post - Pre direction:
    # find the contrast that goes group_b - group_a and grab its sign.
    chosen = None
    for pw in result.pairwise:
        # contrast strings look like "10HZ_DG - 10HZ_DG_CPP"
        if (pw.level_a, pw.level_b) == (group_b, group_a):
            chosen = pw
            sign = 1.0
            break
        if (pw.level_a, pw.level_b) == (group_a, group_b):
            chosen = pw
            sign = -1.0  # flip to report group_b - group_a
            break
    if chosen is None:
        return _paired_t_on_sessions_fallback(
            measurement, group_a_name, group_b_name,
            r_data, group_a, group_b,
            reason="LMM emmeans returned no contrast for requested cohorts",
        )

    t_val = sign * float(chosen.t_value)
    p_val = float(chosen.p_value)
    if not (math.isfinite(t_val) and math.isfinite(p_val)):
        return _paired_t_on_sessions_fallback(
            measurement, group_a_name, group_b_name,
            r_data, group_a, group_b,
            reason="LMM produced non-finite t or p",
        )

    return PairedTestResult(
        measurement=measurement,
        test_name="LMM (Satterthwaite)",
        statistic=t_val, p_value=p_val,
        n_pairs=n_pairs, descriptives=descriptives,
    )
