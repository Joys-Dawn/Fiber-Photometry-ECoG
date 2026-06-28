"""
Per-pair multiplicative/log-scale paired tests.

Three branches based on the sign pattern of the cell:

- All-positive: ratio paired t-test on log(b/a). Reports GMR = exp(mean
  log r) with a multiplicative CI.
- All-negative (sign consistent across all pairs, typical for `min`):
  same test on |x|. GMR represents a ratio of magnitudes.
- Mixed-sign or any zero: signed-log paired t-test (sign(x)*log1p(|x|)).
  Effect estimate is on the signed-log scale; GMR back-transform is not
  meaningful, so the GMR fields are left NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, ttest_rel


@dataclass
class RatioTestResult:
    test_name: str
    n_used: int
    gmr: float
    gmr_lo: float
    gmr_hi: float
    signed_log_mean_diff: float
    p_value: float
    notes: str


def gmr_paired_test(
    a_vals: np.ndarray, b_vals: np.ndarray
) -> Optional[RatioTestResult]:
    """Return a RatioTestResult comparing paired arrays b/a, or None if
    fewer than 2 finite pairs are available.
    """
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)
    mask = np.isfinite(a_vals) & np.isfinite(b_vals)
    a = a_vals[mask]
    b = b_vals[mask]
    if a.size < 2:
        return None

    all_neg = bool(np.all(a < 0) and np.all(b < 0))
    all_pos = bool(np.all(a > 0) and np.all(b > 0))

    if all_pos or all_neg:
        notes = ""
        test_name = "ratio paired t-test (log)"
        if all_neg:
            a = -a
            b = -b
            notes = "all values < 0; computed GMR on |x|"
            test_name = "ratio paired t-test (log) on |x|"
        log_r = np.log(b) - np.log(a)
        result = ttest_1samp(log_r, popmean=0.0)
        mean_lr = float(np.mean(log_r))
        sem_lr = float(np.std(log_r, ddof=1) / np.sqrt(log_r.size))
        df = log_r.size - 1
        tcrit = float(student_t.ppf(0.975, df))
        return RatioTestResult(
            test_name=test_name,
            n_used=int(log_r.size),
            gmr=float(np.exp(mean_lr)),
            gmr_lo=float(np.exp(mean_lr - tcrit * sem_lr)),
            gmr_hi=float(np.exp(mean_lr + tcrit * sem_lr)),
            signed_log_mean_diff=float("nan"),
            p_value=float(result.pvalue),
            notes=notes,
        )

    # Mixed signs (or any zero): signed-log paired t-test. sign(x)*log1p(|x|)
    # is continuous at 0 and preserves sign — sign-flip pairs register as
    # large differences (correct biological direction). Effect estimate stays
    # on the signed-log scale; GMR back-transform is not meaningful here.
    sa = np.sign(a) * np.log1p(np.abs(a))
    sb = np.sign(b) * np.log1p(np.abs(b))
    diffs = sb - sa
    result = ttest_rel(sb, sa)
    n_neg_a = int(np.sum(a < 0))
    n_neg_b = int(np.sum(b < 0))
    n_zero = int(np.sum((a == 0) | (b == 0)))
    note_bits = [
        f"mixed signs (a: {n_neg_a}/{a.size} neg, "
        f"b: {n_neg_b}/{b.size} neg",
    ]
    if n_zero:
        note_bits.append(f"; {n_zero} pair(s) contain zero")
    note_bits.append(
        "); signed-log paired t-test used; effect on signed-log scale, "
        "GMR not applicable"
    )
    return RatioTestResult(
        test_name="paired t-test (signed log)",
        n_used=int(diffs.size),
        gmr=float("nan"),
        gmr_lo=float("nan"),
        gmr_hi=float("nan"),
        signed_log_mean_diff=float(np.mean(diffs)),
        p_value=float(result.pvalue),
        notes="".join(note_bits),
    )


def pair_arrays(
    df_long, cohort_a: str, cohort_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair (mouse_id, heating_session) rows across two cohorts. Drops
    pair_ids missing from either cohort. Returns (a_vals, b_vals) aligned
    on shared pair_ids.

    `df_long` must have columns: mouse_id, heating_session, cohort, value.
    """
    import pandas as pd  # local: keep the public surface dependency-free
    df = pd.DataFrame(df_long).dropna(subset=["value"]).copy()
    df = df[df["cohort"].isin([cohort_a, cohort_b])]
    if df.empty:
        return np.array([]), np.array([])
    df["pair_id"] = (
        df["mouse_id"].astype(str)
        + "_s" + df["heating_session"].astype(int).astype(str)
    )
    counts = df.groupby("pair_id")["cohort"].nunique()
    keep = counts[counts == 2].index
    df = df[df["pair_id"].isin(keep)].sort_values(["pair_id", "cohort"])
    a = df.loc[df["cohort"] == cohort_a].set_index("pair_id")["value"]
    b = df.loc[df["cohort"] == cohort_b].set_index("pair_id")["value"]
    common = a.index.intersection(b.index)
    return (
        a.loc[common].to_numpy(dtype=float),
        b.loc[common].to_numpy(dtype=float),
    )
