"""
Paired (within-subject) comparisons for two-group designs.

- Parametric : paired t-test  (`scipy.stats.ttest_rel`)
- Non-parametric : Wilcoxon signed-rank test  (`scipy.stats.wilcoxon`)

Test choice runs `decision.should_use_parametric` on the *paired
differences* (not on the original samples) - the paired-test assumption
is normality of differences, not of either group on its own.

Raw p-values only - no multiple-comparison correction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import math
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from .decision import should_use_parametric
from .independent import GroupDescriptive


@dataclass
class PairedTestResult:
    """Outcome of one paired comparison on a single measurement."""

    measurement: str
    test_name: str            # "Paired t-test" / "Wilcoxon signed-rank" / "None"
    statistic: float
    p_value: float
    n_pairs: int              # number of (subject, measurement) pairs after NaN drop
    descriptives: List[GroupDescriptive] = field(default_factory=list)


def _describe(name: str, values: np.ndarray) -> GroupDescriptive:
    values = values[~np.isnan(values)]
    n = int(len(values))
    if n == 0:
        return GroupDescriptive(name=name, n=0, mean=math.nan, sem=math.nan)
    mean = float(np.mean(values))
    if n >= 2:
        sem = float(np.std(values, ddof=1) / math.sqrt(n))
    else:
        sem = math.nan
    return GroupDescriptive(name=name, n=n, mean=mean, sem=sem)


def _pair_arrays(
    values_by_subject_a: Dict[str, float],
    values_by_subject_b: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align two {subject: value} dicts on shared subject keys, dropping any
    pair where either value is NaN. Returns (a_arr, b_arr, kept_subjects)
    in stable subject order.
    """
    common = sorted(set(values_by_subject_a) & set(values_by_subject_b))
    a_vals: List[float] = []
    b_vals: List[float] = []
    kept: List[str] = []
    for subj in common:
        av = float(values_by_subject_a[subj])
        bv = float(values_by_subject_b[subj])
        if math.isnan(av) or math.isnan(bv):
            continue
        a_vals.append(av)
        b_vals.append(bv)
        kept.append(subj)
    return np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float), kept


def compare_paired(
    measurement: str,
    group_a_name: str,
    values_a_by_subject: Dict[str, float],
    group_b_name: str,
    values_b_by_subject: Dict[str, float],
) -> PairedTestResult:
    """Run one paired comparison.

    Parameters
    ----------
    measurement : measurement label (used only for the result record).
    group_a_name, group_b_name : group labels (typically baseline vs treatment).
    values_a_by_subject, values_b_by_subject : {subject_id: scalar value}.
        Subjects present in both dicts (with finite values in both) define
        the paired sample. Subjects missing from either side are dropped.

    Returns
    -------
    PairedTestResult with stat, p, n_pairs, and descriptives covering
    *only the paired subjects* (so reported means align with the test).
    """
    a, b, kept = _pair_arrays(values_a_by_subject, values_b_by_subject)
    descriptives = [
        _describe(group_a_name, a),
        _describe(group_b_name, b),
    ]
    n = len(kept)
    if n < 2:
        return PairedTestResult(
            measurement=measurement,
            test_name="None",
            statistic=float("nan"),
            p_value=float("nan"),
            n_pairs=n,
            descriptives=descriptives,
        )

    differences = b - a
    # Zero-variance differences (constant response, all-equal differences, or
    # all-zero differences) produce NaN from ttest_rel and an exception from
    # wilcoxon. Both cases carry zero information about the contrast; surface
    # them as "None" rather than a "Paired t-test" row with blank cells.
    if float(np.std(differences)) == 0.0:
        return PairedTestResult(
            measurement=measurement,
            test_name="None",
            statistic=float("nan"),
            p_value=float("nan"),
            n_pairs=n,
            descriptives=descriptives,
        )

    # Decision rule operates on the differences for paired tests.
    use_parametric = should_use_parametric([differences])

    if use_parametric:
        stat, p = ttest_rel(a, b)
        test_name = "Paired t-test"
    else:
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        stat, p = float(res.statistic), float(res.pvalue)
        test_name = "Wilcoxon signed-rank"

    return PairedTestResult(
        measurement=measurement,
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p),
        n_pairs=n,
        descriptives=descriptives,
    )
