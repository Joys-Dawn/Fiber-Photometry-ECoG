"""
Independent-group comparisons (unpaired, between-subjects designs).

- 2 groups : Welch's unpaired t-test (parametric) / Mann-Whitney U (non-parametric)
- N groups : one-way ANOVA (parametric) / Kruskal-Wallis (non-parametric),
             with pairwise post-hoc tests (Welch t / Mann-Whitney U).

Test choice is delegated to `decision.should_use_parametric` and applied
once per measurement (same rule for the omnibus and post-hocs so the
reader doesn't have to track two assumption decisions).

Raw p-values only - no multiple-comparison correction is applied here
or by the runner.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
from scipy.stats import (
    f_oneway,
    kruskal,
    mannwhitneyu,
    ttest_ind,
)

from .decision import should_use_parametric


@dataclass
class GroupDescriptive:
    name: str
    n: int
    mean: float
    sem: float


@dataclass
class PairwisePosthoc:
    group1: str
    group2: str
    test_name: str       # "Welch t-test" or "Mann-Whitney U"
    statistic: float
    p_value: float


@dataclass
class IndependentTestResult:
    """Outcome of comparing N independent groups on a single measurement."""

    measurement: str
    test_name: str                  # "Welch t-test" / "Mann-Whitney U" / "One-way ANOVA" / "Kruskal-Wallis" / "None"
    statistic: float
    p_value: float
    descriptives: List[GroupDescriptive]
    posthocs: List[PairwisePosthoc] = field(default_factory=list)


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


def _drop_nans(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr[~np.isnan(arr)]


def _pairwise_posthocs(
    values_by_group: Dict[str, np.ndarray],
    use_parametric: bool,
) -> List[PairwisePosthoc]:
    """Run pairwise tests across all group pairs (raw p-values only)."""
    names = list(values_by_group.keys())
    out: List[PairwisePosthoc] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = _drop_nans(values_by_group[names[i]])
            b = _drop_nans(values_by_group[names[j]])
            if len(a) < 2 or len(b) < 2:
                continue
            if use_parametric:
                stat, p = ttest_ind(a, b, equal_var=False)
                test_name = "Welch t-test"
            else:
                res = mannwhitneyu(a, b, alternative="two-sided")
                stat, p = float(res.statistic), float(res.pvalue)
                test_name = "Mann-Whitney U"
            out.append(PairwisePosthoc(
                group1=names[i], group2=names[j],
                test_name=test_name,
                statistic=float(stat), p_value=float(p),
            ))
    return out


def compare_independent_groups(
    measurement: str,
    values_by_group: Dict[str, np.ndarray],
) -> IndependentTestResult:
    """Compare 2+ independent groups on a single measurement.

    Parameters
    ----------
    measurement : measurement label (used only for the result record).
    values_by_group : ordered mapping {group_label: 1-D array of values}.
        NaN values are ignored per group. Insertion order defines reporting
        order of descriptives and post-hoc pairs.

    Returns
    -------
    IndependentTestResult with omnibus stat/p, descriptives per group,
    and (for N>2) pairwise post-hoc results.
    """
    if len(values_by_group) < 2:
        raise ValueError(
            f"Need at least 2 groups, got {len(values_by_group)}"
        )

    clean = {name: _drop_nans(arr) for name, arr in values_by_group.items()}
    descriptives = [_describe(n, arr) for n, arr in clean.items()]

    # Need at least 2 values per group to do any test.
    insufficient = [n for n, arr in clean.items() if len(arr) < 2]
    if insufficient:
        return IndependentTestResult(
            measurement=measurement,
            test_name="None",
            statistic=float("nan"),
            p_value=float("nan"),
            descriptives=descriptives,
        )

    use_parametric = should_use_parametric(list(clean.values()))

    arrays = list(clean.values())
    if len(clean) == 2:
        a, b = arrays
        if use_parametric:
            stat, p = ttest_ind(a, b, equal_var=False)
            test_name = "Welch t-test"
        else:
            res = mannwhitneyu(a, b, alternative="two-sided")
            stat, p = float(res.statistic), float(res.pvalue)
            test_name = "Mann-Whitney U"
        posthocs: List[PairwisePosthoc] = []
    else:
        if use_parametric:
            stat, p = f_oneway(*arrays)
            test_name = "One-way ANOVA"
        else:
            res = kruskal(*arrays)
            stat, p = float(res.statistic), float(res.pvalue)
            test_name = "Kruskal-Wallis"
        posthocs = _pairwise_posthocs(clean, use_parametric)

    return IndependentTestResult(
        measurement=measurement,
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p),
        descriptives=descriptives,
        posthocs=posthocs,
    )
