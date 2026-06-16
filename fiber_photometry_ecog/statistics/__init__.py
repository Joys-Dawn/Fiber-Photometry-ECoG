"""
Statistical analysis for cohort comparisons.

Generic unpaired-group stats - picks parametric vs non-parametric tests
per measurement using the same skew/kurtosis/N heuristic as
EphysAutomatedAnalysis (Curran, West & Finch 1996).

For 2 cohorts:    Welch's t-test     | Mann-Whitney U
For >2 cohorts:   one-way ANOVA      | Kruskal-Wallis  (with post-hoc pairs)

Driver `run_group_comparisons` walks every per-cohort analysis result,
extracts per-session scalar measurements, and writes one CSV per
output folder (means / transients).
"""

from .config import StatsConfig
from .decision import should_use_parametric
from .independent import (
    IndependentTestResult,
    compare_independent_groups,
)
from .runner import run_group_comparisons

__all__ = [
    "StatsConfig",
    "should_use_parametric",
    "IndependentTestResult",
    "compare_independent_groups",
    "run_group_comparisons",
]
