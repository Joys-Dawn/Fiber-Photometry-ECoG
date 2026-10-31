"""
Parametric vs non-parametric decision rule.

A measurement is treated as non-parametric when any group fails either the
shape-based thresholds or a Shapiro-Wilk normality test:

1. |skewness| > SKEW_THRESHOLD        -> non-parametric
2. |excess kurtosis| > KURT_THRESHOLD -> non-parametric
3. Shapiro-Wilk p < SHAPIRO_ALPHA     -> non-parametric
4. otherwise                           -> parametric

Shape thresholds follow Curran, West & Finch (1996),
"The robustness of test statistics to nonnormality and specification error
in confirmatory factor analysis", Psychological Methods 1(1).

Groups with fewer than three non-NaN values are not assessed.
"""

from typing import List

import numpy as np
from scipy.stats import kurtosis, shapiro, skew


SKEW_THRESHOLD = 2.0
KURT_THRESHOLD = 7.0
SHAPIRO_ALPHA = 0.05


def should_use_parametric(data_arrays: List[np.ndarray]) -> bool:
    """Return True if no assessed group fails the configured checks.

    data_arrays contains one 1-D array per group. NaNs are removed; groups
    with fewer than three remaining values are skipped. Errors return False.
    """
    for arr in data_arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 3:
            continue

        try:
            if abs(skew(arr)) > SKEW_THRESHOLD:
                return False
            if abs(kurtosis(arr)) > KURT_THRESHOLD:
                return False
            if shapiro(arr).pvalue < SHAPIRO_ALPHA:
                return False
        except Exception:
            return False

    return True
