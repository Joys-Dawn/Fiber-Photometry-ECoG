"""
Parametric vs non-parametric decision rule.

A measurement is treated as non-parametric when any group fails either the
shape-based thresholds or a Shapiro-Wilk normality test:

1. |skewness| > SKEW_THRESHOLD        -> non-parametric
2. |excess kurtosis| > KURT_THRESHOLD -> non-parametric
3. Shapiro-Wilk p < SHAPIRO_ALPHA     -> non-parametric
4. otherwise                           -> parametric

Thresholds follow Curran, West & Finch (1996),
"The robustness of test statistics to nonnormality and specification error
in confirmatory factor analysis", Psychological Methods 1(1).

Note: skew/kurtosis require >=3 values per group; smaller groups are
treated as "no detectable shape violation" so they don't auto-fall to
non-parametric tests on N alone.
"""

from typing import List

import numpy as np
from scipy.stats import kurtosis, shapiro, skew


SKEW_THRESHOLD = 2.0
KURT_THRESHOLD = 7.0
SHAPIRO_ALPHA = 0.05


def should_use_parametric(data_arrays: List[np.ndarray]) -> bool:
    """Return True if all groups satisfy parametric assumptions.

    Parameters
    ----------
    data_arrays : list of 1-D numpy arrays, one per group. NaNs are dropped
        per-group before assessing assumptions.

    Returns
    -------
    True if parametric tests are appropriate, False otherwise. Groups with
    fewer than 3 finite values contribute no shape evidence and never veto
    parametric on their own.
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
