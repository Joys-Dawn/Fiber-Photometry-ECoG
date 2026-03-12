"""
4.1 Cohort Characteristics.

Per session: baseline temperature, seizure threshold (temperature at UEO).
Group means ± SEM.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import get_ueo_temp, compute_sem


@dataclass
class CohortSessionResult:
    mouse_id: str
    baseline_temp: Optional[float]      # degrees C
    seizure_threshold_temp: Optional[float]  # temp at UEO (or equivalent)


@dataclass
class CohortGroupResult:
    session_results: List[CohortSessionResult]
    baseline_temp_mean: float
    baseline_temp_sem: float
    seizure_threshold_mean: float
    seizure_threshold_sem: float


def compute_cohort_characteristics(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> CohortGroupResult:
    """Compute cohort characteristics for a group of sessions."""
    session_results = []
    baseline_temps = []
    threshold_temps = []

    for s in sessions:
        lm = s.landmarks
        bl_temp = lm.baseline_temp if lm is not None else None
        sz_temp = get_ueo_temp(s)

        session_results.append(CohortSessionResult(
            mouse_id=s.mouse_id,
            baseline_temp=bl_temp,
            seizure_threshold_temp=sz_temp,
        ))

        if bl_temp is not None:
            baseline_temps.append(bl_temp)
        if sz_temp is not None:
            threshold_temps.append(sz_temp)

    bl_arr = np.array(baseline_temps) if baseline_temps else np.array([])
    th_arr = np.array(threshold_temps) if threshold_temps else np.array([])

    return CohortGroupResult(
        session_results=session_results,
        baseline_temp_mean=float(np.mean(bl_arr)) if len(bl_arr) > 0 else np.nan,
        baseline_temp_sem=compute_sem(bl_arr) if len(bl_arr) > 0 else np.nan,
        seizure_threshold_mean=float(np.mean(th_arr)) if len(th_arr) > 0 else np.nan,
        seizure_threshold_sem=compute_sem(th_arr) if len(th_arr) > 0 else np.nan,
    )
