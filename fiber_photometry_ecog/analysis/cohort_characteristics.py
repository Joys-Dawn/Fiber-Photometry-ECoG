"""
4.1 Cohort Characteristics.

Per session: baseline temperature, seizure threshold (temperature at UEO),
seizure duration (OFF − UEO). Group means ± SEM.
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
    seizure_duration_s: Optional[float] = None  # OFF − UEO; None for non-seizure
    heating_session: int = 1            # 1, 2, 3... for this mouse


@dataclass
class CohortGroupResult:
    session_results: List[CohortSessionResult]
    baseline_temp_mean: float
    baseline_temp_sem: float
    seizure_threshold_mean: float
    seizure_threshold_sem: float
    duration_mean: float = np.nan
    duration_sem: float = np.nan


def compute_cohort_characteristics(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> CohortGroupResult:
    """Compute cohort characteristics for a group of sessions."""
    sessions = [s for s in sessions if s.include_for_baseline]

    session_results = []
    baseline_temps = []
    threshold_temps = []
    durations = []

    for s in sessions:
        lm = s.landmarks
        bl_temp = lm.baseline_temp if lm is not None else None
        sz_temp = get_ueo_temp(s)

        # Seizure duration: only meaningful when this session actually had a
        # seizure (real UEO + real OFF). Equivalent landmarks for controls
        # don't represent a seizure and are excluded.
        duration = None
        if (lm is not None
                and lm.ueo_time is not None
                and lm.off_time is not None):
            duration = float(lm.off_time - lm.ueo_time)
            if duration <= 0:
                duration = None

        session_results.append(CohortSessionResult(
            mouse_id=s.mouse_id,
            baseline_temp=bl_temp,
            seizure_threshold_temp=sz_temp,
            seizure_duration_s=duration,
            heating_session=s.heating_session,
        ))

        if bl_temp is not None:
            baseline_temps.append(bl_temp)
        if sz_temp is not None:
            threshold_temps.append(sz_temp)
        if duration is not None:
            durations.append(duration)

    bl_arr = np.array(baseline_temps) if baseline_temps else np.array([])
    th_arr = np.array(threshold_temps) if threshold_temps else np.array([])
    dur_arr = np.array(durations) if durations else np.array([])

    return CohortGroupResult(
        session_results=session_results,
        baseline_temp_mean=float(np.mean(bl_arr)) if len(bl_arr) > 0 else np.nan,
        baseline_temp_sem=compute_sem(bl_arr) if len(bl_arr) > 0 else np.nan,
        seizure_threshold_mean=float(np.mean(th_arr)) if len(th_arr) > 0 else np.nan,
        seizure_threshold_sem=compute_sem(th_arr) if len(th_arr) > 0 else np.nan,
        duration_mean=float(np.mean(dur_arr)) if len(dur_arr) > 0 else np.nan,
        duration_sem=compute_sem(dur_arr) if len(dur_arr) > 0 else np.nan,
    )
