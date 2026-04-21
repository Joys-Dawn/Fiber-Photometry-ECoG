"""
4.5 Ictal Mean Signal.

Mean z-ΔF/F within seizure period vs baseline.
Δ z-ΔF/F between pre-ictal and ictal windows.
Event-triggered averages (±30s) for each landmark.
AUC via trapezoidal integration on each triggered average.
Multi-seizure handling: EEC/UEO from 1st seizure, OFF from last seizure.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import (
    get_eec_time,
    get_ueo_time,
    get_behavioral_onset_time,
    get_off_time,
    get_signal_and_time,
    time_to_index,
    compute_sem,
)


@dataclass
class TriggeredAverage:
    """Event-triggered average for one landmark."""
    time_axis: np.ndarray       # seconds relative to event (e.g., -30 to +30)
    mean_trace: np.ndarray      # mean z-ΔF/F across sessions
    sem_trace: np.ndarray       # SEM across sessions
    auc: float                  # trapezoidal AUC of mean trace
    per_session_auc: List[float]    # AUC for each session's triggered trace
    per_session_traces: List[np.ndarray]  # individual session traces


@dataclass
class IctalMeanSessionResult:
    mouse_id: str
    heating_session: int
    seizure_mean: float         # mean z-ΔF/F during seizure/equivalent period
    baseline_mean: float        # mean z-ΔF/F during baseline
    delta_preictal_ictal: float # seizure_mean - late_heat_mean


@dataclass
class IctalMeanGroupResult:
    session_results: List[IctalMeanSessionResult]
    seizure_mean: float
    seizure_sem: float
    baseline_mean: float
    baseline_sem: float
    delta_mean: float
    delta_sem: float
    triggered_averages: Dict[str, TriggeredAverage]  # keyed by landmark name


def _extract_triggered_trace(
    signal: np.ndarray,
    fs: float,
    event_idx: int,
    window_samples: int,
) -> Optional[np.ndarray]:
    """Extract a window of signal centered on event_idx.

    Returns None if the window extends beyond the signal boundaries.
    """
    start = event_idx - window_samples
    end = event_idx + window_samples + 1
    if start < 0 or end > len(signal):
        return None
    return signal[start:end].copy()


def _compute_triggered_average(
    sessions: List[Session],
    landmark_func,
    window_s: float,
) -> TriggeredAverage:
    """Compute event-triggered average across sessions for a given landmark."""
    traces = []
    aucs = []

    for s in sessions:
        t_event = landmark_func(s)
        if t_event is None:
            continue
        signal, time, fs = get_signal_and_time(s)
        window_samples = int(round(window_s * fs))
        event_idx = time_to_index(t_event, fs)
        trace = _extract_triggered_trace(signal, fs, event_idx, window_samples)
        if trace is None:
            continue
        traces.append(trace)
        auc = float(np.trapezoid(trace, dx=1.0 / fs))
        aucs.append(auc)

    if not traces:
        n_samples = 1
        return TriggeredAverage(
            time_axis=np.array([0.0]),
            mean_trace=np.array([np.nan]),
            sem_trace=np.array([np.nan]),
            auc=np.nan,
            per_session_auc=[],
            per_session_traces=[],
        )

    # All traces should be the same length (same fs, same window)
    trace_mat = np.array(traces)
    mean_trace = np.mean(trace_mat, axis=0)
    sem_trace = (np.std(trace_mat, axis=0, ddof=1) / np.sqrt(len(traces))
                 if len(traces) > 1 else np.zeros(trace_mat.shape[1]))

    fs = sessions[0].processed.fs
    n_samples = trace_mat.shape[1]
    time_axis = np.linspace(-window_s, window_s, n_samples)
    auc_mean = float(np.trapezoid(mean_trace, dx=1.0 / fs))

    return TriggeredAverage(
        time_axis=time_axis,
        mean_trace=mean_trace,
        sem_trace=sem_trace,
        auc=auc_mean,
        per_session_auc=aucs,
        per_session_traces=traces,
    )


def compute_ictal_mean(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> IctalMeanGroupResult:
    """Compute ictal mean signal metrics for a group of sessions."""
    if config is None:
        config = AnalysisConfig()

    session_results = []
    sz_means = []
    bl_means = []
    deltas = []

    for s in sessions:
        signal, time, fs = get_signal_and_time(s)
        lm = s.landmarks

        # Baseline mean: start to heating
        i_heat = time_to_index(lm.heating_start_time, fs)
        bl_mean = float(np.mean(signal[:i_heat]))

        # Seizure/equivalent period mean
        ueo_t = get_ueo_time(s)
        off_t = get_off_time(s)

        i_ueo = time_to_index(ueo_t, fs)
        i_off = time_to_index(off_t, fs)
        sz_mean = float(np.mean(signal[i_ueo:i_off]))

        # Late heat mean (midpoint of heating to UEO)
        i_mid = (i_heat + i_ueo) // 2
        late_heat_mean = float(np.mean(signal[i_mid:i_ueo]))

        delta = sz_mean - late_heat_mean

        session_results.append(IctalMeanSessionResult(
            mouse_id=s.mouse_id,
            heating_session=s.heating_session,
            seizure_mean=sz_mean,
            baseline_mean=bl_mean,
            delta_preictal_ictal=delta,
        ))
        sz_means.append(sz_mean)
        bl_means.append(bl_mean)
        deltas.append(delta)

    # Triggered averages for each landmark
    window_s = config.triggered_window_s
    landmarks = {
        "EEC": get_eec_time,
        "UEO": get_ueo_time,
        "behavioral_onset": get_behavioral_onset_time,
        "OFF": get_off_time,
        "max_temp": lambda s: s.landmarks.max_temp_time if s.landmarks else None,
    }

    triggered = {}
    for name, func in landmarks.items():
        triggered[name] = _compute_triggered_average(sessions, func, window_s)

    sz_arr = np.array(sz_means)
    bl_arr = np.array(bl_means)
    d_arr = np.array(deltas)

    return IctalMeanGroupResult(
        session_results=session_results,
        seizure_mean=float(np.mean(sz_arr)),
        seizure_sem=compute_sem(sz_arr),
        baseline_mean=float(np.mean(bl_arr)),
        baseline_sem=compute_sem(bl_arr),
        delta_mean=float(np.mean(d_arr)),
        delta_sem=compute_sem(d_arr),
        triggered_averages=triggered,
    )
