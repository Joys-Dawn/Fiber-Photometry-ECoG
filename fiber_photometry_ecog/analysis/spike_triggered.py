"""
4.8 Spike-Triggered Averages.

Align photometry signal to each detected interictal ECoG spike.
Mean ± SEM across spikes, ±30s window.
AUC via trapezoidal method.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import get_signal_and_time, time_to_index


@dataclass
class SpikeTriggeredSessionResult:
    mouse_id: str
    n_spikes: int
    mean_trace: np.ndarray          # mean photometry aligned to spikes
    sem_trace: np.ndarray
    auc: float                      # AUC of mean trace


@dataclass
class SpikeTriggeredGroupResult:
    session_results: List[SpikeTriggeredSessionResult]
    time_axis: np.ndarray           # seconds relative to spike
    group_mean: np.ndarray          # mean across sessions
    group_sem: np.ndarray
    group_auc: float


def compute_spike_triggered_average(
    sessions: List[Session],
    spike_times_per_session: List[np.ndarray],
    config: AnalysisConfig | None = None,
) -> SpikeTriggeredGroupResult:
    """Compute spike-triggered photometry averages.

    Parameters
    ----------
    sessions : list of Session objects with processed photometry data
    spike_times_per_session : list of arrays, each containing interictal
        spike times (seconds) for the corresponding session. Same length
        as sessions.
    config : analysis configuration
    """
    if config is None:
        config = AnalysisConfig()

    window_s = config.spike_triggered_window_s
    session_results = []
    all_mean_traces = []

    for s, spike_times in zip(sessions, spike_times_per_session):
        signal, time, fs = get_signal_and_time(s)
        window_samples = int(round(window_s * fs))
        n_out = 2 * window_samples + 1

        traces = []
        for t_spike in spike_times:
            center = time_to_index(t_spike, fs)
            start = center - window_samples
            end = center + window_samples + 1
            if start < 0 or end > len(signal):
                continue
            traces.append(signal[start:end])

        n_spikes = len(traces)
        if n_spikes > 0:
            trace_mat = np.array(traces)
            mean_trace = np.mean(trace_mat, axis=0)
            sem_trace = (np.std(trace_mat, axis=0, ddof=1) / np.sqrt(n_spikes)
                         if n_spikes > 1 else np.zeros(n_out))
            auc = float(np.trapezoid(mean_trace, dx=1.0 / fs))
        else:
            mean_trace = np.full(n_out, np.nan)
            sem_trace = np.full(n_out, np.nan)
            auc = np.nan

        all_mean_traces.append(mean_trace)
        session_results.append(SpikeTriggeredSessionResult(
            mouse_id=s.mouse_id,
            n_spikes=n_spikes,
            mean_trace=mean_trace,
            sem_trace=sem_trace,
            auc=auc,
        ))

    # Group average across session means
    fs = sessions[0].processed.fs if sessions else 1000.0
    window_samples = int(round(window_s * fs))
    n_out = 2 * window_samples + 1
    time_axis = np.linspace(-window_s, window_s, n_out)

    valid_traces = [t for t in all_mean_traces if not np.all(np.isnan(t))]
    if valid_traces:
        group_mat = np.array(valid_traces)
        group_mean = np.mean(group_mat, axis=0)
        group_sem = (np.std(group_mat, axis=0, ddof=1) / np.sqrt(len(valid_traces))
                     if len(valid_traces) > 1 else np.zeros(n_out))
        group_auc = float(np.trapezoid(group_mean, dx=1.0 / fs))
    else:
        group_mean = np.full(n_out, np.nan)
        group_sem = np.full(n_out, np.nan)
        group_auc = np.nan

    return SpikeTriggeredGroupResult(
        session_results=session_results,
        time_axis=time_axis,
        group_mean=group_mean,
        group_sem=group_sem,
        group_auc=group_auc,
    )
