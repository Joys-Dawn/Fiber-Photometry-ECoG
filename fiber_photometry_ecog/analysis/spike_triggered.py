"""
4.8 Spike-Triggered Averages.

Align photometry and EEG signals to each detected interictal ECoG spike.
Mean +/- SEM across spikes, +/-30s window.
Photometry segments are baseline-subtracted (mean of pre-event window).
EEG segments are polarity-aligned (flipped if spike peak is negative)
and baseline-subtracted.
AUC via trapezoidal method (photometry only).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from typing import Tuple

from ._helpers import get_signal_and_time, get_ecog_filtered, time_to_index


def _group_mean_sem(
    traces: List[np.ndarray], n_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group mean and SEM from a list of per-session mean traces, filtering NaN traces."""
    valid = [t for t in traces if not np.all(np.isnan(t))]
    if valid:
        mat = np.array(valid)
        mean = np.mean(mat, axis=0)
        sem = (np.std(mat, axis=0, ddof=1) / np.sqrt(len(valid))
               if len(valid) > 1 else np.zeros(n_out))
    else:
        mean = np.full(n_out, np.nan)
        sem = np.full(n_out, np.nan)
    return mean, sem


@dataclass
class SpikeTriggeredSessionResult:
    mouse_id: str
    heating_session: int
    n_spikes: int
    mean_trace: np.ndarray          # mean photometry aligned to spikes
    sem_trace: np.ndarray
    auc: float                      # AUC of mean trace
    eeg_mean_trace: np.ndarray      # mean EEG aligned to spikes (polarity-aligned)
    eeg_sem_trace: np.ndarray


@dataclass
class SpikeTriggeredGroupResult:
    session_results: List[SpikeTriggeredSessionResult]
    time_axis: np.ndarray           # seconds relative to spike
    group_mean: np.ndarray          # mean across sessions
    group_sem: np.ndarray
    group_auc: float
    eeg_group_mean: np.ndarray      # mean EEG across sessions
    eeg_group_sem: np.ndarray


def _polarity_align_eeg(eeg_seg: np.ndarray, center_idx: int, fs: float) -> np.ndarray:
    """Flip EEG segment if spike peak is negative, so all spikes appear upward.

    Checks mean value in +/-50ms window around center. If negative, flips.
    """
    peak_window = int(0.05 * fs)  # 50ms
    lo = max(0, center_idx - peak_window)
    hi = min(len(eeg_seg), center_idx + peak_window)
    peak_val = np.mean(eeg_seg[lo:hi])
    if peak_val < 0:
        return -eeg_seg
    return eeg_seg


def _compute_mean_sem(traces: List[np.ndarray], n_out: int):
    """Compute mean and SEM from a list of trace arrays."""
    n = len(traces)
    if n > 0:
        trace_mat = np.array(traces)
        mean_trace = np.mean(trace_mat, axis=0)
        sem_trace = (np.std(trace_mat, axis=0, ddof=1) / np.sqrt(n)
                     if n > 1 else np.zeros(n_out))
    else:
        mean_trace = np.full(n_out, np.nan)
        sem_trace = np.full(n_out, np.nan)
    return mean_trace, sem_trace


def compute_spike_triggered_average(
    sessions: List[Session],
    spike_times_per_session: List[np.ndarray],
    config: AnalysisConfig | None = None,
) -> SpikeTriggeredGroupResult:
    """Compute spike-triggered photometry and EEG averages.

    Parameters
    ----------
    sessions : list of Session objects with processed photometry and ECoG data
    spike_times_per_session : list of arrays, each containing interictal
        spike times (seconds) for the corresponding session. Same length
        as sessions.
    config : analysis configuration
    """
    if config is None:
        config = AnalysisConfig()

    window_s = config.spike_triggered_window_s
    fs0 = sessions[0].processed.fs
    bl_start_samples = int(round(config.spike_triggered_baseline_start_s * fs0))
    bl_end_samples = int(round(config.spike_triggered_baseline_end_s * fs0))
    auc_end_samples = int(round(config.spike_triggered_auc_end_s * fs0))
    session_results = []
    all_phot_means = []
    all_eeg_means = []

    for s, spike_times in zip(sessions, spike_times_per_session):
        signal, time, fs = get_signal_and_time(s)
        ecog = get_ecog_filtered(s)
        window_samples = int(round(window_s * fs))
        n_out = 2 * window_samples + 1

        phot_traces = []
        eeg_traces = []
        for t_spike in spike_times:
            center = time_to_index(t_spike, fs)
            start = center - window_samples
            end = center + window_samples + 1
            if start < 0 or end > len(signal):
                continue
            if ecog is not None and end > len(ecog):
                continue

            # Photometry: subtract pre-event baseline mean (-5s to -1s before spike)
            phot_seg = signal[start:end].copy()
            bl_lo = window_samples - bl_start_samples
            bl_hi = window_samples - bl_end_samples
            phot_seg = phot_seg - np.mean(phot_seg[bl_lo:bl_hi])
            phot_traces.append(phot_seg)

            # EEG: polarity-align then subtract pre-event baseline mean
            if ecog is not None:
                eeg_seg = ecog[start:end].copy()
                eeg_seg = _polarity_align_eeg(eeg_seg, window_samples, fs)
                eeg_seg = eeg_seg - np.mean(eeg_seg[bl_lo:bl_hi])
                eeg_traces.append(eeg_seg)

        n_spikes = len(phot_traces)

        # Photometry stats
        mean_trace, sem_trace = _compute_mean_sem(phot_traces, n_out)
        if n_spikes > 0:
            auc_slice = mean_trace[window_samples:window_samples + auc_end_samples]
            auc = float(np.trapezoid(auc_slice, dx=1.0 / fs))
        else:
            auc = np.nan

        # EEG stats
        eeg_mean, eeg_sem = _compute_mean_sem(eeg_traces, n_out)

        all_phot_means.append(mean_trace)
        all_eeg_means.append(eeg_mean)
        session_results.append(SpikeTriggeredSessionResult(
            mouse_id=s.mouse_id,
            heating_session=s.heating_session,
            n_spikes=n_spikes,
            mean_trace=mean_trace,
            sem_trace=sem_trace,
            auc=auc,
            eeg_mean_trace=eeg_mean,
            eeg_sem_trace=eeg_sem,
        ))

    # Group average across session means
    fs = sessions[0].processed.fs if sessions else 1000.0
    window_samples = int(round(window_s * fs))
    n_out = 2 * window_samples + 1
    time_axis = np.linspace(-window_s, window_s, n_out)

    # Photometry group
    group_mean, group_sem = _group_mean_sem(all_phot_means, n_out)
    if not np.all(np.isnan(group_mean)):
        auc_slice = group_mean[window_samples:window_samples + auc_end_samples]
        group_auc = float(np.trapezoid(auc_slice, dx=1.0 / fs))
    else:
        group_auc = np.nan

    # EEG group
    eeg_group_mean, eeg_group_sem = _group_mean_sem(all_eeg_means, n_out)

    return SpikeTriggeredGroupResult(
        session_results=session_results,
        time_axis=time_axis,
        group_mean=group_mean,
        group_sem=group_sem,
        group_auc=group_auc,
        eeg_group_mean=eeg_group_mean,
        eeg_group_sem=eeg_group_sem,
    )
