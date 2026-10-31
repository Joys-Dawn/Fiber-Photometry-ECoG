"""
Spike-triggered averages.

Align photometry and EEG signals to each detected interictal ECoG spike.
Mean +/- SEM across spikes within the configured window.
Photometry segments are baseline-subtracted (mean of pre-event window).
EEG segments are polarity-aligned using their mean within +/-50 ms
and baseline-subtracted.
AUC via trapezoidal method (photometry only).
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from typing import Tuple

from ._helpers import get_signal_and_time, get_ecog_filtered, time_to_index

log = logging.getLogger(__name__)


def _group_mean_sem(
    traces: List[np.ndarray], n_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pointwise mean and SEM across available session values."""
    valid = [t for t in traces if not np.all(np.isnan(t))]
    if valid:
        mat = np.array(valid)
        n_per_sample = np.sum(~np.isnan(mat), axis=0)
        mean = np.divide(
            np.nansum(mat, axis=0), n_per_sample,
            out=np.full(n_out, np.nan), where=n_per_sample > 0,
        )
        sem = np.full(n_out, np.nan)
        has_sem = n_per_sample >= 2
        sem[has_sem] = (
            np.nanstd(mat[:, has_sem], axis=0, ddof=1)
            / np.sqrt(n_per_sample[has_sem])
        )
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
    """Flip EEG if its NaN-excluding mean within +/-50 ms is negative."""
    peak_window = int(0.05 * fs)  # 50ms
    lo = max(0, center_idx - peak_window)
    hi = min(len(eeg_seg), center_idx + peak_window)
    peak_val = float(np.nanmean(eeg_seg[lo:hi]))
    if peak_val < 0:
        return -eeg_seg
    return eeg_seg


def _compute_mean_sem(traces: List[np.ndarray], n_out: int):
    """Compute pointwise mean and SEM from NaN-padded spike traces."""
    n = len(traces)
    if n > 0:
        trace_mat = np.array(traces)
        mean_trace = np.nanmean(trace_mat, axis=0)
        n_per_sample = np.sum(~np.isnan(trace_mat), axis=0)
        if n > 1:
            std_trace = np.nanstd(trace_mat, axis=0, ddof=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                sem_trace = std_trace / np.sqrt(n_per_sample)
            sem_trace = np.where(n_per_sample >= 2, sem_trace, np.nan)
        else:
            sem_trace = np.zeros(n_out)
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
        bl_lo = window_samples - bl_start_samples
        bl_hi = window_samples - bl_end_samples
        for t_spike in spike_times:
            center = time_to_index(t_spike, fs)
            start = center - window_samples
            end = center + window_samples + 1

            # NaN-pad photometry segment so edge-clipped spikes still
            # contribute where data exists rather than being dropped.
            phot_seg = np.full(n_out, np.nan, dtype=float)
            sig_start = max(0, start)
            sig_end = min(len(signal), end)
            if sig_end > sig_start:
                phot_seg[sig_start - start: sig_end - start] = signal[sig_start:sig_end]

            # Subtract the mean of the available baseline samples.
            phot_baseline = phot_seg[bl_lo:bl_hi]
            if np.all(np.isnan(phot_baseline)):
                phot_seg.fill(np.nan)
            else:
                phot_seg = phot_seg - float(np.nanmean(phot_baseline))
            phot_traces.append(phot_seg)

            # EEG: NaN-pad, polarity-align, then nanmean baseline-subtract.
            if ecog is not None:
                eeg_seg = np.full(n_out, np.nan, dtype=float)
                ecog_end = min(len(ecog), end)
                if ecog_end > sig_start:
                    eeg_seg[sig_start - start: ecog_end - start] = ecog[sig_start:ecog_end]
                eeg_seg = _polarity_align_eeg(eeg_seg, window_samples, fs)
                eeg_baseline = eeg_seg[bl_lo:bl_hi]
                if np.all(np.isnan(eeg_baseline)):
                    eeg_seg.fill(np.nan)
                else:
                    eeg_seg = eeg_seg - float(np.nanmean(eeg_baseline))
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
