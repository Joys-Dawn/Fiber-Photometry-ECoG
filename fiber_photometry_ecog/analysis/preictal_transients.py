"""
4.4 Pre-Ictal Transients.

Sliding-window moving averages of transient frequency, amplitude,
half-width during heating.
Same metrics binned by configurable temperature bins, relative to
seizure onset temp (or equivalent).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session, TransientEvent
from ._helpers import (
    get_ueo_time,
    get_ueo_temp,
    filter_transients_by_time,
    compute_sem,
)


@dataclass
class MovingAvgPoint:
    """One point in a sliding-window moving average time series."""
    center_time: float          # seconds (relative to heating start)
    frequency_hz: float
    mean_amplitude: float
    mean_half_width_s: float


@dataclass
class PreictalTransientSessionResult:
    mouse_id: str
    heating_session: int
    moving_avg: List[MovingAvgPoint]
    temp_bin_centers: np.ndarray
    temp_bin_frequency: np.ndarray
    temp_bin_amplitude: np.ndarray
    temp_bin_half_width: np.ndarray


@dataclass
class PreictalTransientGroupResult:
    session_results: List[PreictalTransientSessionResult]
    # Group-level moving average arrays (aligned to common time grid)
    moving_avg_times: np.ndarray        # center times relative to heating start
    frequency_mean: np.ndarray
    frequency_sem: np.ndarray
    amplitude_mean: np.ndarray
    amplitude_sem: np.ndarray
    half_width_mean: np.ndarray
    half_width_sem: np.ndarray
    # Group-level temperature-binned (discrete 1C bins, spec row 14)
    temp_bin_centers: np.ndarray
    temp_freq_mean: np.ndarray
    temp_freq_sem: np.ndarray
    temp_amp_mean: np.ndarray
    temp_amp_sem: np.ndarray
    temp_hw_mean: np.ndarray
    temp_hw_sem: np.ndarray
    # Group-level temperature moving average (smooth, spec row 13)
    temp_ma_centers: np.ndarray
    temp_ma_freq_mean: np.ndarray
    temp_ma_freq_sem: np.ndarray
    temp_ma_amp_mean: np.ndarray
    temp_ma_amp_sem: np.ndarray
    temp_ma_hw_mean: np.ndarray
    temp_ma_hw_sem: np.ndarray


def _sliding_window_transients(
    transients: List[TransientEvent],
    t_start: float,
    t_end: float,
    window_s: float,
    step_s: float,
) -> List[MovingAvgPoint]:
    """Compute sliding-window moving average of transient properties."""
    results = []
    center = t_start + window_s / 2
    while center + window_s / 2 <= t_end:
        w_start = center - window_s / 2
        w_end = center + window_s / 2
        w_trans = filter_transients_by_time(transients, w_start, w_end)
        duration = w_end - w_start
        freq = len(w_trans) / duration if duration > 0 else 0.0
        if len(w_trans) > 0:
            amp = float(np.mean([t.z_peak_to_trough if t.z_peak_to_trough is not None
                                else t.peak_to_trough for t in w_trans]))
            hw = float(np.mean([t.half_width for t in w_trans]))
        else:
            amp = 0.0
            hw = 0.0
        results.append(MovingAvgPoint(
            center_time=center - t_start,  # relative to heating start
            frequency_hz=freq,
            mean_amplitude=amp,
            mean_half_width_s=hw,
        ))
        center += step_s
    return results


def _sliding_window_transients_by_temp(
    session: Session,
    transients: List[TransientEvent],
    ueo_temp: float,
    heat_start: float,
    ueo_t: float,
    temp_range: float,
    window_C: float,
    step_C: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding window over temperature axis (rel to seizure onset).

    For each temp window [c-w/2, c+w/2], frequency = events / time_in_window.
    "Time in window" is the duration the smoothed temperature spends inside
    that window during the heating phase, so frequency is in true Hz.
    """
    if session.processed is None or session.processed.temperature_smooth is None:
        return np.array([]), np.array([]), np.array([]), np.array([])
    fs = session.processed.fs
    if fs is None or fs <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    temp = np.asarray(session.processed.temperature_smooth)
    i0 = max(0, int(heat_start * fs))
    i1 = min(len(temp), int(ueo_t * fs))
    if i1 <= i0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    rel_temp = temp[i0:i1] - ueo_temp

    centers = np.arange(-temp_range + window_C / 2,
                        0 - window_C / 2 + step_C / 2, step_C)
    freqs = np.zeros(len(centers))
    amps = np.zeros(len(centers))
    hws = np.zeros(len(centers))

    for i, c in enumerate(centers):
        w_min = c - window_C / 2
        w_max = c + window_C / 2
        n_in = int(np.sum((rel_temp >= w_min) & (rel_temp <= w_max)))
        time_in_window_s = n_in / fs
        in_window = [t for t in transients
                     if t.temperature_at_peak is not None
                     and w_min <= (t.temperature_at_peak - ueo_temp) <= w_max]
        if time_in_window_s > 0:
            freqs[i] = len(in_window) / time_in_window_s
        if in_window:
            amps[i] = float(np.mean([t.z_peak_to_trough if t.z_peak_to_trough is not None
                                     else t.peak_to_trough for t in in_window]))
            hws[i] = float(np.mean([t.half_width for t in in_window]))
        else:
            amps[i] = np.nan
            hws[i] = np.nan
    return centers, freqs, amps, hws


def _bin_transients_by_temperature(
    transients: List[TransientEvent],
    ueo_temp: float,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin transient properties by temperature relative to seizure onset.

    Returns (frequency_per_bin, amplitude_per_bin, half_width_per_bin).
    Frequency is count per bin (not normalized by time — consistent across sessions
    since temperature bins represent similar temp ranges).
    """
    n_bins = len(bin_edges) - 1
    counts = np.zeros(n_bins)
    amp_sums = np.zeros(n_bins)
    hw_sums = np.zeros(n_bins)

    for t in transients:
        if t.temperature_at_peak is None:
            continue
        rel_temp = t.temperature_at_peak - ueo_temp
        idx = np.searchsorted(bin_edges, rel_temp, side='right') - 1
        if 0 <= idx < n_bins:
            counts[idx] += 1
            amp_sums[idx] += (t.z_peak_to_trough if t.z_peak_to_trough is not None
                             else t.peak_to_trough)
            hw_sums[idx] += t.half_width

    freq = counts
    amp = np.where(counts > 0, amp_sums / counts, np.nan)
    hw = np.where(counts > 0, hw_sums / counts, np.nan)
    return freq, amp, hw


def compute_preictal_transients(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> PreictalTransientGroupResult:
    """Compute pre-ictal transient metrics for a group of sessions."""
    if config is None:
        config = AnalysisConfig()

    sessions = [s for s in sessions if s.include_for_transients]

    bin_size = config.temp_bin_size
    temp_range = config.preictal_temp_range
    bin_edges = np.arange(-temp_range, bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2

    session_results = []
    all_moving_avgs = []
    all_temp_freq = []
    all_temp_amp = []
    all_temp_hw = []
    all_tma_freq = []
    all_tma_amp = []
    all_tma_hw = []

    # Temp moving-avg uses 1°C window stepped at half-bin resolution by default
    tma_window_C = bin_size
    tma_step_C = bin_size / 2.0
    tma_centers_global = np.arange(-temp_range + tma_window_C / 2,
                                   0 - tma_window_C / 2 + tma_step_C / 2, tma_step_C)

    for s in sessions:
        lm = s.landmarks
        heat_start = lm.heating_start_time
        ueo_t = get_ueo_time(s)
        ueo_temp = get_ueo_temp(s)

        heat_transients = filter_transients_by_time(s.transients, heat_start, ueo_t)

        # Sliding window moving averages
        ma = _sliding_window_transients(
            heat_transients, heat_start, ueo_t,
            config.moving_avg_window_s, config.moving_avg_step_s,
        )
        all_moving_avgs.append(ma)

        # Temperature-binned (discrete 1C bins, spec row 14)
        freq, amp, hw = _bin_transients_by_temperature(
            heat_transients, ueo_temp, bin_edges,
        )
        all_temp_freq.append(freq)
        all_temp_amp.append(amp)
        all_temp_hw.append(hw)

        # Temperature moving average (smooth, spec row 13)
        _, tma_freq, tma_amp, tma_hw = _sliding_window_transients_by_temp(
            s, heat_transients, ueo_temp, heat_start, ueo_t,
            temp_range, tma_window_C, tma_step_C,
        )
        # Pad/truncate to match common grid length
        if len(tma_freq) != len(tma_centers_global):
            n = min(len(tma_freq), len(tma_centers_global))
            tma_freq = np.concatenate([tma_freq[:n], np.full(len(tma_centers_global) - n, np.nan)])
            tma_amp = np.concatenate([tma_amp[:n], np.full(len(tma_centers_global) - n, np.nan)])
            tma_hw = np.concatenate([tma_hw[:n], np.full(len(tma_centers_global) - n, np.nan)])
        all_tma_freq.append(tma_freq)
        all_tma_amp.append(tma_amp)
        all_tma_hw.append(tma_hw)

        session_results.append(PreictalTransientSessionResult(
            mouse_id=s.mouse_id,
            heating_session=s.heating_session,
            moving_avg=ma,
            temp_bin_centers=bin_centers.copy(),
            temp_bin_frequency=freq,
            temp_bin_amplitude=amp,
            temp_bin_half_width=hw,
        ))

    # Group-level moving average: align to common time grid
    # Use the shortest moving avg series to determine common times
    if all_moving_avgs and all(len(ma) > 0 for ma in all_moving_avgs):
        min_len = min(len(ma) for ma in all_moving_avgs)
        common_times = np.array([all_moving_avgs[0][i].center_time for i in range(min_len)])
        freq_mat = np.array([[ma[i].frequency_hz for i in range(min_len)] for ma in all_moving_avgs])
        amp_mat = np.array([[ma[i].mean_amplitude for i in range(min_len)] for ma in all_moving_avgs])
        hw_mat = np.array([[ma[i].mean_half_width_s for i in range(min_len)] for ma in all_moving_avgs])
    else:
        common_times = np.array([])
        freq_mat = np.empty((0, 0))
        amp_mat = np.empty((0, 0))
        hw_mat = np.empty((0, 0))

    def _mean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if mat.size == 0:
            return np.array([]), np.array([])
        m = np.mean(mat, axis=0)
        sem = np.std(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.zeros(mat.shape[1])
        return m, sem

    freq_mean, freq_sem = _mean_sem(freq_mat)
    amp_mean, amp_sem = _mean_sem(amp_mat)
    hw_mean, hw_sem = _mean_sem(hw_mat)

    # Group-level temperature-binned
    temp_freq_mat = np.array(all_temp_freq)
    temp_amp_mat = np.array(all_temp_amp)
    temp_hw_mat = np.array(all_temp_hw)

    def _nanmean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.nanmean(mat, axis=0)
        sem = np.full(mat.shape[1], np.nan)
        for b in range(mat.shape[1]):
            col = mat[:, b]
            valid = col[~np.isnan(col)]
            if len(valid) >= 2:
                sem[b] = compute_sem(valid)
        return m, sem

    tf_mean, tf_sem = _nanmean_sem(temp_freq_mat)
    ta_mean, ta_sem = _nanmean_sem(temp_amp_mat)
    tw_mean, tw_sem = _nanmean_sem(temp_hw_mat)

    # Group-level temp moving average
    tma_freq_mat = np.array(all_tma_freq) if all_tma_freq else np.empty((0, len(tma_centers_global)))
    tma_amp_mat = np.array(all_tma_amp) if all_tma_amp else np.empty((0, len(tma_centers_global)))
    tma_hw_mat = np.array(all_tma_hw) if all_tma_hw else np.empty((0, len(tma_centers_global)))
    tma_f_mean, tma_f_sem = _nanmean_sem(tma_freq_mat) if tma_freq_mat.size else (np.array([]), np.array([]))
    tma_a_mean, tma_a_sem = _nanmean_sem(tma_amp_mat) if tma_amp_mat.size else (np.array([]), np.array([]))
    tma_w_mean, tma_w_sem = _nanmean_sem(tma_hw_mat) if tma_hw_mat.size else (np.array([]), np.array([]))

    return PreictalTransientGroupResult(
        session_results=session_results,
        moving_avg_times=common_times,
        frequency_mean=freq_mean,
        frequency_sem=freq_sem,
        amplitude_mean=amp_mean,
        amplitude_sem=amp_sem,
        half_width_mean=hw_mean,
        half_width_sem=hw_sem,
        temp_bin_centers=bin_centers,
        temp_freq_mean=tf_mean,
        temp_freq_sem=tf_sem,
        temp_amp_mean=ta_mean,
        temp_amp_sem=ta_sem,
        temp_hw_mean=tw_mean,
        temp_hw_sem=tw_sem,
        temp_ma_centers=tma_centers_global,
        temp_ma_freq_mean=tma_f_mean,
        temp_ma_freq_sem=tma_f_sem,
        temp_ma_amp_mean=tma_a_mean,
        temp_ma_amp_sem=tma_a_sem,
        temp_ma_hw_mean=tma_w_mean,
        temp_ma_hw_sem=tma_w_sem,
    )
