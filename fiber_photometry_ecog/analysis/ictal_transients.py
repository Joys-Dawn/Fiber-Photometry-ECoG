"""
4.6 Ictal Transients.

PSTH: transient frequency in 10s bins, ±60s around UEO/equivalent.
Sliding-window moving averages of transient freq/amp/half-width
zoomed around UEO.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import (
    get_ueo_time,
    filter_transients_by_time,
)


@dataclass
class IctalTransientSessionResult:
    mouse_id: str
    psth_counts: np.ndarray         # transient count per bin
    psth_bin_centers: np.ndarray    # seconds relative to UEO


@dataclass
class IctalTransientGroupResult:
    session_results: List[IctalTransientSessionResult]
    psth_bin_centers: np.ndarray
    psth_mean: np.ndarray           # mean counts across sessions
    psth_sem: np.ndarray
    # Moving average around UEO
    moving_avg_times: np.ndarray    # seconds relative to UEO
    freq_mean: np.ndarray
    freq_sem: np.ndarray
    amp_mean: np.ndarray
    amp_sem: np.ndarray
    hw_mean: np.ndarray
    hw_sem: np.ndarray


def compute_ictal_transients(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> IctalTransientGroupResult:
    """Compute ictal transient metrics for a group of sessions."""
    if config is None:
        config = AnalysisConfig()

    bin_size = config.psth_bin_size_s
    window = config.psth_window_s
    edges = np.arange(-window, window + bin_size, bin_size)
    bin_centers = edges[:-1] + bin_size / 2

    session_results = []
    all_counts = []
    all_freq_ma = []
    all_amp_ma = []
    all_hw_ma = []

    # Moving average parameters
    ma_window = config.moving_avg_window_s
    ma_step = config.moving_avg_step_s
    ma_centers = np.arange(-window + ma_window / 2, window - ma_window / 2 + ma_step, ma_step)

    for s in sessions:
        ueo_t = get_ueo_time(s)

        # PSTH: bin transient times relative to UEO
        t_start = ueo_t - window
        t_end = ueo_t + window
        peri_transients = filter_transients_by_time(s.transients, t_start, t_end)

        # Get peak times relative to UEO
        rel_times = np.array([t.peak_time - ueo_t for t in peri_transients])
        counts, _ = np.histogram(rel_times, bins=edges)
        all_counts.append(counts)

        session_results.append(IctalTransientSessionResult(
            mouse_id=s.mouse_id,
            psth_counts=counts,
            psth_bin_centers=bin_centers.copy(),
        ))

        # Sliding window moving averages around UEO
        freqs = []
        amps = []
        hws = []
        for c in ma_centers:
            abs_start = ueo_t + c - ma_window / 2
            abs_end = ueo_t + c + ma_window / 2
            w_trans = filter_transients_by_time(s.transients, abs_start, abs_end)
            freq = len(w_trans) / ma_window if ma_window > 0 else 0.0
            freqs.append(freq)
            if len(w_trans) > 0:
                amps.append(float(np.mean([t.peak_to_trough for t in w_trans])))
                hws.append(float(np.mean([t.half_width for t in w_trans])))
            else:
                amps.append(0.0)
                hws.append(0.0)

        all_freq_ma.append(freqs)
        all_amp_ma.append(amps)
        all_hw_ma.append(hws)

    # Group PSTH
    count_mat = np.array(all_counts)
    psth_mean = np.mean(count_mat, axis=0)
    psth_sem = (np.std(count_mat, axis=0, ddof=1) / np.sqrt(len(sessions))
                if len(sessions) > 1 else np.zeros(len(bin_centers)))

    # Group moving averages
    def _mean_sem(data: List[List[float]]) -> tuple[np.ndarray, np.ndarray]:
        mat = np.array(data)
        m = np.mean(mat, axis=0)
        sem = (np.std(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
               if mat.shape[0] > 1 else np.zeros(mat.shape[1]))
        return m, sem

    freq_mean, freq_sem = _mean_sem(all_freq_ma)
    amp_mean, amp_sem = _mean_sem(all_amp_ma)
    hw_mean, hw_sem = _mean_sem(all_hw_ma)

    return IctalTransientGroupResult(
        session_results=session_results,
        psth_bin_centers=bin_centers,
        psth_mean=psth_mean,
        psth_sem=psth_sem,
        moving_avg_times=ma_centers,
        freq_mean=freq_mean,
        freq_sem=freq_sem,
        amp_mean=amp_mean,
        amp_sem=amp_sem,
        hw_mean=hw_mean,
        hw_sem=hw_sem,
    )
