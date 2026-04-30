"""
4.3 Pre-Ictal Mean Signal.

Time-binned: mean z-ΔF/F within baseline, early heat, late heat.
End of late heat: mean z-ΔF/F in window immediately before UEO.
Temperature-binned: mean z-ΔF/F in configurable bins during heating,
    relative to seizure onset temperature (or equivalent for controls).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import (
    get_ueo_time,
    get_ueo_temp,
    get_off_time,
    get_signal_and_time,
    get_temperature,
    time_to_index,
    compute_sem,
    bin_signal_by_temperature,
)


@dataclass
class PreictalMeanSessionResult:
    mouse_id: str
    heating_session: int
    baseline_mean: float
    early_heat_mean: float
    late_heat_mean: float
    end_late_heat_mean: float
    heating_mean: float                 # consolidated: entire heating period
    ictal_mean: Optional[float]         # mean within ictal/equivalent period; None if no UEO/OFF
    temp_bin_centers: np.ndarray        # degrees C relative to seizure onset temp
    temp_bin_means: np.ndarray          # mean z-ΔF/F per bin


@dataclass
class PreictalMeanGroupResult:
    session_results: List[PreictalMeanSessionResult]
    baseline_mean: float
    baseline_sem: float
    early_heat_mean: float
    early_heat_sem: float
    late_heat_mean: float
    late_heat_sem: float
    end_late_heat_mean: float
    end_late_heat_sem: float
    heating_mean: float
    heating_sem: float
    ictal_mean: float
    ictal_sem: float
    temp_bin_centers: np.ndarray
    temp_bin_group_mean: np.ndarray     # mean across sessions per bin
    temp_bin_group_sem: np.ndarray      # SEM across sessions per bin


def compute_preictal_mean(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> PreictalMeanGroupResult:
    """Compute pre-ictal mean signal metrics for a group of sessions."""
    if config is None:
        config = AnalysisConfig()

    sessions = [s for s in sessions if s.include_for_baseline]

    session_results = []
    bl_means = []
    eh_means = []
    lh_means = []
    elh_means = []
    heat_means = []
    ictal_means = []

    # Determine temperature bin edges: from -preictal_temp_range to 0 relative to UEO temp
    bin_size = config.temp_bin_size
    temp_range = config.preictal_temp_range
    bin_edges = np.arange(-temp_range, bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    all_temp_bins = []

    for s in sessions:
        signal, time, fs = get_signal_and_time(s)
        temperature = get_temperature(s)
        lm = s.landmarks

        heat_start = lm.heating_start_time
        ueo_t = get_ueo_time(s)
        ueo_temp = get_ueo_temp(s)

        # Time indices
        i_heat = time_to_index(heat_start, fs)
        i_ueo = time_to_index(ueo_t, fs)
        i_mid = (i_heat + i_ueo) // 2

        # Time-binned means
        bl_mean = float(np.mean(signal[:i_heat]))
        eh_mean = float(np.mean(signal[i_heat:i_mid]))
        lh_mean = float(np.mean(signal[i_mid:i_ueo]))
        heating_mean = float(np.mean(signal[i_heat:i_ueo]))
        # End of late heat: fixed 10s window before UEO
        end_late_n = min(int(10.0 * fs), i_ueo - i_heat)
        elh_mean = float(np.mean(signal[i_ueo - end_late_n:i_ueo]))

        # Ictal mean: window from UEO to OFF (uses equivalents for controls)
        off_t = get_off_time(s)
        ictal_mean: Optional[float] = None
        if off_t is not None and ueo_t is not None and off_t > ueo_t:
            i_off = time_to_index(off_t, fs)
            if i_off > i_ueo and i_off <= len(signal):
                ictal_mean = float(np.mean(signal[i_ueo:i_off]))

        bl_means.append(bl_mean)
        eh_means.append(eh_mean)
        lh_means.append(lh_mean)
        elh_means.append(elh_mean)
        heat_means.append(heating_mean)
        if ictal_mean is not None:
            ictal_means.append(ictal_mean)

        # Temperature-binned: heating portion, relative to seizure onset temp
        heat_signal = signal[i_heat:i_ueo]
        heat_temp = temperature[i_heat:i_ueo]
        temp_relative = heat_temp - ueo_temp  # relative to seizure onset
        temp_bin_vals = bin_signal_by_temperature(heat_signal, temp_relative, bin_edges)
        all_temp_bins.append(temp_bin_vals)

        session_results.append(PreictalMeanSessionResult(
            mouse_id=s.mouse_id,
            heating_session=s.heating_session,
            baseline_mean=bl_mean,
            early_heat_mean=eh_mean,
            late_heat_mean=lh_mean,
            end_late_heat_mean=elh_mean,
            heating_mean=heating_mean,
            ictal_mean=ictal_mean,
            temp_bin_centers=bin_centers.copy(),
            temp_bin_means=temp_bin_vals,
        ))

    # Group summaries
    bl_arr = np.array(bl_means)
    eh_arr = np.array(eh_means)
    lh_arr = np.array(lh_means)
    elh_arr = np.array(elh_means)
    heat_arr = np.array(heat_means)
    ictal_arr = np.array(ictal_means) if ictal_means else np.array([])

    temp_mat = np.array(all_temp_bins)  # (n_sessions, n_bins)
    temp_group_mean = np.nanmean(temp_mat, axis=0)
    temp_group_sem = np.full(len(bin_centers), np.nan)
    for b in range(len(bin_centers)):
        col = temp_mat[:, b]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            temp_group_sem[b] = compute_sem(valid)

    return PreictalMeanGroupResult(
        session_results=session_results,
        baseline_mean=float(np.mean(bl_arr)),
        baseline_sem=compute_sem(bl_arr),
        early_heat_mean=float(np.mean(eh_arr)),
        early_heat_sem=compute_sem(eh_arr),
        late_heat_mean=float(np.mean(lh_arr)),
        late_heat_sem=compute_sem(lh_arr),
        end_late_heat_mean=float(np.mean(elh_arr)),
        end_late_heat_sem=compute_sem(elh_arr),
        heating_mean=float(np.mean(heat_arr)),
        heating_sem=compute_sem(heat_arr),
        ictal_mean=float(np.mean(ictal_arr)) if len(ictal_arr) > 0 else np.nan,
        ictal_sem=compute_sem(ictal_arr) if len(ictal_arr) > 1 else np.nan,
        temp_bin_centers=bin_centers,
        temp_bin_group_mean=temp_group_mean,
        temp_bin_group_sem=temp_group_sem,
    )
