"""Postictal cooling curves and terminal recording measurements.

Cooling temperatures are relative to each recording's maximum.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import (
    get_off_time,
    get_signal_and_time,
    get_temperature,
    compute_sem,
    bin_signal_by_temperature,
)


@dataclass
class PostictalSessionResult:
    mouse_id: str
    heating_session: int
    cooling_bin_centers: np.ndarray
    cooling_bin_means: np.ndarray       # mean z-ΔF/F per temp bin (cooling only)
    final_time: float                   # seconds since seizure offset (time[-1] - off_time)
    final_temp: float                   # degrees C
    final_mean_dff: float               # mean z-ΔF/F at end of recording


@dataclass
class PostictalGroupResult:
    session_results: List[PostictalSessionResult]
    cooling_bin_centers: np.ndarray
    cooling_group_mean: np.ndarray
    cooling_group_sem: np.ndarray
    # Final recording metrics
    final_times: np.ndarray
    final_temps: np.ndarray
    final_dffs: np.ndarray


def compute_postictal(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> PostictalGroupResult:
    """Compute postictal recovery metrics for a group of sessions."""
    if config is None:
        config = AnalysisConfig()

    sessions = [s for s in sessions if s.include_for_baseline]

    bin_size = config.temp_bin_size
    bin_edges = np.arange(-config.preictal_temp_range, bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2

    session_results = []
    all_cooling = []
    final_times = []
    final_temps = []
    final_dffs = []

    for s in sessions:
        signal, time, fs = get_signal_and_time(s)
        temperature = get_temperature(s)

        if np.all(np.isnan(temperature)):
            cooling_binned = np.full(len(bin_centers), np.nan)
        else:
    # Cooling starts at the maximum recorded temperature.
            max_idx = int(np.nanargmax(temperature))
            max_temp = float(temperature[max_idx])

            # Cooling portion: from max temp to end
            cool_signal = signal[max_idx:]
            cool_temp = temperature[max_idx:]

            # Temperature relative to the recording maximum.
            cool_temp_rel = cool_temp - max_temp
            cooling_binned = bin_signal_by_temperature(cool_signal, cool_temp_rel, bin_edges)
        all_cooling.append(cooling_binned)

        # Time since offset is NaN when neither actual nor equivalent OFF exists.
        off_t = get_off_time(s)
        final_time = float(time[-1]) - off_t if off_t is not None else np.nan
        valid_temp = temperature[~np.isnan(temperature)]
        final_temp_val = float(valid_temp[-1]) if len(valid_temp) > 0 else np.nan
        # Mean z-ΔF/F over last 30 seconds of recording
        n_final = min(int(30 * fs), len(signal))
        final_dff = float(np.mean(signal[-n_final:]))

        final_times.append(final_time)
        final_temps.append(final_temp_val)
        final_dffs.append(final_dff)

        session_results.append(PostictalSessionResult(
            mouse_id=s.mouse_id,
            heating_session=s.heating_session,
            cooling_bin_centers=bin_centers.copy(),
            cooling_bin_means=cooling_binned,
            final_time=final_time,
            final_temp=final_temp_val,
            final_mean_dff=final_dff,
        ))

    # Group cooling curve
    cool_mat = np.array(all_cooling)
    cool_counts = np.sum(~np.isnan(cool_mat), axis=0)
    cool_mean = np.divide(
        np.nansum(cool_mat, axis=0), cool_counts,
        out=np.full(len(bin_centers), np.nan), where=cool_counts > 0,
    )
    cool_sem = np.full(len(bin_centers), np.nan)
    for b in range(len(bin_centers)):
        col = cool_mat[:, b]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            cool_sem[b] = compute_sem(valid)

    return PostictalGroupResult(
        session_results=session_results,
        cooling_bin_centers=bin_centers,
        cooling_group_mean=cool_mean,
        cooling_group_sem=cool_sem,
        final_times=np.array(final_times),
        final_temps=np.array(final_temps),
        final_dffs=np.array(final_dffs),
    )
