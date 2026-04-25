"""
Control equivalent landmark assignment.

Computes mean seizure landmarks across all seizure sessions, then assigns
equivalent landmarks to each control session via either temperature matching
or time matching.

Reference: SNr analysis MATLAB code (temp_eq_times.m).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..core.data_models import Session


def _temp_at_time(temp: np.ndarray, fs: float, t: float) -> float:
    """Look up temperature value at a given time."""
    idx = min(int(round(t * fs)), len(temp) - 1)
    return float(temp[idx])


@dataclass
class SeizureGroupMeans:
    """Mean landmark values computed across all seizure sessions."""
    mean_eec_time: float          # seconds from heating start
    mean_ueo_time: float
    mean_behavioral_onset_time: Optional[float]
    mean_off_time: float
    mean_eec_temp: float          # degrees C
    mean_ueo_temp: float
    mean_behavioral_onset_temp: Optional[float]
    mean_seizure_duration: float  # seconds (UEO → OFF)


def compute_seizure_group_means(
    seizure_sessions: List[Session],
) -> SeizureGroupMeans:
    """Compute mean landmark times and temperatures across seizure sessions.

    Times are relative to heating start. Temperatures are looked up from
    each session's smoothed temperature trace at the landmark times.

    Parameters
    ----------
    seizure_sessions : sessions with n_seizures >= 1 and landmarks populated

    Returns
    -------
    SeizureGroupMeans with averaged values.

    Raises
    ------
    ValueError if no valid seizure sessions or required landmarks are missing.
    """
    eec_times = []
    ueo_times = []
    beh_times = []
    off_times = []
    eec_temps = []
    ueo_temps = []
    beh_temps = []
    seizure_durations = []

    for sess in seizure_sessions:
        lm = sess.landmarks
        if lm is None:
            continue
        if lm.eec_time is None or lm.ueo_time is None or lm.off_time is None:
            continue

        heat_start = lm.heating_start_time

        eec_times.append(lm.eec_time - heat_start)
        ueo_times.append(lm.ueo_time - heat_start)
        off_times.append(lm.off_time - heat_start)
        seizure_durations.append(lm.off_time - lm.ueo_time)

        if lm.behavioral_onset_time is not None:
            beh_times.append(lm.behavioral_onset_time - heat_start)

        # Use pre-computed landmark temperatures (already NaN-filtered by
        # pipeline.py — None means thermistor dropout at that time).
        if lm.eec_temp is not None:
            eec_temps.append(lm.eec_temp)
        if lm.ueo_temp is not None:
            ueo_temps.append(lm.ueo_temp)
        if lm.behavioral_onset_time is not None and lm.behavioral_onset_temp is not None:
            beh_temps.append(lm.behavioral_onset_temp)

    if not eec_times:
        raise ValueError("No valid seizure sessions with EEC/UEO/OFF landmarks found.")
    if not eec_temps:
        raise ValueError(
            "No temperature data available for seizure sessions. "
            "Temperature processing must run before pairing."
        )

    return SeizureGroupMeans(
        mean_eec_time=float(np.mean(eec_times)),
        mean_ueo_time=float(np.mean(ueo_times)),
        mean_behavioral_onset_time=float(np.mean(beh_times)) if beh_times else None,
        mean_off_time=float(np.mean(off_times)),
        mean_eec_temp=float(np.mean(eec_temps)),
        mean_ueo_temp=float(np.mean(ueo_temps)),
        mean_behavioral_onset_temp=float(np.mean(beh_temps)) if beh_temps else None,
        mean_seizure_duration=float(np.mean(seizure_durations)),
    )


def _find_first_time_at_temp(
    temperature_trace: np.ndarray,
    fs: float,
    target_temp: float,
    max_idx: Optional[int] = None,
) -> Optional[float]:
    """Find the first time (seconds) when temperature reaches target_temp.

    Searches only the heating phase (up to max_idx, or the index of max
    temperature if max_idx is not given).

    Parameters
    ----------
    temperature_trace : smoothed temperature in Celsius
    fs : sampling rate (Hz)
    target_temp : temperature to match (degrees C)
    max_idx : search up to and including this index. If None, uses argmax.

    Returns
    -------
    Time in seconds, or None if the control never reaches the target temp.
    """
    if len(temperature_trace) == 0:
        return None

    if max_idx is None:
        max_idx = int(np.nanargmax(temperature_trace))

    # Only search the heating phase
    heating = temperature_trace[:max_idx + 1]
    if len(heating) == 0:
        return None

    # Take the first sample closest to target_temp, unconditionally.
    # Matches temp_eq_times.m: find(min(abs(temp - target)),1,'first').
    diff = np.abs(heating - target_temp)
    best_idx = int(np.nanargmin(diff))
    return best_idx / fs


def assign_equivalents_temperature(
    control_session: Session,
    group_means: SeizureGroupMeans,
) -> None:
    """Assign equivalent landmarks to a control session via temperature matching.

    Finds the first time during the heating phase when the control's
    temperature matches each mean seizure landmark temperature.
    OFF equivalent = UEO equivalent time + mean seizure duration.

    Modifies control_session.landmarks in place.

    Parameters
    ----------
    control_session : a control session with processed temperature data
    group_means : mean landmark values from the seizure group

    Raises
    ------
    ValueError if the session lacks required data.
    """
    lm = control_session.landmarks
    if lm is None:
        raise ValueError("Control session has no landmarks (heating_start_time needed).")
    if control_session.processed is None or control_session.processed.temperature_smooth is None:
        raise ValueError("Control session has no processed temperature data.")
    if control_session.processed.fs is None:
        raise ValueError("Control session has no sampling rate.")

    temp_trace = control_session.processed.temperature_smooth
    fs = control_session.processed.fs

    # Truncate at max temp (heating phase only)
    max_idx = int(np.nanargmax(temp_trace))

    eec_time = _find_first_time_at_temp(temp_trace, fs, group_means.mean_eec_temp, max_idx)
    ueo_time = _find_first_time_at_temp(temp_trace, fs, group_means.mean_ueo_temp, max_idx)

    beh_time = None
    if group_means.mean_behavioral_onset_temp is not None:
        beh_time = _find_first_time_at_temp(
            temp_trace, fs, group_means.mean_behavioral_onset_temp, max_idx
        )

    # OFF equivalent = UEO equivalent + mean seizure duration
    off_time = None
    if ueo_time is not None:
        off_time = ueo_time + group_means.mean_seizure_duration

    lm.equiv_eec_time = eec_time
    lm.equiv_ueo_time = ueo_time
    lm.equiv_behavioral_onset_time = beh_time
    lm.equiv_off_time = off_time

    # Fill in temperatures at equivalent times
    temp = control_session.processed.temperature_smooth
    fs = control_session.processed.fs
    if temp is not None and fs is not None:
        if eec_time is not None:
            lm.equiv_eec_temp = _temp_at_time(temp, fs, eec_time)
        if ueo_time is not None:
            lm.equiv_ueo_temp = _temp_at_time(temp, fs, ueo_time)
        if beh_time is not None:
            lm.equiv_behavioral_onset_temp = _temp_at_time(temp, fs, beh_time)


def assign_equivalents_time(
    control_session: Session,
    group_means: SeizureGroupMeans,
) -> None:
    """Assign equivalent landmarks to a control session via time matching.

    Each control gets the seizure group's mean elapsed times (from heating
    start) added to its own heating start time.

    Modifies control_session.landmarks in place.

    Parameters
    ----------
    control_session : a control session with landmarks.heating_start_time set
    group_means : mean landmark values from the seizure group

    Raises
    ------
    ValueError if the session lacks landmarks.
    """
    lm = control_session.landmarks
    if lm is None:
        raise ValueError("Control session has no landmarks (heating_start_time needed).")

    heat_start = lm.heating_start_time

    lm.equiv_eec_time = heat_start + group_means.mean_eec_time
    lm.equiv_ueo_time = heat_start + group_means.mean_ueo_time
    lm.equiv_off_time = heat_start + group_means.mean_off_time

    if group_means.mean_behavioral_onset_time is not None:
        lm.equiv_behavioral_onset_time = heat_start + group_means.mean_behavioral_onset_time
    else:
        lm.equiv_behavioral_onset_time = None

    # Fill in temperatures at equivalent times
    temp = control_session.processed.temperature_smooth
    fs = control_session.processed.fs
    if temp is not None and fs is not None:
        lm.equiv_eec_temp = _temp_at_time(temp, fs, lm.equiv_eec_time)
        lm.equiv_ueo_temp = _temp_at_time(temp, fs, lm.equiv_ueo_time)
        if lm.equiv_behavioral_onset_time is not None:
            lm.equiv_behavioral_onset_temp = _temp_at_time(temp, fs, lm.equiv_behavioral_onset_time)


def assign_all_controls(
    seizure_sessions: List[Session],
    control_sessions: List[Session],
    mode: str = "temperature",
) -> SeizureGroupMeans:
    """Assign equivalent landmarks to all control sessions.

    Parameters
    ----------
    seizure_sessions : sessions with seizures and populated landmarks
    control_sessions : sessions without seizures
    mode : "temperature" or "time"

    Returns
    -------
    The computed SeizureGroupMeans (for inspection/reporting).

    Raises
    ------
    ValueError for invalid mode or missing data.
    """
    if mode not in ("temperature", "time"):
        raise ValueError(f"mode must be 'temperature' or 'time', got '{mode}'")

    group_means = compute_seizure_group_means(seizure_sessions)

    assign_fn = assign_equivalents_temperature if mode == "temperature" else assign_equivalents_time

    for control in control_sessions:
        assign_fn(control, group_means)

    return group_means
