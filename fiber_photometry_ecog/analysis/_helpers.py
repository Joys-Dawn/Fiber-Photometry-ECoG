"""
Shared helpers for analysis modules.

Extracts the correct landmark times from Session objects regardless of
whether the session is a seizure trial or a control (using equivalents).
"""

from typing import Optional, List

import numpy as np

from ..core.data_models import Session, TransientEvent


def get_eec_time(s: Session) -> Optional[float]:
    lm = s.landmarks
    if lm is None:
        return None
    if s.n_seizures > 0:
        return lm.eec_time
    return lm.equiv_eec_time


def get_ueo_time(s: Session) -> Optional[float]:
    lm = s.landmarks
    if lm is None:
        return None
    if s.n_seizures > 0:
        return lm.ueo_time
    return lm.equiv_ueo_time


def get_behavioral_onset_time(s: Session) -> Optional[float]:
    lm = s.landmarks
    if lm is None:
        return None
    if s.n_seizures > 0:
        return lm.behavioral_onset_time
    return lm.equiv_behavioral_onset_time


def get_off_time(s: Session) -> Optional[float]:
    lm = s.landmarks
    if lm is None:
        return None
    if s.n_seizures > 0:
        return lm.off_time
    return lm.equiv_off_time


def get_ueo_temp(s: Session) -> Optional[float]:
    """Get seizure onset temperature (or equivalent for controls)."""
    lm = s.landmarks
    if lm is None:
        return None
    if s.n_seizures > 0:
        return lm.ueo_temp
    return lm.equiv_ueo_temp


def get_signal_and_time(s: Session) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (z-scored dF/F, time, fs) from a session's processed data."""
    proc = s.processed
    return proc.photometry.dff_zscore, proc.time, proc.fs


def get_ecog_filtered(s: Session) -> np.ndarray:
    """Return filtered ECoG signal from a session's processed data."""
    return s.processed.ecog_filtered


def get_temperature(s: Session) -> np.ndarray:
    """Return smoothed temperature trace from a session's processed data."""
    return s.processed.temperature_smooth


def time_to_index(t: float, fs: float) -> int:
    return int(round(t * fs))


def filter_transients_by_time(
    transients: List[TransientEvent],
    t_start: float,
    t_end: float,
) -> List[TransientEvent]:
    """Return transients whose peak_time falls within [t_start, t_end]."""
    return [t for t in transients if t_start <= t.peak_time <= t_end]


def compute_sem(values: np.ndarray) -> float:
    """Standard error of the mean."""
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))



def bin_signal_by_temperature(
    signal: np.ndarray,
    temperature: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Compute mean signal within each temperature bin.

    Returns array of length len(bin_edges)-1 with NaN for empty bins.
    """
    n_bins = len(bin_edges) - 1
    result = np.full(n_bins, np.nan)
    indices = np.digitize(temperature, bin_edges) - 1
    for b in range(n_bins):
        mask = indices == b
        if np.any(mask):
            result[b] = np.mean(signal[mask])
    return result
