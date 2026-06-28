"""
ECoG preprocessing: bandpass filter + 60 Hz notch.

All filters use SOS form with sosfiltfilt for numerical stability
and zero-phase distortion.
"""

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from ..core.config import ECoGConfig


def filter_ecog(
    ecog: np.ndarray,
    fs: float,
    config: ECoGConfig | None = None,
) -> np.ndarray:
    """Apply bandpass + notch filter to ECoG signal.

    Parameters
    ----------
    ecog : raw ECoG signal (1-D array, uV). May contain leading/trailing
        NaN padding for opto sessions where the photometry timebase
        extends past the OEP recording.
    fs : sampling rate (Hz)
    config : filter parameters (uses defaults if None)

    Returns
    -------
    Filtered ECoG signal (same length as input). NaN-padded regions are
    preserved as NaN in the output; only the contiguous valid segment is
    filtered.
    """
    if config is None:
        config = ECoGConfig()

    sos_bp = butter(
        config.bandpass_order,
        [config.bandpass_low, config.bandpass_high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    b_notch, a_notch = iirnotch(config.notch_freq, config.notch_q, fs=fs)
    sos_notch = tf2sos(b_notch, a_notch)

    # Opto sessions NaN-pad ECoG outside the OEP overlap (sync.py extends
    # the timebase to cover the full PPD photometry span). sosfiltfilt is
    # an IIR + reverse pass, so a single NaN propagates through the entire
    # output and we end up with an all-NaN trace. Filter the contiguous
    # valid segment only and write the result back into a NaN-padded
    # full-length array. Assumes any NaN padding is at the edges of the
    # array, not in the middle — true for the opto extend path; would
    # need a multi-segment loop if interior gaps ever appear.
    finite = np.isfinite(ecog)
    if finite.all():
        filtered = sosfiltfilt(sos_bp, ecog)
        filtered = sosfiltfilt(sos_notch, filtered)
        return filtered
    if not finite.any():
        return np.full_like(ecog, np.nan, dtype=np.float64)

    valid_idx = np.flatnonzero(finite)
    lo, hi = int(valid_idx[0]), int(valid_idx[-1]) + 1
    segment = ecog[lo:hi]
    if not np.isfinite(segment).all():
        # Interior NaN — bail rather than emit a half-filtered trace that
        # downstream code can't reason about.
        return np.full_like(ecog, np.nan, dtype=np.float64)

    seg_filt = sosfiltfilt(sos_bp, segment)
    seg_filt = sosfiltfilt(sos_notch, seg_filt)

    out = np.full_like(ecog, np.nan, dtype=np.float64)
    out[lo:hi] = seg_filt
    return out
