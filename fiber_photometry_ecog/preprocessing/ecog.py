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
    ecog : raw ECoG signal (1-D array, uV)
    fs : sampling rate (Hz)
    config : filter parameters (uses defaults if None)

    Returns
    -------
    Filtered ECoG signal (same length as input).
    """
    if config is None:
        config = ECoGConfig()

    # --- Butterworth bandpass (SOS form) ---
    sos_bp = butter(
        config.bandpass_order,
        [config.bandpass_low, config.bandpass_high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = sosfiltfilt(sos_bp, ecog)

    # --- 60 Hz notch (SOS form) ---
    b_notch, a_notch = iirnotch(config.notch_freq, config.notch_q, fs=fs)
    sos_notch = tf2sos(b_notch, a_notch)
    filtered = sosfiltfilt(sos_notch, filtered)

    return filtered
