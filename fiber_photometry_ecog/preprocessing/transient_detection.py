"""
Calcium transient detection.

Two methods:
1. Prominence-based: scipy.signal.find_peaks with prominence threshold
2. MAD-based: adaptive threshold using median + k * MAD

Transient properties (amplitude, half-width) are measured on raw dF/F,
NOT z-scored signal, per Wallace et al. 2025.
"""

from typing import Optional, List

import numpy as np
from scipy.signal import find_peaks, peak_widths

from ..core.config import TransientConfig
from ..core.data_models import TransientEvent


def detect_transients(
    dff: np.ndarray,
    fs: float,
    config: TransientConfig | None = None,
    temperature: Optional[np.ndarray] = None,
) -> List[TransientEvent]:
    """Detect calcium transients in a dF/F signal.

    Parameters
    ----------
    dff : 1-D dF/F signal (raw, not z-scored)
    fs : sampling rate (Hz)
    config : detection parameters (uses defaults if None)
    temperature : optional temperature trace (same length as dff) for
        cross-referencing transient events

    Returns
    -------
    List of TransientEvent, sorted by peak_time.
    """
    if config is None:
        config = TransientConfig()

    if config.method == "prominence":
        return _detect_prominence(dff, fs, config, temperature)
    elif config.method == "mad":
        return _detect_mad(dff, fs, config, temperature)
    else:
        raise ValueError(f"Unknown transient detection method: {config.method}")


def _detect_prominence(
    dff: np.ndarray,
    fs: float,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
) -> List[TransientEvent]:
    """Prominence-based detection using scipy.signal.find_peaks."""
    max_width_samples = int(config.max_width_s * fs)

    peaks, properties = find_peaks(
        dff,
        prominence=config.min_prominence,
        width=(None, max_width_samples),
        wlen=max_width_samples * 2,
    )

    return _build_events(dff, fs, peaks, properties, config, temperature)


def _detect_mad(
    dff: np.ndarray,
    fs: float,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
) -> List[TransientEvent]:
    """MAD-based adaptive threshold detection."""
    median_val = np.median(dff)
    mad = np.median(np.abs(dff - median_val))
    threshold = median_val + config.mad_k * mad

    max_width_samples = int(config.max_width_s * fs)

    peaks, properties = find_peaks(
        dff,
        height=threshold,
        width=(None, max_width_samples),
        prominence=0,  # still compute prominence for reporting
        wlen=max_width_samples * 2,
    )

    return _build_events(dff, fs, peaks, properties, config, temperature)


def _build_events(
    dff: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    properties: dict,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
) -> List[TransientEvent]:
    """Convert peak indices + properties into TransientEvent list."""
    events = []
    trough_window = int(config.trough_window_s * fs)

    # Get half-widths at 50% prominence
    if len(peaks) > 0:
        widths, width_heights, left_ips, right_ips = peak_widths(
            dff, peaks, rel_height=0.5
        )
    else:
        widths = np.array([])

    prominences = properties.get("prominences", np.zeros(len(peaks)))

    for i, peak_idx in enumerate(peaks):
        peak_time = peak_idx / fs
        peak_amp = float(dff[peak_idx])

        # Trough: minimum in window around peak
        left = max(0, peak_idx - trough_window)
        right = min(len(dff), peak_idx + trough_window)
        trough_amp = float(np.min(dff[left:right]))

        half_w = float(widths[i] / fs) if i < len(widths) else 0.0
        prom = float(prominences[i]) if i < len(prominences) else 0.0

        temp_at_peak = None
        if temperature is not None and peak_idx < len(temperature):
            temp_at_peak = float(temperature[peak_idx])

        events.append(TransientEvent(
            peak_time=peak_time,
            peak_amplitude=peak_amp,
            trough_amplitude=trough_amp,
            peak_to_trough=peak_amp - trough_amp,
            half_width=half_w,
            prominence=prom,
            temperature_at_peak=temp_at_peak,
        ))

    return events
