"""
Calcium transient detection.

Two methods:
1. Prominence-based: scipy.signal.find_peaks with prominence threshold
2. MAD-based: adaptive threshold using median + k * MAD

Detection is performed on the z-scored HPF signal (prominence threshold
is in z-score units). Transient properties (amplitude, half-width) are
measured on raw dF/F at the detected peak locations, per Wallace et al.
2025 and Chandni's code (detect_transients.m).
"""

from typing import Optional, List

import numpy as np
from scipy.signal import find_peaks, peak_widths

from ..core.config import TransientConfig
from ..core.data_models import TransientEvent


def detect_transients(
    zdff_hpf: np.ndarray,
    dff_raw: np.ndarray,
    fs: float,
    config: TransientConfig | None = None,
    temperature: Optional[np.ndarray] = None,
) -> List[TransientEvent]:
    """Detect calcium transients and measure their properties.

    Peak detection runs on the z-scored HPF signal (zdff_hpf) where the
    prominence threshold (default 1.0) is in z-score units. Properties
    (amplitude, peak-to-trough, half-width) are measured on raw dF/F at
    the same peak locations, per Wallace 2025.

    Parameters
    ----------
    zdff_hpf : z-scored high-pass filtered dF/F (for detection)
    dff_raw : raw dF/F signal (for property measurement)
    fs : sampling rate (Hz)
    config : detection parameters (uses defaults if None)
    temperature : optional temperature trace (same length) for
        cross-referencing transient events

    Returns
    -------
    List of TransientEvent, sorted by peak_time.
    """
    if config is None:
        config = TransientConfig()

    if config.method == "prominence":
        return _detect_prominence(zdff_hpf, dff_raw, fs, config, temperature)
    elif config.method == "mad":
        return _detect_mad(zdff_hpf, dff_raw, fs, config, temperature)
    else:
        raise ValueError(f"Unknown transient detection method: {config.method}")


def _detect_prominence(
    zdff_hpf: np.ndarray,
    dff_raw: np.ndarray,
    fs: float,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
) -> List[TransientEvent]:
    """Prominence-based detection using scipy.signal.find_peaks."""
    max_width_samples = int(config.max_width_s * fs)

    peaks, properties = find_peaks(
        zdff_hpf,
        prominence=config.min_prominence,
        width=(None, max_width_samples),
        wlen=max_width_samples * 2,
    )

    return _build_events(zdff_hpf, dff_raw, fs, peaks, properties, config, temperature)


def _detect_mad(
    zdff_hpf: np.ndarray,
    dff_raw: np.ndarray,
    fs: float,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
) -> List[TransientEvent]:
    """MAD-based adaptive threshold detection."""
    median_val = np.median(zdff_hpf)
    mad = np.median(np.abs(zdff_hpf - median_val))
    threshold = median_val + config.mad_k * mad

    max_width_samples = int(config.max_width_s * fs)

    peaks, properties = find_peaks(
        zdff_hpf,
        height=threshold,
        width=(None, max_width_samples),
        prominence=0,  # still compute prominence for reporting
        wlen=max_width_samples * 2,
    )

    return _build_events(zdff_hpf, dff_raw, fs, peaks, properties, config, temperature)


def _build_events(
    zdff_hpf: np.ndarray,
    dff_raw: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    properties: dict,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
) -> List[TransientEvent]:
    """Convert peak indices + properties into TransientEvent list.

    Peak locations come from zdff_hpf. Amplitude and peak-to-trough are
    measured on dff_raw. Half-width is measured on zdff_hpf (the detection
    signal) since peaks may not be local maxima in the raw signal.
    """
    events = []
    trough_window = int(config.trough_window_s * fs)

    # Get half-widths at 50% prominence on the detection signal (zdff_hpf)
    if len(peaks) > 0:
        widths, _, _, _ = peak_widths(zdff_hpf, peaks, rel_height=0.5)
    else:
        widths = np.array([])

    prominences = properties.get("prominences", np.zeros(len(peaks)))

    for i, peak_idx in enumerate(peaks):
        peak_time = peak_idx / fs

        # Measure amplitude on raw dF/F (per Wallace 2025)
        peak_amp = float(dff_raw[peak_idx])

        # Trough: minimum in window around peak on raw dF/F
        left = max(0, peak_idx - trough_window)
        right = min(len(dff_raw), peak_idx + trough_window)
        trough_amp = float(np.min(dff_raw[left:right]))

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
