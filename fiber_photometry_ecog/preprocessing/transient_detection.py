"""
Calcium transient detection.

Two methods:
1. prominence: scipy find_peaks with prominence (Chandni's original, Strategy A)
2. wallace: two-step — height gate then prominence filter (Wallace 2025 ProM)

Detection is performed on the z-scored HPF signal. Transient properties
(amplitude, half-width) are measured on raw dF/F at the detected peak
locations, per Wallace et al. 2025 and Chandni's code (detect_transients.m).
"""

from typing import Optional, List

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

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
    elif config.method == "wallace":
        return _detect_wallace(zdff_hpf, dff_raw, fs, config, temperature)
    else:
        raise ValueError(f"Unknown transient detection method: {config.method}")


def _detect_prominence(
    zdff_hpf: np.ndarray,
    dff_raw: np.ndarray,
    fs: float,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
) -> List[TransientEvent]:
    """Chandni's prominence-based detection (detect_transients.m).

    1. findpeaks with MinPeakProminence (no wlen — full-signal prominence)
    2. Post-hoc remove peaks wider than max_width_s
    """
    kwargs = {}

    if config.min_prominence is not None:
        kwargs["prominence"] = config.min_prominence
    else:
        kwargs["prominence"] = 0

    if config.min_height is not None:
        kwargs["height"] = config.min_height

    peaks, properties = find_peaks(zdff_hpf, **kwargs)

    if len(peaks) == 0:
        return _build_events(zdff_hpf, dff_raw, fs, peaks, properties, config, temperature)

    # Post-hoc width filter (per Chandni: remove peaks longer than maxWidth)
    max_width_samples = int(config.max_width_s * fs)
    widths, _, _, _ = peak_widths(zdff_hpf, peaks, rel_height=0.5)
    mask = widths <= max_width_samples
    peaks = peaks[mask]
    properties = {k: v[mask] for k, v in properties.items()
                  if isinstance(v, np.ndarray) and len(v) == len(mask)}

    return _build_events(zdff_hpf, dff_raw, fs, peaks, properties, config, temperature)


def _detect_wallace(
    zdff_hpf: np.ndarray,
    dff_raw: np.ndarray,
    fs: float,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
) -> List[TransientEvent]:
    """Wallace 2025 Prominence Method (ProM): two-step sequential detection.

    Step 1: find_peaks on z-scored signal with height >= min_height (z=1.0)
    Step 2: compute prominence on raw %dF/F at surviving peak timestamps,
            keep only peaks with prominence >= min_prominence (2.0)

    Per Wallace 2025:
    - "a z-score threshold was applied to systematically detect signals"
    - "The detected peaks were timestamped, after which they were assessed
       for prominence within the non-normalized %dF/F trace."
    - "We used a prominence value of 2 for inclusion criteria"
    - "The non-normalized, preprocessed %dF/F signals at the same timestamps
       were then quantified for amplitude (the maximal %dF/F value), width
       (the time it took for the signal to exceed then fall back below the
       cutoff)"
    """
    height = config.min_height if config.min_height is not None else 0
    prom_threshold = config.min_prominence if config.min_prominence is not None else 0

    # Step 1: height gate on z-scored signal (no MinPeakDistance — paper doesn't specify)
    peaks, _ = find_peaks(
        zdff_hpf,
        height=height,
    )

    if len(peaks) == 0:
        return []

    # Step 2: compute prominence on raw %dF/F, filter by threshold
    prominences, _, _ = peak_prominences(dff_raw, peaks)
    mask = prominences >= prom_threshold
    peaks = peaks[mask]
    prominences = prominences[mask]

    properties = {"prominences": prominences}
    return _build_events(zdff_hpf, dff_raw, fs, peaks, properties, config, temperature,
                         width_signal=dff_raw)


def _build_events(
    zdff_hpf: np.ndarray,
    dff_raw: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    properties: dict,
    config: TransientConfig,
    temperature: Optional[np.ndarray],
    width_signal: Optional[np.ndarray] = None,
) -> List[TransientEvent]:
    """Convert peak indices + properties into TransientEvent list.

    Amplitude and peak-to-trough are measured on dff_raw.
    Half-width is measured on width_signal (defaults to zdff_hpf for
    Strategy A per Chandni; callers pass dff_raw for Wallace).
    """
    if width_signal is None:
        width_signal = zdff_hpf

    events = []
    trough_window = int(config.trough_window_s * fs)

    # Get half-widths at 50% prominence
    if len(peaks) > 0:
        widths, _, _, _ = peak_widths(width_signal, peaks, rel_height=0.5)
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
