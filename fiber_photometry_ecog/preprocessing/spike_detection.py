"""
Interictal ECoG spike detection.

Z-scored threshold + find_peaks, detecting both positive and negative spikes.
Adapted from Alex's detector (reference_code/fcd-hyperthermia/Spike_Detection).
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from ..core.config import SpikeDetectionConfig
from ..core.data_models import SpikeEvent


def detect_spikes(
    ecog_filtered: np.ndarray,
    fs: float,
    baseline_end_s: float,
    config: SpikeDetectionConfig | None = None,
    exclusion_zones: Optional[List[Tuple[float, float]]] = None,
) -> List[SpikeEvent]:
    """Detect interictal spikes in filtered ECoG signal.

    Parameters
    ----------
    ecog_filtered : bandpass + notch filtered ECoG (1-D)
    fs : sampling rate (Hz)
    baseline_end_s : end of baseline period (seconds), used to compute
        the z-score normalization baseline
    config : detection parameters (uses defaults if None)
    exclusion_zones : list of (start_s, end_s) tuples defining seizure
        periods; spikes within these windows are excluded

    Returns
    -------
    List of SpikeEvent, sorted by time.
    """
    if config is None:
        config = SpikeDetectionConfig()

    # Z-score relative to baseline
    baseline_n = int(baseline_end_s * fs)
    baseline = ecog_filtered[:baseline_n]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    zscored = (ecog_filtered - baseline_mean) / baseline_std

    # Threshold
    threshold = max(config.tmul * 1.0, config.abs_threshold)
    # After z-scoring with baseline, baseline_std ≈ 1.0,
    # so tmul * baseline_std ≈ tmul. Use the raw tmul value.

    # find_peaks parameters
    min_width = config.spkdur_min_ms * fs / 1000
    max_width = config.spkdur_max_ms * fs / 1000
    min_distance = int(config.min_distance_ms * fs / 1000)
    min_prominence = threshold * config.min_prominence_frac

    all_spikes: List[SpikeEvent] = []

    for polarity in ("positive", "negative"):
        signal = -zscored if polarity == "negative" else zscored

        peaks, props = find_peaks(
            signal,
            height=threshold,
            distance=min_distance,
            width=(min_width, max_width),
            prominence=min_prominence,
        )

        for i, peak_idx in enumerate(peaks):
            all_spikes.append(SpikeEvent(
                time=peak_idx / fs,
                amplitude=float(props["peak_heights"][i]),
                width_ms=float(props["widths"][i]) * 1000 / fs,
                prominence=float(props["prominences"][i]),
                polarity=polarity,
            ))

    # Sort by time
    all_spikes.sort(key=lambda s: s.time)

    # Remove duplicates within dedup window (keep higher prominence)
    all_spikes = _remove_duplicates(all_spikes, config.dedup_window_ms, fs)

    # Exclude edge spikes
    edge_margin = config.edge_margin_s
    duration = len(ecog_filtered) / fs
    all_spikes = [
        s for s in all_spikes
        if edge_margin <= s.time <= duration - edge_margin
    ]

    # Exclude spikes within seizure periods
    if exclusion_zones:
        all_spikes = [
            s for s in all_spikes
            if not any(start <= s.time <= end for start, end in exclusion_zones)
        ]

    return all_spikes


def _remove_duplicates(
    spikes: List[SpikeEvent],
    dedup_window_ms: float,
    fs: float,
) -> List[SpikeEvent]:
    """Remove duplicate spikes within dedup_window_ms, keeping higher prominence."""
    if len(spikes) <= 1:
        return spikes

    dedup_window_s = dedup_window_ms / 1000
    filtered: List[SpikeEvent] = []

    for spike in spikes:
        if filtered:
            last = filtered[-1]
            if spike.time - last.time < dedup_window_s:
                if spike.prominence > last.prominence:
                    filtered[-1] = spike
                continue
        filtered.append(spike)

    return filtered
