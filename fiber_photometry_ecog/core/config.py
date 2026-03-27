"""
Configuration dataclasses for the preprocessing and analysis pipeline.

All scientific parameters are exposed here so that every analysis is
fully reproducible from a single config object.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ECoGConfig:
    """ECoG filtering parameters."""
    bandpass_low: float = 1.0       # Hz
    bandpass_high: float = 70.0     # Hz
    bandpass_order: int = 4         # Butterworth order
    notch_freq: float = 60.0       # Hz (line noise)
    notch_q: float = 30.0          # quality factor


@dataclass
class PhotometryConfig:
    """Photometry preprocessing parameters (shared across strategies)."""
    strategy: str = "A"             # "A" (Chandni), "B" (Meiling), "C" (IRLS)

    # Strategy A: Gaussian smoothing
    gaussian_sigma: int = 75        # samples

    # Strategy B: Meiling's low-pass + biexp + OLS
    lowpass_cutoff_b: float = 10.0  # Hz

    # Strategy C: IRLS / Keevers
    lowpass_cutoff_c: float = 3.0   # Hz
    irls_tuning_c: float = 1.4      # Tukey bisquare tuning constant
    irls_max_iter: int = 50         # max IRLS iterations
    irls_tol: float = 1e-6          # convergence tolerance

    # Biexponential fit (Strategy B only — Strategy C per Keevers 2025 does not detrend)
    biexp_crop_s: float = 120.0     # seconds to crop from start if fit fails

    # Post-processing
    baseline_end_s: Optional[float] = None  # seconds; if None, must be set per session (heating start)
    hpf_cutoff: float = 0.01       # Hz, for transient detection stream
    hpf_order: int = 2             # Butterworth order


@dataclass
class SpikeDetectionConfig:
    """Interictal ECoG spike detection parameters."""
    tmul: float = 3.0                  # threshold multiplier (threshold = tmul * baseline_std)
    abs_threshold: float = 0.4         # minimum absolute threshold (z-score units)
    spkdur_min_ms: float = 70.0        # minimum spike width (ms)
    spkdur_max_ms: float = 200.0       # maximum spike width (ms)
    min_prominence_frac: float = 0.5   # prominence >= threshold * this fraction
    min_distance_ms: float = 70.0      # minimum inter-spike distance (ms)
    dedup_window_ms: float = 10.0      # duplicates within this window are merged
    edge_margin_s: float = 0.1         # exclude spikes within this margin of signal edges


@dataclass
class TransientConfig:
    """Transient detection parameters."""
    method: str = "prominence"       # "prominence" or "mad"
    min_prominence: float = 1.0      # z-score units
    max_width_s: float = 8.0         # seconds
    trough_window_s: float = 2.5     # seconds each side of peak
    mad_k: float = 3.0               # for MAD method: median + k * MAD


# Strategy-specific defaults. A uses whole-signal z-score (STD=1, so 1.0 is
# appropriate). B/C use baseline z-score (tiny denominator inflates values,
# so 2.6 SD per Donka et al. 2025 PASTa toolbox).
TRANSIENT_CONFIGS = {
    "A": TransientConfig(min_prominence=1.0),
    "B": TransientConfig(min_prominence=2.6),
    "C": TransientConfig(min_prominence=2.6),
}


@dataclass
class TemperatureConfig:
    """Temperature conversion and landmark extraction parameters."""
    slope: float = 0.0981            # T(C) = slope * V(mV) + intercept
    intercept: float = 8.81
    smoothing_window: int = 300      # samples for moving average
    baseline_duration_s: float = 60.0  # seconds from start to compute baseline temp


@dataclass
class PreprocessingConfig:
    """Top-level config aggregating all preprocessing parameters."""
    ecog: ECoGConfig = field(default_factory=ECoGConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    spike_detection: SpikeDetectionConfig = field(default_factory=SpikeDetectionConfig)


# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Parameters for all Phase 4 analysis modules."""
    # Temperature binning (used by preictal_mean, preictal_transients, postictal)
    temp_bin_size: float = 1.0              # degrees C
    # Triggered averages
    triggered_window_s: float = 30.0        # seconds each side of event
    # PSTH
    psth_bin_size_s: float = 10.0           # seconds per bin
    psth_window_s: float = 60.0             # seconds each side of UEO
    # Moving averages (sliding window for transient properties)
    moving_avg_window_s: float = 30.0       # seconds
    moving_avg_step_s: float = 5.0          # seconds step between window centers
    # Pre-ictal transients: temperature range below seizure onset
    preictal_temp_range: float = 10.0       # degrees C below seizure onset temp
    # Spike-triggered averages
    spike_triggered_window_s: float = 30.0  # seconds each side of spike
