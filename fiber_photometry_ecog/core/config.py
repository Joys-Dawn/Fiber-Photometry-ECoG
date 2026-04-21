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
    # Strategy A detrending: HPF (per Chandni's detect_transients.m)
    hpf_cutoff: float = 0.01       # Hz, Butterworth HPF for Strategy A
    hpf_order: int = 2             # Butterworth order
    # Strategy B/C detrending: moving-average subtraction (per Wallace 2025)
    detrend_window_s: float = 100.0  # seconds; moving-average window


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
    """Transient detection parameters.

    method="prominence": scipy find_peaks with prominence (Chandni's original).
    method="wallace": two-step — height gate then prominence filter
        (Wallace 2025 ProM: findpeaks height>=1.0, then prominence>=2).
    """
    method: str = "prominence"       # "prominence" or "wallace"
    min_prominence: Optional[float] = 1.0   # prominence in z-score units (None to disable)
    min_height: Optional[float] = None      # height in z-score units (None to disable)
    max_width_s: float = 8.0         # seconds (prominence method only, per Chandni)
    trough_window_s: float = 2.5     # seconds each side of peak


# Strategy-specific defaults.
# A: Chandni's original — prominence=1.0 on z-scored HPF signal (per detect_transients.m)
# B/C: Wallace 2025 ProM — height>=1.0 on z-scored, then prominence on raw dF/F.
#   Wallace used prominence=2.0 but on %dF/F in percentage scale (dLight sensor).
#   Our dF/F is fractional (0.08 = 8%), so 0.02 = 2 percentage points equivalent.
TRANSIENT_CONFIGS = {
    "A": TransientConfig(min_prominence=1.0),
    "B": TransientConfig(method="wallace", min_height=1.0, min_prominence=0.02),
    "C": TransientConfig(method="wallace", min_height=1.0, min_prominence=0.02),
}


@dataclass
class TemperatureConfig:
    """Temperature conversion and landmark extraction parameters.

    `slope` / `intercept` give the fallback calibration used by Chandni's
    rig (thermistor on NI-DAQ USB-6009 AI0): T(C) = slope * V(mV) + intercept.
    Meiling's rigs do not have a NI-DAQ; they use date-range-specific
    calibrations from MEILING_RIG_CALIBRATIONS (see below) applied to the
    thermistor on Intan ADC1 of the Acquisition Board stream.
    """
    slope: float = 0.0981            # T(C) = slope * V(mV) + intercept
    intercept: float = 8.81
    smoothing_window: int = 300      # samples for moving average
    baseline_duration_s: float = 60.0  # seconds from start to compute baseline temp


# Date-range calibrations supplied by Meiling for rigs without a NI-DAQ.
# Originals were given as T = a * V_volts + b (temperature_probe = voltage in V).
# Stored here as (start_date, end_date, slope_per_mV, intercept_C). Dates are
# inclusive, ISO format (YYYY-MM-DD). None means open-ended.
MEILING_RIG_CALIBRATIONS = [
    (None,         "2024-03-13",  0.08817, 375.77),   # Before 2024-03-14
    ("2024-03-14", "2024-05-23", -0.10342,   9.69),   # 2024-03-14 to 2024-05-23
    ("2024-05-24", "2025-02-10",  0.09013,   5.79),   # 2024-05-24 to 2025-02-10
    ("2025-02-11", "2025-07-22",  0.09855,   2.86),   # 2025-02-11 to 2025-07-22
    ("2025-07-23", None,          0.09597,   4.36),   # 2025-07-23 onwards
]


def lookup_meiling_calibration(recording_date: str) -> tuple[float, float]:
    """Return (slope_per_mV, intercept_C) for a Meiling rig recording date.

    `recording_date` is a YYYY-MM-DD string. Raises ValueError if no range
    matches (shouldn't happen for in-bounds dates).
    """
    for start, end, slope, intercept in MEILING_RIG_CALIBRATIONS:
        if (start is None or recording_date >= start) and \
           (end is None or recording_date <= end):
            return slope, intercept
    raise ValueError(
        f"No Meiling rig calibration covers recording date {recording_date!r}. "
        f"Check MEILING_RIG_CALIBRATIONS in core/config.py."
    )


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
    # Triggered averages (ictal landmark: EEC, UEO, OFF, etc.)
    triggered_window_s: float = 30.0        # seconds each side of event
    triggered_baseline_start_s: float = 5.0 # baseline begins this many seconds before event
    triggered_baseline_end_s: float = 1.0   # baseline ends this many seconds before event
    triggered_auc_end_s: float = 30.0       # AUC: 0 to this many seconds post-event
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
    spike_triggered_baseline_start_s: float = 5.0  # baseline begins this many seconds before event
    spike_triggered_baseline_end_s: float = 1.0    # baseline ends this many seconds before event
    spike_triggered_auc_end_s: float = 5.0         # AUC integration window: 0 to this many seconds post-event
