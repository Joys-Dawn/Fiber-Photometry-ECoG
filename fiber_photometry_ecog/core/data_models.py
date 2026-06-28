"""
Core data models for session-level data.

These dataclasses hold per-session metadata, landmarks, raw/processed
signals, and detected events. They are the central data containers
passed between preprocessing, analysis, and visualization modules.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

from .config import PreprocessingConfig


@dataclass
class SessionLandmarks:
    """Temporal and temperature landmarks for a recording session."""
    # Time at which the experimental phase begins (heating onset for hyperthermia,
    # auto-set to stim onset for opto). Acts as the baseline-end marker for
    # z-scoring, spike-detection statistics, and pre/post-ictal windowing.
    # Required for hyperthermia (read from the data log); for opto it is
    # auto-populated by stim detection during preprocess_session.
    heating_start_time: Optional[float] = None  # seconds from recording start
    # End of the window used to z-score the photometry signal and to compute
    # spike-detection noise stats. Decoupled from heating_start_time so opto
    # sessions (which auto-set heating_start_time = stim_onset for analysis
    # window math) can use a fixed pre-stim window for normalization
    # consistency across sessions. If None, falls back to heating_start_time.
    zscore_baseline_end_time: Optional[float] = None  # seconds from recording start
    eec_time: Optional[float] = None        # electrographic event criterion (None for controls)
    ueo_time: Optional[float] = None        # unequivocal electrographic onset
    behavioral_onset_time: Optional[float] = None
    off_time: Optional[float] = None        # seizure offset
    baseline_temp: Optional[float] = None   # calculated from temperature trace
    baseline_window_s: Optional[Tuple[float, float]] = None  # (start_s, end_s) for baseline metrics; None = use heating_start_time
    max_temp: Optional[float] = None        # calculated
    max_temp_time: Optional[float] = None   # time of peak temperature (seconds)
    terminal_temp: Optional[float] = None   # temperature at end of recording
    terminal_time: Optional[float] = None   # time at end of recording (seconds)
    # Temperature at seizure landmarks (calculated from temp trace)
    eec_temp: Optional[float] = None
    ueo_temp: Optional[float] = None
    behavioral_onset_temp: Optional[float] = None
    # Equivalent landmarks for controls (filled by pairing engine)
    equiv_eec_time: Optional[float] = None
    equiv_ueo_time: Optional[float] = None
    equiv_behavioral_onset_time: Optional[float] = None
    equiv_off_time: Optional[float] = None
    equiv_eec_temp: Optional[float] = None
    equiv_ueo_temp: Optional[float] = None
    equiv_behavioral_onset_temp: Optional[float] = None


@dataclass
class TransientEvent:
    """A single detected calcium transient."""
    peak_time: float                        # seconds
    peak_amplitude: float                   # raw dF/F (NOT z-scored)
    trough_amplitude: float                 # minimum in window around peak
    peak_to_trough: float                   # peak - trough
    half_width: float                       # seconds, at 50% prominence
    prominence: float                       # scipy prominence value
    temperature_at_peak: Optional[float] = None  # degrees C, cross-referenced
    z_peak_amplitude: Optional[float] = None     # z-scored HPF signal at peak
    z_trough_amplitude: Optional[float] = None   # z-scored HPF min in trough window
    z_peak_to_trough: Optional[float] = None     # z_peak - z_trough


@dataclass
class SpikeEvent:
    """A single detected interictal ECoG spike."""
    time: float                             # seconds
    amplitude: float                        # z-scored amplitude
    width_ms: float                         # spike width in milliseconds
    prominence: float                       # scipy prominence value
    polarity: str                           # "positive" or "negative"


@dataclass
class RawData:
    """Raw signals after sync but before preprocessing."""
    signal_470: np.ndarray
    signal_405: np.ndarray
    ecog: np.ndarray
    emg: Optional[np.ndarray]
    temperature_raw: np.ndarray
    temp_bit_volts: float
    temp_slope: float                     # T(C) = slope * V(mV) + intercept
    temp_intercept: float
    time: np.ndarray
    fs: float
    # Opto-stim analog channel, post-sync trimming/decimation. None for non-opto
    # sessions. Carried in volts (already bit-scaled by oep_reader).
    stim_raw: Optional[np.ndarray] = None


@dataclass
class PhotometryResult:
    """Output of a photometry preprocessing strategy."""
    dff: np.ndarray                         # dF/F signal
    dff_zscore: Optional[np.ndarray] = None # z-scored dF/F (relative to baseline)
    dff_hpf: Optional[np.ndarray] = None    # high-pass filtered dF/F (for transient detection)


@dataclass
class ProcessedData:
    """Signals after preprocessing."""
    photometry: Optional[PhotometryResult] = None
    ecog_filtered: Optional[np.ndarray] = None
    temperature_c: Optional[np.ndarray] = None   # degrees Celsius
    temperature_smooth: Optional[np.ndarray] = None  # smoothed temperature
    time: Optional[np.ndarray] = None
    fs: Optional[float] = None


@dataclass
class Session:
    """Complete data container for one recording session."""
    mouse_id: str
    genotype: str                           # "Scn1a" | "WT"
    heating_session: int = 1
    n_seizures: int = 0                     # 0, 1, >1
    sudep: bool = False                    # Sudden Unexpected Death in Epilepsy
    include_session: bool = True            # whether to include in analysis
    exclusion_reason: Optional[str] = None  # if include_session=False, why
    include_for_baseline: bool = True
    include_for_transients: bool = True
    transient_prominence: Optional[float] = None
    experiment_label: str = ""              # e.g. "GCaMP / mPFC / PV"
    landmarks: Optional[SessionLandmarks] = None
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    raw: Optional[RawData] = None
    processed: Optional[ProcessedData] = None
    transients: List[TransientEvent] = field(default_factory=list)
    spikes: List[SpikeEvent] = field(default_factory=list)
    cohort: str = ""
    date: Optional[str] = None
    session_name: Optional[str] = None
    # If set, classify_session returns this string instead of the
    # genotype-by-seizure analytical bucket. Used by paradigms where the
    # meaningful group axis isn't DS/WT × seized (e.g. opto, where the
    # cohort is the subfolder name like "10HZ_DG_CPP").
    cohort_override: Optional[str] = None
