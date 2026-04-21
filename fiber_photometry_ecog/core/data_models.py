"""
Core data models for session-level data.

These dataclasses hold per-session metadata, landmarks, raw/processed
signals, and detected events. They are the central data containers
passed between preprocessing, analysis, and visualization modules.
"""

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from .config import PreprocessingConfig


@dataclass
class SessionLandmarks:
    """Temporal and temperature landmarks for a recording session."""
    heating_start_time: float               # seconds from recording start
    eec_time: Optional[float] = None        # electrographic event criterion (None for controls)
    ueo_time: Optional[float] = None        # unequivocal electrographic onset
    behavioral_onset_time: Optional[float] = None
    off_time: Optional[float] = None        # seizure offset
    baseline_temp: Optional[float] = None   # calculated from temperature trace
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
