# Fiber Photometry & ECoG Tool — Implementation Plan

## Architecture Overview

Mirror the EphysAutomatedAnalysis pattern: **strategy-based preprocessing**, **pluggable analyzers**, **dataclass-driven pipeline**, **Tkinter GUI shell**. Every scientific decision point becomes a swappable strategy class so the user can compare approaches and the codebase stays open to future methods.

```
fiber_photometry_ecog/
├── app.py                          # Tkinter GUI entry point (3 tabs)
├── core/
│   ├── config.py                   # Dataclass configs (PreprocessingConfig, AnalysisConfig, PlotConfig)
│   ├── data_models.py              # Session, CohortGroup, PairingResult, TransientEvent, etc.
│   ├── session_io.py               # Save/load session state (JSON)
│   └── utils.py                    # Shared helpers
├── data_loading/
│   ├── ppd_reader.py               # pyPhotometry .ppd binary parser
│   ├── oep_reader.py               # Open Ephys binary reader (ECoG stream 1, temp stream 2)
│   └── sync.py                     # TTL barcode alignment engine
├── preprocessing/
│   ├── base.py                     # PreprocessingStrategy protocol + PreprocessingResult dataclass
│   ├── ecog.py                     # ECoG filter pipeline (bandpass + notch)
│   ├── emg.py                      # EMG alignment (analysis out of scope for now)
│   ├── temperature.py              # Voltage→temp conversion + landmark extraction
│   ├── photometry/
│   │   ├── strategy_a_chandni.py   # Gaussian smooth → simple ΔF/F
│   │   ├── strategy_b_meiling.py   # Low-pass → biexp detrend → OLS motion correction
│   │   ├── strategy_c_irls.py      # Biexp detrend → IRLS robust regression
│   │   └── common.py              # Shared: z-scoring, HPF, biexponential fitting
│   ├── transient_detection.py      # Prominence-based + MAD-based detectors
│   └── spike_detection.py          # Interictal ECoG spike detector (adapted from Alex's)
├── pairing/
│   └── engine.py                   # Control equivalent landmark assignment (time-matched / temp-matched)
├── analysis/
│   ├── base.py                     # Analyzer protocol
│   ├── cohort_characteristics.py   # Baseline temp, seizure threshold
│   ├── baseline_transients.py      # Interictal transient freq/amp/half-width
│   ├── preictal_mean.py            # Time-binned + temp-binned mean signal
│   ├── preictal_transients.py      # Moving averages + temp-binned transient props
│   ├── ictal_mean.py               # Triggered averages, AUC, signal change
│   ├── ictal_transients.py         # PSTH + moving averages around UEO
│   ├── postictal.py                # Cooling curve analysis
│   └── spike_triggered.py          # Interictal spike-triggered photometry averages
├── visualization/
│   ├── trace_plots.py              # Display v1 (sanity, full) + Display v2 (zoomed)
│   ├── group_plots.py              # Group-level summary figures (auto-save PNG + CSV)
│   └── colors.py                   # Cohort color scheme constants
└── tests/
    ├── test_sync.py
    ├── test_photometry_strategies.py
    ├── test_transient_detection.py
    ├── test_pairing.py
    ├── test_temperature.py
    └── test_analysis.py
```

---

## Phase 1: Data Loading & Synchronization

### 1.1 PPD Reader (`data_loading/ppd_reader.py`)
- Parse pyPhotometry `.ppd` binary format
- Extract `analog_1` (GCaMP/470nm), `analog_2` (isosbestic/405nm), `digital_1` (sync TTL)
- Extract sampling rate from file metadata; expose as user-overridable
- Return typed dataclass: `PPDData(signal_470, signal_405, digital, fs, metadata)`

### 1.2 Open Ephys Reader (`data_loading/oep_reader.py`)
- Parse Open Ephys binary format (continuous/*.dat + structure.oebin)
- Stream 1 → ECoG channels (default ch3, user-selectable) + EMG channel(s)
- Stream 2 → NI-DAQ ADC ch1 (temperature voltage)
- Extract sync TTL from digital channel
- Return typed dataclass: `OEPData(ecog, emg, temperature_voltage, digital, fs_ecog, fs_temp, metadata)`

### 1.3 TTL Barcode Synchronization (`data_loading/sync.py`)
- **Scientific rigor point:** Single-edge alignment (as in fcd-hyperthermia) cannot detect clock drift. Multi-pulse matching with linear regression is required.
- Detect rising edges in both PPD and OEP digital channels
- Multi-pulse barcode matching: identify corresponding pulse patterns across systems
- Linear regression on matched pulse times → slope (drift rate) + intercept (offset)
- PCHIP interpolation: upsample photometry to ECoG sampling rate using drift-corrected timebase
- Trim unmatched leading/trailing segments
- Validation: report drift rate, residual jitter, number of matched pulses
- Return `SyncResult(time_vector, drift_ppm, n_matched, residual_ms)`

**Test:** Synthetic TTL with known drift → verify recovery within tolerance.

---

## Phase 2: Preprocessing Pipeline

### 2.1 Core Data Models (`core/data_models.py`)
```python
@dataclass
class Session:
    mouse_id: str
    genotype: str  # "Scn1a" | "WT"
    heating_session: int
    n_seizures: int  # 0, 1, >1
    survived: bool
    experiment_label: str  # e.g. "GCaMP / mPFC / PV"
    landmarks: SessionLandmarks
    preprocessing_config: PreprocessingConfig
    raw: RawData          # loaded signals
    processed: ProcessedData  # after preprocessing
    transients: List[TransientEvent]
    pairing: Optional[PairingResult]

@dataclass
class SessionLandmarks:
    heating_start_time: float       # seconds
    eec_time: Optional[float]       # None for controls
    ueo_time: Optional[float]
    behavioral_onset_time: Optional[float]
    off_time: Optional[float]
    baseline_temp: float            # calculated
    max_temp: float                 # calculated
    # ... equivalent times/temps for controls filled by pairing engine

@dataclass
class TransientEvent:
    peak_time: float
    peak_amplitude: float           # raw ΔF/F (not z-scored)
    trough_amplitude: float
    peak_to_trough: float
    half_width: float
    prominence: float
    temperature_at_peak: float
```

### 2.2 ECoG Preprocessing (`preprocessing/ecog.py`)
- 4th order Butterworth bandpass 1–70 Hz (SOS form)
- 60 Hz notch filter Q=30 (SOS form)
- **Both applied with `sosfiltfilt` (zero-phase)**
- **Scientific rigor point:** Alex's code uses `filtfilt` with ba-form notch, which is numerically less stable. SOS form is mandatory for high-order or narrow-band filters.
- All filter parameters exposed in `PreprocessingConfig` for reproducibility

### 2.3 Photometry Strategies (`preprocessing/photometry/`)

All three strategies implement the same protocol:
```python
class PhotometryStrategy(Protocol):
    def preprocess(self, signal_470: ndarray, signal_405: ndarray,
                   fs: float, config: PreprocessingConfig) -> PhotometryResult
```

**Strategy A — Chandni's** (`strategy_a_chandni.py`):
1. Gaussian smoothing (σ=75 samples, configurable) on both channels via `filtfilt` with Gaussian kernel
2. `ΔF/F = (smoothed_470 - smoothed_405) / smoothed_405`
- Simplest approach; no explicit photobleaching or motion correction
- Gaussian acts as low-pass, isosbestic subtraction handles shared noise
- **Limitation:** No photobleaching model — assumes Gaussian smoothing adequately handles slow drift. Works when bleaching is mild and recording is short.

**Strategy B — Meiling's** (`strategy_b_meiling.py`):
1. Butterworth low-pass (4th order, 10 Hz default per Meiling's code, configurable) on both channels
2. Biexponential fit on each channel independently → subtract fitted curve (photobleaching correction)
3. OLS linear regression: `detrended_iso → detrended_signal` → estimated motion = slope × iso + intercept
4. `ΔF/F = (detrended_signal - estimated_motion) / expfit`
- Explicit photobleaching model + explicit motion model
- **Limitation:** OLS produces "downshifted" corrected signals (Keevers 2025) because it treats calcium transients as part of the isosbestic fit, biasing the motion estimate upward.

**Strategy C — IRLS / Keevers** (`strategy_c_irls.py`):
Reproduces the pipeline from Keevers & Jean-Richard-dit-Bressel 2025 (Neurophotonics 12(2):025003, PMID 40166421).
Reference implementation: https://github.com/philjrdb/RegressionSim (see `IRLS_dFF.m`)
1. Butterworth low-pass (3 Hz, configurable) on both channels
2. IRLS robust regression on **filtered** signals (no biexponential detrending): Tukey's bisquare weighting, tuning constant c=1.4 (configurable)
   - Down-weights samples where residuals are large (i.e., genuine calcium transients)
   - Converges to a motion estimate that is not distorted by neural activity
   - The regression directly captures the shared bleaching trend + motion artifacts
3. `ΔF/F = (filtered_signal - fitted_iso) / fitted_iso` (dF/F, not dF — Keevers showed dF/F outperforms dF)
   - The fitted_iso retains the original signal scale (including bleaching), so dividing by it normalizes for photobleaching
   - Keevers explicitly warns: "detrending and/or z-scoring experimental and isosbestic signals preclude their use in dF/F calculations as the original scale...is necessary"
- **Source:** Keevers 2025 is the only published paper comparing OLS vs IRLS for isosbestic correction. They showed OLS produces "downshifted" signals due to overfitting neural dynamics. IRLS with c=1.4 was the most accurate of the tested configurations.
- **Key distinction from Strategy B:** No biexponential detrending step. The IRLS regression + dF/F division handles photobleaching implicitly via the isosbestic channel's own bleaching trend.

**Shared post-processing** (`preprocessing/photometry/common.py`):
- z-score relative to baseline window (recording start → heating onset) → z-ΔF/F for mean analyses
- HPF (2nd order Butterworth, 0.01 Hz, configurable) → transient detection stream
- Biexponential fitting function used by Strategy B

### 2.4 Transient Detection (`preprocessing/transient_detection.py`)
- **Primary:** `scipy.signal.find_peaks` with prominence ≥ 1 (Chandni's default), max width ≤ 8s, all configurable. No literature consensus on a specific prominence default — Wallace 2025 uses z=1.0 and z=2.6 for different methods; GuPPy uses a 2-step MAD process (2 MAD filter + 3 MAD threshold).
- Peak-to-trough amplitude: minimum in ±2.5s window around peak
- Half-width at 50% prominence
- **Alternative:** MAD-based adaptive threshold (median + k × MAD, configurable k)
- **Scientific rigor point:** Transient properties (amplitude, half-width) measured on raw ΔF/F, NOT z-scored signal, per Wallace et al. 2025. Z-scoring distorts waveform shape.
- All detected transients stored as `List[TransientEvent]` with temperature cross-referenced

**Test:** Inject synthetic transients of known amplitude/width into noise → verify recovery.

### 2.5 EMG Preprocessing (`preprocessing/emg.py`)
- Align to other signals via shared sync (same timebase as ECoG after sync)
- Store aligned EMG in session data for future analysis
- Additional preprocessing TBD per Excel spec

### 2.6 Temperature Processing (`preprocessing/temperature.py`)
- Linear calibration: `T(°C) = slope × V(mV) + intercept` (defaults: 0.0981, 8.81; user-configurable)
- 300-sample moving average smoothing (configurable window)
- Extract landmarks: baseline temp (mean of first N seconds), max temp, time of max, terminal temp
- Heating start detection: user-entered or auto-detected (sustained derivative > threshold)
- Temperature lookup function: `temp_at_time(t) → °C` for cross-referencing transient events

### 2.7 Interictal Spike Detection (`preprocessing/spike_detection.py`)
- Adapted from Alex's existing detector (same general approach, parameters reviewed)
- Target: interictal spikes (70–200ms duration, sharp morphology)
- **Algorithm** (z-scored threshold + `find_peaks`):
  1. Z-score the filtered ECoG signal relative to baseline period
  2. Compute adaptive threshold: `tmul × baseline_std` (default tmul=3, configurable)
  3. Apply absolute minimum threshold floor (default 0.4, configurable)
  4. `scipy.signal.find_peaks` with: height ≥ threshold, width 70–200ms, prominence ≥ 0.5 × threshold, min inter-spike distance = 70ms
  5. Detect both positive and negative spikes (invert signal for negative detection)
  6. Remove duplicates within 10ms (keep higher prominence)
  7. Exclude spikes within seizure periods (user-entered EEC→OFF window)
- Configurable via `SpikeDetectionConfig`: tmul, absolute threshold, spike duration range, edge margin, dedup window
- Output: `List[SpikeEvent(time, amplitude, width_ms, prominence, polarity)]`
- GUI review mode: display detected spikes overlaid on ECoG so the user can visually approve/reject detections and tune parameters

---

## Phase 3: Control Equivalent Landmark Assignment

### 3.1 Modes (`pairing/engine.py`)

Two modes, user-selectable:

**Time-matched:** Compute the mean EEC, UEO, behavioral onset, and OFF elapsed times (from heating start) across all seizure mice. Each control gets those mean times as its equivalent landmarks.

**Temperature-matched:** Compute the mean EEC, UEO, behavioral onset, and OFF temperatures across all seizure mice. For each control, find the first time during heating when the control reached each mean temperature. OFF equivalent = UEO equivalent + mean seizure duration. This is the approach used in the existing SNr analysis MATLAB code (`temp_eq_times.m`).

### 3.2 Implementation

1. Collect all seizure sessions → compute mean landmark times and temperatures
2. Compute mean seizure duration (UEO → OFF)
3. For each control session:
   - **Temperature-matched:** Truncate temperature trace at max temp (heating phase only). Find the first sample where temperature equals each mean landmark temperature. OFF equivalent = UEO equivalent time + mean seizure duration.
   - **Time-matched:** Assign the mean elapsed times directly as equivalent landmarks.
4. Store equivalent landmarks in session data for downstream analysis

Output: `PairingResult` with per-session equivalent times/temps.

**Test:** Synthetic temperature curves with known profiles → verify correct time lookups.

### 3.3 Future (v2): One-to-One Pairing

One-to-one pairing with specific seizure trials is the preferred long-term approach but is more complicated since sample sizes are not necessarily matched across groups. Prioritize matching the group mean over the standard deviation. Handle unequal N (leftover seizure mice when N_seizure > N_control; redundant pairing or population-mean assignment when N_control > N_seizure). v2 could optionally match both means and standard deviations.

---

## Phase 4: Analysis Modules

Each analyzer implements:
```python
class Analyzer(Protocol):
    def analyze(self, sessions: List[Session], config: AnalysisConfig) -> AnalysisResult
```

Results are dataclass containers with both per-session and group-level values + the data needed for plotting.

### 4.1 Cohort Characteristics (`analysis/cohort_characteristics.py`)
- Per session: baseline temperature, seizure threshold (temp at UEO)
- Group means ± SEM

### 4.2 Baseline / Interictal Transients (`analysis/baseline_transients.py`)
- Filter transients to baseline period (start → heating onset)
- Compute: frequency (count / duration), mean amplitude, mean half-width
- Per session + group summaries

### 4.3 Pre-Ictal Mean Signal (`analysis/preictal_mean.py`)
- **Time-binned:** mean z-ΔF/F within baseline, early heat (heat start → midpoint to UEO), late heat (midpoint → UEO)
- **End of late heat:** mean z-ΔF/F in window immediately before UEO
- **Temperature-binned:** mean z-ΔF/F in configurable bins (default 1°C) during heating, relative to seizure onset temperature (or equivalent for controls)
- For controls: use equivalent landmarks from pairing
- Temperature bin size is configurable (default 1°C per spec)

### 4.4 Pre-Ictal Transients (`analysis/preictal_transients.py`)
- **Sliding-window moving averages** of transient frequency, amplitude, half-width during heating (true moving average, not period-binned)
- Same metrics binned by configurable temperature bins (default 1°C), relative to seizure onset temp (or equivalent)
- Temperature range: seizure onset temp − X to seizure onset temp (X configurable)

### 4.5 Ictal Mean Signal (`analysis/ictal_mean.py`)
- Mean z-ΔF/F within seizure period vs baseline
- Δ z-ΔF/F between pre-ictal and ictal windows
- **Event-triggered averages (±30s)** for each landmark: EEC, UEO, behavioral onset, OFF, max temp (and equivalents)
- AUC via trapezoidal integration on each triggered average
- Multi-seizure handling: EEC/UEO from 1st seizure, OFF from last seizure

### 4.6 Ictal Transients (`analysis/ictal_transients.py`)
- PSTH: transient frequency in 10s bins, ±60s around UEO/equivalent
- Sliding-window moving averages of transient freq/amp/half-width zoomed around UEO

### 4.7 Postictal Recovery (`analysis/postictal.py`)
- Cooling curve: mean z-ΔF/F vs temperature (relative to seizure onset temp), configurable bins (default 1°C), cooling portion only
- Final recording metrics: time, temp, mean ΔF/F, and their pairwise relationships

### 4.8 Spike-Triggered Averages (`analysis/spike_triggered.py`)
- Align photometry signal to each detected interictal ECoG spike
- Mean ± SEM across spikes, ±30s window
- AUC via trapezoidal method

---

## Phase 5: Visualization

Following the EphysAutomatedAnalysis pattern: each plot function takes an `output_dir`, auto-creates it with `os.makedirs(exist_ok=True)`, saves PNG directly with `plt.savefig()` + `plt.close('all')` (frees memory), and returns the saved path. No separate export module — saving is baked into each plot function. CSVs saved to the same output directory where applicable.

### 5.1 Display v1 — Full Sanity Check (`visualization/trace_plots.py`)
- 6 vertically stacked subplots: raw 470 & 405, corrected z-ΔF/F, HPF + transients marked, raw ECoG, filtered ECoG, temperature
- x-axis: seconds from t=0 (recording start), full trace

### 5.2 Display v2 — Zoomed Processed (`visualization/trace_plots.py`)
- 2 subplots: corrected z-ΔF/F, filtered ECoG
- x-axis: centered on seizure onset / equivalent (t=0), configurable window
- Vertical dashed line at t=0

### 5.3 Group Summary Plots (`visualization/group_plots.py`)
- Color-coded by cohort (red/purple/blue per spec)
- Temperature-binned line plots with SEM shading
- Triggered average plots with SEM
- PSTH bar charts
- Box/violin plots for scalar comparisons

---

## Phase 6: GUI

### 6.1 Tkinter Application (`app.py`)
Following the EphysAutomatedAnalysis pattern: all sessions/groups loaded and processed together, not one at a time.

- **Tab 1 — Data Loading & Cohort Management:** File selectors (PPD + OEP folders), metadata entry panel, sync preview. User loads sessions and assigns each to a cohort group (Scn1a seizure, Scn1a failed seizure, WT). All loaded sessions visible in a session list with group labels.
- **Tab 2 — Preprocessing:** Strategy selector (A/B/C), parameter panel, sanity check plots (v1 + v2), interictal spike review overlay (detected spikes on ECoG with approve/reject/tune), approve/reject per session
- **Tab 3 — Extraction:** Pairing mode selector, extraction params (temp bin size, triggered window, PSTH bins, etc.), run extraction (the 8 analysis modules), results display + plots, auto-save PNG/CSV. Statistical analysis is a future addition.

### 6.2 Session Management
- Save/load full session state (JSON serialization of all dataclasses)
- Batch processing: queue multiple sessions, run preprocessing + analysis sequentially

---

## Phase 7: Testing & Validation

### 7.1 Unit Tests
- Sync: synthetic TTL with known drift
- Each photometry strategy: synthetic signal with known bleaching curve + transients → verify ΔF/F recovery
- Transient detection: synthetic peaks of known amplitude/width
- Temperature: known voltage → expected °C
- Pairing: synthetic cohorts → mean preservation

### 7.2 Cross-Strategy Validation
- Run all 3 strategies on the same real recording
- Compare: ΔF/F traces, detected transient counts, transient amplitudes
- Flag divergences > threshold for user review
- **Scientific rigor point:** This is the primary way users validate that their strategy choice doesn't drive their conclusions.

### 7.3 Integration Test
- End-to-end: real PPD + OEP files → full pipeline → output parameters + plots
- Compare key metrics against Chandni's MATLAB output on same data

---

## Error Handling

| Edge Case | Handling |
|---|---|
| Biexponential fit fails to converge | Crop first 2 min (per PASTa, Donka 2025). Fallback: monoexponential → linear detrend. Warn user. |
| No transients detected in a session | Return empty list, warn user. Mean-signal analyses still run. Flag session in output. |
| Sync fails (too few matched pulses) | If < 3 matched pulses, reject session with clear error. If drift > 100 ppm, warn but allow. |
| Signal is all NaN or flat | Reject at load time with clear error message. |
| Baseline std = 0 (z-scoring) | Return zeros, warn user (session likely has no signal). |

---

## Dependencies

**Core scientific:** `numpy`, `scipy` (signal, optimize, interpolate, stats), `matplotlib`

**Data I/O:** `struct` (stdlib, PPD parsing), `json` (stdlib, OEP metadata + session save/load), `openpyxl` or `pandas` (Excel/CSV export)

**GUI:** `tkinter` (stdlib), `matplotlib.backends.backend_tkagg`

**Optional:** `pandas` (tabular parameter handling)

