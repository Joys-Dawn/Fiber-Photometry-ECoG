# Fiber Photometry & ECoG Analysis

A Python GUI application for automated analysis of simultaneous fiber photometry, ECoG, and temperature recordings in a mouse hyperthermia-induced seizure model. The application preprocesses multi-modal data, detects calcium transients and interictal spikes, extracts experimental parameters across seizure phases, and generates publication-style plots.

## Installation

Requires Python 3.12+.

```bash
pip install -r requirements.txt
```

Launch the GUI:

```bash
python run.py
```

## Expected Data Directory Structure

The application expects data organized in a parent experiment folder:

```
experiment/
├── data_log.xlsx                   (experiment log with metadata)
├── Scn1a_seizure/                  (cohort subfolder)
│   ├── mouse1234_session1/         (one sub-subfolder per mouse × session)
│   │   ├── Record Node XXX/        (Open Ephys recording folder)
│   │   │   └── experiment1/
│   │   │       └── recording1/
│   │   │           ├── continuous/  (ECoG, EMG, temperature)
│   │   │           └── events/     (TTL sync pulses)
│   │   └── photometry.ppd          (pyPhotometry file)
│   └── mouse5678_session2/
├── Scn1a_no_seizure/
└── WT/
```

Each cohort folder contains session subfolders with co-located Open Ephys and pyPhotometry recordings. The tool scans the experiment folder automatically — individual file picking is not required.

### Data Log

The Excel data log (`data_log.xlsx`) provides per-session metadata:

| Column | Description |
|--------|-------------|
| Mouse ID | Matched to folder name |
| Date | Matched to Open Ephys folder date |
| Genotype | `H` = Scn1a, `W` = WT |
| Seizure | Number of seizures (0, 1, >1) |
| Fatal | SUDEP (yes/no) |
| Exclude | Whether to exclude session |
| Reason | Exclusion reason |

## Application Workflow

The GUI has three tabs, used sequentially:

### Tab 1: Load & Preprocess

1. **Browse** to the experiment folder. The app discovers cohort subfolders, session folders, and matches them to the data log.
2. Set hardware parameters (ECoG channel, EMG channel, temperature channel, sampling rates) — defaults are auto-detected from file metadata.
3. Select a **preprocessing strategy** for fiber photometry (A, B, or C — see below).
4. Click **Load All Sessions**. The app batch-loads, aligns, and preprocesses all included sessions.
5. Optionally **save preprocessed data** to avoid recomputing on future loads. Saved data is stored per-strategy in `.sessions/<strategy>/` inside the experiment folder.

### Tab 2: Review & Mark Landmarks

An interactive viewer displays the preprocessed ECoG trace for each session. The user clicks on the trace to mark seizure landmarks:

- **EEC** — Earliest electrographic change
- **UEO** — Unequivocal electrographic onset
- **Behavioral onset** — Behavioral seizure onset
- **OFF** — Seizure offset

Control sessions (no seizures) receive equivalent landmarks derived from the seizure group mean, with either time-matching or temperature-matching.

### Tab 3: Analysis & Output

Runs all analyses (see Parameters below), generates plots, and exports results. This step requires all sessions to have completed preprocessing and landmark marking.

## Preprocessing

### Data Alignment

Photometry and Open Ephys recordings run on independent clocks at different sampling rates. Alignment uses:

1. TTL barcode matching between pyPhotometry digital output and Open Ephys events
2. Linear regression on matched pulse times to correct for clock drift
3. PCHIP interpolation to upsample photometry to the ECoG sampling rate
4. Trim unmatched ends

### ECoG

- 4th-order Butterworth bandpass: 1–70 Hz
- 60 Hz notch filter (Q=30)
- Zero-phase filtering (`sosfiltfilt`)
- Interictal spike detection outside seizure periods

### Temperature

- Voltage-to-temperature conversion: `T(°C) = 0.0981 × V(mV) + 8.81` (configurable)
- 300-sample moving average smoothing
- Automatic extraction of baseline, peak, and terminal temperatures

### Fiber Photometry

Three isosbestic correction strategies are available:

#### Strategy A (Chandni)

1. Gaussian smoothing (σ = 75 samples) on both 470 nm and 405 nm channels
2. Simple isosbestic subtraction: `ΔF/F = (GCaMP − Iso) / Iso`

#### Strategy B (Meiling)

1. Butterworth low-pass filter (4th order, 10 Hz) on both channels
2. Biexponential fit for photobleaching correction
3. OLS linear regression of isosbestic onto GCaMP for motion correction
4. `ΔF/F = (detrended − estimated_motion) / expfit`

#### Strategy C (IRLS / Robust)

1. Biexponential fit for photobleaching correction
2. IRLS robust regression of isosbestic onto GCaMP (down-weights outliers, more resistant to transient artifacts than OLS)
3. `ΔF/F` with baseline normalization

#### Post-Correction Processing (All Strategies)

Two parallel signal streams are derived from the corrected `ΔF/F`:

| Stream | Purpose | Processing |
|--------|---------|------------|
| **Mean signal** | Analyses involving mean ΔF/F | z-score relative to baseline period (recording start → heating start) |
| **Transient signal** | Transient detection | Strategy A: high-pass filter (0.01 Hz Butterworth) → whole-session z-score. Strategy B/C: moving-average detrend → baseline z-score |

Baseline z-scoring for B/C avoids inflating the standard deviation during the heating period, which would compress z-scores and cause the height gate to miss real transients (per PASTa/Donka 2025 recommendation for paradigms that change signal variance).

### Transient Detection

Two detection methods are used:

**Strategy A** — `scipy.signal.find_peaks` with prominence ≥ 1.0 on the z-scored HPF signal.

**Strategy B/C** — Wallace 2025 ProM two-step detection:
1. **Height gate**: z-score ≥ 1.0 on the z-scored detrended signal (identifies candidate peaks above noise)
2. **Prominence gate**: prominence ≥ threshold on the raw `ΔF/F` signal (confirms peaks are real transients, not noise fluctuations riding on a slow trend)

Prominence thresholds: B = 0.035, C = 0.02 (fractional `ΔF/F`; equivalent to 3.5% and 2.0% in percentage scale).

Transient properties measured on raw `ΔF/F` (per Wallace et al. 2025):
- Peak amplitude
- Peak-to-trough amplitude (within ±2.5 s window)
- Half-width at 50% prominence
- Temperature at peak

### Spike Detection

Interictal ECoG spikes are detected outside seizure periods using prominence-based peak finding on the filtered ECoG signal, with polarity-aware detection (both positive and negative deflections).

## Output Parameters

### Cohort Characteristics
- Baseline temperature, seizure threshold temperature

### Baseline / Interictal Transients
- Transient frequency, amplitude, and half-width across the baseline period

### Pre-Ictal Dynamics (Mean Signal)
- Mean z-ΔF/F binned by time period (baseline, early heat, late heat)
- Mean z-ΔF/F binned by temperature (1°C bins during heating)

### Pre-Ictal Dynamics (Transients)
- Moving averages of transient frequency, amplitude, and half-width during heating
- Transient properties binned by temperature (1°C bins)

### Spike-Triggered Averages
- Interictal spike-triggered average of photometry signal (±30 s window)
- Associated AUC quantification

### Ictal Dynamics (Mean Signal)
- Mean z-ΔF/F within seizure period relative to baseline
- Triggered averages (±30 s) at EEC, UEO, behavioral onset, OFF, and max temperature
- Associated AUC for each triggered average

### Ictal Dynamics (Transients)
- PSTH of transient frequency around UEO (10 s bins, ±60 s window)
- Moving averages of transient properties around UEO

### Postictal Recovery
- Temperature vs. mean z-ΔF/F during cooling
- Final recording time, temperature, and ΔF/F

## Project Structure

```
fiber_photometry_ecog/
├── app.py                          # Main GUI application (Tkinter)
├── core/
│   ├── config.py                   # Preprocessing, transient, temperature configs
│   ├── data_models.py              # Session, RawData, ProcessedData, TransientEvent, etc.
│   └── session_io.py               # Save/load preprocessed sessions (.npz)
├── data_loading/
│   ├── experiment_scanner.py       # Discover sessions from experiment folder + data log
│   ├── oep_reader.py               # Read Open Ephys binary format
│   ├── ppd_reader.py               # Read pyPhotometry .ppd files
│   └── sync.py                     # TTL-based cross-device alignment
├── preprocessing/
│   ├── ecog.py                     # ECoG bandpass + notch filtering
│   ├── emg.py                      # EMG preprocessing (placeholder)
│   ├── temperature.py              # Voltage → °C conversion, landmark extraction
│   ├── transient_detection.py      # find_peaks + Wallace ProM detection
│   ├── spike_detection.py          # Interictal spike detection
│   └── photometry/
│       ├── common.py               # Shared utilities (z-score, HPF, detrend, biexp fit)
│       ├── strategy_a_chandni.py   # Gaussian smoothing + subtraction
│       ├── strategy_b_meiling.py   # LPF + biexp + OLS regression
│       └── strategy_c_irls.py      # Biexp + IRLS robust regression
├── analysis/                       # Per-phase parameter extraction
│   ├── baseline_transients.py
│   ├── cohort_characteristics.py
│   ├── ictal_mean.py
│   ├── ictal_transients.py
│   ├── postictal.py
│   ├── preictal_mean.py
│   ├── preictal_transients.py
│   └── spike_triggered.py
├── visualization/                  # Plot generation
│   ├── colors.py                   # Cohort color scheme
│   ├── group_plots.py              # Group-level comparison plots
│   └── trace_plots.py              # Per-session trace displays
├── pairing/                        # Control ↔ seizure landmark matching
└── tests/

Exploratory/                        # Strategy comparison scripts and figures
docs/                               # Specification document
run.py                              # Entry point
```

## References

- **Wallace et al. 2025** — ProM transient detection method. PMC11888193.
- **Donka & Bhatt 2025 (PASTa)** — Photometry analysis standardized tools; baseline z-scoring recommendation for paradigms that change signal variance.
- **Simpson et al. 2024** — Guidelines for z-scoring fiber photometry data.
- **TDT** — 6 Hz lowpass recommendation for population GCaMP signals; 100–300 ms rise times.
