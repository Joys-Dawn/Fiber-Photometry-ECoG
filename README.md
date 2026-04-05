# Fiber Photometry & ECoG Analysis

A Python GUI for analyzing simultaneous fiber photometry, ECoG, and temperature recordings from a mouse hyperthermia-induced seizure model. It aligns and preprocesses multi-modal data, detects calcium transients and interictal spikes, extracts parameters across seizure phases, and generates publication-style plots.

## Installation

Requires Python 3.12+.

```bash
pip install -r requirements.txt
python run.py
```

## Data Organization

Point the app at a parent experiment folder structured like this:

```
experiment/
├── data_log.xlsx                   (experiment metadata)
├── Scn1a_seizure/                  (cohort)
│   ├── mouse1234_session1/         (one folder per mouse × session)
│   │   ├── Record Node XXX/        (Open Ephys recording)
│   │   │   └── experiment1/recording1/
│   │   │       ├── continuous/     (ECoG, EMG, temperature)
│   │   │       └── events/         (TTL sync pulses)
│   │   └── photometry.ppd          (pyPhotometry file)
│   └── mouse5678_session2/
├── Scn1a_no_seizure/
└── WT/
```

Sessions are discovered automatically — no individual file picking required.

### Data log

`data_log.xlsx` provides per-session metadata:

| Column | Description |
|--------|-------------|
| Mouse ID | Matched to folder name |
| Date | Matched to Open Ephys folder date |
| Genotype | `H` = Scn1a, `W` = WT for example |
| Seizure | Number of seizures (0, 1, >1) |
| Fatal | SUDEP (yes/no) |
| Exclude | Whether to exclude session |
| Reason | Exclusion reason |

## Workflow

The GUI has three tabs, used in order.

### 1. Load & Preprocess

1. Browse to the experiment folder. The app discovers cohorts, sessions, and matches them to the data log.
2. Confirm hardware parameters (ECoG / EMG / temperature channels, sampling rates). Defaults are auto-detected.
3. Pick a fiber photometry preprocessing strategy (see below).
4. **Load All Sessions** to batch-load, align, and preprocess.
5. Optionally save preprocessed data so future loads are instant.

### 2. Review & Mark Landmarks

An interactive viewer shows the preprocessed ECoG trace for each seizure session. Click the trace to mark:

- **EEC** — earliest electrographic change
- **UEO** — unequivocal electrographic onset
- **Behavioral onset**
- **OFF** — seizure offset

Control sessions (no seizures) get matched landmarks automatically, either time-matched or temperature-matched to the seizure group mean.

### 3. Analysis & Output

Runs all analyses, generates plots, and exports results. Requires every included session to have completed preprocessing and landmark marking.

## Preprocessing

### Alignment

Photometry and Open Ephys run on independent clocks. Alignment uses:

1. TTL barcode matching between pyPhotometry digital output and Open Ephys events
2. Linear regression on matched pulse times to correct clock drift
3. PCHIP interpolation to upsample photometry to the ECoG rate
4. Trim of unmatched ends

### ECoG

- 4th-order Butterworth bandpass, 1–70 Hz
- 60 Hz notch (Q=30)
- Zero-phase filtering (`sosfiltfilt`)
- Interictal spike detection outside seizure periods (prominence-based, polarity-aware)

### Temperature

- Voltage-to-temperature conversion: `T(°C) = 0.0981 × V(mV) + 8.81` (configurable)
- 300-sample moving average
- Automatic extraction of baseline, peak, and terminal temperatures

### Fiber photometry

Three isosbestic correction strategies are available:

- **Strategy A** — Gaussian smoothing (σ = 75 samples) on both channels, then simple isosbestic subtraction `ΔF/F = (GCaMP − Iso) / Iso`.
- **Strategy B** — 10 Hz low-pass, biexponential photobleaching fit, OLS regression of isosbestic onto GCaMP for motion correction.
- **Strategy C** — Biexponential photobleaching fit, IRLS robust regression of isosbestic onto GCaMP (down-weights outliers; more resistant to transient artifacts than OLS).

Two signals are derived from the corrected `ΔF/F`:

| Signal | Used for | Normalization |
|--------|----------|---------------|
| **Mean** | Mean ΔF/F analyses | Z-scored to the pre-heating baseline |
| **Transient** | Transient detection | Detrended, then z-scored to baseline |

Baseline z-scoring is used for strategies B and C to avoid inflating the standard deviation during heating, which would compress z-scores and cause the detection height gate to miss real transients (per PASTa / Donka 2025).

### Transient detection

Strategy A uses `scipy.signal.find_peaks` with prominence ≥ 1.0 on the z-scored signal.

Strategies B and C use the Wallace 2025 two-gate method:

1. **Height gate** — z-score ≥ 1.0 on the detrended signal (identifies candidates above noise)
2. **Prominence gate** — prominence threshold on raw `ΔF/F` (confirms real transients, not fluctuations on a slow trend)

Transient properties are measured on the raw `ΔF/F` (per Wallace et al. 2025): peak amplitude, peak-to-trough amplitude (±2.5 s), half-width at 50% prominence, and temperature at peak.

## Output Parameters

**Cohort characteristics** — Baseline temperature, seizure threshold temperature.

**Baseline / interictal transients** — Frequency, amplitude, and half-width across the baseline period.

**Pre-ictal mean signal** — Mean z-ΔF/F binned by time period (baseline, early heat, late heat) and by temperature (1 °C bins during heating).

**Pre-ictal transients** — Moving averages of frequency, amplitude, and half-width during heating; transient properties binned by temperature (1 °C bins).

**Spike-triggered averages** — Interictal spike-triggered average of the photometry signal (±30 s) with AUC quantification.

**Ictal mean signal** — Mean z-ΔF/F within the seizure period relative to baseline; triggered averages (±30 s) at EEC, UEO, behavioral onset, OFF, and max temperature, each with AUC.

**Ictal transients** — PSTH of transient frequency around UEO (10 s bins, ±60 s); moving averages of transient properties around UEO.

**Postictal recovery** — Temperature vs. mean z-ΔF/F during cooling; final recording time, temperature, and ΔF/F.

## References

- **Wallace et al. 2025** — Two-gate transient detection method. PMC11888193.
- **Donka & Bhatt 2025 (PASTa)** — Photometry analysis standardized tools; baseline z-scoring for paradigms that change signal variance.
- **Simpson et al. 2024** — Guidelines for z-scoring fiber photometry data.