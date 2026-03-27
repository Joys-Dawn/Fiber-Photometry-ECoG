# Fiber Photometry & ECoG Automated Analysis Tool — Specification

## 1. Eperimental Goals

1. Record the activity of defined neuronal circuit elements (somatic activity or neurotransmitter release) in vivo
2. Compare that activity before, during, and after hyperthermia-evoked seizures WITHIN experimental animals
3. Compare that activity BETWEEN experimental animals and controls (heated but do not exhibit seizures)

## 2. Cohorts

- **Experimental:** Scn1a+/- mice that have seizures
- **Control 1:** Scn1a+/- mice that fail to have seizures
- **Control 2:** WT mice that don't have seizures

## 3. Data Types Collected (Synchronous)

1. **Fiber photometry** — emitted fluorescence via 470 nm (signal) and 405 nm (isosbestic control)
   - GCaMP: reflects intracellular calcium / neuronal firing
   - GRAB-NE: reflects norepinephrine concentration
   - GRAB-ACh: reflects acetylcholine concentration
   - Pipeline is sensor-agnostic; label is configurable
2. **ECoG** — electrical brain activity; used to identify spikes and seizures
3. **Temperature** — internal body temperature of the mouse
4. **EMG** — muscle activity / movement (OUT OF SCOPE for now)

## 4. Data Formats & Acquisition

| Signal | Source | Format | Default Sampling Rate |
|--------|--------|--------|-----------------------|
| Photometry (470 + 405) | pyPhotometry | `.ppd` (analog_1, analog_2) | From PPD metadata, user-editable |
| Sync signal | pyPhotometry | `.ppd` (digital_1) | — |
| ECoG | Open Ephys | Binary (stream 1) | From OEP metadata, user-editable |
| Temperature | NI-USB 6009 DAQ | Open Ephys binary (stream 1, ch17) | From recording metadata |

| Channel | Default | User-Editable |
|---------|---------|---------------|
| ECoG | Channel 2 | Yes |
| EMG | Channel 3 | Yes |
| Temperature | Channel 17 | Yes |

### Folder Structure

Data is organized in a parent folder per experiment:

```
experiment/                    (e.g., "LC GCaMP")
├── data_log.xlsx              (experiment log — see Workflow below)
├── Scn1a_seizure/             (cohort subfolder)
│   ├── mouse1234_session1/    (one sub-subfolder per mouse_session)
│   │   ├── OpenEphys_folder/  (ECoG, EMG, temperature)
│   │   └── photometry.ppd     (fiber photometry file)
│   └── mouse5678_session2/
├── Scn1a_no_seizure/
└── WT/
```

The tool scans the selected experiment folder, discovers cohort subfolders and session sub-subfolders automatically. The user selects the experiment folder; individual file picking is not required.

## 5. Metadata (Per Session)

### Mouse Info (from data log)

| Field | Source |
|-------|--------|
| Experiment (fluorescent tool / brain region / cell type) | Data log |
| Mouse ID # | Data log (matched to folder name) |
| Heating session # | Data log (matched by date to OEP folder) |
| Genotype | Data log (`genotype` column: H=Scn1a, W=WT) |
| Number of seizures recorded (0, 1, >1) | Data log (`seizure` column) |
| SUDEP (yes/no) | Data log (`fatal` column) |
| Include session (yes/no) | Data log (`exclude` column, inverted) |
| Exclusion reason | Data log (`reason` column) |

### Data Collection Info

| Field | Source |
|-------|--------|
| Sampling frequency: photometry | Default from file → user can edit |
| Sampling frequency: ECoG | Default from file → user can edit |
| Channel: ECoG | Default ch2 → user can edit |
| Channel: EMG | Default ch3 → user can edit |
| Channel: Temperature | Default ch17 → user can edit |

### Key Experimental Landmarks — All Mice

| Field | Source |
|-------|--------|
| Baseline temperature | User enter or calculation from temp curve |
| Heating start time | User enter or calculate from temp curve |
| Max temperature | Calculation from temp curve |
| Time to max temperature | Calculation from temp curve |
| Temperature at experiment end | Calculation from temp curve |
| Time to experiment end | Calculation from temp curve |

### Key Experimental Landmarks — Seizure Trials

| Field | Source |
|-------|--------|
| EEC (earliest electrographic change, 1st seizure) — time | User enter |
| UEO (unequivocal electrographic onset, 1st seizure) — time | User enter |
| Behavioral onset — time | User enter |
| OFF (seizure offset, last seizure) — time | User enter |
| EEC — temperature | Calculation (temp at EEC time) |
| UEO — temperature | Calculation (temp at UEO time) |
| Behavioral onset — temperature | Calculation (temp at behavioral onset time) |
| OFF — temperature | Omitted (doesn't make sense to use) |

### Key Experimental Temperatures — No-Seizure Trials

Equivalents are assigned by matching controls to the seizure group's **mean** landmark values. User may choose **time-matched** or **temperature-matched** mode (both supported).

**v1 — Mean matching:** Compute the mean EEC, UEO, behavioral onset, and OFF times (or temperatures) across all seizure mice. For each control mouse, find the corresponding time point when the control reached that mean temperature (temperature-matched mode) or assign the mean elapsed time from heating start (time-matched mode). All controls get equivalent landmarks derived from the same seizure-group means. This is the approach used in the existing SNr analysis code.

**v2 (future) — One-to-one pairing:** One-to-one pairing with specific seizure trials is the "best" approach but is more complicated since sample sizes are not necessarily matched across groups. Prioritize matching the group mean over the standard deviation. If more seizure mice than controls exist, there will be leftover seizure mice and the means might be skewed. If more controls than seizure mice exist, controls could be redundantly paired (which could skew the means) or leftover controls could be assigned to the population mean. v1 matches means only; v2 could optionally match both means and standard deviations.

| Field | Source |
|-------|--------|
| EEC equivalent — time | Derived from seizure group mean |
| UEO equivalent — time | Derived from seizure group mean |
| Behavioral onset equivalent — time | Derived from seizure group mean |
| OFF equivalent — time | Derived from seizure group mean |
| EEC equivalent — temperature | Derived from seizure group mean |
| UEO equivalent — temperature | Derived from seizure group mean |
| Behavioral onset equivalent — temperature | Derived from seizure group mean |
| OFF equivalent — temperature | Omitted (doesn't make sense to use) |

Both control groups should have same mean & similar distribution to experimental group.

## 6. Preprocessing

### Step 0: Align Data Streams

- Account for differences in sampling rates between photometry and ECoG
- Align using TTL barcode pulses (multi-pulse matching with linear regression for clock drift correction)
- PCHIP interpolation for upsampling photometry to ECoG rate
- Trim dangling ends that don't align

### Step 1: ECoG

- Select the channel (default ch2, user-configurable)
- Filter raw signal: 4th order Butterworth bandpass 1–70 Hz + 60 Hz notch (Q=30), zero-phase (`sosfiltfilt`)
- Identify interictal spikes (spikes occurring outside the seizure period)

### Step 2: Fiber Photometry

Three correction strategies available (user selects):

**Strategy A — Chandni's:**
- Gaussian smoothing (σ=75 samples) on both channels
- Simple isosbestic subtraction: `ΔF/F = (GCaMP − Isosbestic) / Isosbestic`

**Strategy B — Meiling's:**
- Butterworth low-pass filter (4th order, 10 Hz cutoff) on both channels
- Biexponential fit for photobleaching correction (subtract fitted curve from each channel)
- OLS linear regression of isosbestic → GCaMP for motion correction
- `ΔF/F = (detrended_photo − estimated_motion) / expfit`

**Strategy C — IRLS (robust):**
- Biexponential fit for photobleaching correction
- IRLS robust regression of isosbestic → GCaMP for motion/artifact correction (down-weights outliers, more resistant to transient artifacts than OLS)
- `ΔF/F` calculated with baseline normalization

**Then for all strategies:**
- z-score relative to baseline (recording start → heating start) → **use for all analyses involving means**
- High-pass filter (2nd order Butterworth, 0.01 Hz cutoff, configurable), then z-score relative to baseline, then identify transients → **use for all analyses involving transients** (HPF before z-score so baseline stats are not contaminated by slow drift)

**Transient detection parameters (configurable):**
- Minimum peak prominence: 1 (in z-score units)
- Maximum peak width: 8 seconds
- Peak-to-trough amplitude window: 2.5 seconds
- Half-width at 50% prominence
- MAD-based adaptive thresholding available as alternative

**Important:** z-ΔF/F for mean signal analyses; raw ΔF/F for transient property measurements (amplitude, half-width) per Wallace et al. 2025.

### Step 3: Temperature

- Align to other signals (via shared sync)
- Convert voltage → temperature: `T(°C) = 0.0981 × V(mV) + 8.81` (linear calibration, 300-sample moving average smoothing; coefficients user-configurable)
- Calculate baseline, peak (max), and terminal temperatures

### Step 4: EMG

- Align to other signals (via shared sync)
- Additional preprocessing TBD (analysis is out of scope for now)

### User Decisions Before Analysis

1. Strategy for fiber photometry preprocessing (A, B, or C)
2. Whether controls should be defined by heating time or by temperature

## 6b. Workflow (Data Log Steps)

The tool follows a 3-step workflow, mirroring a data log:

### Step 1: Load & Preprocess

User selects the experiment folder. The tool:
1. Finds the data log (Excel file) in the experiment folder
2. Scans cohort subfolders and session sub-subfolders for OEP + PPD data
3. Matches each data log row to a session folder by mouse ID + date
4. Populates all metadata automatically (genotype, seizure, SUDEP, include/exclude, heating start, etc.)
5. User sets channel numbers (defaults: ECoG=2, EMG=3, Temp=17) — applied to all sessions
6. User clicks "Load All Sessions" to batch-load and sync all included sessions

The code then preprocesses the data and outputs raw + preprocessed traces.

### Step 2: Review & Mark Landmarks

The preprocessed ECoG trace is displayed in an **interactive GUI** where the user can click on the trace to mark seizure landmarks:
- EEC (earliest electrographic change)
- UEO (unequivocal electrographic onset)
- OFF (seizure offset)

User also enters:
- Include yes/no (if no, reason for exclusion)

### Step 3: Analysis & Output

The code calculates all parameters (Section 8), runs statistics, and generates plots. This step runs after all sessions have completed Steps 1-2.

## 7. Output — Traces

### Color Scheme

| Trace Type | Color |
|------------|-------|
| ECoG traces | Dark gray |
| Photometry, Scn1a seizure trials | Red |
| Photometry, Scn1a failed seizure trials | Purple |
| Photometry, WT no seizure trials | Blue |
| Temperature traces | Black (gradient heating→cooling if possible) |

### Display v1: Sanity Check — Zoomed Out, Including Raw

| Data Type | Traces |
|-----------|--------|
| Photometry | Raw 470 & 405 signals (volts) |
| | Corrected photometry signal (z-ΔF/F) for mean calculation |
| | Corrected + HPF photometry signal with detected transients marked (z-ΔF/F) |
| ECoG | Raw ECoG (mV) |
| | Corrected ECoG (mV) |
| Temperature | Temperature curve (°C) |

**Display notes:**
- All x-axes in seconds, starting at t=0
- Display the entire trace

### Display v2: Sanity Check — Zoomed In, Processed Only

| Data Type | Traces |
|-----------|--------|
| Photometry | Corrected photometry signal (z-ΔF/F) for mean calculation |
| ECoG | Corrected ECoG (mV) |

**Display notes:**
- All x-axes in seconds, with 0 = seizure onset or equivalent
- Display X seconds (300 before & after?), centered on t=0
- Vertical dashed line at t=0

## 8. Output — Parameters

All parameters calculated for each individual trace, plus group means (seizure, failed seizure, WT).

### 8.1 Cohort Characteristics

- Baseline temperature
- Seizure threshold

### 8.2 Baseline / Interictal Characteristics (Transients)

- Transient frequency across baseline period
- Transient amplitude (mean across baseline period)
- Transient half-widths (mean across baseline period)

### 8.3 Pre-Ictal Dynamics — Mean Signal

- By time period: mean signal binned within baseline, early heat, late heat
  - Early heat = heating start → midpoint between heating start and UEO
  - Late heat = midpoint → UEO
- By time period: mean signal at end of late heat (immediately before seizure onset)
- By temperature: mean signal during heating, relative to seizure onset temperature / equivalent for each trace, binned by 1°C

### 8.4 Pre-Ictal Dynamics — Transients

- Moving averages of transient properties while heating (from "seizure onset temp - X" to "seizure onset temp"):
  - Frequency
  - Amplitude
  - Half-width
- Transient properties binned by temperature while heating (from "seizure onset temp - X" to "seizure onset temp", binned by 1°C):
  - Frequency
  - Amplitude
  - Half-width

### 8.5 Interictal / Pre-Ictal Spike Dynamics

- Interictal ECoG spike-triggered average of photometry signal (±30s window)
- Associated AUC quantification (trapezoidal method)

### 8.6 Ictal Dynamics — Mean Signal

- Signal relative to baseline: mean z-ΔF/F within seizure period / equivalent
- Signal relative to immediate pre-ictal: change in z-ΔF/F between pre-ictal and ictal period
- Triggered averages (±30s) based on key experimental timepoints & associated AUC:
  - EEC / equivalent
  - UEO / equivalent
  - Behavioral onset / equivalent
  - OFF / equivalent
  - Max temp / equivalent

### 8.7 Ictal Dynamics — Transients

- PSTH of transient frequency before vs after UEO / equivalent (10s bins, ±60s window)
- Moving averages of transient properties zoomed around UEO / equivalent:
  - Frequency
  - Amplitude
  - Half-width

### 8.8 Postictal Recovery (*needs further specification*)

- Temp relative to max vs mean z-ΔF/F (cooling portion only), binned by 1°C
- Final recording time vs final recording temp
- Final recording time vs final mean ΔF/F
- Final recording temp vs final mean ΔF/F

## 9. Output — Statistics

Statistics should reflect:

1. Overall experiment structure (note: typically will be multiple unique experimental groups, such as (1) Scn1a seizure, (2) Scn1a failed seizure, (3) WT)
2. Repeated measurements from individual animals
3. Type of parameter

## 10. Scope Exclusions

- EMG analysis (backburner; preprocessing/alignment is in scope)
- Automated seizure detection (manual entry for now)
