# Command-Line Analysis

The runners apply experiment-specific settings to the shared processing and
analysis modules. Run the following PowerShell commands from the repository
directory after installing `requirements.txt`. The runners are in `LC_paper/`
and are invoked with `python -m LC_paper.<runner>`.

## Primary Runners

| Runner | Analysis | Input root |
| --- | --- | --- |
| `run_hyperthermia.py` | Hyperthermia and PTZ | Parent containing named datasets |
| `run_opto.py` | Single-burst optogenetics, including CPP | Parent containing `Hotspot_GANE/` |
| `run_grab_opto.py` | Repeated stimulation before/after Yohimbine | `GRAB_opto/` itself |
| `run_green_red_ne.py` | Dual-color green/red NE | `DG_greenANDred_NE/` itself |

Each runner provides `--help`. The examples use the default cache and output
directories under `LC_paper/`. Replace the `data/...` paths with the locations
of your recordings.

## Signal Processing

Recordings are synchronized using TTL pulses and a linear clock-drift model.
Photometry is interpolated onto the common timebase. Hyperthermia and PTZ
analyses use the overlapping recording interval; optogenetic analyses retain
the full photometry interval and pad unavailable Open Ephys samples with NaN.

The isosbestic-correction method used by these runners applies a 3 Hz low-pass
filter to both photometry channels, then fits the 405 nm control to the signal
with IRLS regression (Tukey bisquare, tuning constant 1.4):

```text
dF/F = (filtered signal - fitted control) / fitted control
```

ECoG processing uses a fourth-order 1-70 Hz Butterworth bandpass and a 60 Hz
notch filter. Temperature is calibrated per recording and smoothed with a
300-sample moving average. Dual-color photometry uses the independent-channel
method described below.

## Experiment Settings

### Hyperthermia and PTZ

```powershell
.venv/Scripts/python.exe -m LC_paper.run_hyperthermia --data-root data/Hyperthermia_seizure --only LC_GCaMP
```

`--only` accepts one or more of `LC_GCaMP`, `DG_GRABne`, `DG_GRABneMutant`,
`DG_GRABne_NRI`, `FCD_GRABne`, `PVT_GRABne`, and `PTZ_GRABne`. PTZ generates
split and pooled analyses. Without `--only`, all configured datasets are
attempted. Missing datasets or failed runs produce a nonzero exit status.

`--cohorts` accepts comma-separated keys such as `DS_sz,DS_no_sz,WT_no_sz`.
Filtering changes the sessions used for control matching and group analysis.
`--no-stats` disables statistics.

Photometry is z-scored to each session's baseline endpoint, which is heating
onset for hyperthermia recordings. Transient detection uses a z-score height
threshold of 1 and a raw-dF/F prominence threshold of 0.015 for LC_GCaMP or
0.030 for the other configured datasets, without high-pass detrending.
Hyperthermia control landmarks are assigned from seizure-group mean
temperatures; equivalent offset is equivalent onset plus mean seizure duration.

| Dataset | Event-triggered AUC |
| --- | --- |
| LC_GCaMP | Signed AUC over 100 s after each configured landmark, including offset; shortened to the available interval when a recording ends earlier |
| DG_GRABne | EEC/UEO AUC includes only positive baseline-corrected signal over each seizure's duration; controls use the seizure-group mean duration. Offset AUC is signed over 100 s. |

These AUCs use a local baseline from -5 to -1 s relative to the landmark.
LC and DG ictal extrema must remain at or beyond the reported value for
200 ms. Plotting windows are configured separately from integration windows.

Between-cohort statistics are enabled for LC_GCaMP and DG_GRABne. Mean-signal
comparisons use seizure versus no-seizure groups, pooling the no-seizure
genotypes. LC baseline-transient comparisons use DS versus WT. Aligned traces
and heatmaps retain separate genotype/outcome groups.

### Single-Burst Optogenetics and CPP

```powershell
.venv/Scripts/python.exe -m LC_paper.run_opto --data-root data/OPTO_stim --only Hotspot_GANE/CPP_treatment
```

Available experiment names:
- `Hotspot_GANE/CPP_treatment`
- `Hotspot_GANE/DG_local_release`
- `Hotspot_GANE/DG_local_release _Atomoxetine` (contains a space; quote it)

CPP uses non-z-scored dF/F; the other configured roots use baseline-z-scored
dF/F. Extrema and signed AUC are calculated over 0-40 s after stimulation
onset, following subtraction of the mean from -5 to -1 s. CPP traces display
-40 to +40 s. `--no-paired` skips paired statistics.

Paired comparisons use a mixed model with mouse and session-pair random
intercepts. The reference environment uses R 4.6.0, `lmerTest` 3.2.1,
`emmeans` 2.0.3, and `lme4` 2.0.1. Install R and its `lmerTest` and
`emmeans` packages (`lme4` is installed as a dependency):

```r
install.packages(c("lmerTest", "emmeans"), repos="https://cloud.r-project.org")
```

`Rscript` is located through PATH, `R_HOME`, or the Windows R installation.
Missing R packages may be installed automatically on first use. If R is
unavailable or the model fit is unusable, the analysis issues a warning and
uses a paired t-test or Wilcoxon signed-rank test on matched sessions. This
fallback does not model within-mouse correlation across sessions. The `Test`
column identifies the method used.

Extrema and AUC also receive paired ratio tests: a t-test on log ratios for
sign-consistent values, or a paired t-test on signed-log values when zeros
or mixed signs are present.

### Repeated Stimulation Before/After Yohimbine

```powershell
.venv/Scripts/python.exe -m LC_paper.run_grab_opto --data-root data/OPTO_stim/GRAB_opto
```

The first 10 trains are analyzed using non-z-scored IRLS dF/F. The mean over
the complete pre-first-train interval is subtracted from each recording.
Peak and signed AUC are calculated over [0, 10 s) after each train. Trains
are technical replicates; paired tests use mouse-level summaries.
Outputs include tables, figures, and
`statistics/GRAB_opto_paired_statistics.csv`. `--rebuild-cache` forces processing.

Mouse IDs and stimulation-protocol checks are defined in the runner.

### Dual-Color Green/Red NE

```powershell
.venv/Scripts/python.exe -m LC_paper.run_green_red_ne --data-root data/DG_greenANDred_NE
```

Each channel is independently processed with a fourth-order 10 Hz low-pass
filter, biexponential baseline correction, and pre-heating z-scoring. The
runner produces four event-aligned summary plots. `--rebuild-cache` forces
reprocessing. Group definitions are specified in the runner.

## Statistical Tests

Independent two-group comparisons use Welch's t-test or Mann-Whitney U.
Mann-Whitney U is selected if either group has absolute skewness above 2,
absolute excess kurtosis above 7, or Shapiro-Wilk `p < 0.05`. Groups with
fewer than three observations are omitted from these distribution checks.

Paired t-test/Wilcoxon comparisons apply the same checks to paired differences.
The mixed-model and ratio analyses described above use their own methods.
Reported p-values are unadjusted for multiple comparisons. Trace shading shows
pointwise mean +/- SEM.

## Input Data and Metadata

Recording directories follow this layout:

```text
dataset/
  data_log.xlsx
  cohort/
    mouse_session/
      recording.ppd
      dated_open_ephys_recording/
        Record Node .../
          experiment1/recording1/
            structure.oebin
            continuous/
            events/
```

A flat layout with mouse/session folders directly under the dataset root is
also supported. Retain Open Ephys metadata, samples, timestamps, sample numbers,
and TTL events. Recordings within a mouse directory are matched by acquisition
date. Place one `.xlsx` data log in the dataset root; all worksheets are read.

The log supplies:
- Mouse ID, acquisition date, cohort/genotype, and session number.
- Seizure status, inclusion flags, and exclusion reasons.
- ECoG channel and any non-default temperature/EMG/stimulation channels.
- Heating onset and EEC, UEO, and OFF times for seizure experiments.
- Stimulation channel for optogenetic experiments without heating.
- Optional baseline/transient inclusion flags, prominence, and baseline interval.

Use the LC example workbook as a column template. Landmark times are seconds
on the aligned recording timebase. EEC denotes earliest electrographic change,
UEO unequivocal electrographic onset, and OFF seizure offset.

For a new experiment, configure channel assignments, temperature calibration,
event definitions, cohorts, and analysis windows in the corresponding runner.

## Caches and Outputs

Default output and cache directories are under `LC_paper/`.
Hyperthermia and single-burst optogenetic caches are indexed by dataset,
session, and strategy. Input and parameter changes do not automatically
invalidate these caches; use a new cache directory after either changes.
Use a separate output directory to retain results from earlier runs.

Hyperthermia/opto output contains `quality check`, `parameter detection`,
`plots and values_means`, and `plots and values_transients` below
`preprocessing_IRLS`. Statistics are stored alongside the measurements.
Repeated stimulation and dual-color runners have the layouts described above.

## Auxiliary Scripts

`run_threshold_sweep.py` compares transient thresholds and preprocessing
strategies. Set input paths in `DATASETS`. Options include `--strategies`,
`--refresh-temp`, `--clear-cache` to remove the sweep cache, and `--test`
to process one session per dataset.

`run_mutant_supplemental.py` generates mutant-sensor plots with strategies A
and B. Input and output paths are configured as module constants.
