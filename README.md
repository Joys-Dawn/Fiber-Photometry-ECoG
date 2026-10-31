[LC_GCaMP example data and demo guide](LC_paper/docs/REVIEWER_GUIDE.md)

# Fiber Photometry & ECoG Analysis

A Python desktop application for analyzing simultaneous fiber photometry,
electrocorticography (ECoG), and temperature recordings. It synchronizes
recordings, preprocesses signals, detects photometry transients and interictal
spikes, and extracts measurements around seizure events.

The graphical interface is under development. Paper-specific analyses and the
LC data demo use the command-line runners in [LC_paper/](LC_paper/).

## Installation

Tested with Python 3.13.9 on Windows 11 Pro. Python dependencies and their
versions are listed in [requirements.txt](requirements.txt). The application
uses Tkinter for its interface.

From the repository directory, run in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe run.py
```

## Data Organization

Select a parent experiment folder containing an Excel data log and recordings:

```text
experiment/
  data_log.xlsx
  Scn1a_seizure/
    mouse_session/
      photometry.ppd
      dated_open_ephys_recording/
        Record Node .../
          experiment1/recording1/
            structure.oebin
            continuous/
            events/
  Scn1a_no_seizure/
  WT/
```

The scanner discovers recording pairs automatically. A flat layout with
mouse/session folders directly under the experiment folder is also supported.
Keep the Open Ephys metadata, continuous samples, timestamps, and TTL events.

The Excel log supplies mouse IDs, acquisition dates, genotype/cohort, session
numbers, seizure status, inclusion flags, channel assignments, heating onset,
and seizure landmarks. Optional columns specify transient thresholds and
baseline intervals. All worksheets are read. Sessions are matched by mouse ID
and recording date.

## Workflow

The application has four tabs:

1. **Data Loading & Cohort Management:** scan the experiment folder, review
   session metadata and channel assignments, then load raw recordings or saved
   preprocessed sessions. Select the output directory here.
2. **Preprocessing:** select one or more photometry strategies and run
   preprocessing. Strategy C is selected by default. Review the ECoG trace and
   mark EEC, UEO, behavioral onset, and seizure offset.
3. **Extraction:** select time- or temperature-based control matching and the
   cohorts to plot, then run session and group analyses.
4. **Statistics:** select cohorts and run comparisons on the extracted
   measurements. Two groups use Welch's t-test or Mann-Whitney U; three or more
   use ANOVA or Kruskal-Wallis with pairwise comparisons.

EEC denotes earliest electrographic change, UEO unequivocal electrographic
onset, and OFF seizure offset. Controls receive equivalent event landmarks
through the selected matching method.

## Preprocessing

**Synchronization:** TTL pulses establish a linear clock-offset and drift
correction. Photometry is interpolated onto the common timebase. Higher-rate
ECoG is downsampled to 1 kHz. Standard recordings are trimmed to their overlap;
optogenetic recordings retain the full photometry interval and use NaN padding
where Open Ephys data are unavailable.

**ECoG:** fourth-order 1-70 Hz Butterworth bandpass and a 60 Hz notch filter
(Q = 30), applied without phase shift. Spike detection uses baseline-derived
thresholds. The shared pipeline excludes samples from 30 s before seizure or
equivalent onset through the end of the recording.

**Temperature:** voltage-to-temperature calibration is selected from recording
metadata and acquisition date, followed by a 300-sample moving average.
Baseline, peak, and terminal temperatures are extracted.

**Photometry:** four preprocessing strategies are available:

| Strategy | Processing |
| --- | --- |
| A | Gaussian smoothing of both channels, then isosbestic subtraction and division |
| B | 10 Hz low-pass filtering, biexponential bleaching correction, and ordinary least-squares isosbestic regression |
| C (default) | 3 Hz low-pass filtering and IRLS isosbestic regression using Tukey bisquare weights; no separate biexponential correction |
| D | 10 Hz low-pass filtering and biexponential correction of the signal channel without an isosbestic control |

The shared pipeline z-scores dF/F from recording start to the session's
baseline endpoint.
High-pass filtering/detrending is disabled by default. Strategy A detects
transients by prominence on the z-scored signal; B, C, and D apply a z-score
height threshold followed by a prominence threshold on dF/F. Thresholds come
from the strategy configuration or session metadata.

## Measurements and Outputs

Analyses cover cohort characteristics, baseline transients, preictal mean
signal and transients, ictal mean signal and transients, postictal recovery,
and spike-triggered photometry responses. Outputs include session measurements,
event-aligned mean/SEM traces, heatmaps, scatterplots, and statistical tables.

Within the selected output directory, each preprocessing strategy has separate
folders for quality checks, parameter detection, mean-signal plots and values,
and transient plots and values. Saved session files can be reloaded without
repeating raw-data preprocessing.

Paper-specific cohort definitions, AUC windows, and figure exports are documented
in [LC_paper/docs/HEADLESS_RUNNERS.md](LC_paper/docs/HEADLESS_RUNNERS.md); they are
not all defaults of the graphical application.

## Repository Structure

| Path | Contents |
| --- | --- |
| `run.py` | Application entry point |
| `fiber_photometry_ecog/app.py` | Graphical interface |
| `fiber_photometry_ecog/data_loading/` | Readers, metadata import, and synchronization |
| `fiber_photometry_ecog/preprocessing/` | Signal processing and event detection |
| `fiber_photometry_ecog/analysis/` | Session and group measurements |
| `fiber_photometry_ecog/pairing/` | Control landmark matching |
| `fiber_photometry_ecog/statistics/` | Statistical comparisons |
| `fiber_photometry_ecog/visualization/` | Quality-control and analysis figures |
| `fiber_photometry_ecog/tests/` | Shared application and analysis tests |
| `LC_paper/` | Paper runners, figure scripts, guides, data assets, and results |

## Tests

```powershell
.venv/Scripts/python.exe -m pip install pytest==9.0.0
.venv/Scripts/python.exe -m pytest fiber_photometry_ecog/tests LC_paper/tests -q
```

Some integration tests require local recording files.

## License

The software is distributed under [GNU GPL version 3](LICENSE).
