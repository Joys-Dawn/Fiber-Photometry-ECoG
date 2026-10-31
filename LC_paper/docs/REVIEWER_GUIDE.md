# LC_GCaMP Example Data

This guide describes how to analyze the 15-session LC_GCaMP dataset with
`LC_paper.run_hyperthermia` and compare the results with the supplied reference outputs.

Cloning or downloading the source repository does not include the recordings
or reference outputs.

## System Requirements

- Tested operating system: Windows 11 Pro, version 10.0.26200, 64-bit.
- Tested interpreter: CPython 3.13.9, 64-bit.
- Python packages: listed in [requirements.txt](../../requirements.txt).
- Hardware: CPU processing, with approximately 4.9 GiB peak resident memory
  in the reference run. No specialized hardware is required.
- Disk space: allow 8 GB for the environment, archives, data, caches, and results.

## Installation

Clone or download the repository, then open PowerShell in its root directory
(the folder containing `run.py` and `LC_paper/`). With Python 3.13 installed, run:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Estimated installation time is approximately 3.2 minutes, with internet access
required for package downloads. Analysis runs offline after installation.

## Dataset and Files

Download these three files from the **Assets** section on
[GitHub Releases](https://github.com/Joys-Dawn/Fiber-Photometry-ECoG/releases):

| File | Contents |
| --- | --- |
| `LC_GCaMP.zip` | All 15 LC sessions and the original data log |
| `LC_GCaMP_sessions.csv` | Session identifiers, cohorts, dates, and landmarks |
| `LC_GCaMP_expected_outputs.zip` | Reference CSVs and representative PNG figures |

The dataset contains 6 DS seizure, 5 DS no-seizure, and 4 WT no-seizure sessions.
The data download is approximately 872 MiB and expands to 1.34 GiB.

## Run the Analysis

Place the three downloaded files in `LC_paper/`. From the repository root, run:

```powershell
.venv/Scripts/python.exe -m zipfile -e LC_paper/LC_GCaMP.zip LC_paper
.venv/Scripts/python.exe -m LC_paper.run_hyperthermia --only LC_GCaMP --data-root LC_paper
```

The extracted layout is:

```text
LC_paper/
  LC_GCaMP/
    LC_ GCaMP_log.xlsx
    scn1a_seizure/
    scn1a_NOseizure/
    wt/
```

If the recordings are elsewhere, set `--data-root` to the folder containing
`LC_GCaMP`.

The first run processes the raw recordings; subsequent runs reuse
`LC_paper/sweep_cache/`. Use a new cache directory after changing input data or
preprocessing parameters. The command above includes all sessions and both
statistics tables.

Estimated runtime for all 15 sessions without an existing preprocessing cache
is 4.8-6.4 minutes.

## Expected Outputs

```text
LC_paper/results/results_hyperthermia_LC_GCaMP/preprocessing_IRLS/
  quality check/
  parameter detection/
  plots and values_means/
  plots and values_transients/
```

Results include onset-aligned mean/SEM traces, heatmaps, session measurements,
group comparisons, and interactive quality-check plots.

The reference-output archive contains all 7 CSVs and 10 representative PNGs,
with paths matching the generated results. Expected CSV row counts are:

| Directory | File | Rows |
| --- | --- | --- |
| `plots and values_means` | `cohort_characteristics_LC_GCaMP.csv` | 15 |
| `plots and values_means` | `values_means_LC_GCaMP.csv` | 15 |
| `plots and values_means` | `values_auc_LC_GCaMP.csv` | 15 |
| `plots and values_means` | `values_auc_spikes_LC_GCaMP.csv` | 14 |
| `plots and values_means` | `stats_LC_GCaMP.csv` | 22 |
| `plots and values_transients` | `baseline_transients.csv` | 14 |
| `plots and values_transients` | `stats_LC_GCaMP.csv` | 3 |

Measurement-specific inclusion flags determine session counts. Baseline-transient
statistics use 10 DS and 4 WT sessions; mean-signal comparisons use 6 seizure
and 9 no-seizure sessions.

## Use with Other Data

See [Command-Line Analysis](HEADLESS_RUNNERS.md) for input layouts, metadata,
experiment settings, and runner commands.

## License and Source

Software is distributed under GNU GPL version 3; see [LICENSE](../../LICENSE). Source:
<https://github.com/Joys-Dawn/Fiber-Photometry-ECoG>.

