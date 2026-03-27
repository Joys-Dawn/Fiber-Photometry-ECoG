"""
Group-level summary plots.

Each function receives one or more cohort result objects (keyed by cohort name),
plots with cohort colors, saves PNG + companion CSV, and returns the saved path.
"""

import csv
import os
from typing import Dict

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.cohort_characteristics import CohortGroupResult as CohortResult
from ..analysis.baseline_transients import BaselineTransientGroupResult
from ..analysis.preictal_mean import PreictalMeanGroupResult
from ..analysis.preictal_transients import PreictalTransientGroupResult
from ..analysis.ictal_mean import IctalMeanGroupResult
from ..analysis.ictal_transients import IctalTransientGroupResult
from ..analysis.postictal import PostictalGroupResult
from ..analysis.spike_triggered import SpikeTriggeredGroupResult
from .colors import COHORT_COLORS


def _color(cohort: str) -> str:
    return COHORT_COLORS.get(cohort, "gray")


# ---- helpers ----

def _save_csv(path: str, header: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _bar_scatter(ax, cohorts: Dict[str, tuple], ylabel: str) -> None:
    """Bar + individual scatter overlay.

    cohorts: {name: (per_session_values, mean, sem)}
    """
    for i, (name, (vals, mean, sem)) in enumerate(cohorts.items()):
        color = _color(name)
        ax.bar(i, mean, yerr=sem, color=color, alpha=0.5, capsize=4, width=0.6)
        ax.scatter(
            np.full(len(vals), i) + np.random.default_rng(42).uniform(-0.15, 0.15, len(vals)),
            vals, color=color, s=20, zorder=3,
        )
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts.keys())
    ax.set_ylabel(ylabel)
    ax.spines[["right", "top"]].set_visible(False)


def _line_sem(ax, x, mean, sem, color, label) -> None:
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25)


# ---- 1. Cohort characteristics ----

def plot_cohort_characteristics(
    results: Dict[str, CohortResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    baseline = {}
    threshold = {}
    csv_rows = []
    for name, r in results.items():
        bl_vals = [s.baseline_temp for s in r.session_results if s.baseline_temp is not None]
        th_vals = [s.seizure_threshold_temp for s in r.session_results if s.seizure_threshold_temp is not None]
        baseline[name] = (bl_vals, r.baseline_temp_mean, r.baseline_temp_sem)
        threshold[name] = (th_vals, r.seizure_threshold_mean, r.seizure_threshold_sem)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.baseline_temp, s.seizure_threshold_temp])

    _bar_scatter(axes[0], baseline, "Baseline Temp (°C)")
    axes[0].set_title("Baseline Temperature")
    _bar_scatter(axes[1], threshold, "Seizure Threshold (°C)")
    axes[1].set_title("Seizure Threshold")

    fig.tight_layout()
    path = os.path.join(output_dir, "cohort_characteristics.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "cohort_characteristics.csv"),
        ["cohort", "mouse_id", "baseline_temp", "seizure_threshold_temp"],
        csv_rows,
    )
    return path


# ---- 2. Baseline transients ----

def plot_baseline_transients(
    results: Dict[str, BaselineTransientGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    freq, amp, hw = {}, {}, {}
    csv_rows = []
    for name, r in results.items():
        f_vals = [s.frequency_hz for s in r.session_results]
        a_vals = [s.mean_amplitude for s in r.session_results]
        h_vals = [s.mean_half_width_s for s in r.session_results]
        freq[name] = (f_vals, r.frequency_mean, r.frequency_sem)
        amp[name] = (a_vals, r.amplitude_mean, r.amplitude_sem)
        hw[name] = (h_vals, r.half_width_mean, r.half_width_sem)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.frequency_hz, s.mean_amplitude, s.mean_half_width_s])

    _bar_scatter(axes[0], freq, "Frequency (Hz)")
    axes[0].set_title("Transient Frequency")
    _bar_scatter(axes[1], amp, "Amplitude (ΔF/F)")
    axes[1].set_title("Transient Amplitude")
    _bar_scatter(axes[2], hw, "Half-width (s)")
    axes[2].set_title("Transient Half-width")

    fig.tight_layout()
    path = os.path.join(output_dir, "baseline_transients.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "baseline_transients.csv"),
        ["cohort", "mouse_id", "frequency_hz", "mean_amplitude", "mean_half_width_s"],
        csv_rows,
    )
    return path


# ---- 3. Pre-ictal mean signal ----

def plot_preictal_mean(
    results: Dict[str, PreictalMeanGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time-binned bar plot
    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.baseline_mean, s.early_heat_mean,
                             s.late_heat_mean, s.end_late_heat_mean])

    ax = axes[0]
    labels = ["Baseline", "Early Heat", "Late Heat", "End Late Heat"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(results), 1)
    for i, (name, r) in enumerate(results.items()):
        means = [r.baseline_mean, r.early_heat_mean, r.late_heat_mean, r.end_late_heat_mean]
        sems = [r.baseline_sem, r.early_heat_sem, r.late_heat_sem, r.end_late_heat_sem]
        offset = (i - (len(results) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=sems, color=_color(name), alpha=0.7,
               capsize=3, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean z-ΔF/F")
    ax.set_title("Time-binned Mean Signal")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Temperature-binned line plot
    ax = axes[1]
    for name, r in results.items():
        _line_sem(ax, r.temp_bin_centers, r.temp_bin_group_mean,
                  r.temp_bin_group_sem, _color(name), name)
    ax.set_xlabel("Temp relative to seizure onset (°C)")
    ax.set_ylabel("Mean z-ΔF/F")
    ax.set_title("Temp-binned Mean Signal")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "preictal_mean.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "preictal_mean.csv"),
        ["cohort", "mouse_id", "baseline_mean", "early_heat_mean",
         "late_heat_mean", "end_late_heat_mean"],
        csv_rows,
    )
    return path


# ---- 4. Pre-ictal transients ----

def plot_preictal_transients(
    results: Dict[str, PreictalTransientGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: moving averages (time)
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[0, 0], r.moving_avg_times, r.frequency_mean, r.frequency_sem, c, name)
        _line_sem(axes[0, 1], r.moving_avg_times, r.amplitude_mean, r.amplitude_sem, c, name)
        _line_sem(axes[0, 2], r.moving_avg_times, r.half_width_mean, r.half_width_sem, c, name)

    axes[0, 0].set_ylabel("Frequency (Hz)")
    axes[0, 0].set_title("Moving Avg — Frequency")
    axes[0, 1].set_ylabel("Amplitude (ΔF/F)")
    axes[0, 1].set_title("Moving Avg — Amplitude")
    axes[0, 2].set_ylabel("Half-width (s)")
    axes[0, 2].set_title("Moving Avg — Half-width")
    for ax in axes[0]:
        ax.set_xlabel("Time from heating start (s)")
        ax.legend(fontsize=7)
        ax.spines[["right", "top"]].set_visible(False)

    # Row 2: temperature-binned
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[1, 0], r.temp_bin_centers, r.temp_freq_mean, r.temp_freq_sem, c, name)
        _line_sem(axes[1, 1], r.temp_bin_centers, r.temp_amp_mean, r.temp_amp_sem, c, name)
        _line_sem(axes[1, 2], r.temp_bin_centers, r.temp_hw_mean, r.temp_hw_sem, c, name)

    axes[1, 0].set_ylabel("Frequency (Hz)")
    axes[1, 0].set_title("Temp-binned — Frequency")
    axes[1, 1].set_ylabel("Amplitude (ΔF/F)")
    axes[1, 1].set_title("Temp-binned — Amplitude")
    axes[1, 2].set_ylabel("Half-width (s)")
    axes[1, 2].set_title("Temp-binned — Half-width")
    for ax in axes[1]:
        ax.set_xlabel("Temp rel. to seizure onset (°C)")
        ax.legend(fontsize=7)
        ax.spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "preictal_transients.png")
    fig.savefig(path, dpi=150)
    plt.close("all")
    return path


# ---- 5. Ictal mean signal ----

def plot_ictal_mean(
    results: Dict[str, IctalMeanGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # Count triggered average landmarks across all cohorts
    landmark_names = set()
    for r in results.values():
        landmark_names.update(r.triggered_averages.keys())
    landmark_names = sorted(landmark_names)

    n_ta = len(landmark_names)
    fig, axes = plt.subplots(1 + 1, max(n_ta, 1), figsize=(5 * max(n_ta, 1), 10),
                             squeeze=False)

    # Row 0: scalar bar plots (seizure mean, delta)
    scalars_seizure = {}
    scalars_delta = {}
    csv_rows = []
    for name, r in results.items():
        sz_vals = [s.seizure_mean for s in r.session_results]
        d_vals = [s.delta_preictal_ictal for s in r.session_results]
        scalars_seizure[name] = (sz_vals, r.seizure_mean, r.seizure_sem)
        scalars_delta[name] = (d_vals, r.delta_mean, r.delta_sem)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.seizure_mean, s.baseline_mean,
                             s.delta_preictal_ictal])

    if n_ta >= 2:
        _bar_scatter(axes[0, 0], scalars_seizure, "Mean z-ΔF/F")
        axes[0, 0].set_title("Seizure Period Mean")
        _bar_scatter(axes[0, 1], scalars_delta, "Δ z-ΔF/F")
        axes[0, 1].set_title("Pre-ictal → Ictal Δ")
        for j in range(2, n_ta):
            axes[0, j].axis("off")
    else:
        _bar_scatter(axes[0, 0], scalars_seizure, "Mean z-ΔF/F")
        axes[0, 0].set_title("Seizure Period Mean")

    # Row 1: triggered averages
    for j, lm in enumerate(landmark_names):
        ax = axes[1, j]
        for name, r in results.items():
            if lm in r.triggered_averages:
                ta = r.triggered_averages[lm]
                _line_sem(ax, ta.time_axis, ta.mean_trace, ta.sem_trace, _color(name), name)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("z-ΔF/F")
        ax.set_title(f"Triggered Avg — {lm}")
        ax.legend(fontsize=7)
        ax.spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "ictal_mean.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "ictal_mean.csv"),
        ["cohort", "mouse_id", "seizure_mean", "baseline_mean", "delta_preictal_ictal"],
        csv_rows,
    )
    return path


# ---- 6. Ictal transients ----

def plot_ictal_transients(
    results: Dict[str, IctalTransientGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PSTH bar chart
    ax = axes[0, 0]
    width = 0.8 / max(len(results), 1)
    for i, (name, r) in enumerate(results.items()):
        x = r.psth_bin_centers
        offset = (i - (len(results) - 1) / 2) * width
        ax.bar(x + offset, r.psth_mean, width, yerr=r.psth_sem,
               color=_color(name), alpha=0.7, capsize=2, label=name)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time rel. to UEO (s)")
    ax.set_ylabel("Transient Count")
    ax.set_title("PSTH")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Moving averages around UEO
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[0, 1], r.moving_avg_times, r.freq_mean, r.freq_sem, c, name)
        _line_sem(axes[1, 0], r.moving_avg_times, r.amp_mean, r.amp_sem, c, name)
        _line_sem(axes[1, 1], r.moving_avg_times, r.hw_mean, r.hw_sem, c, name)

    axes[0, 1].set_title("Moving Avg — Frequency")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    axes[1, 0].set_title("Moving Avg — Amplitude")
    axes[1, 0].set_ylabel("Amplitude (ΔF/F)")
    axes[1, 1].set_title("Moving Avg — Half-width")
    axes[1, 1].set_ylabel("Half-width (s)")
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time rel. to UEO (s)")
        ax.legend(fontsize=7)
        ax.spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "ictal_transients.png")
    fig.savefig(path, dpi=150)
    plt.close("all")
    return path


# ---- 7. Postictal recovery ----

def plot_postictal(
    results: Dict[str, PostictalGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    csv_rows = []

    # Cooling curve (temp-binned)
    ax = axes[0, 0]
    for name, r in results.items():
        _line_sem(ax, r.cooling_bin_centers, r.cooling_group_mean,
                  r.cooling_group_sem, _color(name), name)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.final_time, s.final_temp, s.final_mean_dff])
    ax.set_xlabel("Temp rel. to seizure onset (°C)")
    ax.set_ylabel("Mean z-ΔF/F")
    ax.set_title("Cooling Curve")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Scatter: final time vs final temp
    ax = axes[0, 1]
    for name, r in results.items():
        ax.scatter(r.final_times, r.final_temps, color=_color(name), label=name, s=30)
    ax.set_xlabel("Final Recording Time (s)")
    ax.set_ylabel("Final Temp (°C)")
    ax.set_title("Final Time vs Temp")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Scatter: final time vs final dff
    ax = axes[1, 0]
    for name, r in results.items():
        ax.scatter(r.final_times, r.final_dffs, color=_color(name), label=name, s=30)
    ax.set_xlabel("Final Recording Time (s)")
    ax.set_ylabel("Final Mean ΔF/F")
    ax.set_title("Final Time vs ΔF/F")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Scatter: final temp vs final dff
    ax = axes[1, 1]
    for name, r in results.items():
        ax.scatter(r.final_temps, r.final_dffs, color=_color(name), label=name, s=30)
    ax.set_xlabel("Final Temp (°C)")
    ax.set_ylabel("Final Mean ΔF/F")
    ax.set_title("Final Temp vs ΔF/F")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "postictal.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "postictal.csv"),
        ["cohort", "mouse_id", "final_time", "final_temp", "final_mean_dff"],
        csv_rows,
    )
    return path


# ---- 8. Spike-triggered averages ----

def plot_spike_triggered(
    results: Dict[str, SpikeTriggeredGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    csv_rows = []

    # Top-left: Photometry STA (mean +/- 95% CI)
    ax = axes[0, 0]
    for name, r in results.items():
        c = _color(name)
        ci95 = 1.96 * r.group_sem
        ax.plot(r.time_axis, r.group_mean, color=c, label=name)
        ax.fill_between(r.time_axis, r.group_mean - ci95, r.group_mean + ci95,
                        color=c, alpha=0.25)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.n_spikes, s.auc])
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time rel. to spike (s)")
    ax.set_ylabel("z-ΔF/F")
    ax.set_title("Photometry STA (mean +/- 95% CI)")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Top-right: EEG STA (mean +/- 95% CI)
    ax = axes[0, 1]
    for name, r in results.items():
        c = _color(name)
        ci95 = 1.96 * r.eeg_group_sem
        ax.plot(r.time_axis, r.eeg_group_mean, color=c, label=name)
        ax.fill_between(r.time_axis, r.eeg_group_mean - ci95,
                        r.eeg_group_mean + ci95, color=c, alpha=0.25)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time rel. to spike (s)")
    ax.set_ylabel("ECoG (polarity-aligned)")
    ax.set_title("EEG STA (mean +/- 95% CI)")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Bottom-left: Photometry AUC bar plot
    auc_data = {}
    for name, r in results.items():
        auc_vals = [s.auc for s in r.session_results]
        auc_sem = np.std(auc_vals) / np.sqrt(len(auc_vals)) if len(auc_vals) > 1 else 0.0
        auc_data[name] = (auc_vals, r.group_auc, auc_sem)
    _bar_scatter(axes[1, 0], auc_data, "AUC")
    axes[1, 0].set_title("Spike-Triggered AUC")

    # Bottom-right: empty (reserved)
    axes[1, 1].axis("off")

    fig.tight_layout()
    path = os.path.join(output_dir, "spike_triggered.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "spike_triggered.csv"),
        ["cohort", "mouse_id", "n_spikes", "auc"],
        csv_rows,
    )
    return path
