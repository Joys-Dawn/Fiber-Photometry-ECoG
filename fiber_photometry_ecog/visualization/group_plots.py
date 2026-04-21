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


_COHORT_ORDER = ["seizure", "failed_seizure", "wt"]


def _ordered(results: Dict) -> Dict:
    """Return results dict ordered seizure | failed_seizure | wt, with any extras appended."""
    keys = list(results.keys())
    ordered = [k for k in _COHORT_ORDER if k in keys]
    ordered += [k for k in keys if k not in _COHORT_ORDER]
    return {k: results[k] for k in ordered}


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


def _scatter_mean_sem(ax, cohorts: Dict[str, tuple], ylabel: str) -> None:
    """Pure scatter plot: jittered per-session dots, mean tick, SEM whiskers.

    Per spec text: "scatter plot w/ means +/- SEM".
    cohorts: {name: (per_session_values, mean, sem)}
    """
    rng = np.random.default_rng(42)
    for i, (name, (vals, mean, sem)) in enumerate(cohorts.items()):
        color = _color(name)
        if len(vals) > 0:
            x = np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(x, vals, color=color, s=25, zorder=3, alpha=0.8,
                       edgecolor="black", linewidth=0.4)
        ax.errorbar(i, mean, yerr=sem, fmt="_", color=color, markersize=28,
                    capsize=6, elinewidth=2, markeredgewidth=2, zorder=4)
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
    """Row 1: baseline temperature (scatter w/ mean+/-SEM, all cohorts).
    Row 2: seizure threshold by session # with per-mouse lines (seizure cohort only).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    baseline = {}
    csv_rows = []
    for name, r in results.items():
        bl_vals = [s.baseline_temp for s in r.session_results if s.baseline_temp is not None]
        baseline[name] = (bl_vals, r.baseline_temp_mean, r.baseline_temp_sem)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.baseline_temp, s.seizure_threshold_temp])

    _scatter_mean_sem(axes[0], baseline, "Baseline Temp (°C)")
    axes[0].set_title("Baseline Temperature")

    # Seizure threshold: Scn1a-only per-mouse lines across sessions
    ax = axes[1]
    seizure_key = "seizure" if "seizure" in results else next(iter(results.keys()))
    sz_r = results[seizure_key]
    by_mouse: Dict[str, list] = {}
    for s in sz_r.session_results:
        if s.seizure_threshold_temp is None:
            continue
        by_mouse.setdefault(s.mouse_id, []).append((s.heating_session, s.seizure_threshold_temp))
    color = _color(seizure_key)
    all_by_session: Dict[int, list] = {}
    for mouse_id, pts in by_mouse.items():
        pts.sort(key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=color, alpha=0.5, markersize=5, linewidth=1)
        for x, y in pts:
            all_by_session.setdefault(x, []).append(y)
    for x, ys in sorted(all_by_session.items()):
        mean = float(np.mean(ys))
        sem = float(np.std(ys, ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0
        ax.errorbar(x, mean, yerr=sem, fmt="_", color=color,
                    markersize=32, capsize=7, elinewidth=2.5,
                    markeredgewidth=2.5, zorder=5)
    ax.set_xlabel("Session #")
    ax.set_ylabel("Seizure Threshold (°C)")
    ax.set_title(f"Seizure Threshold ({seizure_key} only)")
    if all_by_session:
        ax.set_xticks(sorted(all_by_session.keys()))
    ax.spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "cohort_characteristics.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "cohort_characteristics.csv"),
        ["cohort", "mouse_id", "heating_session", "baseline_temp", "seizure_threshold_temp"],
        csv_rows,
    )
    return path


# ---- 2. Baseline transients ----

def plot_baseline_transients(
    results: Dict[str, BaselineTransientGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
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
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.frequency_hz, s.mean_amplitude, s.mean_half_width_s])

    _scatter_mean_sem(axes[0], freq, "Frequency (Hz)")
    axes[0].set_title("Transient Frequency")
    _scatter_mean_sem(axes[1], amp, "Amplitude (ΔF/F)")
    axes[1].set_title("Transient Amplitude")
    _scatter_mean_sem(axes[2], hw, "Half-width (s)")
    axes[2].set_title("Transient Half-width")

    fig.tight_layout()
    path = os.path.join(output_dir, "baseline_transients.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "baseline_transients.csv"),
        ["cohort", "mouse_id", "heating_session",
         "frequency_hz", "mean_amplitude", "mean_half_width_s"],
        csv_rows,
    )
    return path


# ---- 3. Pre-ictal mean signal ----

def plot_preictal_mean(
    results: Dict[str, PreictalMeanGroupResult],
    output_dir: str,
) -> str:
    """Row 8: time-binned bars {early heat, late heat} (baseline omitted —
        z-scored to baseline so it is identically 0 by construction).
    Row 9: end-of-late-heat scatter (separate plot).
    Row 10: temp-binned signal during heating, rel to seizure onset temp.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.baseline_mean, s.early_heat_mean,
                             s.late_heat_mean, s.end_late_heat_mean])

    # Row 8: time-binned bars (early heat + late heat; baseline = 0 by definition)
    ax = axes[0]
    labels = ["Early Heat", "Late Heat"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(results), 1)
    for i, (name, r) in enumerate(results.items()):
        means = [r.early_heat_mean, r.late_heat_mean]
        sems = [r.early_heat_sem, r.late_heat_sem]
        offset = (i - (len(results) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=sems, color=_color(name), alpha=0.7,
               capsize=3, label=name)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean z-ΔF/F (vs baseline)")
    ax.set_title("Time-binned Mean Signal")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Row 9: end of late heat scatter
    elh = {}
    for name, r in results.items():
        vals = [s.end_late_heat_mean for s in r.session_results]
        elh[name] = (vals, r.end_late_heat_mean, r.end_late_heat_sem)
    _scatter_mean_sem(axes[1], elh, "Mean z-ΔF/F")
    axes[1].set_title("End of Late Heat")

    # Row 10: temp-binned during heating, rel to seizure onset temp
    ax = axes[2]
    for name, r in results.items():
        _line_sem(ax, r.temp_bin_centers, r.temp_bin_group_mean,
                  r.temp_bin_group_sem, _color(name), name)
    ax.set_xlabel("Temp rel. to seizure onset (°C)")
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
        ["cohort", "mouse_id", "heating_session",
         "baseline_mean", "early_heat_mean",
         "late_heat_mean", "end_late_heat_mean"],
        csv_rows,
    )
    return path


# ---- 4. Pre-ictal transients ----

def plot_preictal_transients(
    results: Dict[str, PreictalTransientGroupResult],
    output_dir: str,
) -> str:
    """Spec rows 13-15: three sets of curves x three properties.
      Row 0: temp moving average (spec row 13)
      Row 1: temp-binned (spec row 14)
      Row 2: time moving average (spec row 15)
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))

    # Row 0: TEMP moving averages (spec row 13)
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[0, 0], r.temp_ma_centers, r.temp_ma_freq_mean, r.temp_ma_freq_sem, c, name)
        _line_sem(axes[0, 1], r.temp_ma_centers, r.temp_ma_amp_mean, r.temp_ma_amp_sem, c, name)
        _line_sem(axes[0, 2], r.temp_ma_centers, r.temp_ma_hw_mean, r.temp_ma_hw_sem, c, name)
    axes[0, 0].set_ylabel("Frequency (Hz)")
    axes[0, 0].set_title("Temp Moving Avg — Frequency")
    axes[0, 1].set_ylabel("Amplitude (ΔF/F)")
    axes[0, 1].set_title("Temp Moving Avg — Amplitude")
    axes[0, 2].set_ylabel("Half-width (s)")
    axes[0, 2].set_title("Temp Moving Avg — Half-width")
    for ax in axes[0]:
        ax.set_xlabel("Temp rel. to seizure onset (°C)")
        ax.legend(fontsize=7)
        ax.spines[["right", "top"]].set_visible(False)

    # Row 1: temperature-binned (discrete bins, spec row 14)
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[1, 0], r.temp_bin_centers, r.temp_freq_mean, r.temp_freq_sem, c, name)
        _line_sem(axes[1, 1], r.temp_bin_centers, r.temp_amp_mean, r.temp_amp_sem, c, name)
        _line_sem(axes[1, 2], r.temp_bin_centers, r.temp_hw_mean, r.temp_hw_sem, c, name)
    axes[1, 0].set_ylabel("Frequency (count)")
    axes[1, 0].set_title("Temp-binned (1°C) — Frequency")
    axes[1, 1].set_ylabel("Amplitude (ΔF/F)")
    axes[1, 1].set_title("Temp-binned (1°C) — Amplitude")
    axes[1, 2].set_ylabel("Half-width (s)")
    axes[1, 2].set_title("Temp-binned (1°C) — Half-width")
    for ax in axes[1]:
        ax.set_xlabel("Temp rel. to seizure onset (°C)")
        ax.legend(fontsize=7)
        ax.spines[["right", "top"]].set_visible(False)

    # Row 2: TIME moving averages (spec row 15)
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[2, 0], r.moving_avg_times, r.frequency_mean, r.frequency_sem, c, name)
        _line_sem(axes[2, 1], r.moving_avg_times, r.amplitude_mean, r.amplitude_sem, c, name)
        _line_sem(axes[2, 2], r.moving_avg_times, r.half_width_mean, r.half_width_sem, c, name)
    axes[2, 0].set_ylabel("Frequency (Hz)")
    axes[2, 0].set_title("Time Moving Avg — Frequency")
    axes[2, 1].set_ylabel("Amplitude (ΔF/F)")
    axes[2, 1].set_title("Time Moving Avg — Amplitude")
    axes[2, 2].set_ylabel("Half-width (s)")
    axes[2, 2].set_title("Time Moving Avg — Half-width")
    for ax in axes[2]:
        ax.set_xlabel("Time from heating start (s)")
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
    """Row 18: seizure mean scatter. Row 19: preictal->ictal Δ scatter.
    Row 20: per-landmark triggered averages + AUC scatter.

    Layout: left column = scatters (seizure mean, delta), right side = vertical
    stack of (triggered average trace + AUC scatter) per landmark.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

    landmark_names = set()
    for r in results.values():
        landmark_names.update(r.triggered_averages.keys())
    # Preferred order per spec row 20
    preferred = ["EEC", "UEO", "behavioral_onset", "OFF", "max_temp"]
    landmark_names = [lm for lm in preferred if lm in landmark_names] + \
                     sorted(lm for lm in landmark_names if lm not in preferred)
    # Skip landmarks with no real data from any cohort (e.g. behavioral onset
    # when none of the sessions had it scored).
    landmark_names = [
        lm for lm in landmark_names
        if any(len(r.triggered_averages[lm].per_session_traces) > 0
               for r in results.values() if lm in r.triggered_averages)
    ]
    n_lm = len(landmark_names)

    # Layout: 2 + n_lm rows x 2 cols. Rows 0-1: scalar scatters (col 0)
    # + legend/spacer (col 1). Rows 2..: triggered avg (col 0) + AUC (col 1).
    n_rows = max(n_lm, 2)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.2 * n_rows), squeeze=False)

    # Left column (col 0-1): scatters stacked
    scalars_seizure = {}
    scalars_delta = {}
    csv_rows = []
    for name, r in results.items():
        sz_vals = [s.seizure_mean for s in r.session_results]
        d_vals = [s.delta_preictal_ictal for s in r.session_results]
        scalars_seizure[name] = (sz_vals, r.seizure_mean, r.seizure_sem)
        scalars_delta[name] = (d_vals, r.delta_mean, r.delta_sem)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.seizure_mean, s.baseline_mean,
                             s.delta_preictal_ictal])

    _scatter_mean_sem(axes[0, 0], scalars_seizure, "Mean z-ΔF/F")
    axes[0, 0].set_title("Seizure Period Mean")
    _scatter_mean_sem(axes[1, 0], scalars_delta, "Δ z-ΔF/F")
    axes[1, 0].set_title("Pre-ictal → Ictal Δ")
    # Hide unused cells in col 0
    for i in range(2, n_rows):
        axes[i, 0].axis("off")

    # Middle column (col 1): triggered averages per landmark
    # Right column (col 2): AUC scatter per landmark
    auc_csv = []
    for i, lm in enumerate(landmark_names):
        ax = axes[i, 1]
        for name, r in results.items():
            if lm in r.triggered_averages:
                ta = r.triggered_averages[lm]
                _line_sem(ax, ta.time_axis, ta.mean_trace, ta.sem_trace,
                          _color(name), name)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("z-ΔF/F")
        ax.set_title(f"Triggered Avg — {lm}")
        ax.legend(fontsize=7)
        ax.spines[["right", "top"]].set_visible(False)

        # AUC scatter per landmark
        ax_auc = axes[i, 2]
        auc_dict = {}
        for name, r in results.items():
            if lm not in r.triggered_averages:
                continue
            ta = r.triggered_averages[lm]
            vals = list(ta.per_session_auc)
            if len(vals) > 1:
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                sem = 0.0
            mean = float(np.mean(vals)) if vals else np.nan
            auc_dict[name] = (vals, mean, sem)
            for v in vals:
                auc_csv.append([name, lm, v])
        if auc_dict:
            _scatter_mean_sem(ax_auc, auc_dict, "AUC")
        ax_auc.set_title(f"AUC — {lm}")

    # Hide unused landmark rows in middle/right
    for i in range(n_lm, n_rows):
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")

    fig.tight_layout()
    path = os.path.join(output_dir, "ictal_mean.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "ictal_mean.csv"),
        ["cohort", "mouse_id", "heating_session",
         "seizure_mean", "baseline_mean", "delta_preictal_ictal"],
        csv_rows,
    )
    _save_csv(
        os.path.join(output_dir, "ictal_mean_auc.csv"),
        ["cohort", "landmark", "auc"],
        auc_csv,
    )
    return path


# ---- 6. Ictal transients ----

def plot_ictal_transients(
    results: Dict[str, IctalTransientGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PSTH bar chart — frequency (Hz) per 10s bin, grouped per cohort
    ax = axes[0, 0]
    n_groups = max(len(results), 1)
    # Bin spacing drives bar width so bars are visible regardless of bin size
    first_centers = next(iter(results.values())).psth_bin_centers
    bin_step = float(first_centers[1] - first_centers[0]) if len(first_centers) > 1 else 1.0
    width = (bin_step * 0.8) / n_groups
    for i, (name, r) in enumerate(results.items()):
        x = r.psth_bin_centers
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, r.psth_mean, width, yerr=r.psth_sem,
               color=_color(name), alpha=0.7, capsize=2, label=name)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time rel. to UEO (s)")
    ax.set_ylabel("Frequency (Hz)")
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
    """Row 26: cooling curve (rel to max temp, binned by 1C) as mean line + SEM band.
    Rows 27-29: pairwise scatters of final time, temp, and mean ΔF/F.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.final_time, s.final_temp, s.final_mean_dff])

    # Row 26: cooling curve — mean line + SEM band per cohort (matches preictal style)
    ax = axes[0, 0]
    for name, r in results.items():
        color = _color(name)
        sem = getattr(r, "cooling_group_sem", None)
        if sem is None:
            sem = np.zeros_like(r.cooling_group_mean)
        _line_sem(ax, r.cooling_bin_centers, r.cooling_group_mean, sem, color, name)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Temp rel. to max (°C)")
    ax.set_ylabel("Mean z-ΔF/F")
    ax.set_title("Cooling Curve")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Row 26: final recording time vs final recording temp
    ax = axes[0, 1]
    for name, r in results.items():
        ax.scatter(r.final_times, r.final_temps, color=_color(name), label=name,
                   s=30, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Final Recording Time (s)")
    ax.set_ylabel("Final Temp (°C)")
    ax.set_title("Final Time vs Temp")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Row 27: final recording time vs final mean ΔF/F
    ax = axes[1, 0]
    for name, r in results.items():
        ax.scatter(r.final_times, r.final_dffs, color=_color(name), label=name,
                   s=30, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Final Recording Time (s)")
    ax.set_ylabel("Final Mean z-ΔF/F")
    ax.set_title("Final Time vs ΔF/F")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    # Row 28: final recording temp vs final mean ΔF/F
    ax = axes[1, 1]
    for name, r in results.items():
        ax.scatter(r.final_temps, r.final_dffs, color=_color(name), label=name,
                   s=30, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Final Temp (°C)")
    ax.set_ylabel("Final Mean z-ΔF/F")
    ax.set_title("Final Temp vs ΔF/F")
    ax.legend(fontsize=7)
    ax.spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "postictal.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "postictal.csv"),
        ["cohort", "mouse_id", "heating_session",
         "final_time", "final_temp", "final_mean_dff"],
        csv_rows,
    )
    return path


# ---- 8. Spike-triggered averages ----

def plot_spike_triggered(
    results: Dict[str, SpikeTriggeredGroupResult],
    output_dir: str,
) -> str:
    """Spec row 17: interictal ECoG spike-triggered photometry average + AUC.

    Saves two PNGs: spike_triggered.png (full ±30s STA + AUC scatter) and
    spike_triggered_zoomed.png (same STA zoomed to ±10s).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session, s.n_spikes, s.auc])

    def _draw_sta(ax, xlim=None):
        ylo, yhi = np.inf, -np.inf
        for name, r in results.items():
            c = _color(name)
            ci95 = 1.96 * r.group_sem
            ax.plot(r.time_axis, r.group_mean, color=c, label=name)
            ax.fill_between(r.time_axis, r.group_mean - ci95, r.group_mean + ci95,
                            color=c, alpha=0.25)
            mask = (np.ones_like(r.time_axis, dtype=bool) if xlim is None
                    else (r.time_axis >= xlim[0]) & (r.time_axis <= xlim[1]))
            if mask.any():
                lo = np.nanmin((r.group_mean - ci95)[mask])
                hi = np.nanmax((r.group_mean + ci95)[mask])
                ylo = min(ylo, lo)
                yhi = max(yhi, hi)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time rel. to spike (s)")
        ax.set_ylabel("z-ΔF/F")
        ax.legend(fontsize=7)
        ax.spines[["right", "top"]].set_visible(False)
        if xlim is not None:
            ax.set_xlim(xlim)
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            pad = 0.05 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)

    # --- Main figure: full STA + AUC scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_sta(axes[0])
    axes[0].set_title("Photometry STA (mean ± 95% CI)")

    auc_data = {}
    for name, r in results.items():
        auc_vals = [s.auc for s in r.session_results]
        auc_sem = float(np.std(auc_vals, ddof=1) / np.sqrt(len(auc_vals))) if len(auc_vals) > 1 else 0.0
        auc_data[name] = (auc_vals, r.group_auc, auc_sem)
    _scatter_mean_sem(axes[1], auc_data, "AUC")
    axes[1].set_title("Spike-Triggered AUC")

    fig.tight_layout()
    path = os.path.join(output_dir, "spike_triggered.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # --- Zoomed companion figure: same STA, matching AUC window ---
    fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(8, 5))
    _draw_sta(ax_zoom, xlim=(-5, 5))
    ax_zoom.set_title("Photometry STA — zoomed (−5s to +5s)")
    fig_zoom.tight_layout()
    fig_zoom.savefig(os.path.join(output_dir, "spike_triggered_zoomed.png"), dpi=150)
    plt.close(fig_zoom)

    _save_csv(
        os.path.join(output_dir, "spike_triggered.csv"),
        ["cohort", "mouse_id", "heating_session", "n_spikes", "auc"],
        csv_rows,
    )
    return path
