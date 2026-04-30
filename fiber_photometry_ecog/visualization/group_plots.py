"""
Group-level summary plots.

Each function receives one or more cohort result objects (keyed by cohort name),
plots with cohort colors, saves PNG + companion CSV, and returns the saved path.
"""

import csv
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.cohort_characteristics import CohortGroupResult as CohortResult
from ..analysis.baseline_transients import BaselineTransientGroupResult
from ..analysis.preictal_mean import PreictalMeanGroupResult
from ..analysis.preictal_transients import PreictalTransientGroupResult
from ..analysis.ictal_mean import IctalMeanGroupResult, TriggeredAverage
from ..analysis.ictal_transients import IctalTransientGroupResult
from ..analysis.postictal import PostictalGroupResult
from ..analysis.spike_triggered import SpikeTriggeredGroupResult
from ..core.data_models import Session
from .colors import COHORT_COLORS, COHORT_DISPLAY_LABELS

# Larger default fonts so axis labels are readable in slide-deck contexts.
matplotlib.rcParams.update({
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})


def _color(cohort: str) -> str:
    return COHORT_COLORS.get(cohort, "gray")


def _label(cohort: str) -> str:
    """Return the user-facing display label for an internal cohort key."""
    return COHORT_DISPLAY_LABELS.get(cohort, cohort)


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
    ax.set_xticklabels([_label(k) for k in cohorts.keys()])
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
    ax.set_xticklabels([_label(k) for k in cohorts.keys()])
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
    """Panel 0: baseline temperature scatter ± SEM across cohorts.
    Panel 1: seizure threshold by session #, per-mouse lines (DS sz only).
    Panel 2: seizure duration by session #, per-mouse lines (DS sz only).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    baseline = {}
    csv_rows = []
    for name, r in results.items():
        bl_vals = [s.baseline_temp for s in r.session_results if s.baseline_temp is not None]
        baseline[name] = (bl_vals, r.baseline_temp_mean, r.baseline_temp_sem)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.baseline_temp, s.seizure_threshold_temp,
                             s.seizure_duration_s])

    _scatter_mean_sem(axes[0], baseline, "Baseline Temp (°C)")

    # Helper for "session # vs metric, per-mouse lines + cohort mean tick"
    def _per_mouse_panel(ax, attr: str, ylabel: str):
        seizure_key = "seizure" if "seizure" in results else next(iter(results.keys()))
        sz_r = results[seizure_key]
        by_mouse: Dict[str, list] = {}
        for s in sz_r.session_results:
            v = getattr(s, attr)
            if v is None:
                continue
            by_mouse.setdefault(s.mouse_id, []).append((s.heating_session, v))
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
        ax.set_ylabel(ylabel)
        if all_by_session:
            ax.set_xticks(sorted(all_by_session.keys()))
        ax.spines[["right", "top"]].set_visible(False)

    _per_mouse_panel(axes[1], "seizure_threshold_temp", "Seizure Threshold (°C)")
    _per_mouse_panel(axes[2], "seizure_duration_s", "Seizure Duration (s)")

    fig.tight_layout()
    path = os.path.join(output_dir, "cohort_characteristics.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "cohort_characteristics.csv"),
        ["cohort", "mouse_id", "heating_session",
         "baseline_temp", "seizure_threshold_temp", "seizure_duration_s"],
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

    _scatter_mean_sem(axes[0], freq, "Transient Frequency (Hz)")
    _scatter_mean_sem(axes[1], amp, "Transient Amplitude (z-ΔF/F)")
    _scatter_mean_sem(axes[2], hw, "Transient Half-width (s)")

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

def _grouped_scatter_mean_sem(
    ax,
    period_data: List[Dict[str, tuple]],
    period_labels: List[str],
    ylabel: str,
) -> None:
    """Multi-period grouped scatter with mean/SEM ticks per cohort.

    period_data: list (one entry per period) of {cohort_name: (vals, mean, sem)}.
    All periods use the same set of cohorts. Cohorts are clustered within each
    period; periods are spaced one unit apart on the x-axis.
    """
    # Use the first period's cohort order; all periods are expected to use the
    # same cohorts in the same order (caller responsibility).
    cohorts = list(period_data[0].keys()) if period_data else []
    n_cohorts = max(len(cohorts), 1)
    cluster_w = 0.8
    sub_w = cluster_w / n_cohorts
    rng = np.random.default_rng(42)
    for p_idx, pdata in enumerate(period_data):
        for c_idx, name in enumerate(cohorts):
            if name not in pdata:
                continue
            vals, mean, sem = pdata[name]
            color = _color(name)
            x_center = p_idx + (c_idx - (n_cohorts - 1) / 2) * sub_w
            if len(vals) > 0:
                jitter = rng.uniform(-sub_w * 0.3, sub_w * 0.3, len(vals))
                ax.scatter(np.full(len(vals), x_center) + jitter, vals,
                           color=color, s=22, zorder=3, alpha=0.8,
                           edgecolor="black", linewidth=0.4,
                           label=_label(name) if p_idx == 0 else None)
            if not np.isnan(mean):
                ax.errorbar(x_center, mean, yerr=sem if not np.isnan(sem) else 0,
                            fmt="_", color=color, markersize=22,
                            capsize=4, elinewidth=2, markeredgewidth=2, zorder=4)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(period_labels)))
    ax.set_xticklabels(period_labels)
    ax.set_ylabel(ylabel)
    ax.spines[["right", "top"]].set_visible(False)
    if cohorts:
        ax.legend(loc="best")


def plot_preictal_mean(
    results: Dict[str, PreictalMeanGroupResult],
    output_dir: str,
) -> str:
    """Panel 0: consolidated [Heating, Ictal] scatter.
    Panel 1: split [Early Heat, Late Heat, Ictal] scatter.
    Panel 2: end-of-late-heat scatter.
    Panel 3: temp-binned signal (mean line ± SEM band).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.baseline_mean, s.early_heat_mean,
                             s.late_heat_mean, s.end_late_heat_mean,
                             s.heating_mean,
                             "" if s.ictal_mean is None else s.ictal_mean])

    # Helper: per-cohort (vals, mean, sem) for a session attribute that may be None
    def _attr_data(attr: str):
        out = {}
        for name, r in results.items():
            vals = [getattr(s, attr) for s in r.session_results
                    if getattr(s, attr) is not None]
            mean = (float(np.mean(vals)) if vals else np.nan)
            if len(vals) > 1:
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                sem = 0.0 if vals else np.nan
            out[name] = (vals, mean, sem)
        return out

    # Panel 0: consolidated [Heating, Ictal]
    heating_data = _attr_data("heating_mean")
    ictal_data = _attr_data("ictal_mean")
    _grouped_scatter_mean_sem(
        axes[0],
        [heating_data, ictal_data],
        ["Heating", "Ictal"],
        "Mean z-ΔF/F",
    )

    # Panel 1: split [Early Heat, Late Heat, Ictal]
    eh_data = _attr_data("early_heat_mean")
    lh_data = _attr_data("late_heat_mean")
    _grouped_scatter_mean_sem(
        axes[1],
        [eh_data, lh_data, ictal_data],
        ["Early Heat", "Late Heat", "Ictal"],
        "Mean z-ΔF/F",
    )

    # Panel 2: end of late heat scatter
    elh = {}
    for name, r in results.items():
        vals = [s.end_late_heat_mean for s in r.session_results]
        elh[name] = (vals, r.end_late_heat_mean, r.end_late_heat_sem)
    _scatter_mean_sem(axes[2], elh, "End-of-Late-Heat (z-ΔF/F)")

    # Panel 3: temp-binned during heating, rel to seizure onset temp
    ax = axes[3]
    for name, r in results.items():
        _line_sem(ax, r.temp_bin_centers, r.temp_bin_group_mean,
                  r.temp_bin_group_sem, _color(name), _label(name))
    ax.set_xlabel("Temp rel. to seizure onset (°C)")
    ax.set_ylabel("Mean z-ΔF/F")
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "preictal_mean.png")
    fig.savefig(path, dpi=150)
    plt.close("all")

    _save_csv(
        os.path.join(output_dir, "preictal_mean.csv"),
        ["cohort", "mouse_id", "heating_session",
         "baseline_mean", "early_heat_mean",
         "late_heat_mean", "end_late_heat_mean", "heating_mean",
         "ictal_mean"],
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
        _line_sem(axes[0, 0], r.temp_ma_centers, r.temp_ma_freq_mean, r.temp_ma_freq_sem, c, _label(name))
        _line_sem(axes[0, 1], r.temp_ma_centers, r.temp_ma_amp_mean, r.temp_ma_amp_sem, c, _label(name))
        _line_sem(axes[0, 2], r.temp_ma_centers, r.temp_ma_hw_mean, r.temp_ma_hw_sem, c, _label(name))
    axes[0, 0].set_ylabel("Temp MA — Frequency (Hz)")
    axes[0, 1].set_ylabel("Temp MA — Amplitude (z-ΔF/F)")
    axes[0, 2].set_ylabel("Temp MA — Half-width (s)")
    for ax in axes[0]:
        ax.set_xlabel("Temp rel. to seizure onset (°C)")
        ax.legend()
        ax.spines[["right", "top"]].set_visible(False)

    # Row 1: temperature-binned (discrete bins, spec row 14)
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[1, 0], r.temp_bin_centers, r.temp_freq_mean, r.temp_freq_sem, c, _label(name))
        _line_sem(axes[1, 1], r.temp_bin_centers, r.temp_amp_mean, r.temp_amp_sem, c, _label(name))
        _line_sem(axes[1, 2], r.temp_bin_centers, r.temp_hw_mean, r.temp_hw_sem, c, _label(name))
    axes[1, 0].set_ylabel("Temp-binned — Frequency (count)")
    axes[1, 1].set_ylabel("Temp-binned — Amplitude (z-ΔF/F)")
    axes[1, 2].set_ylabel("Temp-binned — Half-width (s)")
    for ax in axes[1]:
        ax.set_xlabel("Temp rel. to seizure onset (°C)")
        ax.legend()
        ax.spines[["right", "top"]].set_visible(False)

    # Row 2: TIME moving averages (spec row 15)
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[2, 0], r.moving_avg_times, r.frequency_mean, r.frequency_sem, c, _label(name))
        _line_sem(axes[2, 1], r.moving_avg_times, r.amplitude_mean, r.amplitude_sem, c, _label(name))
        _line_sem(axes[2, 2], r.moving_avg_times, r.half_width_mean, r.half_width_sem, c, _label(name))
    axes[2, 0].set_ylabel("Time MA — Frequency (Hz)")
    axes[2, 1].set_ylabel("Time MA — Amplitude (z-ΔF/F)")
    axes[2, 2].set_ylabel("Time MA — Half-width (s)")
    for ax in axes[2]:
        ax.set_xlabel("Time from heating start (s)")
        ax.legend()
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

    _scatter_mean_sem(axes[0, 0], scalars_seizure, "Seizure-Period Mean (z-ΔF/F)")
    _scatter_mean_sem(axes[1, 0], scalars_delta, "Pre-ictal → Ictal Δ (z-ΔF/F)")
    # Hide unused cells in col 0
    for i in range(2, n_rows):
        axes[i, 0].axis("off")

    # Determine the AUC window used (varies per cohort under the new spec but
    # all cohorts share the same window in a single run via the driver). Pull
    # from the first available landmark of the first available cohort.
    auc_window_s = None
    for r in results.values():
        for ta in r.triggered_averages.values():
            if ta.auc_window_s and not np.isnan(ta.auc_window_s):
                auc_window_s = ta.auc_window_s
                break
        if auc_window_s:
            break
    auc_label_suffix = (f" (0–{auc_window_s:.0f}s)"
                        if auc_window_s else "")

    # Middle column (col 1): triggered averages per landmark
    # Right column (col 2): AUC scatter per landmark
    auc_csv = []
    trace_axes_for_landmarks = []
    auc_axes_for_landmarks = []
    trace_lo, trace_hi = np.inf, -np.inf
    auc_lo, auc_hi = np.inf, -np.inf

    for i, lm in enumerate(landmark_names):
        ax = axes[i, 1]
        for name, r in results.items():
            if lm in r.triggered_averages:
                ta = r.triggered_averages[lm]
                _line_sem(ax, ta.time_axis, ta.mean_trace, ta.sem_trace,
                          _color(name), _label(name))
                # Track full y-extent including SEM band
                lo = float(np.nanmin(ta.mean_trace - ta.sem_trace))
                hi = float(np.nanmax(ta.mean_trace + ta.sem_trace))
                if np.isfinite(lo): trace_lo = min(trace_lo, lo)
                if np.isfinite(hi): trace_hi = max(trace_hi, hi)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlabel(f"Time rel. to {lm} (s)")
        ax.set_ylabel(f"z-ΔF/F — {lm}")
        ax.set_xlim(-5, 65)
        ax.legend()
        ax.spines[["right", "top"]].set_visible(False)
        trace_axes_for_landmarks.append(ax)

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
                auc_csv.append([name, lm, v, ta.auc_window_s])
            if vals:
                auc_lo = min(auc_lo, float(np.nanmin(vals)))
                auc_hi = max(auc_hi, float(np.nanmax(vals)))
        if auc_dict:
            _scatter_mean_sem(ax_auc, auc_dict,
                              f"AUC — {lm}{auc_label_suffix}")
        auc_axes_for_landmarks.append(ax_auc)

    # Apply matched y-axes across landmark rows (item 15)
    if np.isfinite(trace_lo) and np.isfinite(trace_hi) and trace_hi > trace_lo:
        pad = 0.05 * (trace_hi - trace_lo)
        for ax in trace_axes_for_landmarks:
            ax.set_ylim(trace_lo - pad, trace_hi + pad)
    if np.isfinite(auc_lo) and np.isfinite(auc_hi) and auc_hi > auc_lo:
        pad = 0.05 * (auc_hi - auc_lo)
        for ax in auc_axes_for_landmarks:
            ax.set_ylim(auc_lo - pad, auc_hi + pad)

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
        ["cohort", "landmark", "auc", "auc_window_s"],
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
    rng = np.random.default_rng(42)
    for i, (name, r) in enumerate(results.items()):
        x = r.psth_bin_centers
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, r.psth_mean, width, yerr=r.psth_sem,
               color=_color(name), alpha=0.35, capsize=2, label=_label(name))
        if r.per_session_psth_freq:
            for sess_freq in r.per_session_psth_freq:
                jitter = rng.uniform(-width * 0.3, width * 0.3, len(x))
                ax.scatter(x + offset + jitter, sess_freq,
                           color=_color(name), s=8, alpha=0.6, zorder=3, edgecolor='none')
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time rel. to UEO (s)")
    ax.set_ylabel("PSTH Frequency (Hz)")
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)

    # Moving averages around UEO
    for name, r in results.items():
        c = _color(name)
        _line_sem(axes[0, 1], r.moving_avg_times, r.freq_mean, r.freq_sem, c, _label(name))
        _line_sem(axes[1, 0], r.moving_avg_times, r.amp_mean, r.amp_sem, c, _label(name))
        _line_sem(axes[1, 1], r.moving_avg_times, r.hw_mean, r.hw_sem, c, _label(name))

    axes[0, 1].set_ylabel("Moving Avg — Frequency (Hz)")
    axes[1, 0].set_ylabel("Moving Avg — Amplitude (z-ΔF/F)")
    axes[1, 1].set_ylabel("Moving Avg — Half-width (s)")
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time rel. to UEO (s)")
        ax.legend()
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
        _line_sem(ax, r.cooling_bin_centers, r.cooling_group_mean, sem, color, _label(name))
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Temp rel. to max (°C) — cooling →")
    ax.set_ylabel("Cooling Mean z-ΔF/F")
    ax.invert_xaxis()  # 0 on the right, more negative going left
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)

    # Row 26: final recording time vs final recording temp
    ax = axes[0, 1]
    for name, r in results.items():
        ax.scatter(r.final_times, r.final_temps, color=_color(name), label=_label(name),
                   s=30, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Post-Seizure Time (s)")
    ax.set_ylabel("Final Temp (°C)")
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)

    # Row 27: final recording time vs final mean ΔF/F
    ax = axes[1, 0]
    for name, r in results.items():
        ax.scatter(r.final_times, r.final_dffs, color=_color(name), label=_label(name),
                   s=30, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Post-Seizure Time (s)")
    ax.set_ylabel("Final Mean z-ΔF/F")
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)

    # Row 28: final recording temp vs final mean ΔF/F
    ax = axes[1, 1]
    for name, r in results.items():
        ax.scatter(r.final_temps, r.final_dffs, color=_color(name), label=_label(name),
                   s=30, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Final Temp (°C)")
    ax.set_ylabel("Final Mean z-ΔF/F")
    ax.legend()
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
            sem = r.group_sem
            ax.plot(r.time_axis, r.group_mean, color=c, label=_label(name))
            ax.fill_between(r.time_axis, r.group_mean - sem, r.group_mean + sem,
                            color=c, alpha=0.25)
            mask = (np.ones_like(r.time_axis, dtype=bool) if xlim is None
                    else (r.time_axis >= xlim[0]) & (r.time_axis <= xlim[1]))
            if mask.any():
                lo = np.nanmin((r.group_mean - sem)[mask])
                hi = np.nanmax((r.group_mean + sem)[mask])
                ylo = min(ylo, lo)
                yhi = max(yhi, hi)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Time rel. to spike (s)")
        ax.set_ylabel("Spike-Triggered z-ΔF/F (mean ± SE)")
        ax.legend()
        ax.spines[["right", "top"]].set_visible(False)
        if xlim is not None:
            ax.set_xlim(xlim)
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            pad = 0.05 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)

    # --- Main figure: full STA + AUC scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_sta(axes[0])

    auc_data = {}
    for name, r in results.items():
        auc_vals = [s.auc for s in r.session_results]
        auc_sem = float(np.std(auc_vals, ddof=1) / np.sqrt(len(auc_vals))) if len(auc_vals) > 1 else 0.0
        auc_data[name] = (auc_vals, r.group_auc, auc_sem)
    _scatter_mean_sem(axes[1], auc_data, "Spike-Triggered AUC")

    fig.tight_layout()
    path = os.path.join(output_dir, "spike_triggered.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # --- Zoomed companion figure: same STA, matching AUC window ---
    fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(8, 5))
    _draw_sta(ax_zoom, xlim=(-2, 10))
    ax_zoom.set_xlabel("Time rel. to spike (s) — zoomed")
    fig_zoom.tight_layout()
    fig_zoom.savefig(os.path.join(output_dir, "spike_triggered_zoomed.png"), dpi=150)
    plt.close(fig_zoom)

    _save_csv(
        os.path.join(output_dir, "spike_triggered.csv"),
        ["cohort", "mouse_id", "heating_session", "n_spikes", "auc"],
        csv_rows,
    )
    return path


# ---- 9. Per-cohort UEO-aligned mean trace ----

def plot_ueo_per_cohort(
    results: Dict[str, TriggeredAverage],
    output_dir: str,
) -> str:
    """Wide-window (±150s) UEO-triggered mean trace ± SE, one panel per cohort.

    Y-axes are matched across all cohort panels to make magnitude comparisons
    visible at a glance.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    n = max(len(results), 1)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    # First pass: compute global y-extent including ±SE band
    ylo, yhi = np.inf, -np.inf
    for ta in results.values():
        if ta.mean_trace is None or len(ta.mean_trace) == 0:
            continue
        sem = ta.sem_trace
        lo = float(np.nanmin(ta.mean_trace - sem))
        hi = float(np.nanmax(ta.mean_trace + sem))
        if np.isfinite(lo): ylo = min(ylo, lo)
        if np.isfinite(hi): yhi = max(yhi, hi)

    for i, (name, ta) in enumerate(results.items()):
        ax = axes[0, i]
        color = _color(name)
        sem = ta.sem_trace
        ax.plot(ta.time_axis, ta.mean_trace, color=color, label=_label(name))
        ax.fill_between(ta.time_axis, ta.mean_trace - sem,
                        ta.mean_trace + sem, color=color, alpha=0.25)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Time rel. to UEO (s)")
        ax.set_ylabel(f"{_label(name)}: UEO-aligned z-ΔF/F (mean ± SE)")
        ax.spines[["right", "top"]].set_visible(False)

    if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
        pad = 0.05 * (yhi - ylo)
        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(ylo - pad, yhi + pad)

    fig.tight_layout()
    path = os.path.join(output_dir, "ueo_per_cohort.png")
    fig.savefig(path, dpi=150)
    plt.close("all")
    return path


# ---- 10. Pre-ictal mean diagnostic ----

def plot_preictal_mean_diagnostic(
    session: Session,
    output_dir: str,
) -> Optional[str]:
    """Diagnostic overlay: shows the bin windows and means used by the
    pre-ictal-mean analysis directly on the session's processed trace, plus
    a zoomed view that annotates the delta-of-mean calculation.

    The goal is to make the binning + delta calculation auditable by eye.
    Returns the saved PNG path, or None if landmarks are missing.
    """
    from ..analysis._helpers import (
        get_signal_and_time, get_ueo_time, get_off_time, time_to_index,
    )

    lm = session.landmarks
    if lm is None or lm.heating_start_time is None:
        return None
    ueo_t = get_ueo_time(session)
    if ueo_t is None:
        return None

    signal, time_axis, fs = get_signal_and_time(session)
    if signal is None:
        return None

    heat_t = lm.heating_start_time
    off_t = get_off_time(session)
    end_t = off_t if off_t is not None else min(time_axis[-1], ueo_t + 30.0)

    i_heat = time_to_index(heat_t, fs)
    i_ueo = time_to_index(ueo_t, fs)
    i_mid = (i_heat + i_ueo) // 2
    i_end = min(time_to_index(end_t, fs), len(signal))

    # Bin means (must mirror analysis/preictal_mean.py + analysis/ictal_mean.py)
    eh_mean = float(np.mean(signal[i_heat:i_mid])) if i_mid > i_heat else np.nan
    lh_mean = float(np.mean(signal[i_mid:i_ueo])) if i_ueo > i_mid else np.nan
    end_late_n = min(int(10.0 * fs), i_ueo - i_heat)
    elh_mean = (float(np.mean(signal[i_ueo - end_late_n:i_ueo]))
                if end_late_n > 0 else np.nan)
    ictal_window = signal[i_ueo:i_end] if i_end > i_ueo else np.array([])
    ictal_mean = float(np.mean(ictal_window)) if len(ictal_window) else np.nan
    if len(ictal_window):
        ictal_max = float(np.max(ictal_window))
        ictal_min = float(np.min(ictal_window))
        if abs(ictal_max - elh_mean) >= abs(ictal_min - elh_mean):
            delta = ictal_max - elh_mean
            delta_idx_local = int(np.argmax(ictal_window))
            delta_label = "max"
        else:
            delta = ictal_min - elh_mean
            delta_idx_local = int(np.argmin(ictal_window))
            delta_label = "min"
        delta_idx = i_ueo + delta_idx_local
    else:
        delta = np.nan
        delta_idx = None
        delta_label = ""

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel 0: full heating + ictal trace with bin shading and mean lines
    ax = axes[0]
    color = _color(session.cohort) if session.cohort else "black"
    ax.plot(time_axis, signal, color=color, linewidth=0.4)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)

    region_specs = [
        (heat_t, time_axis[i_mid] if i_mid < len(time_axis) else heat_t,
         "tab:blue", 0.10, "Early heat", eh_mean),
        (time_axis[i_mid] if i_mid < len(time_axis) else heat_t, ueo_t,
         "tab:orange", 0.10, "Late heat", lh_mean),
        (ueo_t - end_late_n / fs if end_late_n else ueo_t, ueo_t,
         "tab:red", 0.20, "End-of-late-heat (10s)", elh_mean),
        (ueo_t, end_t, "tab:purple", 0.10, "Ictal", ictal_mean),
    ]
    for x0, x1, c, alpha, lbl, m in region_specs:
        if x1 <= x0:
            continue
        ax.axvspan(x0, x1, color=c, alpha=alpha)
        if not np.isnan(m):
            ax.plot([x0, x1], [m, m], color=c, linewidth=2.2,
                    label=f"{lbl}: {m:.3f}")

    ax.set_xlim(max(0, heat_t - 30), min(time_axis[-1], end_t + 10))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-ΔF/F (full window)")
    ax.legend(loc="upper left")
    ax.spines[["right", "top"]].set_visible(False)

    # Panel 1: zoom on [ueo - 15s, end] with delta annotation
    ax2 = axes[1]
    z0, z1 = ueo_t - 15.0, end_t + 2.0
    ax2.plot(time_axis, signal, color=color, linewidth=0.6)
    ax2.set_xlim(z0, z1)
    if end_late_n > 0:
        ax2.axvspan(ueo_t - end_late_n / fs, ueo_t,
                    color="tab:red", alpha=0.20, label="End-of-late-heat (10s)")
    if not np.isnan(elh_mean):
        ax2.axhline(elh_mean, color="tab:red", linestyle="--", linewidth=1.5,
                    label=f"End-of-late-heat mean = {elh_mean:.3f}")
    if not np.isnan(delta) and delta_idx is not None and delta_idx < len(time_axis):
        peak_t = time_axis[delta_idx]
        peak_v = signal[delta_idx]
        ax2.scatter([peak_t], [peak_v], color="black", s=60, zorder=5,
                    label=f"Ictal {delta_label} = {peak_v:.3f}")
        ax2.annotate(
            f"Δ = {delta_label} − end-late-heat\n   = {peak_v:.3f} − {elh_mean:.3f}\n   = {delta:+.3f}",
            xy=(peak_t, peak_v),
            xytext=(peak_t + 1.0, peak_v + 0.05 * abs(peak_v + 1)),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"),
        )
    ax2.axvline(ueo_t, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(f"Time (s) — zoomed around UEO = {ueo_t:.1f}s")
    ax2.set_ylabel("z-ΔF/F (zoomed)")
    ax2.legend(loc="upper left")
    ax2.spines[["right", "top"]].set_visible(False)

    fig.suptitle(
        f"Pre-ictal mean diagnostic — {session.mouse_id} S{session.heating_session} "
        f"({_label(session.cohort) if session.cohort else 'unknown cohort'})",
        fontsize=11,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = (
        f"preictal_mean_diagnostic_"
        f"{session.mouse_id}_S{session.heating_session}.png"
    )
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close("all")
    return path
