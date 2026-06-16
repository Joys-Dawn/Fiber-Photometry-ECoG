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

from . import _style  # applies shared rcParams on import; also exports canonical figsizes
from ..analysis.cohort_characteristics import CohortGroupResult as CohortResult
from ..analysis.baseline_transients import BaselineTransientGroupResult
from ..analysis.preictal_mean import PreictalMeanGroupResult
from ..analysis.preictal_transients import PreictalTransientGroupResult
from ..analysis.ictal_mean import IctalMeanGroupResult, TriggeredAverage
from ..analysis.ictal_transients import IctalTransientGroupResult
from ..analysis.postictal import PostictalGroupResult
from ..analysis.spike_triggered import SpikeTriggeredGroupResult
from ..core.cohort import COHORT_ORDER
from ..core.data_models import Session
from .colors import (
    COHORT_COLORS, COHORT_DISPLAY_LABELS,
    LANDMARK_DISPLAY_LABELS, DFF_LABEL, landmark_label,
)


def _color(cohort: str) -> str:
    return COHORT_COLORS.get(cohort, "gray")


def _label(cohort: str) -> str:
    """Return the user-facing display label for an internal cohort key."""
    return COHORT_DISPLAY_LABELS.get(cohort, cohort)


_COHORT_ORDER = list(COHORT_ORDER)


def _ordered(results: Dict) -> Dict:
    """Return results dict in canonical cohort order, extras appended at end."""
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


def _legend_above(ax, ncol: Optional[int] = None, y_offset: float = 1.02) -> None:
    """Move the legend out of the data area to sit centered above the axes.

    Mirrors the "figure labels above (rather than within) the plots" request
    from the analysis spec. Pass `y_offset` > 1.02 when the panel also has a
    title - the legend then sits above the title rather than overlapping it.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    if ncol is None:
        ncol = max(1, min(len(labels), 4))
    ax.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, y_offset),
        ncol=ncol, frameon=False, borderaxespad=0.0,
        handletextpad=0.5, columnspacing=2.0,
    )


def _save_standalone_scatter(cohorts: Dict[str, tuple], ylabel: str, path: str) -> None:
    """Save a single-panel cohort scatter (mean ± SEM) at `path`.

    Used for plots that double as a panel in a multi-panel figure AND as a
    standalone summary plot the user wants to drop directly into slides.
    """
    fig, ax = plt.subplots(figsize=_style.FIG_SCATTER)
    _scatter_mean_sem(ax, cohorts, ylabel)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_standalone(draw_fn, path: str, figsize=None) -> str:
    """Generic standalone-export helper.

    Creates a single-axis figure, calls draw_fn(ax), saves and closes it.
    Returns the saved path. The draw_fn is responsible for ALL axis content
    including labels, title, legend, etc.
    """
    if figsize is None:
        figsize = _style.FIG_SCATTER
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- 1. Cohort characteristics ----

@_style.styled
def plot_cohort_characteristics(
    results: Dict[str, CohortResult],
    output_dir: str,
) -> str:
    """Three standalone files (no combo): Cohort_characteristics_baseline_temp.png,
    _seizure_threshold.png, _seizure_duration.png. The previous composite
    cohort_characteristics.png has been retired per request.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

    baseline = {}
    csv_rows = []
    for name, r in results.items():
        bl_vals = [s.baseline_temp for s in r.session_results if s.baseline_temp is not None]
        baseline[name] = (bl_vals, r.baseline_temp_mean, r.baseline_temp_sem)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.baseline_temp, s.seizure_threshold_temp,
                             s.seizure_duration_s])

    # Mouse body temperature physiologically bounded - fix axes so cohort
    # plots are visually comparable across batches.
    TEMP_YLIM = (35.0, 43.0)

    def _draw_baseline(ax):
        _scatter_mean_sem(ax, baseline, "Baseline Temp (°C)")
        ax.set_ylim(*TEMP_YLIM)

    def _draw_per_mouse(ax, attr: str, ylabel: str, fixed_ylim=None):
        # Prefer the DS-sz cohort; fall back to WT-sz (e.g. PTZ runs) so
        # this panel always draws the seized cohort when one exists.
        seizure_key = next(
            (k for k in ("DS_sz", "WT_sz") if k in results),
            next(iter(results.keys())),
        )
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
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        ax.spines[["right", "top"]].set_visible(False)

    # Standalone per-panel exports (combo cohort_characteristics.png retired
    # per spec - the 3 individual panels are the canonical view now).
    baseline_path = _save_standalone(
        _draw_baseline,
        os.path.join(output_dir, "Cohort_characteristics_baseline_temp.png"))
    _save_standalone(
        lambda ax: _draw_per_mouse(ax, "seizure_threshold_temp",
                                   "Seizure Threshold (°C)", TEMP_YLIM),
        os.path.join(output_dir, "Cohort_characteristics_seizure_threshold.png"))
    _save_standalone(
        lambda ax: _draw_per_mouse(ax, "seizure_duration_s", "Seizure Duration (s)"),
        os.path.join(output_dir, "Cohort_characteristics_seizure_duration.png"))

    _save_csv(
        os.path.join(output_dir, "cohort_characteristics.csv"),
        ["cohort", "mouse_id", "heating_session",
         "baseline_temp", "seizure_threshold_temp", "seizure_duration_s"],
        csv_rows,
    )
    return baseline_path


# ---- 2. Baseline transients ----

@_style.styled
def plot_baseline_transients(
    results: Dict[str, BaselineTransientGroupResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

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

    fig, axes = plt.subplots(1, 3, figsize=_style.panel_figsize(3))
    _scatter_mean_sem(axes[0], freq, "Transient Frequency (Hz)")
    _scatter_mean_sem(axes[1], amp, "Transient Amplitude (z_ΔF/F)")
    _scatter_mean_sem(axes[2], hw, "Transient Half-width (s)")
    path = os.path.join(output_dir, "summary_baseline_transients.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    _save_standalone_scatter(freq, "Transient Frequency (Hz)",
        os.path.join(output_dir, "Transient_frequency_baseline.png"))
    _save_standalone_scatter(amp, "Transient Amplitude (z_ΔF/F)",
        os.path.join(output_dir, "Transient_amplitude_baseline.png"))
    _save_standalone_scatter(hw, "Transient Half-width (s)",
        os.path.join(output_dir, "Transient_halfwidth_baseline.png"))

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
    period_spacing: float = 1.8,
) -> None:
    """Multi-period grouped scatter with mean/SEM ticks per cohort.

    period_data: list (one entry per period) of {cohort_name: (vals, mean, sem)}.
    All periods use the same set of cohorts. Cohorts are clustered within each
    period; periods are spaced `period_spacing` units apart on the x-axis
    (default 1.8 - wider than the cohort cluster width so periods read as
    distinct visual groups).
    """
    cohorts = list(period_data[0].keys()) if period_data else []
    n_cohorts = max(len(cohorts), 1)
    cluster_w = 0.8
    sub_w = cluster_w / n_cohorts
    rng = np.random.default_rng(42)
    period_centers = np.arange(len(period_data), dtype=float) * period_spacing
    for p_idx, pdata in enumerate(period_data):
        for c_idx, name in enumerate(cohorts):
            if name not in pdata:
                continue
            vals, mean, sem = pdata[name]
            color = _color(name)
            x_center = period_centers[p_idx] + (c_idx - (n_cohorts - 1) / 2) * sub_w
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
    ax.set_xticks(period_centers)
    ax.set_xticklabels(period_labels)
    ax.set_ylabel(ylabel)
    ax.spines[["right", "top"]].set_visible(False)
    if cohorts:
        _legend_above(ax)


@_style.styled
def plot_preictal_mean(
    results: Dict[str, PreictalMeanGroupResult],
    output_dir: str,
) -> str:
    """Panel 0: consolidated [Heating, Ictal] scatter.
    Panel 1: split [Early Heat, Late Heat, Ictal] scatter.
    Panel 2: end-of-late-heat scatter.
    Panel 3: temp-binned mean signal during heating (rel. to seizure onset temp).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(1, 4, figsize=_style.panel_figsize(4))

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
        "Mean z_ΔF/F",
    )

    # Panel 1: split [Early Heat, Late Heat, Ictal]
    eh_data = _attr_data("early_heat_mean")
    lh_data = _attr_data("late_heat_mean")
    _grouped_scatter_mean_sem(
        axes[1],
        [eh_data, lh_data, ictal_data],
        ["Early Heat", "Late Heat", "Ictal"],
        "Mean z_ΔF/F",
    )

    # Panel 2: end of late heat scatter
    elh = {}
    for name, r in results.items():
        vals = [s.end_late_heat_mean for s in r.session_results]
        elh[name] = (vals, r.end_late_heat_mean, r.end_late_heat_sem)
    _scatter_mean_sem(axes[2], elh, "End-of-Late-Heat (z_ΔF/F)")

    # Panel 3: mean signal vs temperature during heating
    ax = axes[3]
    for name, r in results.items():
        _line_sem(ax, r.temp_bin_centers, r.temp_bin_group_mean,
                  r.temp_bin_group_sem, _color(name), _label(name))
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Temp rel. to seizure onset (°C)")
    ax.set_ylabel("Mean z_ΔF/F")
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)

    path = os.path.join(output_dir, "summary_preictal_mean.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Companion standalone plots: by-cohort paired-line views.
    _plot_paired_by_cohort(
        results,
        attrs=["heating_mean", "ictal_mean"],
        period_labels=["Heating", "Ictal"],
        ylabel="Mean z_ΔF/F",
        path=os.path.join(output_dir, "Mean_values_all_heat_ictal_by_cohort.png"),
    )
    _plot_paired_by_cohort(
        results,
        attrs=["early_heat_mean", "late_heat_mean", "ictal_mean"],
        period_labels=["Early Heat", "Late Heat", "Ictal"],
        ylabel="Mean z_ΔF/F",
        path=os.path.join(
            output_dir, "Mean_values_early_heat_late_heat_ictal_by_cohort.png"
        ),
    )

    # By-period grouped scatters (companion to by-cohort views)
    _save_standalone(
        lambda ax: _grouped_scatter_mean_sem(
            ax, [heating_data, ictal_data], ["Heating", "Ictal"], "Mean z_ΔF/F"),
        os.path.join(output_dir, "Mean_values_all_heat_ictal_by_period.png"),
    )
    _save_standalone(
        lambda ax: _grouped_scatter_mean_sem(
            ax, [eh_data, lh_data, ictal_data],
            ["Early Heat", "Late Heat", "Ictal"], "Mean z_ΔF/F"),
        os.path.join(output_dir,
                     "Mean_values_early_heat_late_heat_ictal_by_period.png"),
    )

    # Single-period cohort scatter. Note: Mean_values_ictal.png is owned by
    # plot_ictal_mean (the canonical source for the ictal mean metric); we
    # deliberately don't write it here to avoid silent overwrites if the two
    # modules ever drift.
    _save_standalone_scatter(
        elh, "z_ΔF/F,\nimmediate preictal period",
        os.path.join(output_dir, "Mean_values_immediate_preictal.png"),
    )

    # Mean signal vs temperature during heating (panel 3)
    def _draw_vs_temp_heating(ax):
        for name, r in results.items():
            _line_sem(ax, r.temp_bin_centers, r.temp_bin_group_mean,
                      r.temp_bin_group_sem, _color(name), _label(name))
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_xlabel("Temp rel. to seizure onset (°C)")
        ax.set_ylabel("Mean z_ΔF/F")
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)
    _save_standalone(_draw_vs_temp_heating,
                     os.path.join(output_dir, "Mean_values_vs_temp_heating.png"),
                     figsize=_style.FIG_WIDE)

    _save_csv(
        os.path.join(output_dir, "preictal_mean.csv"),
        ["cohort", "mouse_id", "heating_session",
         "baseline_mean", "early_heat_mean",
         "late_heat_mean", "end_late_heat_mean", "heating_mean",
         "ictal_mean"],
        csv_rows,
    )
    return path


def _build_ueo_heatmap_data(
    sessions: List[Session],
    pre_s: float,
    post_s: float,
    bl_start_s: float = 5.0,
    bl_end_s: float = 1.0,
) -> tuple:
    """Per-session UEO-aligned matrix for the heatmap, padded with NaN
    outside each session's recording.

    The aggregation path (`compute_wide_ueo_triggered`) stays strict so
    cohort mean/SEM only reflect sessions that span the requested window.
    The heatmap, by contrast, just displays each row to whatever extent
    its recording covers - NaN cells render as transparent via the
    colormap's `set_bad`. Different concerns, separate code path.

    Returns (matrix, time_axis, session_labels).
    """
    from ..analysis._helpers import get_ueo_time

    if not sessions:
        return np.zeros((0, 0)), np.array([]), []
    fs = sessions[0].processed.fs
    if fs is None or fs <= 0:
        return np.zeros((0, 0)), np.array([]), []
    pre_samples = int(round(pre_s * fs))
    post_samples = int(round(post_s * fs))
    total = pre_samples + post_samples + 1

    rows: List[np.ndarray] = []
    labels: List[str] = []
    for s in sessions:
        if s.processed is None or s.processed.photometry is None:
            continue
        t_event = get_ueo_time(s)
        if t_event is None or not np.isfinite(t_event):
            continue
        signal = s.processed.photometry.dff_zscore
        event_idx = int(round(t_event * fs))
        start = event_idx - pre_samples
        end = event_idx + post_samples + 1

        row = np.full(total, np.nan, dtype=float)
        sig_start = max(0, start)
        sig_end = min(len(signal), end)
        if sig_end <= sig_start:
            continue
        out_start = sig_start - start
        out_end = out_start + (sig_end - sig_start)
        row[out_start:out_end] = signal[sig_start:sig_end]

        # Pre-event baseline subtract using whatever pre-window samples
        # exist in this row (nanmean tolerates partial coverage).
        bl_lo = pre_samples - int(round(bl_start_s * fs))
        bl_hi = pre_samples - int(round(bl_end_s * fs))
        bl_window = row[bl_lo:bl_hi]
        if np.any(~np.isnan(bl_window)):
            row = row - float(np.nanmean(bl_window))

        rows.append(row)
        labels.append(f"{s.mouse_id} S{s.heating_session}")

    if not rows:
        return np.zeros((0, total)), np.linspace(-pre_s, post_s, total), []
    return np.array(rows), np.linspace(-pre_s, post_s, total), labels


@_style.styled
def plot_ueo_aligned_heatmaps(
    cohort_sessions: Dict[str, List[Session]],
    output_dir: str,
    pre_s: float = 150.0,
    post_s: float = 150.0,
) -> Dict[str, str]:
    """UEO-aligned per-session heatmaps, one PNG per cohort.

    Each row is a session in that cohort; cells past the end of the
    recording render as transparent gaps. Stats (the wide UEO mean ±
    SEM line) live in `compute_wide_ueo_triggered` and remain strict -
    this heatmap-only path doesn't feed those aggregations.

    Generates TWO sets so each can sit under the matching traces plot:
      standard  → x-axis  -pre_s..+post_s          (-150..+150 by default)
      extended  → x-axis  -pre_s..+1000s           (long tail past UEO)

    Files written:
      UEO_aligned_heatmap_DS_sz[_extended].png
      UEO_aligned_heatmap_DS_nosz[_extended].png
      UEO_aligned_heatmap_WT[_extended].png

    The color scale is global across all cohorts within each window so
    the cohort plots are directly visually comparable. Diverging RdBu_r
    colormap; vlim is clipped at the 98th percentile of |values| so
    outlier sessions don't compress the dynamic range. Returns
    {cohort_key: path_of_standard_window}.
    """
    import matplotlib.cm as cm

    os.makedirs(output_dir, exist_ok=True)
    cohort_fname = {
        "DS_sz": ("UEO_aligned_heatmap_DS_sz.png",
                  "UEO_aligned_heatmap_DS_sz_extended.png"),
        "DS_no_sz": ("UEO_aligned_heatmap_DS_nosz.png",
                     "UEO_aligned_heatmap_DS_nosz_extended.png"),
        "WT_sz": ("UEO_aligned_heatmap_WT_sz.png",
                  "UEO_aligned_heatmap_WT_sz_extended.png"),
        "WT_no_sz": ("UEO_aligned_heatmap_WT.png",
                     "UEO_aligned_heatmap_WT_extended.png"),
    }

    WINDOWS = [
        (-pre_s, post_s, 0),     # standard
        (-pre_s, 1000.0, 1),     # extended (-150..+1000)
    ]

    # Pre-build each cohort's (matrix, time_axis, labels) at each window
    # so we can compute a global cmap range across cohorts within window.
    per_window_cohort_data: List[Dict[str, tuple]] = [{}, {}]
    for x_lo, x_hi, w_idx in WINDOWS:
        win_post = x_hi  # x_hi is the post-event extent for this window
        for cohort_key in cohort_fname:
            sessions = cohort_sessions.get(cohort_key, [])
            mat, time_axis, sess_labels = _build_ueo_heatmap_data(
                sessions, pre_s=pre_s, post_s=win_post)
            if mat.size == 0:
                continue
            per_window_cohort_data[w_idx][cohort_key] = (mat, time_axis, sess_labels)

    saved: Dict[str, str] = {}

    # Diverging colormap (RdBu_r: blue=negative, white=zero, red=positive)
    # - signed data reads more clearly than under sequential viridis. NaN
    # cells stay transparent so missing-recording tails appear as gaps.
    cmap = cm.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white", alpha=0.0)

    # Clip the color range at a percentile so a single outlier session
    # doesn't compress everyone else into the middle band. 98th-percentile
    # of |values| means at most 2% of cells saturate to the colormap ends,
    # which trades a little outlier fidelity for a lot more contrast.
    CLIP_PERCENTILE = 98.0

    for x_lo, x_hi, w_idx in WINDOWS:
        cohort_data = per_window_cohort_data[w_idx]
        if not cohort_data:
            continue
        # Global symmetric scale across all cohorts in this window
        # (NaN-aware so missing tails don't poison vlim).
        all_vals = np.concatenate([m[0].ravel() for m in cohort_data.values()
                                   if m[0].size])
        if all_vals.size:
            finite = all_vals[np.isfinite(all_vals)]
            vlim = float(np.percentile(np.abs(finite), CLIP_PERCENTILE)) \
                if finite.size else 1.0
            vlim = max(vlim, 1e-6)  # avoid zero range when data is all-NaN
        else:
            vlim = 1.0

        for cohort_key, (mat, t_clip, sess_labels) in cohort_data.items():
            n_sessions = mat.shape[0]
            fname = cohort_fname[cohort_key][w_idx]
            fig, ax = plt.subplots(figsize=(_style.FIG_WIDE[0],
                                            max(_style.FIG_WIDE[1],
                                                0.55 * n_sessions + 3.5)))
            im = ax.imshow(
                mat,
                aspect="auto",
                origin="lower",
                extent=(t_clip[0], t_clip[-1], -0.5, n_sessions - 0.5),
                cmap=cmap,
                vmin=-vlim,
                vmax=vlim,
                interpolation="nearest",
            )
            ax.axvline(0, color="white", linestyle="--", linewidth=1.4)
            ax.set_xlabel("Time rel. to seizure onset (s)")
            ax.set_ylabel("Session #")
            ax.set_xlim(x_lo, x_hi)
            ax.set_yticks(np.arange(n_sessions))
            if len(sess_labels) == n_sessions:
                ax.set_yticklabels(sess_labels, fontsize=24)
            ax.set_title(f"UEO-aligned z_ΔF/F - {_label(cohort_key)}")
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label(DFF_LABEL)
            path = os.path.join(output_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            if w_idx == 0:
                saved[cohort_key] = path

    return saved


def _plot_paired_by_cohort(
    results: Dict[str, PreictalMeanGroupResult],
    attrs: List[str],
    period_labels: List[str],
    ylabel: str,
    path: str,
) -> None:
    """Single-axes view, cohorts as x-axis groups, periods nested inside.

    Each cohort gets its own x-cluster; within a cluster, every session
    contributes a polyline across the requested period attributes. The
    cohort mean ± SEM is overlaid as a thick line for reference. Sessions
    missing any of the requested values are skipped.
    """
    results = _ordered(results)
    n_cohorts = len(results)
    if n_cohorts == 0:
        return
    n_attrs = len(attrs)

    fig, ax = plt.subplots(figsize=(max(_style.FIG_2X2[0],
                                        _style.PANEL_W * n_cohorts),
                                    _style.FIG_2X2[1]))
    rng = np.random.default_rng(42)

    cohort_centers = np.arange(n_cohorts, dtype=float)
    cluster_width = 0.7 if n_attrs > 1 else 0.0
    if n_attrs > 1:
        period_offsets = (np.arange(n_attrs) - (n_attrs - 1) / 2.0) * (
            cluster_width / (n_attrs - 1))
    else:
        period_offsets = np.zeros(1)

    for c_idx, (name, r) in enumerate(results.items()):
        color = _color(name)
        x_positions = cohort_centers[c_idx] + period_offsets

        per_session_vectors = []
        for s in r.session_results:
            vals = [getattr(s, a) for a in attrs]
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
                continue
            per_session_vectors.append(vals)
            jitter = rng.uniform(-0.03, 0.03, n_attrs)
            ax.plot(x_positions + jitter, vals,
                    "-o", color=color, alpha=0.45,
                    markersize=4, linewidth=0.9)
        if per_session_vectors:
            mat = np.array(per_session_vectors)
            means = np.mean(mat, axis=0)
            sems = (np.std(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
                    if mat.shape[0] > 1 else np.zeros(mat.shape[1]))
            ax.errorbar(x_positions, means, yerr=sems,
                        fmt="-o", color=color, linewidth=2.8,
                        markersize=9, capsize=5, zorder=5)

    # Period sub-labels just under the data area, above the cohort xticklabels.
    for c_idx in range(n_cohorts):
        for p_idx, period_label in enumerate(period_labels):
            x = cohort_centers[c_idx] + period_offsets[p_idx]
            ax.text(x, -0.03, period_label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=28, color="dimgray",
                    clip_on=False)

    ax.set_xticks(cohort_centers)
    ax.set_xticklabels([_label(name) for name in results.keys()])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, n_cohorts - 1 + 0.6)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.spines[["right", "top"]].set_visible(False)
    # Push cohort xticklabels down past the period sub-labels.
    ax.tick_params(axis="x", which="major", pad=58)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- 4. Pre-ictal transients ----

@_style.styled
def plot_preictal_transients(
    results: Dict[str, PreictalTransientGroupResult],
    output_dir: str,
) -> str:
    """Pre-ictal transient properties as temperature-aligned moving averages.

    Per spec, the legacy "time from heating start" moving averages and the
    "temp-binned" panels are not part of the current output - only the
    temperature moving averages (rel. to seizure onset) are retained.

    Also writes 3 standalone files (Transient_frequency_moving_average_vs_temp_heating.png,
    _amplitude_*.png, _halfwidth_*.png).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

    def _draw_panel(ax, attr_mean: str, attr_sem: str, ylabel: str):
        for name, r in results.items():
            c = _color(name)
            _line_sem(ax, r.temp_ma_centers,
                      getattr(r, attr_mean), getattr(r, attr_sem),
                      c, _label(name))
        ax.set_xlabel("Temp rel. to seizure onset (°C)")
        ax.set_ylabel(ylabel)
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)

    fig, axes = plt.subplots(1, 3, figsize=_style.panel_figsize(3))
    _draw_panel(axes[0], "temp_ma_freq_mean", "temp_ma_freq_sem",
                "Temp moving average - Frequency (Hz)")
    _draw_panel(axes[1], "temp_ma_amp_mean", "temp_ma_amp_sem",
                "Temp moving average - Amplitude (z_ΔF/F)")
    _draw_panel(axes[2], "temp_ma_hw_mean", "temp_ma_hw_sem",
                "Temp moving average - Half-width (s)")

    path = os.path.join(output_dir, "summary_preictal_transients.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    _save_standalone(
        lambda ax: _draw_panel(ax, "temp_ma_freq_mean", "temp_ma_freq_sem",
                               "Frequency (Hz) - moving average vs temp"),
        os.path.join(output_dir, "Transient_frequency_moving_average_vs_temp_heating.png"),
        figsize=_style.FIG_WIDE)
    _save_standalone(
        lambda ax: _draw_panel(ax, "temp_ma_amp_mean", "temp_ma_amp_sem",
                               "Amplitude (z_ΔF/F) - moving average vs temp"),
        os.path.join(output_dir, "Transient_amplitude_moving_average_vs_temp_heating.png"),
        figsize=_style.FIG_WIDE)
    _save_standalone(
        lambda ax: _draw_panel(ax, "temp_ma_hw_mean", "temp_ma_hw_sem",
                               "Half-width (s) - moving average vs temp"),
        os.path.join(output_dir, "Transient_halfwidth_moving_average_vs_temp_heating.png"),
        figsize=_style.FIG_WIDE)

    return path


# ---- 5. Ictal mean signal ----

@_style.styled
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

    # Layout: 3 + n_lm rows x 3 cols. Rows 0-2: scalar scatters (col 0):
    # seizure mean, delta_max, delta_min. Rows 3..: triggered avg (col 1) +
    # AUC (col 2). Width bumped to accommodate large axis-label fonts so the
    # left-column ylabels (which sit close together vertically) don't run on
    # top of one another.
    n_rows = max(n_lm, 3)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(_style.PANEL_W * 3, _style.PANEL_H * n_rows),
                             squeeze=False)

    # Left column (col 0): scatters stacked
    scalars_seizure = {}
    scalars_delta = {}
    scalars_delta_max = {}
    scalars_delta_min = {}
    csv_rows = []
    for name, r in results.items():
        sz_vals = [s.seizure_mean for s in r.session_results]
        d_vals = [s.delta_preictal_ictal for s in r.session_results]
        d_max_vals = [s.delta_preictal_ictal_max for s in r.session_results]
        d_min_vals = [s.delta_preictal_ictal_min for s in r.session_results]
        scalars_seizure[name] = (sz_vals, r.seizure_mean, r.seizure_sem)
        scalars_delta[name] = (d_vals, r.delta_mean, r.delta_sem)
        scalars_delta_max[name] = (d_max_vals, r.delta_max_mean, r.delta_max_sem)
        scalars_delta_min[name] = (d_min_vals, r.delta_min_mean, r.delta_min_sem)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.seizure_mean, s.baseline_mean,
                             s.delta_preictal_ictal,
                             s.delta_preictal_ictal_max,
                             s.delta_preictal_ictal_min])

    # Multi-line ylabels in the combo so adjacent rows' labels don't
    # collide vertically. Standalone exports below keep single-line labels.
    _scatter_mean_sem(axes[0, 0], scalars_seizure,
                      "Seizure-Period Mean\n(z_ΔF/F)")
    _scatter_mean_sem(axes[1, 0], scalars_delta_max,
                      "Δ immediate pre-ictal\n→ ictal max (z_ΔF/F)")
    _scatter_mean_sem(axes[2, 0], scalars_delta_min,
                      "Δ immediate pre-ictal\n→ ictal min (z_ΔF/F)")
    # Hide unused cells in col 0
    for i in range(3, n_rows):
        axes[i, 0].axis("off")

    # Middle column (col 1): triggered averages per landmark
    # Right column (col 2): AUC scatter per landmark
    auc_csv = []
    trace_axes_for_landmarks = []
    auc_axes_for_landmarks = []
    trace_lo, trace_hi = np.inf, -np.inf
    auc_lo, auc_hi = np.inf, -np.inf
    # Per-landmark data captured so we can also write standalone files.
    per_lm_auc: Dict[str, Dict[str, tuple]] = {}
    per_lm_window: Dict[str, Optional[float]] = {}

    def _draw_trace_panel(ax, lm: str, lm_auc_window: Optional[float],
                          show_title: bool = False, legend: bool = True):
        for name, r in results.items():
            if lm in r.triggered_averages:
                ta = r.triggered_averages[lm]
                _line_sem(ax, ta.time_axis, ta.mean_trace, ta.sem_trace,
                          _color(name), _label(name))
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
        lm_disp = landmark_label(lm)
        ax.set_xlabel(f"Time rel. to {lm_disp} (s)")
        ax.set_ylabel(DFF_LABEL)
        if show_title:
            ax.set_title(lm_disp)
        xlim_hi = max(65.0, (lm_auc_window or 60.0) + 5.0)
        ax.set_xlim(-5, xlim_hi)
        ax.spines[["right", "top"]].set_visible(False)
        if legend:
            # Push the legend above the title. Add one extra step per
            # wrapped title line so multi-line titles (e.g. "earliest\n
            # electrographic change") don't collide with the legend.
            if show_title:
                n_lines = lm_disp.count("\n") + 1
                offset = 1.15 + 0.18 * (n_lines - 1)
            else:
                offset = 1.02
            _legend_above(ax, y_offset=offset)

    for i, lm in enumerate(landmark_names):
        ax = axes[i, 1]
        lm_auc_window = None
        for name, r in results.items():
            if lm in r.triggered_averages:
                ta = r.triggered_averages[lm]
                lo = float(np.nanmin(ta.mean_trace - ta.sem_trace))
                hi = float(np.nanmax(ta.mean_trace + ta.sem_trace))
                if np.isfinite(lo): trace_lo = min(trace_lo, lo)
                if np.isfinite(hi): trace_hi = max(trace_hi, hi)
                if lm_auc_window is None and ta.auc_window_s and not np.isnan(ta.auc_window_s):
                    lm_auc_window = float(ta.auc_window_s)
        per_lm_window[lm] = lm_auc_window
        _draw_trace_panel(ax, lm, lm_auc_window,
                          show_title=True, legend=(i == 0))
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
        per_lm_auc[lm] = auc_dict
        if auc_dict:
            _scatter_mean_sem(ax_auc, auc_dict, "AUC")
            suffix = (f" (0–{lm_auc_window:.0f}s)"
                      if lm_auc_window else "")
            ax_auc.set_title(f"AUC - {landmark_label(lm)}{suffix}")
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

    path = os.path.join(output_dir, "summary_ictal_mean.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Separate standalone plots for the delta_max / delta_min companions, so
    # downstream summary slides can pull them individually.
    # Multi-line y-labels so they fit inside the plot rather than overflowing.
    _save_standalone_scatter(
        scalars_delta_max,
        "Δ immediate pre-ictal\n→ ictal max (z_ΔF/F)",
        os.path.join(output_dir, "Mean_delta_immediate_preictal_to_ictal_max.png"),
    )
    _save_standalone_scatter(
        scalars_delta_min,
        "Δ immediate pre-ictal\n→ ictal min (z_ΔF/F)",
        os.path.join(output_dir, "Mean_delta_immediate_preictal_to_ictal_min.png"),
    )
    # Standalone "Mean_values_ictal.png" mirroring the seizure-mean scalar panel.
    _save_standalone_scatter(
        scalars_seizure,
        "Mean z_ΔF/F,\nseizure period",
        os.path.join(output_dir, "Mean_values_ictal.png"),
    )

    # Per-landmark standalone files (AUC_<lm>_traces.png + _scatterplot.png).
    # Per spec, only EEC / UEO / OFF get their own files - behavioral_onset
    # and max_temp remain visible in the composite figure only.
    STANDALONE_AUC_LANDMARKS = {"EEC", "UEO", "OFF"}
    for lm in landmark_names:
        if lm not in STANDALONE_AUC_LANDMARKS:
            continue
        lm_auc_window = per_lm_window.get(lm)
        # Trace standalone
        _save_standalone(
            lambda ax, _lm=lm, _w=lm_auc_window: _draw_trace_panel(ax, _lm, _w),
            os.path.join(output_dir, f"AUC_{lm}_traces.png"),
        )
        # AUC scatter standalone
        auc_dict = per_lm_auc.get(lm)
        if auc_dict:
            suffix = (f" (0–{lm_auc_window:.0f}s)" if lm_auc_window else "")
            _save_standalone(
                lambda ax, _d=auc_dict, _lm=lm, _suf=suffix: (
                    _scatter_mean_sem(ax, _d, "AUC"),
                    ax.set_title(f"AUC - {landmark_label(_lm)}{_suf}"),
                ),
                os.path.join(output_dir, f"AUC_{lm}_scatterplot.png"),
            )

    _save_csv(
        os.path.join(output_dir, "ictal_mean.csv"),
        ["cohort", "mouse_id", "heating_session",
         "seizure_mean", "baseline_mean", "delta_preictal_ictal",
         "delta_preictal_ictal_max", "delta_preictal_ictal_min"],
        csv_rows,
    )
    _save_csv(
        os.path.join(output_dir, "Mean_delta_immediate_preictal_to_ictal_max.csv"),
        ["cohort", "mouse_id", "heating_session", "delta_max"],
        [[name, s.mouse_id, s.heating_session, s.delta_preictal_ictal_max]
         for name, r in results.items() for s in r.session_results],
    )
    _save_csv(
        os.path.join(output_dir, "Mean_delta_immediate_preictal_to_ictal_min.csv"),
        ["cohort", "mouse_id", "heating_session", "delta_min"],
        [[name, s.mouse_id, s.heating_session, s.delta_preictal_ictal_min]
         for name, r in results.items() for s in r.session_results],
    )
    _save_csv(
        os.path.join(output_dir, "ictal_mean_auc.csv"),
        ["cohort", "landmark", "auc", "auc_window_s"],
        auc_csv,
    )
    return path


# ---- 6. Ictal transients ----

@_style.styled
def plot_ictal_transients(
    results: Dict[str, IctalTransientGroupResult],
    output_dir: str,
) -> str:
    """Composite ictal_transients.png + 4 standalone files:
    Transient_frequency_PSTH.png and Transient_{frequency,amplitude,halfwidth}_vs_time.png.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

    def _draw_psth(ax):
        n_groups = max(len(results), 1)
        first_centers = next(iter(results.values())).psth_bin_centers
        bin_step = (float(first_centers[1] - first_centers[0])
                    if len(first_centers) > 1 else 1.0)
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
                               color=_color(name), s=8, alpha=0.6,
                               zorder=3, edgecolor='none')
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time rel. to seizure onset (s)")
        ax.set_ylabel("PSTH Frequency (Hz)")
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)

    def _draw_ma(ax, attr_mean: str, attr_sem: str, ylabel: str):
        for name, r in results.items():
            c = _color(name)
            _line_sem(ax, r.moving_avg_times,
                      getattr(r, attr_mean), getattr(r, attr_sem),
                      c, _label(name))
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time rel. to seizure onset (s)")
        ax.set_ylabel(ylabel)
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)

    fig, axes = plt.subplots(2, 2, figsize=_style.FIG_2X2)
    _draw_psth(axes[0, 0])
    _draw_ma(axes[0, 1], "freq_mean", "freq_sem", "Moving average - Frequency (Hz)")
    _draw_ma(axes[1, 0], "amp_mean", "amp_sem", "Moving average - Amplitude (z_ΔF/F)")
    _draw_ma(axes[1, 1], "hw_mean", "hw_sem", "Moving average - Half-width (s)")
    path = os.path.join(output_dir, "summary_ictal_transients.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    _save_standalone(_draw_psth,
        os.path.join(output_dir, "Transient_frequency_PSTH.png"))
    _save_standalone(
        lambda ax: _draw_ma(ax, "freq_mean", "freq_sem", "Frequency (Hz)"),
        os.path.join(output_dir, "Transient_frequency_vs_time.png"))
    _save_standalone(
        lambda ax: _draw_ma(ax, "amp_mean", "amp_sem", "Amplitude (z_ΔF/F)"),
        os.path.join(output_dir, "Transient_amplitude_vs_time.png"))
    _save_standalone(
        lambda ax: _draw_ma(ax, "hw_mean", "hw_sem", "Half-width (s)"),
        os.path.join(output_dir, "Transient_halfwidth_vs_time.png"))

    return path


# ---- 7. Postictal recovery ----

@_style.styled
def plot_postictal(
    results: Dict[str, PostictalGroupResult],
    output_dir: str,
) -> str:
    """Row 26: cooling curve (rel to max temp, binned by 1C) as mean line + SEM band.
    Rows 27-29: pairwise scatters of final time, temp, and mean ΔF/F.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    fig, axes = plt.subplots(2, 2, figsize=_style.FIG_2X2)

    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.final_time, s.final_temp, s.final_mean_dff])

    # Row 26: cooling curve - mean line + SEM band per cohort.
    # Per spec the x-axis starts at 0 on the LEFT and decreases (more negative)
    # going right - i.e. cooling progression reads left-to-right.
    ax = axes[0, 0]
    for name, r in results.items():
        color = _color(name)
        sem = getattr(r, "cooling_group_sem", None)
        if sem is None:
            sem = np.zeros_like(r.cooling_group_mean)
        _line_sem(ax, r.cooling_bin_centers, r.cooling_group_mean, sem, color, _label(name))
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("Temp rel. to max (°C) - cooling →")
    ax.set_ylabel("Cooling Mean z_ΔF/F")
    ax.invert_xaxis()  # 0 on the LEFT, more negative going right (per spec)
    ax.spines[["right", "top"]].set_visible(False)
    _legend_above(ax)

    def _draw_scatter_pair(ax, x_attr: str, y_attr: str, xlabel: str,
                           ylabel: str, legend: bool = True):
        for name, r in results.items():
            ax.scatter(getattr(r, x_attr), getattr(r, y_attr),
                       color=_color(name), label=_label(name),
                       s=30, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines[["right", "top"]].set_visible(False)
        if legend:
            _legend_above(ax)

    # Summary panels: only the cooling curve carries a legend - cohort
    # colors are obvious from there, so the three scatter panels skip it.
    _draw_scatter_pair(axes[0, 1], "final_times", "final_temps",
                       "Post-Seizure Time (s)", "Final Temp (°C)", legend=False)
    _draw_scatter_pair(axes[1, 0], "final_times", "final_dffs",
                       "Post-Seizure Time (s)", "Final Mean z_ΔF/F", legend=False)
    _draw_scatter_pair(axes[1, 1], "final_temps", "final_dffs",
                       "Final Temp (°C)", "Final Mean z_ΔF/F", legend=False)

    path = os.path.join(output_dir, "summary_postictal_mean.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Standalone export of the cooling curve so it can be dropped directly
    # into slide decks. Same data and (flipped) x-axis as panel [0,0].
    fig_c, ax_c = plt.subplots(figsize=_style.FIG_WIDE)
    for name, r in results.items():
        color = _color(name)
        sem = getattr(r, "cooling_group_sem", None)
        if sem is None:
            sem = np.zeros_like(r.cooling_group_mean)
        _line_sem(ax_c, r.cooling_bin_centers, r.cooling_group_mean, sem,
                  color, _label(name))
    ax_c.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_c.set_xlabel("Temp rel. to max (°C) - cooling →")
    ax_c.set_ylabel("Cooling Mean z_ΔF/F")
    ax_c.invert_xaxis()
    ax_c.spines[["right", "top"]].set_visible(False)
    _legend_above(ax_c)
    fig_c.savefig(os.path.join(output_dir, "Mean_values_vs_temp_cooling.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(fig_c)

    # Standalone final-recording scatter exports.
    _save_standalone(
        lambda ax: _draw_scatter_pair(ax, "final_times", "final_temps",
                                       "Post-Seizure Time (s)", "Final Temp (°C)"),
        os.path.join(output_dir, "Final_time_vs_temp.png"))
    _save_standalone(
        lambda ax: _draw_scatter_pair(ax, "final_times", "final_dffs",
                                       "Post-Seizure Time (s)", "Final Mean z_ΔF/F"),
        os.path.join(output_dir, "Final_time_vs_mean_signal.png"))
    _save_standalone(
        lambda ax: _draw_scatter_pair(ax, "final_temps", "final_dffs",
                                       "Final Temp (°C)", "Final Mean z_ΔF/F"),
        os.path.join(output_dir, "Final_temp_vs_mean_signal.png"))

    _save_csv(
        os.path.join(output_dir, "postictal.csv"),
        ["cohort", "mouse_id", "heating_session",
         "final_time", "final_temp", "final_mean_dff"],
        csv_rows,
    )
    return path


# ---- 7b. Experimental vs isosbestic spike-triggered overlay ----

@_style.styled
def plot_experimental_vs_isosbestic_spike_triggered(
    sessions_by_cohort: Dict[str, List[Session]],
    spike_times_by_cohort: Dict[str, List[np.ndarray]],
    output_dir: str,
    window_s: float = 30.0,
    bl_start_s: float = 5.0,
    bl_end_s: float = 1.0,
) -> str:
    """Spike-triggered overlay of 470 (experimental) and 405 (isosbestic).

    A true neural response should appear in 470 but NOT in 405. Co-modulation
    in 405 flags motion artifact contamination. Saves one panel per cohort.

    Both channels are baseline-subtracted on the per-spike window using the
    pre-event window [-bl_start_s, -bl_end_s]. We do not z-score because the
    relative magnitude of 470 vs 405 swings is the diagnostic of interest.
    Returns the saved PNG path.
    """
    os.makedirs(output_dir, exist_ok=True)
    cohorts = [c for c in _COHORT_ORDER if c in sessions_by_cohort]
    cohorts += [c for c in sessions_by_cohort if c not in _COHORT_ORDER]
    if not cohorts:
        return ""

    fig, axes = plt.subplots(1, len(cohorts),
                             figsize=_style.panel_figsize(len(cohorts)),
                             squeeze=False, sharey=True)
    ylo, yhi = np.inf, -np.inf

    for ax, cohort in zip(axes[0], cohorts):
        sessions = sessions_by_cohort[cohort]
        spikes_list = spike_times_by_cohort.get(cohort, [])
        if not sessions or len(spikes_list) != len(sessions):
            ax.set_title(_label(cohort) + " (no data)")
            ax.axis("off")
            continue

        per_session_470 = []
        per_session_405 = []
        time_axis_ref = None

        for s, spike_times in zip(sessions, spikes_list):
            if s.raw is None or s.processed is None:
                continue
            sig_470 = s.raw.signal_470
            sig_405 = s.raw.signal_405
            fs = s.processed.fs
            if fs is None:
                continue
            w = int(round(window_s * fs))
            bl_lo = w - int(round(bl_start_s * fs))
            bl_hi = w - int(round(bl_end_s * fs))
            seg_470, seg_405 = [], []
            for t_spike in spike_times:
                center = int(round(t_spike * fs))
                start = center - w
                end = center + w + 1
                if start < 0 or end > len(sig_470) or end > len(sig_405):
                    continue
                a = sig_470[start:end].astype(float)
                b = sig_405[start:end].astype(float)
                a = a - float(np.mean(a[bl_lo:bl_hi]))
                b = b - float(np.mean(b[bl_lo:bl_hi]))
                seg_470.append(a)
                seg_405.append(b)
            if seg_470:
                per_session_470.append(np.mean(seg_470, axis=0))
                per_session_405.append(np.mean(seg_405, axis=0))
                if time_axis_ref is None:
                    time_axis_ref = np.linspace(-window_s, window_s, 2 * w + 1)

        if not per_session_470:
            ax.set_title(_label(cohort) + " (no spikes)")
            ax.axis("off")
            continue

        mat_470 = np.array(per_session_470)
        mat_405 = np.array(per_session_405)
        n = mat_470.shape[0]
        mean_470 = np.mean(mat_470, axis=0)
        mean_405 = np.mean(mat_405, axis=0)
        if n > 1:
            sem_470 = np.std(mat_470, axis=0, ddof=1) / np.sqrt(n)
            sem_405 = np.std(mat_405, axis=0, ddof=1) / np.sqrt(n)
        else:
            sem_470 = np.zeros_like(mean_470)
            sem_405 = np.zeros_like(mean_405)

        _line_sem(ax, time_axis_ref, mean_470, sem_470,
                  "tab:green", "470 (experimental)")
        _line_sem(ax, time_axis_ref, mean_405, sem_405,
                  "tab:purple", "405 (isosbestic)")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Time rel. to spike (s)")
        ax.set_ylabel("Δ raw signal (a.u.)")
        ax.set_title(f"{_label(cohort)} (n={n} sessions)")
        # Single-line 48pt title needs ~1.20 to clear the legend.
        _legend_above(ax, y_offset=1.20)
        ax.spines[["right", "top"]].set_visible(False)

        lo = float(np.nanmin([np.nanmin(mean_470 - sem_470),
                              np.nanmin(mean_405 - sem_405)]))
        hi = float(np.nanmax([np.nanmax(mean_470 + sem_470),
                              np.nanmax(mean_405 + sem_405)]))
        ylo = min(ylo, lo)
        yhi = max(yhi, hi)

    if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
        pad = 0.05 * (yhi - ylo)
        for ax in axes[0]:
            if ax.has_data():
                ax.set_ylim(ylo - pad, yhi + pad)

    path = os.path.join(
        output_dir, "AUC_interictal_spikes_experimental_vs_isosbestic.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- 8. Spike-triggered averages ----

@_style.styled
def plot_spike_triggered(
    results: Dict[str, SpikeTriggeredGroupResult],
    output_dir: str,
) -> str:
    """Spec row 17: interictal ECoG spike-triggered photometry average + AUC.

    Saves two standalone PNGs:
      AUC_interictal_spikes_traces.png      (STA traces over ±30s)
      AUC_interictal_spikes_scatterplot.png (per-cohort AUC scatter)
    Plus the companion CSV. The earlier `spike_triggered.png` combo and
    `spike_triggered_zoomed.png` outputs have been retired per request.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session, s.n_spikes, s.auc])

    def _draw_sta(ax, xlim=(-30.0, 30.0)):
        ylo, yhi = np.inf, -np.inf
        for name, r in results.items():
            c = _color(name)
            sem = r.group_sem
            ax.plot(r.time_axis, r.group_mean, color=c, label=_label(name))
            ax.fill_between(r.time_axis, r.group_mean - sem, r.group_mean + sem,
                            color=c, alpha=0.25)
            mask = (r.time_axis >= xlim[0]) & (r.time_axis <= xlim[1])
            if mask.any():
                lo = np.nanmin((r.group_mean - sem)[mask])
                hi = np.nanmax((r.group_mean + sem)[mask])
                ylo = min(ylo, lo)
                yhi = max(yhi, hi)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Time rel. to spike (s)")
        ax.set_ylabel("Spike-Triggered z_ΔF/F (mean ± SEM)")
        ax.set_xlim(xlim)
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            pad = 0.05 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)

    auc_data = {}
    for name, r in results.items():
        auc_vals = [s.auc for s in r.session_results]
        auc_sem = float(np.std(auc_vals, ddof=1) / np.sqrt(len(auc_vals))) if len(auc_vals) > 1 else 0.0
        auc_data[name] = (auc_vals, r.group_auc, auc_sem)

    traces_path = _save_standalone(
        _draw_sta,
        os.path.join(output_dir, "AUC_interictal_spikes_traces.png"),
        figsize=_style.FIG_WIDE,
    )
    _save_standalone_scatter(
        auc_data, "Spike-Triggered AUC",
        os.path.join(output_dir, "AUC_interictal_spikes_scatterplot.png"),
    )

    _save_csv(
        os.path.join(output_dir, "spike_triggered.csv"),
        ["cohort", "mouse_id", "heating_session", "n_spikes", "auc"],
        csv_rows,
    )
    return traces_path


# ---- 9. Per-cohort UEO-aligned mean trace ----

@_style.styled
def plot_ueo_per_cohort(
    results: Dict[str, TriggeredAverage],
    output_dir: str,
    results_extended: Optional[Dict[str, TriggeredAverage]] = None,
    y_lim_override: Optional[tuple] = None,
) -> str:
    """Per-cohort UEO-aligned mean ± SEM trace.

    Generates TWO windows so each can sit alongside the matching heatmaps:
      standard  → -150..+150 s   (driven by `results`)
      extended  → -150..+1000 s  (driven by `results_extended` if given,
                                  else falls back to `results`)

    `results` and `results_extended` should be the output of TWO separate
    `compute_wide_ueo_triggered` calls - one with `post_window_s=150`, one
    with `post_window_s=1000`. That keeps each window's mean/SEM honest
    for whatever cohort N actually spans it, instead of letting a small
    long-tail subset hijack the standard-window stats.

    `y_lim_override` (tuple of (lo, hi)) hard-clamps the y-axis on every
    panel so plots from different subfolders are directly comparable -
    useful when one cohort's signal magnitude would otherwise dwarf
    another's autoscaled axis.

    Outputs per window:
      summary_UEO_aligned_traces[_extended].png        (combo, one panel per cohort)
      UEO_aligned_traces_DS_sz[_extended].png          (standalone per cohort)
      UEO_aligned_traces_DS_nosz[_extended].png
      UEO_aligned_traces_WT[_extended].png

    Y-axes are matched within each window to make magnitudes comparable.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    if results_extended is None:
        results_extended = results
    else:
        results_extended = _ordered(results_extended)

    cohort_fname = {
        "DS_sz": ("UEO_aligned_traces_DS_sz.png",
                  "UEO_aligned_traces_DS_sz_extended.png"),
        "DS_no_sz": ("UEO_aligned_traces_DS_nosz.png",
                     "UEO_aligned_traces_DS_nosz_extended.png"),
        "WT_sz": ("UEO_aligned_traces_WT_sz.png",
                  "UEO_aligned_traces_WT_sz_extended.png"),
        "WT_no_sz": ("UEO_aligned_traces_WT.png",
                     "UEO_aligned_traces_WT_extended.png"),
    }
    WINDOWS = [
        (-150.0, 150.0, "summary_UEO_aligned_traces.png", 0, results),
        (-150.0, 1000.0, "summary_UEO_aligned_traces_extended.png", 1,
         results_extended),
    ]

    primary_path = None

    for x_lo, x_hi, combo_name, w_idx, window_results in WINDOWS:
        # Window-local ylim across all cohorts so the three panels remain
        # directly comparable for THIS window.
        ylo, yhi = np.inf, -np.inf
        for ta in window_results.values():
            if ta.mean_trace is None or len(ta.mean_trace) == 0:
                continue
            mask = (ta.time_axis >= x_lo) & (ta.time_axis <= x_hi)
            if not np.any(mask):
                continue
            sem = ta.sem_trace
            lo = float(np.nanmin((ta.mean_trace - sem)[mask]))
            hi = float(np.nanmax((ta.mean_trace + sem)[mask]))
            if np.isfinite(lo): ylo = min(ylo, lo)
            if np.isfinite(hi): yhi = max(yhi, hi)
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            pad = 0.05 * (yhi - ylo)
            final_ylim = (ylo - pad, yhi + pad)
        else:
            final_ylim = None
        # Hard override wins - cross-subfolder comparability beats
        # per-cohort autoscaling when the caller explicitly asks for it.
        if y_lim_override is not None:
            final_ylim = tuple(y_lim_override)

        # Combo figure
        n = max(len(window_results), 1)
        fig, axes = plt.subplots(1, n,
                                 figsize=_style.panel_figsize(n),
                                 squeeze=False)
        for i, (name, ta) in enumerate(window_results.items()):
            ax = axes[0, i]
            color = _color(name)
            sem = ta.sem_trace
            ax.plot(ta.time_axis, ta.mean_trace, color=color, label=_label(name))
            ax.fill_between(ta.time_axis, ta.mean_trace - sem,
                            ta.mean_trace + sem, color=color, alpha=0.25)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
            ax.set_xlabel("Time rel. to seizure onset (s)")
            ax.set_ylabel(f"{DFF_LABEL} (mean ± SEM)")
            ax.set_title(f"{_label(name)} - aligned to seizure onset")
            ax.set_xlim(x_lo, x_hi)
            if final_ylim is not None:
                ax.set_ylim(*final_ylim)
            ax.spines[["right", "top"]].set_visible(False)

            combo_path = os.path.join(output_dir, combo_name)
        fig.savefig(combo_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if w_idx == 0:
            primary_path = combo_path

        # Per-cohort standalones for this window. No legend - each standalone
        # shows a single line, so the title carries the cohort identity.
        def _draw_single(ax, ta, name, ylim, xl, xh):
            color = _color(name)
            sem = ta.sem_trace
            ax.plot(ta.time_axis, ta.mean_trace, color=color)
            ax.fill_between(ta.time_axis, ta.mean_trace - sem,
                            ta.mean_trace + sem, color=color, alpha=0.25)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4)
            ax.set_xlabel("Time rel. to seizure onset (s)")
            ax.set_ylabel(f"{DFF_LABEL} (mean ± SEM)")
            ax.set_title(f"{_label(name)} - aligned to seizure onset")
            ax.set_xlim(xl, xh)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.spines[["right", "top"]].set_visible(False)

        for name, ta in window_results.items():
            fname = cohort_fname.get(name,
                                     (f"UEO_aligned_traces_{name}.png",
                                      f"UEO_aligned_traces_{name}_extended.png"))[w_idx]
            _save_standalone(
                lambda ax, _t=ta, _n=name, _y=final_ylim, _xl=x_lo, _xh=x_hi:
                    _draw_single(ax, _t, _n, _y, _xl, _xh),
                os.path.join(output_dir, fname),
                figsize=_style.FIG_WIDE,
            )

    return primary_path or os.path.join(output_dir, WINDOWS[0][2])


# ---- 10. Pre-ictal mean diagnostic ----

@_style.styled
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
    # Per spec: mark BOTH ictal max and ictal min on the diagnostic - the
    # experimenter wants to inspect either direction regardless of which has
    # larger magnitude.
    if len(ictal_window):
        ictal_max_val = float(np.max(ictal_window))
        ictal_min_val = float(np.min(ictal_window))
        delta_max = ictal_max_val - elh_mean
        delta_min = ictal_min_val - elh_mean
        max_idx = i_ueo + int(np.argmax(ictal_window))
        min_idx = i_ueo + int(np.argmin(ictal_window))
    else:
        ictal_max_val = np.nan
        ictal_min_val = np.nan
        delta_max = np.nan
        delta_min = np.nan
        max_idx = None
        min_idx = None

    # 2-row stacked with suptitle + huge labels + per-axis annotations
    # (the Δ_max / Δ_min callouts with boxed text) - well outside the canon
    # FIG_2X2 budget. Constrained_layout collapses axes to zero at smaller
    # sizes here, so use a generously oversized canvas.
    fig, axes = plt.subplots(2, 1, figsize=(_style.FIG_2X2[0] * 1.4,
                                            _style.FIG_2X2[1] * 2.0))

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
    ax.set_ylabel("z_ΔF/F (full window)")
    _legend_above(ax)
    ax.spines[["right", "top"]].set_visible(False)

    # Panel 1: zoom on [ueo - 15s, end] with BOTH min/max delta annotations.
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

    def _mark_peak(idx, val, delta, mark_color, marker, label_prefix, offset_sign):
        if idx is None or idx >= len(time_axis) or np.isnan(val):
            return
        peak_t = time_axis[idx]
        peak_v = signal[idx]
        ax2.scatter([peak_t], [peak_v], color=mark_color, s=60, zorder=5,
                    marker=marker,
                    label=f"Ictal {label_prefix} = {peak_v:.3f}")
        ax2.annotate(
            f"Δ_{label_prefix} = {peak_v:.3f} − {elh_mean:.3f} = {delta:+.3f}",
            xy=(peak_t, peak_v),
            xytext=(peak_t + 1.0, peak_v + offset_sign * 0.10 * (abs(peak_v) + 1)),
            fontsize=24,
            arrowprops=dict(arrowstyle="->", color=mark_color, lw=1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=mark_color),
        )

    _mark_peak(max_idx, ictal_max_val, delta_max, "tab:green", "^", "max", +1)
    _mark_peak(min_idx, ictal_min_val, delta_min, "tab:purple", "v", "min", -1)
    ax2.axvline(ueo_t, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel(f"Time (s) - zoomed around seizure onset = {ueo_t:.1f}s")
    ax2.set_ylabel("z_ΔF/F (zoomed)")
    _legend_above(ax2)
    ax2.spines[["right", "top"]].set_visible(False)

    fig.suptitle(
        f"Parameter detection - {session.mouse_id} S{session.heating_session} "
        f"({_label(session.cohort) if session.cohort else 'unknown cohort'})",
    )

    os.makedirs(output_dir, exist_ok=True)
    fname = (
        f"{session.mouse_id}_S{session.heating_session}_parameter_detection.png"
    )
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path
