"""
Group-level summary plots.

Each function receives one or more cohort result objects (keyed by cohort name),
plots with cohort colors, saves PNG + companion CSV, and returns the saved path.
"""

import csv
import math
import os
import zlib
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from . import _style
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
    COHORT_DISPLAY_LABELS, cohort_color, DFF_LABEL, landmark_label,
)
from .trace_plots import _cohort_filename_tag


def _color(cohort: str) -> str:
    return cohort_color(cohort)


_TEMP_SCALAR_FIELDS = (
    "baseline_temp", "max_temp", "final_temp",
    "seizure_threshold_temp", "max_seizure_temp",
)


def _has_finite_value(v) -> bool:
    """True if v contains at least one finite (non-None, non-NaN) number."""
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and math.isnan(v))
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return False
        try:
            return bool(np.any(np.isfinite(v)))
        except (TypeError, ValueError):
            return False
    return False


def _any_temperature_data(results) -> bool:
    """True if any cohort in `results` carries finite temperature data.

    Gates temperature-related plots (vs_temp scatters, baseline_temp bars,
    etc.) when sessions were recorded without a thermistor (e.g. opto-stim
    experiments). Polymorphic across GroupResult types: checks both the
    standard scalar per-session temp fields (CohortSessionResult,
    PostictalSessionResult) and group-level temp_*-prefixed arrays
    (PreictalMeanGroupResult, PreictalTransientGroupResult).
    """
    for r in results.values():
        # Group-level temp_*-prefixed arrays cover preictal mean/transient
        # GroupResults where per-session scalars don't exist but group-level
        # arrays are computed from per-session temperature traces.
        for attr in dir(r):
            if not attr.startswith("temp_"):
                continue
            if attr in ("temp_bin_centers", "temp_ma_centers"):
                # These are bin edge grids derived from config; populated
                # regardless of whether any session has temperature data.
                continue
            if _has_finite_value(getattr(r, attr, None)):
                return True

        for sr in getattr(r, "session_results", []) or []:
            for attr in _TEMP_SCALAR_FIELDS:
                v = getattr(sr, attr, None)
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                return True
    return False


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


def _seizure_cohort_key(results: Dict) -> str:
    """Return a seizure cohort from genotype-specific or pooled groups."""
    fallback = next(iter(results.keys()))
    return next(
        (
            key for key in ("Seizure", "seizure", "DS_sz", "WT_sz")
            if key in results
        ),
        fallback,
    )


# ---- helpers ----

def _is_nan_like(v) -> bool:
    """True for None, NaN floats, and empty/'nan'/'none' strings.

    Used to detect param-CSV columns that carry no information for the
    current run (e.g. temperature-derived measurements in opto runs).
    """
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "nan", "none"):
        return True
    return False


def _save_csv(path: str, header: list, rows: list) -> None:
    # Omit columns containing only NaN-like values; retain populated metadata.
    if rows:
        keep = [
            ci for ci in range(len(header))
            if not all(_is_nan_like(r[ci]) for r in rows)
        ]
        if len(keep) < len(header):
            header = [header[ci] for ci in keep]
            rows = [[r[ci] for ci in keep] for r in rows]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _bar_scatter(ax, cohorts: Dict[str, tuple], ylabel: str) -> None:
    """Bar + individual scatter overlay.

    cohorts: {name: (per_session_values, mean, sem)}
    """
    rng = np.random.default_rng(42)
    for i, (name, (vals, mean, sem)) in enumerate(cohorts.items()):
        color = _color(name)
        ax.bar(i, mean, yerr=sem, color=color, alpha=0.5, capsize=8,
               width=0.6, error_kw={"elinewidth": 3, "capthick": 3})
        ax.scatter(
            np.full(len(vals), i) + rng.uniform(-0.08, 0.08, len(vals)),
            vals, color=color, s=150, zorder=3, alpha=0.85,
            edgecolor="black", linewidth=0.6,
        )
    _set_categorical_xticklabels(
        ax, range(len(cohorts)), [_label(k) for k in cohorts.keys()],
    )
    ax.set_ylabel(ylabel)
    ax.spines[["right", "top"]].set_visible(False)


def _session_id(result) -> tuple:
    """Stable pairing/color key shared by all per-session result types."""
    return (result.mouse_id, result.heating_session)


def _stable_jitter(key, width: float) -> float:
    """Deterministic horizontal jitter so paired points share an offset."""
    checksum = zlib.crc32(repr(key).encode("utf-8"))
    return ((checksum / 0xFFFFFFFF) * 2.0 - 1.0) * width


def _scatter_entry(entry: tuple) -> tuple:
    """Return values, mean, SEM, and optional per-value session IDs."""
    if len(entry) == 3:
        vals, mean, sem = entry
        return vals, mean, sem, None
    if len(entry) == 4:
        vals, mean, sem, ids = entry
        if ids is not None and len(ids) != len(vals):
            raise ValueError("Scatter metadata length must match values length")
        return vals, mean, sem, ids
    raise ValueError("Scatter entries must contain 3 or 4 values")


def _set_categorical_xticklabels(
    ax,
    positions,
    labels,
    rotation: float = 45.0,
) -> None:
    """Apply the publication-style categorical x-axis label layout."""
    ax.set_xticks(positions)
    rotation_kwargs = {"rotation": rotation}
    if rotation != 0:
        rotation_kwargs.update(ha="right", rotation_mode="anchor")
    ax.set_xticklabels(labels, **rotation_kwargs)


def _scatter_mean_sem(
    ax,
    cohorts: Dict[str, tuple],
    ylabel: str,
    xtick_rotation: Optional[float] = None,
    scatter_options: Optional[dict] = None,
    zero_line: bool = False,
) -> None:
    """Pure scatter plot: jittered per-session dots, mean tick, SEM whiskers.

    cohorts: {name: (per_session_values, mean, sem[, session_ids])}
    `xtick_rotation`: x-tick label rotation in degrees; None uses 45 degrees.
    """
    labels = [_label(k) for k in cohorts.keys()]
    if xtick_rotation is None:
        xtick_rotation = 45.0
    scatter_options = scatter_options or {}
    connect_pairs = bool(scatter_options.get("connect_pairs"))
    point_colors = scatter_options.get("point_colors_by_id", {})
    pair_color = scatter_options.get("pair_line_color", "#6b6b6b")
    pair_alpha = float(scatter_options.get("pair_line_alpha", 0.55))
    rng = np.random.default_rng(42)
    n = max(len(cohorts), 1)
    prepared = []
    pair_positions = {}
    for i, (name, entry) in enumerate(cohorts.items()):
        vals, mean, sem, ids = _scatter_entry(entry)
        color = _color(name)
        if len(vals) > 0:
            if connect_pairs and ids is not None:
                jitter = np.array([_stable_jitter(key, 0.06) for key in ids])
            else:
                jitter = rng.uniform(-0.06, 0.06, len(vals))
            x = np.full(len(vals), i, dtype=float) + jitter
            colors = [
                point_colors.get((name, key), point_colors.get(key, color))
                for key in ids
            ] if ids else color
            prepared.append((x, vals, colors))
            if connect_pairs and ids is not None:
                for key, x_val, y_val in zip(ids, x, vals):
                    if y_val is not None and np.isfinite(y_val):
                        pair_positions.setdefault(key, []).append((x_val, y_val))
        else:
            prepared.append((np.array([]), vals, color))
        ax.errorbar(i, mean, yerr=sem, fmt="_", color="black", markersize=60,
                    capsize=12, elinewidth=4, markeredgewidth=5, zorder=5)
    if connect_pairs:
        for points in pair_positions.values():
            if len(points) < 2:
                continue
            points = sorted(points)
            ax.plot(
                [p[0] for p in points], [p[1] for p in points],
                color=pair_color, alpha=pair_alpha, linewidth=1.8, zorder=2,
            )
    for x, vals, colors in prepared:
        if len(vals) > 0:
            ax.scatter(
                x, vals, color=colors, s=180, zorder=3, alpha=0.85,
                edgecolor="black", linewidth=0.6,
            )
    _set_categorical_xticklabels(ax, range(n), labels, xtick_rotation)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylabel(ylabel)
    if zero_line:
        _style.add_zero_y(ax, ensure_visible=True)
    ax.spines[["right", "top"]].set_visible(False)


def _line_sem(ax, x, mean, sem, color, label) -> None:
    ax.plot(x, mean, color=color, label=label, linewidth=3.0)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25)


def _legend_above(ax, ncol: Optional[int] = None, y_offset: float = 1.02) -> None:
    """Center the legend above the axes."""
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


def _save_standalone_scatter(cohorts: Dict[str, tuple], ylabel: str, path: str,
                             xtick_rotation: Optional[float] = None,
                             title: Optional[str] = None,
                             scatter_options: Optional[dict] = None,
                             zero_line: bool = False) -> None:
    """Save a single-panel cohort scatter (mean ± SEM) at `path`.

    Figure width increases with cohort count above the two-cohort base size.

    `xtick_rotation`: pass-through to `_scatter_mean_sem`. None (default)
    applies the default 45-degree rotation.
    `title`: retained for caller compatibility; publication export suppresses it.
    """
    n_cohorts = max(len(cohorts), 1)
    base_w, base_h = _style.FIG_SCATTER
    requested_width = float((scatter_options or {}).get("standalone_width", 0.0))
    width = max(base_w, 3.0 + 2.5 * n_cohorts, requested_width)
    fig, ax = plt.subplots(figsize=(width, base_h))
    _scatter_mean_sem(
        ax, cohorts, ylabel, xtick_rotation=xtick_rotation,
        scatter_options=scatter_options, zero_line=zero_line,
    )
    if title:
        ax.set_title(title)
    _save_publication_figure(fig, path)
    plt.close(fig)


def _legend_offset_above_title(title: str) -> float:
    """Return a legend anchor that clears a publication-size axis title."""
    return 1.27 + 0.18 * title.count("\n")


def _save_publication_figure(fig, path, *, dpi: int = 150) -> None:
    _style.finalize_publication_figure(fig)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _save_standalone(draw_fn, path: str, figsize=None, axes_size_inches=None) -> str:
    """Generic standalone-export helper.

    Creates a single-axis figure, calls draw_fn(ax), saves and closes it.
    Returns the saved path. The draw_fn is responsible for axis content;
    publication export removes titles and legends.

    The default size is FIG_WIDE. Scatter panels pass
    `_scatter_figsize(n_cohorts)` to scale width with the number of cohorts.
    """
    if figsize is None:
        figsize = _style.FIG_WIDE
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    if axes_size_inches is not None:
        _style.fix_axes_size(ax, axes_size_inches)
    _save_publication_figure(fig, path)
    plt.close(fig)
    return path


def _scatter_figsize(n_cohorts: int) -> tuple:
    """Scatter figsize scaled to cohort count.

    Uses FIG_SCATTER height and max(FIG_SCATTER[0], 3.0 + 2.5 * n_cohorts) width.
    """
    n = max(int(n_cohorts), 1)
    base_w, base_h = _style.FIG_SCATTER
    return (max(base_w, 3.0 + 2.5 * n), base_h)


def _grouped_scatter_figsize(n_cohorts: int, n_periods: int) -> tuple:
    """Scatter size that also leaves room for rotated period labels."""
    cohort_width, height = _scatter_figsize(n_cohorts)
    period_width = 3.0 + 3.0 * max(int(n_periods), 1)
    return (max(cohort_width, period_width), height)


# ---- 1. Cohort characteristics ----

@_style.styled
def plot_cohort_characteristics(
    results: Dict[str, CohortResult],
    output_dir: str,
    name_prefix: str = "",
    scatter_options: Optional[dict] = None,
) -> str:
    """Write standalone baseline-temperature, seizure-threshold, and duration plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

    baseline = {}
    csv_rows = []
    for name, r in results.items():
        selected = [s for s in r.session_results if s.baseline_temp is not None]
        bl_vals = [s.baseline_temp for s in selected]
        ids = [_session_id(s) for s in selected]
        baseline[name] = (
            bl_vals, r.baseline_temp_mean, r.baseline_temp_sem, ids,
        )
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.baseline_temp, s.seizure_threshold_temp,
                             s.seizure_duration_s])

    # Shared temperature limits across cohort plots.
    TEMP_YLIM = (35.0, 43.0)

    def _draw_baseline(ax):
        _scatter_mean_sem(
            ax, baseline, "Baseline Temp (°C)",
            scatter_options=scatter_options,
        )
        ax.set_ylim(*TEMP_YLIM)

    def _draw_per_mouse(ax, attr: str, ylabel: str, fixed_ylim=None):
        # Prefer the DS-sz cohort; fall back to WT-sz (e.g. PTZ runs) so
        # this panel always draws the seized cohort when one exists.
        seizure_key = _seizure_cohort_key(results)
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
            ax.plot(xs, ys, "-o", color=color, alpha=0.7,
                    markersize=14, linewidth=2.5,
                    markeredgecolor="black", markeredgewidth=0.6)
            for x, y in pts:
                all_by_session.setdefault(x, []).append(y)
        for x, ys in sorted(all_by_session.items()):
            mean = float(np.mean(ys))
            sem = float(np.std(ys, ddof=1) / np.sqrt(len(ys))) if len(ys) > 1 else 0.0
            ax.errorbar(x, mean, yerr=sem, fmt="_", color="black",
                        markersize=60, capsize=12, elinewidth=4,
                        markeredgewidth=5, zorder=5)
        ax.set_xlabel("Session #")
        ax.set_ylabel(ylabel)
        if all_by_session:
            ax.set_xticks(sorted(all_by_session.keys()))
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        ax.spines[["right", "top"]].set_visible(False)

    # Export individual panels; omit temperature panels when data is absent.
    has_temp = _any_temperature_data(results)
    baseline_path = None
    if has_temp:
        baseline_path = _save_standalone(
            _draw_baseline,
            os.path.join(output_dir, "Cohort_characteristics_baseline_temp.png"),
            figsize=_scatter_figsize(len(results)))
        _save_standalone(
            lambda ax: _draw_per_mouse(ax, "seizure_threshold_temp",
                                       "Seizure Threshold (°C)", TEMP_YLIM),
            os.path.join(output_dir, "Cohort_characteristics_seizure_threshold.png"),
            figsize=_scatter_figsize(len(results)))
    duration_path = _save_standalone(
        lambda ax: _draw_per_mouse(ax, "seizure_duration_s", "Seizure Duration (s)"),
        os.path.join(output_dir, "Cohort_characteristics_seizure_duration.png"),
        figsize=_scatter_figsize(len(results)))

    csv_name = (
        f"cohort_characteristics_{name_prefix}.csv"
        if name_prefix else "cohort_characteristics.csv"
    )
    _save_csv(
        os.path.join(output_dir, csv_name),
        ["cohort", "mouse_id", "heating_session",
         "baseline_temp", "seizure_threshold_temp", "seizure_duration_s"],
        csv_rows,
    )
    return baseline_path or duration_path


# ---- 2. Baseline transients ----

@_style.styled
def plot_baseline_transients(
    results: Dict[str, BaselineTransientGroupResult],
    output_dir: str,
    scatter_options: Optional[dict] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)

    freq, amp, hw = {}, {}, {}
    csv_rows = []
    for name, r in results.items():
        f_vals = [s.frequency_hz for s in r.session_results]
        a_vals = [s.mean_amplitude for s in r.session_results]
        h_vals = [s.mean_half_width_s for s in r.session_results]
        ids = [_session_id(s) for s in r.session_results]
        freq[name] = (f_vals, r.frequency_mean, r.frequency_sem, ids)
        amp[name] = (a_vals, r.amplitude_mean, r.amplitude_sem, ids)
        hw[name] = (h_vals, r.half_width_mean, r.half_width_sem, ids)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.frequency_hz, s.mean_amplitude, s.mean_half_width_s])

    fig, axes = plt.subplots(1, 3, figsize=_style.panel_figsize(3))
    _scatter_mean_sem(
        axes[0], freq, "Transient\nFrequency\n(Hz)",
        scatter_options=scatter_options,
    )
    _scatter_mean_sem(
        axes[1], amp, "Transient\nAmplitude\n(z_ΔF/F)",
        scatter_options=scatter_options,
    )
    _scatter_mean_sem(
        axes[2], hw, "Transient\nHalf-width\n(s)",
        scatter_options=scatter_options,
    )
    path = os.path.join(output_dir, "summary_baseline_transients.png")
    _save_publication_figure(fig, path)
    plt.close("all")

    _save_standalone_scatter(
        freq, "Transient\nFrequency\n(Hz)",
        os.path.join(output_dir, "Transient_frequency_baseline.png"),
        scatter_options=scatter_options,
    )
    _save_standalone_scatter(
        amp, "Transient\nAmplitude\n(z_ΔF/F)",
        os.path.join(output_dir, "Transient_amplitude_baseline.png"),
        scatter_options=scatter_options,
    )
    _save_standalone_scatter(
        hw, "Transient\nHalf-width\n(s)",
        os.path.join(output_dir, "Transient_halfwidth_baseline.png"),
        scatter_options=scatter_options,
    )

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
    scatter_options: Optional[dict] = None,
    zero_line: bool = True,
) -> None:
    """Multi-period grouped scatter with mean/SEM ticks per cohort.

    period_data: list (one entry per period) of {cohort_name: (vals, mean, sem)}.
    All periods use the same set of cohorts. Cohorts are clustered within each
    period; periods are spaced `period_spacing` units apart on the x-axis
    (default 1.8).
    """
    cohorts = list(period_data[0].keys()) if period_data else []
    n_cohorts = max(len(cohorts), 1)
    cluster_w = 0.8
    sub_w = cluster_w / n_cohorts
    scatter_options = scatter_options or {}
    connect_pairs = bool(scatter_options.get("connect_pairs"))
    point_colors = scatter_options.get("point_colors_by_id", {})
    pair_color = scatter_options.get("pair_line_color", "#6b6b6b")
    pair_alpha = float(scatter_options.get("pair_line_alpha", 0.55))
    rng = np.random.default_rng(42)
    period_centers = np.arange(len(period_data), dtype=float) * period_spacing
    prepared = []
    pair_positions = {}
    for p_idx, pdata in enumerate(period_data):
        for c_idx, name in enumerate(cohorts):
            if name not in pdata:
                continue
            vals, mean, sem, ids = _scatter_entry(pdata[name])
            color = _color(name)
            x_center = period_centers[p_idx] + (c_idx - (n_cohorts - 1) / 2) * sub_w
            if len(vals) > 0:
                if connect_pairs and ids is not None:
                    jitter = np.array([
                        _stable_jitter(key, sub_w * 0.15) for key in ids
                    ])
                else:
                    jitter = rng.uniform(-sub_w * 0.15, sub_w * 0.15, len(vals))
                x = np.full(len(vals), x_center) + jitter
                colors = [
                    point_colors.get((name, key), point_colors.get(key, color))
                    for key in ids
                ] if ids else color
                prepared.append((x, vals, colors, name, p_idx))
                if connect_pairs and ids is not None:
                    for key, x_val, y_val in zip(ids, x, vals):
                        if y_val is not None and np.isfinite(y_val):
                            # Pair treatment/control cohorts within the same
                            # analysis period. Do not draw one polyline across
                            # different biological periods.
                            pair_positions.setdefault((key, p_idx), []).append(
                                (x_val, y_val)
                            )
            if not np.isnan(mean):
                ax.errorbar(x_center, mean, yerr=sem if not np.isnan(sem) else 0,
                            fmt="_", color="black", markersize=40,
                            capsize=8, elinewidth=3, markeredgewidth=4, zorder=5)
    if connect_pairs:
        for points in pair_positions.values():
            if len(points) < 2:
                continue
            points = sorted(points)
            ax.plot(
                [p[0] for p in points], [p[1] for p in points],
                color=pair_color, alpha=pair_alpha, linewidth=1.8, zorder=2,
            )
    for x, vals, colors, name, p_idx in prepared:
        ax.scatter(
            x, vals, color=colors, s=120, zorder=3, alpha=0.85,
            edgecolor="black", linewidth=0.6,
            label=_label(name) if p_idx == 0 else None,
        )
    if zero_line:
        _style.add_zero_y(ax, ensure_visible=True)
    _set_categorical_xticklabels(ax, period_centers, period_labels)
    ax.set_ylabel(ylabel)
    ax.spines[["right", "top"]].set_visible(False)
    if cohorts:
        _legend_above(ax)


@_style.styled
def plot_preictal_mean(
    results: Dict[str, PreictalMeanGroupResult],
    output_dir: str,
    name_prefix: str = "",
    scatter_options: Optional[dict] = None,
    trace_results: Optional[Dict[str, PreictalMeanGroupResult]] = None,
) -> str:
    """Panel 0: consolidated [Heating, Ictal] scatter.
    Panel 1: split [Early Heat, Late Heat, Ictal] scatter.
    Panel 2: end-of-late-heat scatter.
    Panel 3: temp-binned mean signal during heating (rel. to seizure onset temp).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    line_results = _ordered(trace_results or results)
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
            selected = [s for s in r.session_results
                        if getattr(s, attr) is not None]
            vals = [getattr(s, attr) for s in selected]
            ids = [_session_id(s) for s in selected]
            mean = (float(np.mean(vals)) if vals else np.nan)
            if len(vals) > 1:
                sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            else:
                sem = 0.0 if vals else np.nan
            out[name] = (vals, mean, sem, ids)
        return out

    # Panel 0: consolidated [Heating, Ictal]
    heating_data = _attr_data("heating_mean")
    ictal_data = _attr_data("ictal_mean")
    _grouped_scatter_mean_sem(
        axes[0],
        [heating_data, ictal_data],
        ["Heating", "Ictal"],
        "Mean z_ΔF/F",
        scatter_options=scatter_options,
    )

    # Panel 1: split [Early Heat, Late Heat, Ictal]
    eh_data = _attr_data("early_heat_mean")
    lh_data = _attr_data("late_heat_mean")
    _grouped_scatter_mean_sem(
        axes[1],
        [eh_data, lh_data, ictal_data],
        ["Early Heat", "Late Heat", "Ictal"],
        "Mean z_ΔF/F",
        scatter_options=scatter_options,
    )

    # Panel 2: end of late heat scatter
    elh = {}
    for name, r in results.items():
        vals = [s.end_late_heat_mean for s in r.session_results]
        ids = [_session_id(s) for s in r.session_results]
        elh[name] = (vals, r.end_late_heat_mean, r.end_late_heat_sem, ids)
    _scatter_mean_sem(
        axes[2], elh, "End late heat\n(z_ΔF/F)",
        scatter_options=scatter_options, zero_line=True,
    )

    # Panel 3: mean signal vs temperature during heating
    ax = axes[3]
    for name, r in line_results.items():
        _line_sem(ax, r.temp_bin_centers, r.temp_bin_group_mean,
                  r.temp_bin_group_sem, _color(name), _label(name))
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_xlabel("Temp rel. to\nseizure onset (°C)")
    ax.set_ylabel("Mean z_ΔF/F")
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)

    path = os.path.join(output_dir, "summary_preictal_mean.png")
    _save_publication_figure(fig, path)
    plt.close("all")

    if _any_temperature_data(line_results):
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

    # By-period grouped scatters.
        _save_standalone(
            lambda ax: _grouped_scatter_mean_sem(
                ax, [heating_data, ictal_data], ["Heating", "Ictal"],
                "Mean z_ΔF/F", scatter_options=scatter_options),
            os.path.join(output_dir, "Mean_values_all_heat_ictal_by_period.png"),
            figsize=_grouped_scatter_figsize(len(results), 2),
        )
        _save_standalone(
            lambda ax: _grouped_scatter_mean_sem(
                ax, [eh_data, lh_data, ictal_data],
                ["Early Heat", "Late Heat", "Ictal"], "Mean z_ΔF/F",
                scatter_options=scatter_options),
            os.path.join(output_dir,
                         "Mean_values_early_heat_late_heat_ictal_by_period.png"),
            figsize=_grouped_scatter_figsize(len(results), 3),
        )

        # Immediate-preictal scatter; plot_ictal_mean owns Mean_values_ictal.png.
        _save_standalone_scatter(
            elh, "Immediate preictal\n(z_ΔF/F)",
            os.path.join(output_dir, "Mean_values_immediate_preictal.png"),
            scatter_options=scatter_options,
            zero_line=True,
        )

    # Plot signal versus temperature only when temperature data is available.
    if _any_temperature_data(line_results):
        def _draw_vs_temp_heating(ax):
            for name, r in line_results.items():
                _line_sem(ax, r.temp_bin_centers, r.temp_bin_group_mean,
                          r.temp_bin_group_sem, _color(name), _label(name))
            _style.add_zero_y(ax, ensure_visible=True)
            ax.set_xlabel("Temp rel. to seizure onset (°C)")
            ax.set_ylabel("Mean z_ΔF/F")
            ax.spines[["right", "top"]].set_visible(False)
            _legend_above(ax)
        _save_standalone(_draw_vs_temp_heating,
                         os.path.join(output_dir, "Mean_values_vs_temp_heating.png"),
                         figsize=_style.FIG_WIDE)

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

    NaN cells render transparently via the colormap's `set_bad` setting.
    Mean/SEM traces are computed separately by `compute_wide_ueo_triggered`,
    which also NaN-pads incomplete windows and uses pointwise coverage.

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
    vlim_override: Optional[float] = None,
    diagnostics_out: Optional[Dict[str, float]] = None,
    filename_prefix: str = "UEO_aligned",
    xlabel: str = "Time rel. to seizure onset (s)",
    dff_label: Optional[str] = None,
    exclude_from_vlim: Optional[List[str]] = None,
    standard_axes_size_inches: Optional[tuple] = None,
) -> Dict[str, str]:
    """Plot standard and extended per-session heatmaps by cohort.

    Color limits are shared across cohorts within each window. diagnostics_out
    receives the standard-window color limit; the return value maps cohorts
    to standard-window image paths.
    """
    if exclude_from_vlim is None:
        exclude_from_vlim = []
    exclude_set = set(exclude_from_vlim)
    import matplotlib.cm as cm

    os.makedirs(output_dir, exist_ok=True)
    canonical_fname = {
        "DS_sz": (f"{filename_prefix}_heatmap_DS_sz.png",
                  f"{filename_prefix}_heatmap_DS_sz_extended.png"),
        "DS_no_sz": (f"{filename_prefix}_heatmap_DS_nosz.png",
                     f"{filename_prefix}_heatmap_DS_nosz_extended.png"),
        "WT_sz": (f"{filename_prefix}_heatmap_WT_sz.png",
                  f"{filename_prefix}_heatmap_WT_sz_extended.png"),
        "WT_no_sz": (f"{filename_prefix}_heatmap_WT.png",
                     f"{filename_prefix}_heatmap_WT_extended.png"),
    }

    def _fname_for(cohort_key: str) -> tuple:
        return canonical_fname.get(
            cohort_key,
            (f"{filename_prefix}_heatmap_{cohort_key}.png",
             f"{filename_prefix}_heatmap_{cohort_key}_extended.png"),
        )

    def _xlabel_for(cohort_key: str) -> str:
        normalized = cohort_key.strip().lower().replace(" ", "_")
        if (xlabel == "Time rel. to seizure onset (s)"
                and (normalized.endswith("_no_sz")
                     or normalized == "no_seizure")):
            return "Time rel. to event onset (s)"
        return xlabel

    WINDOWS = [
        (-pre_s, post_s, 0),     # standard
        (-pre_s, 1000.0, 1),     # extended (-150..+1000)
    ]

    # Collect cohort data before setting shared color limits for each window.
    per_window_cohort_data: List[Dict[str, tuple]] = [{}, {}]
    for x_lo, x_hi, w_idx in WINDOWS:
        win_post = x_hi  # x_hi is the post-event extent for this window
        for cohort_key in cohort_sessions:
            sessions = cohort_sessions.get(cohort_key, [])
            mat, time_axis, sess_labels = _build_ueo_heatmap_data(
                sessions, pre_s=pre_s, post_s=win_post)
            if mat.size == 0:
                continue
            per_window_cohort_data[w_idx][cohort_key] = (mat, time_axis, sess_labels)

    saved: Dict[str, str] = {}

    # Signed color scale with transparent NaN cells.
    cmap = cm.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white", alpha=0.0)

    CLIP_PERCENTILE = 98.0

    for x_lo, x_hi, w_idx in WINDOWS:
        cohort_data = per_window_cohort_data[w_idx]
        if not cohort_data:
            continue
        if vlim_override is not None:
            # Use the caller's shared color limit.
            vlim = float(vlim_override)
        else:
            # Compute a shared symmetric scale from finite values. Excluded
            # cohorts are plotted but do not contribute to scale estimation.
            scale_data = [m[0].ravel() for ck, m in cohort_data.items()
                          if m[0].size and ck not in exclude_set]
            if not scale_data:
                # If all contributing cohorts were excluded, use all data.
                scale_data = [m[0].ravel() for m in cohort_data.values()
                              if m[0].size]
            all_vals = (np.concatenate(scale_data) if scale_data
                        else np.array([]))
            if all_vals.size:
                finite = all_vals[np.isfinite(all_vals)]
                vlim = float(np.percentile(np.abs(finite), CLIP_PERCENTILE)) \
                    if finite.size else 1.0
                vlim = max(vlim, 1e-6)  # avoid zero range when data is all-NaN
            else:
                vlim = 1.0
        if w_idx == 0 and diagnostics_out is not None:
            diagnostics_out["vlim_standard"] = vlim

        for cohort_key, (mat, t_clip, sess_labels) in cohort_data.items():
            n_sessions = mat.shape[0]
            fname = _fname_for(cohort_key)[w_idx]
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
            _style.add_zero_x(ax)
            ax.set_xlabel(_xlabel_for(cohort_key))
            ax.set_xlim(x_lo, x_hi)
            ax.set_yticks(np.arange(n_sessions))
            ax.set_yticklabels([str(i + 1) for i in range(n_sessions)],
                               fontsize=_style.TICK_SIZE)
            title_prefix = filename_prefix.replace("_", " ")
            cbar_label = dff_label if dff_label else DFF_LABEL
            ax.set_title(f"{title_prefix} {cbar_label} - {_label(cohort_key)}")
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label(cbar_label)
            if w_idx == 0 and standard_axes_size_inches is not None:
                _style.fix_axes_size(ax, standard_axes_size_inches, colorbar=cbar)
            path = os.path.join(output_dir, fname)
            _save_publication_figure(fig, path)
            plt.close(fig)
            if w_idx == 0:
                saved[cohort_key] = path

    return saved


def _add_nested_period_labels(
    ax,
    cohort_centers,
    period_offsets,
    period_labels,
    cohort_labels,
) -> list:
    """Add rotated period sublabels above cohort labels."""
    texts = []
    for cohort_center in cohort_centers:
        for period_offset, period_label in zip(period_offsets, period_labels):
            texts.append(ax.text(
                cohort_center + period_offset,
                -0.03,
                period_label,
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=_style.TICK_SIZE,
                color="dimgray",
                rotation=45,
                rotation_mode="anchor",
                clip_on=False,
            ))
    ax.set_xticks(cohort_centers)
    ax.set_xticklabels(cohort_labels)
    ax.tick_params(axis="x", which="major", pad=240)
    return texts


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
                    "-o", color=color, alpha=0.55,
                    markersize=12, linewidth=2.0,
                    markeredgecolor="black", markeredgewidth=0.6)
        if per_session_vectors:
            mat = np.array(per_session_vectors)
            means = np.mean(mat, axis=0)
            sems = (np.std(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
                    if mat.shape[0] > 1 else np.zeros(mat.shape[1]))
            ax.errorbar(x_positions, means, yerr=sems,
                        fmt="-o", color=color, linewidth=4.0,
                        markersize=18, capsize=8,
                        elinewidth=3, markeredgewidth=2, zorder=5)

    # Period sub-labels sit under the data and above the cohort group names.
    _add_nested_period_labels(
        ax,
        cohort_centers,
        period_offsets,
        period_labels,
        [_label(name) for name in results.keys()],
    )
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, n_cohorts - 1 + 0.6)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.spines[["right", "top"]].set_visible(False)
    _save_publication_figure(fig, path)
    plt.close(fig)


# ---- 4. Pre-ictal transients ----

@_style.styled
def plot_preictal_transients(
    results: Dict[str, PreictalTransientGroupResult],
    output_dir: str,
    trace_results: Optional[Dict[str, PreictalTransientGroupResult]] = None,
) -> str:
    """Pre-ictal transient properties as temperature-aligned moving averages.

    Plots temperature moving averages relative to seizure onset, without
    time-aligned or temperature-binned panels.

    Also writes 3 standalone files (Transient_frequency_moving_average_vs_temp_heating.png,
    _amplitude_*.png, _halfwidth_*.png).

    Entirely temperature-based: returns "" without writing any files when
    no cohort has temperature data (e.g. opto sessions).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(trace_results or results)

    if not _any_temperature_data(results):
        return ""

    def _draw_panel(ax, attr_mean: str, attr_sem: str, ylabel: str):
        for name, r in results.items():
            c = _color(name)
            _line_sem(ax, r.temp_ma_centers,
                      getattr(r, attr_mean), getattr(r, attr_sem),
                      c, _label(name))
        ax.set_xlabel("Temp rel. to\nseizure onset (°C)")
        ax.set_ylabel(ylabel)
        _style.add_zero_y(ax, ensure_visible=True)
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)

    fig, axes = plt.subplots(1, 3, figsize=_style.panel_figsize(3))
    _draw_panel(axes[0], "temp_ma_freq_mean", "temp_ma_freq_sem",
                "Moving avg\nfrequency\n(Hz)")
    _draw_panel(axes[1], "temp_ma_amp_mean", "temp_ma_amp_sem",
                "Moving avg\namplitude\n(z_ΔF/F)")
    _draw_panel(axes[2], "temp_ma_hw_mean", "temp_ma_hw_sem",
                "Moving avg\nhalf-width\n(s)")

    path = os.path.join(output_dir, "summary_preictal_transients.png")
    _save_publication_figure(fig, path)
    plt.close("all")

    _save_standalone(
        lambda ax: _draw_panel(ax, "temp_ma_freq_mean", "temp_ma_freq_sem",
                               "Moving avg\nfrequency\n(Hz)"),
        os.path.join(output_dir, "Transient_frequency_moving_average_vs_temp_heating.png"),
        figsize=_style.FIG_WIDE)
    _save_standalone(
        lambda ax: _draw_panel(ax, "temp_ma_amp_mean", "temp_ma_amp_sem",
                               "Moving avg\namplitude\n(z_ΔF/F)"),
        os.path.join(output_dir, "Transient_amplitude_moving_average_vs_temp_heating.png"),
        figsize=_style.FIG_WIDE)
    _save_standalone(
        lambda ax: _draw_panel(ax, "temp_ma_hw_mean", "temp_ma_hw_sem",
                               "Moving avg\nhalf-width\n(s)"),
        os.path.join(output_dir, "Transient_halfwidth_moving_average_vs_temp_heating.png"),
        figsize=_style.FIG_WIDE)

    return path


# ---- 5. Ictal mean signal ----

@_style.styled
def plot_ictal_mean(
    results: Dict[str, IctalMeanGroupResult],
    output_dir: str,
    name_prefix: str = "",
    landmark_labels: Optional[Dict[str, str]] = None,
    scatter_options: Optional[dict] = None,
    trace_results: Optional[Dict[str, IctalMeanGroupResult]] = None,
    dff_label: Optional[str] = None,
) -> str:
    """Plot seizure means, preictal-to-ictal changes, and landmark-aligned AUCs.

    Layout: left column = scatters (seizure mean, delta), right side = vertical
    stack of (triggered average trace + AUC scatter) per landmark.

    `landmark_labels`: optional per-landmark display-label override. Used by
    opto pipelines to rename ``UEO`` → ``Stim on`` for the standalone
    filename slug (``AUC_stim_on_traces.png``), the trace-panel title,
    and the values_auc CSV column header. The override display label is
    lowercased and space→underscored for the filename slug.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    line_results = _ordered(trace_results or results)
    landmark_labels = landmark_labels or {}
    signal_label = dff_label or DFF_LABEL

    def _auc_ylabel(positive_only: bool) -> str:
        if positive_only:
            return f"AUC above 0\n({signal_label} × s)"
        return f"AUC\n({signal_label} × s)"

    def _lm_display(lm: str) -> str:
        """Display label: override wins, otherwise fall back to the
        canonical LANDMARK_DISPLAY_LABELS mapping."""
        return landmark_labels.get(lm, landmark_label(lm))

    def _lm_slug(lm: str) -> str:
        """Filename slug: raw key by default, override lowercased with
        spaces→underscores when supplied."""
        if lm in landmark_labels:
            return landmark_labels[lm].lower().replace(" ", "_")
        return lm

    landmark_names = set()
    for r in line_results.values():
        landmark_names.update(r.triggered_averages.keys())
    # Display order for event landmarks.
    preferred = ["EEC", "UEO", "behavioral_onset", "OFF", "max_temp"]
    landmark_names = [lm for lm in preferred if lm in landmark_names] + \
                     sorted(lm for lm in landmark_names if lm not in preferred)
    # Skip landmarks with no real data from any cohort (e.g. behavioral onset
    # when none of the sessions had it scored).
    landmark_names = [
        lm for lm in landmark_names
        if any(len(r.triggered_averages[lm].per_session_traces) > 0
               for r in line_results.values() if lm in r.triggered_averages)
    ]
    n_lm = len(landmark_names)

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
        ids = [_session_id(s) for s in r.session_results]
        scalars_seizure[name] = (sz_vals, r.seizure_mean, r.seizure_sem, ids)
        scalars_delta[name] = (d_vals, r.delta_mean, r.delta_sem, ids)
        scalars_delta_max[name] = (d_max_vals, r.delta_max_mean, r.delta_max_sem, ids)
        scalars_delta_min[name] = (d_min_vals, r.delta_min_mean, r.delta_min_sem, ids)
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.seizure_mean, s.baseline_mean,
                             s.delta_preictal_ictal,
                             s.delta_preictal_ictal_max,
                             s.delta_preictal_ictal_min])

    # Multi-line ylabels in the combo so adjacent rows' labels don't
    # collide vertically. Standalone exports below keep single-line labels.
    _scatter_mean_sem(
        axes[0, 0], scalars_seizure, "Seizure mean\n(z_ΔF/F)",
        scatter_options=scatter_options, zero_line=True,
    )
    _scatter_mean_sem(
        axes[1, 0], scalars_delta_max,
        "Preictal-to-ictal\nΔ max (z_ΔF/F)",
        scatter_options=scatter_options, zero_line=True,
    )
    _scatter_mean_sem(
        axes[2, 0], scalars_delta_min,
        "Preictal-to-ictal\nΔ min (z_ΔF/F)",
        scatter_options=scatter_options, zero_line=True,
    )
    # Hide unused cells in col 0
    for i in range(3, n_rows):
        axes[i, 0].axis("off")

    # Middle column (col 1): triggered averages per landmark
    # Right column (col 2): AUC scatter per landmark
    trace_axes_for_landmarks = []
    auc_axes_for_landmarks = []
    trace_lo, trace_hi = np.inf, -np.inf
    auc_lo, auc_hi = np.inf, -np.inf
    per_lm_auc: Dict[str, Dict[str, tuple]] = {}
    per_lm_window: Dict[str, Optional[float]] = {}
    per_lm_variable_window: Dict[str, bool] = {}
    per_lm_positive_only: Dict[str, bool] = {}

    def _draw_trace_panel(ax, lm: str, lm_auc_window: Optional[float],
                          show_title: bool = False, legend: bool = True):
        for name, r in line_results.items():
            if lm in r.triggered_averages:
                ta = r.triggered_averages[lm]
                _line_sem(ax, ta.time_axis, ta.mean_trace, ta.sem_trace,
                          _color(name), _label(name))
        _style.add_zero_x(ax)
        _style.add_zero_y(ax, ensure_visible=True)
        lm_disp = _lm_display(lm)
        prefix = "Time rel. to\n" if show_title else "Time rel. to "
        ax.set_xlabel(f"{prefix}{lm_disp} (s)")
        ax.set_ylabel(signal_label)
        if show_title:
            ax.set_title(lm_disp)
        xlim_hi = max(65.0, (lm_auc_window or 60.0) + 5.0)
        ax.set_xlim(-5, xlim_hi)
        ax.spines[["right", "top"]].set_visible(False)
        if legend:
            if show_title:
                offset = _legend_offset_above_title(lm_disp)
            else:
                offset = 1.02
            _legend_above(ax, y_offset=offset)

    for i, lm in enumerate(landmark_names):
        ax = axes[i, 1]
        lm_auc_windows = []
        lm_positive_flags = []
        for name, r in line_results.items():
            if lm in r.triggered_averages:
                ta = r.triggered_averages[lm]
                lm_positive_flags.append(bool(ta.auc_positive_only))
                # Keep the mean within the axes even where SEM is undefined.
                lo = float(np.nanmin(np.fmin(ta.mean_trace, ta.mean_trace - ta.sem_trace)))
                hi = float(np.nanmax(np.fmax(ta.mean_trace, ta.mean_trace + ta.sem_trace)))
                if np.isfinite(lo):
                    trace_lo = min(trace_lo, lo)
                if np.isfinite(hi):
                    trace_hi = max(trace_hi, hi)
                session_windows = getattr(
                    ta, "per_session_auc_window_s", None,
                )
                if session_windows:
                    lm_auc_windows.extend(
                        float(w) for w in session_windows if np.isfinite(w)
                    )
                elif ta.auc_window_s and np.isfinite(ta.auc_window_s):
                    lm_auc_windows.append(float(ta.auc_window_s))
        lm_variable_window = (
            len(lm_auc_windows) > 1
            and not np.allclose(lm_auc_windows, lm_auc_windows[0])
        )
        lm_auc_window = max(lm_auc_windows) if lm_auc_windows else None
        if lm_positive_flags and len(set(lm_positive_flags)) != 1:
            raise ValueError(
                f"Landmark {lm} mixes signed and positive-only AUC results."
            )
        lm_positive_only = bool(lm_positive_flags[0]) if lm_positive_flags else False
        per_lm_window[lm] = lm_auc_window
        per_lm_variable_window[lm] = lm_variable_window
        per_lm_positive_only[lm] = lm_positive_only
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
            ids = list(ta.per_session_meta or [])
            auc_dict[name] = (vals, mean, sem, ids)
            if vals:
                auc_lo = min(auc_lo, float(np.nanmin(vals)))
                auc_hi = max(auc_hi, float(np.nanmax(vals)))
        per_lm_auc[lm] = auc_dict
        if auc_dict:
            _scatter_mean_sem(
                ax_auc, auc_dict, _auc_ylabel(lm_positive_only),
                scatter_options=scatter_options, zero_line=True,
            )
            if lm_variable_window:
                window_label = "variable window"
            else:
                window_label = (f"0–{lm_auc_window:.0f} s"
                                if lm_auc_window else "")
            auc_title = "Positive-only AUC" if lm_positive_only else "AUC"
            title = f"{auc_title} - {_lm_display(lm)}"
            if window_label:
                title += f"\n{window_label}"
            ax_auc.set_title(title)
        auc_axes_for_landmarks.append(ax_auc)

    # Match y-limits across landmark rows.
    if np.isfinite(trace_lo) and np.isfinite(trace_hi) and trace_hi > trace_lo:
        trace_lo = min(trace_lo, 0.0)
        trace_hi = max(trace_hi, 0.0)
        pad = 0.05 * (trace_hi - trace_lo)
        for ax in trace_axes_for_landmarks:
            ax.set_ylim(trace_lo - pad, trace_hi + pad)
    if np.isfinite(auc_lo) and np.isfinite(auc_hi) and auc_hi > auc_lo:
        auc_lo = min(auc_lo, 0.0)
        auc_hi = max(auc_hi, 0.0)
        pad = 0.05 * (auc_hi - auc_lo)
        for ax in auc_axes_for_landmarks:
            ax.set_ylim(auc_lo - pad, auc_hi + pad)

    # Hide unused landmark rows in middle/right
    for i in range(n_lm, n_rows):
        axes[i, 1].axis("off")
        axes[i, 2].axis("off")

    path = os.path.join(output_dir, "summary_ictal_mean.png")
    _save_publication_figure(fig, path)
    plt.close("all")

    # Export maximum/minimum deflection plots only when finite values exist.
    # Test the deltas directly; IctalMeanGroupResult has no temperature fields.
    def _has_any_finite(vals):
        return any(v is not None and not (isinstance(v, float) and math.isnan(v))
                   for v in vals)

    any_finite_max = any(_has_any_finite(v[0]) for v in scalars_delta_max.values())
    any_finite_min = any(_has_any_finite(v[0]) for v in scalars_delta_min.values())
    if any_finite_max:
        _save_standalone_scatter(
            scalars_delta_max,
            "Preictal-to-ictal\nΔ max (z_ΔF/F)",
            os.path.join(output_dir, "Mean_delta_immediate_preictal_to_ictal_max.png"),
            scatter_options=scatter_options,
            zero_line=True,
        )
    if any_finite_min:
        _save_standalone_scatter(
            scalars_delta_min,
            "Preictal-to-ictal\nΔ min (z_ΔF/F)",
            os.path.join(output_dir, "Mean_delta_immediate_preictal_to_ictal_min.png"),
            scatter_options=scatter_options,
            zero_line=True,
        )
    # Standalone "Mean_values_ictal.png" mirroring the seizure-mean scalar panel.
    _save_standalone_scatter(
        scalars_seizure,
        "Seizure mean\n(z_ΔF/F)",
        os.path.join(output_dir, "Mean_values_ictal.png"),
        scatter_options=scatter_options,
        zero_line=True,
    )

    STANDALONE_AUC_LANDMARKS = {"EEC", "UEO", "OFF"}
    for lm in landmark_names:
        if lm not in STANDALONE_AUC_LANDMARKS:
            continue
        lm_auc_window = per_lm_window.get(lm)
        slug = _lm_slug(lm)
        # Trace standalone
        _save_standalone(
            lambda ax, _lm=lm, _w=lm_auc_window: _draw_trace_panel(ax, _lm, _w),
            os.path.join(output_dir, f"AUC_{slug}_traces.png"),
        )
        # AUC scatter standalone
        auc_dict = per_lm_auc.get(lm)
        if auc_dict:
            if per_lm_variable_window.get(lm, False):
                suffix = " (variable window)"
            else:
                suffix = (f" (0–{lm_auc_window:.0f}s)"
                          if lm_auc_window else "")
            _save_standalone_scatter(
                auc_dict, _auc_ylabel(per_lm_positive_only.get(lm, False)),
                os.path.join(output_dir, f"AUC_{slug}_scatterplot.png"),
                title=(
                    f"{'Positive-only AUC' if per_lm_positive_only.get(lm, False) else 'AUC'}"
                    f" - {_lm_display(lm)}{suffix}"
                ),
                scatter_options=scatter_options,
                zero_line=True,
            )


    # Fixed AUC durations go in headers; variable durations get a per-session column.
    landmarks_in_order = [
        lm for lm in landmark_names
        if any(lm in r.triggered_averages
               and r.triggered_averages[lm].per_session_auc
               for r in results.values())
    ]
    landmark_windows = {
        lm: per_lm_window.get(lm) for lm in landmarks_in_order
    }

    def _col_label(lm: str, w: Optional[float], variable: bool) -> str:
        # Column header uses the override display label for opto ("Stim on
        # AUC (40s)"), the raw key for hyperthermia ("UEO AUC (10s)").
        head = landmark_labels.get(lm, lm)
        auc_text = (
            "positive-only AUC"
            if per_lm_positive_only.get(lm, False) else "AUC"
        )
        if variable:
            return f"{head} {auc_text} (variable window)"
        suffix = f" ({w:.0f}s)" if w is not None else ""
        return f"{head} {auc_text}{suffix}"

    # (cohort, mouse_id, heating_session) -> {landmark: auc}
    auc_by_session: Dict[tuple, Dict[str, float]] = {}
    auc_window_by_session: Dict[tuple, Dict[str, float]] = {}
    for cohort_name, r in results.items():
        for lm in landmarks_in_order:
            ta = r.triggered_averages.get(lm)
            if ta is None:
                continue
            session_windows = getattr(
                ta, "per_session_auc_window_s", None,
            ) or [ta.auc_window_s] * len(ta.per_session_auc)
            for (mouse_id, h_sess), auc, window_s in zip(
                ta.per_session_meta, ta.per_session_auc, session_windows,
            ):
                key = (cohort_name, mouse_id, h_sess)
                auc_by_session.setdefault(key, {})[lm] = float(auc)
                auc_window_by_session.setdefault(key, {})[lm] = float(window_s)

    auc_rows = []
    for (cohort_name, mouse_id, h_sess), lm_aucs in sorted(
        auc_by_session.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])
    ):
        row = [cohort_name, mouse_id, h_sess]
        for lm in landmarks_in_order:
            row.append(lm_aucs.get(lm, float("nan")))
            if per_lm_variable_window.get(lm, False):
                row.append(
                    auc_window_by_session.get(
                        (cohort_name, mouse_id, h_sess), {}
                    ).get(lm, float("nan"))
                )
        auc_rows.append(row)

    auc_csv_name = (
        f"values_auc_{name_prefix}.csv"
        if name_prefix else "values_auc.csv"
    )
    auc_header = ["cohort", "mouse_id", "heating_session"]
    for lm in landmarks_in_order:
        variable = per_lm_variable_window.get(lm, False)
        auc_header.append(_col_label(lm, landmark_windows[lm], variable))
        if variable:
            head = landmark_labels.get(lm, lm)
            auc_header.append(f"{head} AUC window (s)")
    _save_csv(os.path.join(output_dir, auc_csv_name), auc_header, auc_rows)
    return path


# ---- 6. Ictal transients ----

@_style.styled
def plot_ictal_transients(
    results: Dict[str, IctalTransientGroupResult],
    output_dir: str,
    trace_results: Optional[Dict[str, IctalTransientGroupResult]] = None,
) -> str:
    """Composite ictal_transients.png + 4 standalone files:
    Transient_frequency_PSTH.png and Transient_{frequency,amplitude,halfwidth}_vs_time.png.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(trace_results or results)

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
        _style.add_zero_x(ax)
        _style.add_zero_y(ax, ensure_visible=True)
        ax.set_xlabel("Time rel. to seizure onset (s)")
        ax.set_ylabel("PSTH Frequency\n(Hz)")
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)

    def _draw_ma(ax, attr_mean: str, attr_sem: str, ylabel: str):
        for name, r in results.items():
            c = _color(name)
            _line_sem(ax, r.moving_avg_times,
                      getattr(r, attr_mean), getattr(r, attr_sem),
                      c, _label(name))
        _style.add_zero_x(ax)
        _style.add_zero_y(ax, ensure_visible=True)
        ax.set_xlabel("Time rel. to seizure onset (s)")
        ax.set_ylabel(ylabel)
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)

    fig, axes = plt.subplots(2, 2, figsize=_style.FIG_2X2)
    _draw_psth(axes[0, 0])
    _draw_ma(axes[0, 1], "freq_mean", "freq_sem", "Moving avg\nfrequency (Hz)")
    _draw_ma(axes[1, 0], "amp_mean", "amp_sem", "Moving avg\namplitude (z_ΔF/F)")
    _draw_ma(axes[1, 1], "hw_mean", "hw_sem", "Moving avg\nhalf-width (s)")
    for ax in axes.flat:
        ax.set_xlabel("Time rel. to\nseizure onset (s)")
    path = os.path.join(output_dir, "summary_ictal_transients.png")
    _save_publication_figure(fig, path)
    plt.close("all")

    _save_standalone(_draw_psth,
        os.path.join(output_dir, "Transient_frequency_PSTH.png"))
    _save_standalone(
        lambda ax: _draw_ma(ax, "freq_mean", "freq_sem", "Frequency\n(Hz)"),
        os.path.join(output_dir, "Transient_frequency_vs_time.png"))
    _save_standalone(
        lambda ax: _draw_ma(ax, "amp_mean", "amp_sem", "Amplitude\n(z_ΔF/F)"),
        os.path.join(output_dir, "Transient_amplitude_vs_time.png"))
    _save_standalone(
        lambda ax: _draw_ma(ax, "hw_mean", "hw_sem", "Half-width\n(s)"),
        os.path.join(output_dir, "Transient_halfwidth_vs_time.png"))

    return path


# ---- 7. Postictal recovery ----

@_style.styled
def plot_postictal(
    results: Dict[str, PostictalGroupResult],
    output_dir: str,
    name_prefix: str = "",
    scatter_options: Optional[dict] = None,
    trace_results: Optional[Dict[str, PostictalGroupResult]] = None,
) -> str:
    """Plot cooling mean +/- SEM in 1 C bins relative to maximum temperature.

    Also plot pairwise relationships between final time, temperature, and ΔF/F.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    line_results = _ordered(trace_results or results)
    fig, axes = plt.subplots(2, 2, figsize=_style.FIG_2X2)

    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session,
                             s.final_time, s.final_temp, s.final_mean_dff])

    # Cooling curves show decreasing temperature from left to right.
    ax = axes[0, 0]
    for name, r in line_results.items():
        color = _color(name)
        sem = getattr(r, "cooling_group_sem", None)
        if sem is None:
            sem = np.zeros_like(r.cooling_group_mean)
        _line_sem(ax, r.cooling_bin_centers, r.cooling_group_mean, sem, color, _label(name))
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_xlabel("Temp rel. to max (°C) - cooling →")
    ax.set_ylabel("Cooling mean\n(z_ΔF/F)")
    ax.invert_xaxis()  # Zero on the left; temperature decreases to the right.
    ax.spines[["right", "top"]].set_visible(False)
    _legend_above(ax)

    def _draw_scatter_pair(ax, x_attr: str, y_attr: str, xlabel: str,
                           ylabel: str, legend: bool = True):
        options = scatter_options or {}
        point_colors = options.get("point_colors_by_id", {})
        pair_positions = {}
        prepared = []
        for name, r in results.items():
            xs = getattr(r, x_attr)
            ys = getattr(r, y_attr)
            ids = [_session_id(s) for s in r.session_results]
            colors = [
                point_colors.get(
                    (name, key), point_colors.get(key, _color(name))
                )
                for key in ids
            ]
            prepared.append((xs, ys, colors, name))
            if options.get("connect_pairs"):
                for key, x_val, y_val in zip(ids, xs, ys):
                    if np.isfinite(x_val) and np.isfinite(y_val):
                        pair_positions.setdefault(key, []).append((x_val, y_val))
        if options.get("connect_pairs"):
            for points in pair_positions.values():
                if len(points) < 2:
                    continue
                ax.plot(
                    [p[0] for p in points], [p[1] for p in points],
                    color=options.get("pair_line_color", "#6b6b6b"),
                    alpha=float(options.get("pair_line_alpha", 0.55)),
                    linewidth=1.8, zorder=2,
                )
        for xs, ys, colors, name in prepared:
            ax.scatter(
                xs, ys, color=colors, label=_label(name),
                s=180, alpha=0.85, edgecolor="black", linewidth=0.6,
                zorder=3,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _style.add_zero_y(ax, ensure_visible=True)
        ax.spines[["right", "top"]].set_visible(False)
        if legend:
            _legend_above(ax)

    _draw_scatter_pair(axes[0, 1], "final_times", "final_temps",
                       "Post-Seizure Time (s)", "Final Temp (°C)", legend=False)
    _draw_scatter_pair(axes[1, 0], "final_times", "final_dffs",
                       "Post-Seizure Time (s)", "Final mean\n(z_ΔF/F)", legend=False)
    _draw_scatter_pair(axes[1, 1], "final_temps", "final_dffs",
                       "Final Temp (°C)", "Final mean\n(z_ΔF/F)", legend=False)

    path = os.path.join(output_dir, "summary_postictal_mean.png")
    _save_publication_figure(fig, path)
    plt.close("all")

    # Export cooling panels only when temperature data is available.
    has_temp = _any_temperature_data(line_results)
    if has_temp:
        fig_c, ax_c = plt.subplots(figsize=_style.FIG_WIDE)
        for name, r in line_results.items():
            color = _color(name)
            sem = getattr(r, "cooling_group_sem", None)
            if sem is None:
                sem = np.zeros_like(r.cooling_group_mean)
            _line_sem(ax_c, r.cooling_bin_centers, r.cooling_group_mean, sem,
                      color, _label(name))
        _style.add_zero_y(ax_c, ensure_visible=True)
        ax_c.set_xlabel("Temp rel. to max (°C) - cooling →")
        ax_c.set_ylabel("Cooling mean\n(z_ΔF/F)")
        ax_c.invert_xaxis()
        ax_c.spines[["right", "top"]].set_visible(False)
        _legend_above(ax_c)
        _save_publication_figure(
            fig_c,
            os.path.join(output_dir, "Mean_values_vs_temp_cooling.png"),
        )
        plt.close(fig_c)

        _save_standalone(
            lambda ax: _draw_scatter_pair(ax, "final_times", "final_temps",
                                           "Post-Seizure Time (s)", "Final Temp (°C)"),
            os.path.join(output_dir, "Final_time_vs_temp.png"))
        _save_standalone(
            lambda ax: _draw_scatter_pair(ax, "final_temps", "final_dffs",
                                           "Final Temp (°C)", "Final mean\n(z_ΔF/F)"),
            os.path.join(output_dir, "Final_temp_vs_mean_signal.png"))

    # Final time vs. signal is independent of temperature and always emitted.
    _save_standalone(
        lambda ax: _draw_scatter_pair(ax, "final_times", "final_dffs",
                                       "Post-Seizure Time (s)", "Final mean\n(z_ΔF/F)"),
        os.path.join(output_dir, "Final_time_vs_mean_signal.png"))

    # Per-session postictal columns are written by the consolidator into
    # values_means_<name_prefix>.csv (see write_values_means below).
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

    Both channels use the [-bl_start_s, -bl_end_s] baseline without z-scoring.
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
        _style.add_zero_x(ax)
        _style.add_zero_y(ax, ensure_visible=True)
        ax.set_xlabel("Time rel. to spike (s)")
        ax.set_ylabel("Δ raw signal\n(a.u.)")
        ax.set_title(f"{_label(cohort)} (n={n} sessions)")
        _legend_above(ax, y_offset=1.20)
        ax.spines[["right", "top"]].set_visible(False)

        lo = float(np.nanmin([np.nanmin(mean_470 - sem_470),
                              np.nanmin(mean_405 - sem_405)]))
        hi = float(np.nanmax([np.nanmax(mean_470 + sem_470),
                              np.nanmax(mean_405 + sem_405)]))
        ylo = min(ylo, lo)
        yhi = max(yhi, hi)

    if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
        ylo = min(ylo, 0.0)
        yhi = max(yhi, 0.0)
        pad = 0.05 * (yhi - ylo)
        for ax in axes[0]:
            if ax.has_data():
                ax.set_ylim(ylo - pad, yhi + pad)

    path = os.path.join(
        output_dir, "AUC_interictal_spikes_experimental_vs_isosbestic.png"
    )
    _save_publication_figure(fig, path)
    plt.close(fig)
    return path


# ---- 8. Spike-triggered averages ----

@_style.styled
def plot_spike_triggered(
    results: Dict[str, SpikeTriggeredGroupResult],
    output_dir: str,
    name_prefix: str = "",
    scatter_options: Optional[dict] = None,
    trace_results: Optional[Dict[str, SpikeTriggeredGroupResult]] = None,
    dff_label: Optional[str] = None,
) -> str:
    """Interictal ECoG spike-triggered photometry averages and AUCs.

    Saves two standalone PNGs:
      AUC_interictal_spikes_traces.png      (STA traces)
      AUC_interictal_spikes_scatterplot.png (per-cohort AUC scatter)
    Also writes the companion CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    line_results = _ordered(trace_results or results)
    signal_label = dff_label or DFF_LABEL

    csv_rows = []
    for name, r in results.items():
        for s in r.session_results:
            csv_rows.append([name, s.mouse_id, s.heating_session, s.n_spikes, s.auc])

    def _draw_sta(ax, xlim=(-30.0, 30.0)):
        ylo, yhi = np.inf, -np.inf
        for name, r in line_results.items():
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
        _style.add_zero_x(ax)
        _style.add_zero_y(ax, ensure_visible=True)
        ax.set_xlabel("Time rel. to spike (s)")
        ax.set_ylabel(f"Spike-triggered\n{signal_label}")
        ax.set_xlim(xlim)
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            ylo = min(ylo, 0.0)
            yhi = max(yhi, 0.0)
            pad = 0.05 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + pad)
        ax.spines[["right", "top"]].set_visible(False)
        _legend_above(ax)

    auc_data = {}
    for name, r in results.items():
        auc_vals = [s.auc for s in r.session_results]
        auc_sem = float(np.std(auc_vals, ddof=1) / np.sqrt(len(auc_vals))) if len(auc_vals) > 1 else 0.0
        ids = [_session_id(s) for s in r.session_results]
        auc_data[name] = (auc_vals, r.group_auc, auc_sem, ids)

    traces_path = _save_standalone(
        _draw_sta,
        os.path.join(output_dir, "AUC_interictal_spikes_traces.png"),
        figsize=_style.FIG_WIDE,
    )
    _save_standalone_scatter(
        auc_data, f"Spike-Triggered AUC\n({signal_label} × s)",
        os.path.join(output_dir, "AUC_interictal_spikes_scatterplot.png"),
        scatter_options=scatter_options,
        zero_line=True,
    )

    spikes_csv_name = (
        f"values_auc_spikes_{name_prefix}.csv"
        if name_prefix else "values_auc_spikes.csv"
    )
    _save_csv(
        os.path.join(output_dir, spikes_csv_name),
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
    y_ticks_override: Optional[tuple] = None,
    standard_window_s: Optional[tuple] = None,
    xlabel: str = "Time rel. to seizure onset (s)",
    title_suffix: str = "aligned to seizure onset",
    filename_prefix: str = "UEO_aligned",
    dff_label: Optional[str] = None,
    yscale_exclude: Optional[List[str]] = None,
    trace_colors: Optional[Dict[str, str]] = None,
    standard_axes_size_inches: Optional[tuple] = None,
) -> str:
    """Plot per-cohort mean +/- SEM for standard and extended windows.

    Y-limits are shared within each window. If results_extended is omitted,
    both windows use results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = _ordered(results)
    if results_extended is None:
        results_extended = results
    else:
        results_extended = _ordered(results_extended)

    std_lo, std_hi = standard_window_s if standard_window_s else (-150.0, 150.0)
    yscale_exclude_set = set(yscale_exclude) if yscale_exclude else set()
    y_axis_label = dff_label if dff_label else DFF_LABEL

    def _is_no_seizure(cohort_name: str) -> bool:
        normalized = cohort_name.strip().lower().replace(" ", "_")
        return normalized.endswith("_no_sz") or normalized == "no_seizure"

    def _xlabel_for(cohort_name: str) -> str:
        if (xlabel == "Time rel. to seizure onset (s)"
                and _is_no_seizure(cohort_name)):
            return "Time rel. to event onset (s)"
        return xlabel

    def _combo_xlabel_for(cohort_name: str) -> str:
        return _xlabel_for(cohort_name).replace(
            "Time rel. to ", "Time rel. to\n", 1,
        )

    def _title_suffix_for(cohort_name: str) -> str:
        if (title_suffix == "aligned to seizure onset"
                and _is_no_seizure(cohort_name)):
            return "aligned to event onset"
        return title_suffix

    cohort_fname = {
        "DS_sz": (f"{filename_prefix}_traces_DS_sz.png",
                  f"{filename_prefix}_traces_DS_sz_extended.png"),
        "DS_no_sz": (f"{filename_prefix}_traces_DS_nosz.png",
                     f"{filename_prefix}_traces_DS_nosz_extended.png"),
        "WT_sz": (f"{filename_prefix}_traces_WT_sz.png",
                  f"{filename_prefix}_traces_WT_sz_extended.png"),
        "WT_no_sz": (f"{filename_prefix}_traces_WT.png",
                     f"{filename_prefix}_traces_WT_extended.png"),
    }
    WINDOWS = [
        (std_lo, std_hi, f"summary_{filename_prefix}_traces.png", 0, results),
        (-150.0, 1000.0, f"summary_{filename_prefix}_traces_extended.png", 1,
         results_extended),
    ]

    primary_path = None

    for x_lo, x_hi, combo_name, w_idx, window_results in WINDOWS:
        # Share y-limits within each window. Excluded cohorts are plotted
        # but do not contribute to automatic scale estimation.
        ylo, yhi = np.inf, -np.inf
        for cohort_name, ta in window_results.items():
            if cohort_name in yscale_exclude_set:
                continue
            if ta.mean_trace is None or len(ta.mean_trace) == 0:
                continue
            mask = (ta.time_axis >= x_lo) & (ta.time_axis <= x_hi)
            if not np.any(mask):
                continue
            sem = ta.sem_trace
            # Keep the mean within the axes even where SEM is undefined.
            lo = float(np.nanmin(np.fmin(ta.mean_trace, ta.mean_trace - sem)[mask]))
            hi = float(np.nanmax(np.fmax(ta.mean_trace, ta.mean_trace + sem)[mask]))
            if np.isfinite(lo):
                ylo = min(ylo, lo)
            if np.isfinite(hi):
                yhi = max(yhi, hi)
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            pad = 0.05 * (yhi - ylo)
            final_ylim = (ylo - pad, yhi + pad)
        else:
            final_ylim = None
        # Explicit limits override autoscaling.
        if y_lim_override is not None:
            final_ylim = tuple(y_lim_override)

        # Combo figure
        n = max(len(window_results), 1)
        fig, axes = plt.subplots(1, n,
                                 figsize=_style.panel_figsize(n),
                                 squeeze=False)
        for i, (name, ta) in enumerate(window_results.items()):
            ax = axes[0, i]
            color = (trace_colors or {}).get(name, _color(name))
            sem = ta.sem_trace
            ax.plot(ta.time_axis, ta.mean_trace, color=color, label=_label(name))
            ax.fill_between(ta.time_axis, ta.mean_trace - sem,
                            ta.mean_trace + sem, color=color, alpha=0.25)
            _style.add_zero_x(ax)
            ax.set_xlabel(_combo_xlabel_for(name))
            ax.set_ylabel(y_axis_label)
            ax.set_title(f"{_label(name)} - {_title_suffix_for(name)}")
            ax.set_xlim(x_lo, x_hi)
            if final_ylim is not None:
                ax.set_ylim(*final_ylim)
            if y_ticks_override is not None:
                ax.set_yticks(y_ticks_override)
            _style.add_zero_y(ax, ensure_visible=True)
            ax.spines[["right", "top"]].set_visible(False)

            combo_path = os.path.join(output_dir, combo_name)
        _save_publication_figure(fig, combo_path)
        plt.close(fig)
        if w_idx == 0:
            primary_path = combo_path

        # Per-cohort standalone plots.
        def _draw_single(ax, ta, name, ylim, xl, xh):
            color = (trace_colors or {}).get(name, _color(name))
            sem = ta.sem_trace
            ax.plot(ta.time_axis, ta.mean_trace, color=color)
            ax.fill_between(ta.time_axis, ta.mean_trace - sem,
                            ta.mean_trace + sem, color=color, alpha=0.25)
            _style.add_zero_x(ax)
            ax.set_xlabel(_xlabel_for(name))
            ax.set_ylabel(y_axis_label)
            ax.set_title(f"{_label(name)} - {_title_suffix_for(name)}")
            ax.set_xlim(xl, xh)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if y_ticks_override is not None:
                ax.set_yticks(y_ticks_override)
            _style.add_zero_y(ax, ensure_visible=True)
            ax.spines[["right", "top"]].set_visible(False)

        for name, ta in window_results.items():
            fname = cohort_fname.get(name,
                                     (f"{filename_prefix}_traces_{name}.png",
                                      f"{filename_prefix}_traces_{name}_extended.png"))[w_idx]
            _save_standalone(
                lambda ax, _t=ta, _n=name, _y=final_ylim, _xl=x_lo, _xh=x_hi:
                    _draw_single(ax, _t, _n, _y, _xl, _xh),
                os.path.join(output_dir, fname),
                figsize=_style.FIG_WIDE,
                axes_size_inches=standard_axes_size_inches if w_idx == 0 else None,
            )

    return primary_path or os.path.join(output_dir, WINDOWS[0][2])


# ---- 9b. Opto stim response magnitude ----

@_style.styled
def plot_opto_response(
    results,
    output_dir: str,
    scatter_options: Optional[dict] = None,
    dff_label: Optional[str] = None,
) -> Optional[str]:
    """Three standalone scatters: max / min / AUC of the 40 s post-stim window.

    Each panel: one column per cohort, per-session dots jittered around the
    cohort x, mean ± SEM whisker. Returns without plots when no cohort has data
    (non-opto runs).
    """
    from ..analysis.opto_response import OptoResponseGroupResult

    results = _ordered(results)
    if not any(isinstance(r, OptoResponseGroupResult) and r.session_results
               for r in results.values()):
        return None

    os.makedirs(output_dir, exist_ok=True)
    signal_label = dff_label or DFF_LABEL

    metrics = [
        ("stim_response_max_40s",
         f"Stim max\n({signal_label})",
         "Opto_stim_response_max_40s.png"),
        ("stim_response_min_40s",
         f"Stim min\n({signal_label})",
         "Opto_stim_response_min_40s.png"),
        ("stim_auc_40s",
         f"Stim AUC\n({signal_label} × s)",
         "Opto_stim_auc_40s.png"),
    ]

    last_path = None
    for attr, ylabel, fname in metrics:
        cohorts = {}
        for name, r in results.items():
            selected = [sr for sr in r.session_results
                        if np.isfinite(getattr(sr, attr))]
            vals = np.array([getattr(sr, attr) for sr in selected], dtype=float)
            if vals.size == 0:
                continue
            mean = float(np.nanmean(vals))
            sem = float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) \
                if vals.size > 1 else 0.0
            ids = [_session_id(sr) for sr in selected]
            cohorts[name] = (vals, mean, sem, ids)

        if not cohorts:
            continue
        path = os.path.join(output_dir, fname)
        _save_standalone_scatter(
            cohorts, ylabel, path,
            scatter_options=scatter_options,
            zero_line=True,
        )
        last_path = path

    return last_path


# ---- 10. Pre-ictal mean diagnostic ----

@_style.styled
def plot_preictal_mean_diagnostic(
    session: Session,
    output_dir: str,
    is_opto: bool = False,
    dff_label: Optional[str] = None,
    extrema_min_duration_s: float | None = None,
) -> Optional[str]:
    """Diagnostic overlay: shows the bin windows and means used by the
    pre-ictal-mean analysis directly on the session's processed trace, plus
    a zoomed view that annotates the delta-of-mean calculation.

    Returns the saved PNG path, or None if landmarks are missing.

    ``extrema_min_duration_s`` applies the same sustained max/min definition
    used by ``compute_ictal_mean``.

    `is_opto=True` retitles the regions for stim experiments (no heating, no
    seizure) and uses ``[stim_on, stim_on + 40 s]`` for extrema detection.
    `dff_label` overrides the y-axis label.
    """
    from ..analysis._helpers import (
        get_signal_and_time, get_ueo_time, get_off_time, time_to_index,
    )
    from ..analysis.ictal_mean import compute_sustained_extrema

    # Use the same post-stimulation window as opto_response.
    OPTO_DETECTION_WINDOW_S = 40.0

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
    if is_opto:
        # Detect extrema over the 40 s post-stimulation window.
        end_t = ueo_t + OPTO_DETECTION_WINDOW_S
        if len(time_axis):
            end_t = min(end_t, float(time_axis[-1]))
    else:
        end_t = off_t if off_t is not None else min(time_axis[-1], ueo_t + 30.0)
    y_axis_label = dff_label if dff_label else "z_ΔF/F"

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
    # Mark both maximum and minimum values in the diagnostic plot.
    if len(ictal_window):
        if extrema_min_duration_s is None:
            maximum_index = int(np.argmax(ictal_window))
            minimum_index = int(np.argmin(ictal_window))
            ictal_max_val = float(ictal_window[maximum_index])
            ictal_min_val = float(ictal_window[minimum_index])
            max_idx = i_ueo + maximum_index
            min_idx = i_ueo + minimum_index
            max_window_start = max_idx
            min_window_start = min_idx
            extrema_duration_samples = 1
        else:
            extrema = compute_sustained_extrema(
                ictal_window,
                fs,
                extrema_min_duration_s,
            )
            ictal_max_val = extrema.maximum
            ictal_min_val = extrema.minimum
            max_idx = i_ueo + extrema.maximum_index
            min_idx = i_ueo + extrema.minimum_index
            max_window_start = i_ueo + extrema.maximum_window_start
            min_window_start = i_ueo + extrema.minimum_window_start
            extrema_duration_samples = extrema.duration_samples
        delta_max = ictal_max_val - elh_mean
        delta_min = ictal_min_val - elh_mean
    else:
        ictal_max_val = np.nan
        ictal_min_val = np.nan
        delta_max = np.nan
        delta_min = np.nan
        max_idx = None
        min_idx = None
        max_window_start = None
        min_window_start = None
        extrema_duration_samples = 1

    fig, axes = plt.subplots(2, 1, figsize=(_style.FIG_2X2[0] * 1.4,
                                            _style.FIG_2X2[1] * 2.0))

    # Panel 0: full heating + ictal trace with bin shading and mean lines
    ax = axes[0]
    color = _color(session.cohort) if session.cohort else "black"
    ax.plot(time_axis, signal, color=color, linewidth=0.4)
    _style.add_zero_y(ax, ensure_visible=True)

    # Opto diagnostics show a 10 s pre-stimulation window and the
    # post-stimulation detection window instead of heating periods.
    if is_opto:
        region_specs = [
            (ueo_t - end_late_n / fs if end_late_n else ueo_t, ueo_t,
             "tab:red", 0.20, "Pre-stim baseline (10s)", elh_mean),
            (ueo_t, end_t, "tab:purple", 0.10,
             "Post-stim detection window (40s)", ictal_mean),
        ]
    else:
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

    # Add stimulation onset/offset lines to optogenetic diagnostics.
    if is_opto:
        ax.axvline(ueo_t, color="black", linestyle=":", linewidth=1.2,
                   label="stim on")
        if off_t is not None and np.isfinite(off_t):
            ax.axvline(off_t, color="black", linestyle="-.", linewidth=1.2,
                       label="stim off")

    ax.set_xlim(max(0, heat_t - 30), min(time_axis[-1], end_t + 10))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{y_axis_label} (full window)")
    _legend_above(ax)
    ax.spines[["right", "top"]].set_visible(False)

    # Panel 1: zoom on [ueo - 15s, end] with BOTH min/max delta annotations.
    ax2 = axes[1]
    z0, z1 = ueo_t - 15.0, end_t + 2.0
    ax2.plot(time_axis, signal, color=color, linewidth=0.6)
    _style.add_zero_y(ax2, ensure_visible=True)
    ax2.set_xlim(z0, z1)
    baseline_label = "Pre-stim baseline (10s)" if is_opto else "End-of-late-heat (10s)"
    if end_late_n > 0:
        ax2.axvspan(ueo_t - end_late_n / fs, ueo_t,
                    color="tab:red", alpha=0.20, label=baseline_label)
    if not np.isnan(elh_mean):
        baseline_mean_label = (
            "Pre-stim baseline mean" if is_opto else "End-of-late-heat mean"
        )
        ax2.axhline(elh_mean, color="tab:red", linestyle="--", linewidth=1.5,
                    label=f"{baseline_mean_label} = {elh_mean:.3f}")

    if extrema_duration_samples > 1:
        duration_s = extrema_duration_samples / fs
        for window_start, span_color in (
            (max_window_start, "tab:green"),
            (min_window_start, "tab:purple"),
        ):
            if window_start is not None:
                ax2.axvspan(
                    time_axis[window_start],
                    time_axis[window_start] + duration_s,
                    color=span_color,
                    alpha=0.10,
                )

    peak_word = "Evoked" if is_opto else "Ictal"
    duration_label = (
        f"{extrema_duration_samples / fs * 1000:g} ms sustained "
        if extrema_duration_samples > 1 else ""
    )

    def _mark_peak(idx, val, delta, mark_color, marker, label_prefix, offset_sign):
        if idx is None or idx >= len(time_axis) or np.isnan(val):
            return
        peak_t = time_axis[idx]
        peak_v = signal[idx]
        horizontal_sign = -1 if peak_t > (z0 + z1) / 2 else 1
        ax2.scatter([peak_t], [peak_v], color=mark_color, s=60, zorder=5,
                    marker=marker,
                    label=(f"{peak_word} {duration_label}{label_prefix} "
                           f"= {peak_v:.3f}"))
        ax2.annotate(
            (f"Δ_{label_prefix} ({duration_label.strip()}) = "
             if duration_label else f"Δ_{label_prefix} = ")
            + f"{peak_v:.3f} − {elh_mean:.3f} = {delta:+.3f}",
            xy=(peak_t, peak_v),
            xytext=(horizontal_sign * 18, offset_sign * 40),
            textcoords="offset points",
            ha="right" if horizontal_sign < 0 else "left",
            va="top" if offset_sign < 0 else "bottom",
            fontsize=24,
            arrowprops=dict(arrowstyle="->", color=mark_color, lw=1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=mark_color),
        )

    # Keep callouts inside the plot: the maximum label goes below its point,
    # away from the legend, while the minimum label goes above its point.
    _mark_peak(max_idx, ictal_max_val, delta_max, "tab:green", "^", "max", -1)
    _mark_peak(min_idx, ictal_min_val, delta_min, "tab:purple", "v", "min", +1)
    ax2.axvline(ueo_t, color="black", linestyle="--", linewidth=0.8,
                label=("stim on" if is_opto else None))
    if is_opto and off_t is not None and np.isfinite(off_t) and z0 <= off_t <= z1:
        ax2.axvline(off_t, color="black", linestyle="-.", linewidth=0.8,
                    label="stim off")
    onset_word = "stim" if is_opto else "seizure"
    ax2.set_xlabel(
        f"Time (s) - zoomed around {onset_word} onset = {ueo_t:.1f}s"
    )
    ax2.set_ylabel(f"{y_axis_label} (zoomed)")
    _legend_above(ax2)
    ax2.spines[["right", "top"]].set_visible(False)

    fig.suptitle(
        f"Parameter detection - {session.mouse_id} S{session.heating_session} "
        f"({_label(session.cohort) if session.cohort else 'unknown cohort'})",
    )

    os.makedirs(output_dir, exist_ok=True)
    fname = (
        f"{session.mouse_id}_S{session.heating_session}"
        f"_{_cohort_filename_tag(session)}_parameter_detection.png"
    )
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


# ---------------------------------------------------------------------------
# Consolidated per-session means CSV
# ---------------------------------------------------------------------------

def write_values_means(
    cohort_module_results: Dict[str, Dict[str, object]],
    output_dir: str,
    name_prefix: str = "",
) -> Optional[str]:
    """Write values_means_<name_prefix>.csv: one row per (cohort, mouse,
    heating_session), columns merged from Pre-ictal mean, Ictal mean, and
    Postictal modules.

    Pre-ictal `ictal_mean` and `baseline_mean` columns take precedence over
    the duplicate Ictal `seizure_mean` and `baseline_mean` columns.

    Args:
        cohort_module_results: ``{cohort_name: {module_name: GroupResult}}``
        output_dir: CSV destination directory.
        name_prefix: appended to the filename.
    """
    columns = [
        "cohort", "mouse_id", "heating_session",
        "baseline_mean", "early_heat_mean", "late_heat_mean",
        "end_late_heat_mean", "heating_mean", "ictal_mean",
        "delta_preictal_ictal",
        "delta_preictal_ictal_max", "delta_preictal_ictal_min",
        "final_time", "final_temp", "final_mean_dff",
    ]
    # (cohort, mouse_id, heating_session) -> dict of values
    rows: Dict[tuple, Dict[str, object]] = {}

    def _slot(cohort_name: str, mouse_id: str, h_sess: int) -> Dict[str, object]:
        key = (cohort_name, mouse_id, int(h_sess))
        return rows.setdefault(key, {})

    for cohort_name, modules in cohort_module_results.items():
        pre = modules.get("Pre-ictal mean")
        if pre is not None:
            for s in pre.session_results:
                slot = _slot(cohort_name, s.mouse_id, s.heating_session)
                slot["baseline_mean"]      = s.baseline_mean
                slot["early_heat_mean"]    = s.early_heat_mean
                slot["late_heat_mean"]     = s.late_heat_mean
                slot["end_late_heat_mean"] = s.end_late_heat_mean
                slot["heating_mean"]       = s.heating_mean
                slot["ictal_mean"]         = s.ictal_mean
        ict = modules.get("Ictal mean")
        if ict is not None:
            for s in ict.session_results:
                slot = _slot(cohort_name, s.mouse_id, s.heating_session)
                # Baseline and seizure means are already in the preictal columns.
                slot["delta_preictal_ictal"]     = s.delta_preictal_ictal
                slot["delta_preictal_ictal_max"] = s.delta_preictal_ictal_max
                slot["delta_preictal_ictal_min"] = s.delta_preictal_ictal_min
        post = modules.get("Postictal")
        if post is not None:
            for s in post.session_results:
                slot = _slot(cohort_name, s.mouse_id, s.heating_session)
                slot["final_time"]     = s.final_time
                slot["final_temp"]     = s.final_temp
                slot["final_mean_dff"] = s.final_mean_dff

    if not rows:
        return None

    # Sort by (cohort, mouse_id, heating_session) for stable output.
    out_rows = []
    for (cohort_name, mouse_id, h_sess), vals in sorted(rows.items()):
        row = [cohort_name, mouse_id, h_sess]
        for col in columns[3:]:
            row.append(vals.get(col, float("nan")))
        out_rows.append(row)

    csv_name = (
        f"values_means_{name_prefix}.csv"
        if name_prefix else "values_means.csv"
    )
    path = os.path.join(output_dir, csv_name)
    _save_csv(path, columns, out_rows)
    return path
