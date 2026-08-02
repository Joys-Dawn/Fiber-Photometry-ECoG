"""
Per-session trace plots.

Display v1: full sanity check (6 subplots, entire trace).
Display v2: zoomed processed (2 subplots, centered on seizure onset).
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from . import _style  # provides the @styled decorator used on PNG writers
from ..core.cohort import classify_session
from ..core.data_models import Session, TransientEvent
from .colors import (
    ECOG_COLOR, TEMP_COLOR, cohort_color,
    LANDMARK_COLORS, LANDMARK_DISPLAY_LABELS, DFF_LABEL,
)

_LANDMARK_LABEL_MIN_SEPARATION = 0.045
_LANDMARK_LABEL_LEVEL_STEP_PT = 40
_LANDMARK_LABEL_SIZE = 12


def _open_qc_axes(axes) -> None:
    """Keep only the left and bottom axes on quality-check figures."""
    for ax in np.atleast_1d(axes).flat:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)


def _layout_landmark_labels(entries, x_range, offset: float = 0.0) -> list:
    """Return visible landmarks with non-overlapping vertical label levels."""
    x_min, x_max = sorted(map(float, x_range))
    min_separation = (x_max - x_min) * _LANDMARK_LABEL_MIN_SEPARATION
    visible = []
    for key, time_s in entries:
        if time_s is None:
            continue
        x = float(time_s) - offset
        if x_min <= x <= x_max:
            visible.append((key, x))

    lane_last_x = []
    laid_out = []
    for key, x in sorted(visible, key=lambda item: item[1]):
        level = next(
            (
                index for index, last_x in enumerate(lane_last_x)
                if x - last_x >= min_separation
            ),
            len(lane_last_x),
        )
        if level == len(lane_last_x):
            lane_last_x.append(x)
        else:
            lane_last_x[level] = x
        laid_out.append((key, x, level))
    return laid_out


def _draw_labeled_landmarks(axes, entries, labels, offset: float = 0.0) -> None:
    """Draw landmark lines and stagger labels that would occupy the same x area."""
    xlim = axes[0].get_xlim()
    top_ax = axes[0]
    for key, x, level in _layout_landmark_labels(entries, xlim, offset):
        for ax in axes:
            ax.axvline(
                x, color=LANDMARK_COLORS[key], linestyle="--",
                linewidth=0.8, alpha=0.7,
            )
        top_ax.annotate(
            labels[key],
            xy=(x, 1.0),
            xycoords=top_ax.get_xaxis_transform(),
            xytext=(0, 4 + level * _LANDMARK_LABEL_LEVEL_STEP_PT),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_LANDMARK_LABEL_SIZE,
            color=LANDMARK_COLORS[key],
            clip_on=False,
            annotation_clip=False,
        )


def _draw_landmark_lines(
    axes,
    landmarks,
    offset: float = 0.0,
    label_axis: int = 0,
):
    """Draw vertical lines for all non-None landmarks across *axes*.

    Per-line text labels (e.g. "heating start") are placed just above the top
    axes (the one at index 0) rather than inside the data area. Lines outside
    the first axis's current xlim are skipped.

    Parameters
    ----------
    axes : array of Axes
    landmarks : SessionLandmarks
    offset : value subtracted from each time (0 for absolute, t_zero for relative)
    label_axis : retained for backwards compatibility; labels always sit above
        axes[0] in this implementation.
    """
    entries = [
        ("heating_start", landmarks.heating_start_time),
        ("eec", landmarks.eec_time or landmarks.equiv_eec_time),
        ("ueo", landmarks.ueo_time or landmarks.equiv_ueo_time),
        ("behavioral_onset", landmarks.behavioral_onset_time or landmarks.equiv_behavioral_onset_time),
        ("off", landmarks.off_time or landmarks.equiv_off_time),
    ]
    _draw_labeled_landmarks(
        axes, entries, LANDMARK_DISPLAY_LABELS, offset=offset,
    )


def _cohort_color(session: Session) -> str:
    return cohort_color(classify_session(session))


# Filenames want "DS_nosz"/"WT" (no underscore between parts) while the
# cohort keys themselves are "DS_no_sz"/"WT_no_sz". Map between the two
# so QC filenames stay stable across the cohort-key refactor.
_FILENAME_TAGS = {
    "DS_sz": "DS_sz",
    "DS_no_sz": "DS_nosz",
    "WT_sz": "WT_sz",
    "WT_no_sz": "WT",
}


def _cohort_filename_tag(session: Session) -> str:
    """Short filesystem-safe cohort label for per-session QC filenames.

    Keeps DS_sz / DS_nosz / WT in sync with the group-plot naming convention
    (cf. UEO_aligned_traces_DS_sz.png, etc.). Without this, two sessions that
    share `(mouse_id, S#)` across cohorts - e.g. 2575_S1 in scn1a_seizure vs
    2575_S1 in scn1a_NOseizure - would silently overwrite each other's QC
    plots. Custom cohorts (opto subfolders like "10HZ_DG_CPP") use the
    classify_session string as-is.
    """
    key = classify_session(session)
    return _FILENAME_TAGS.get(key, key)


def _baseline_window_s(session: Session) -> Optional[tuple]:
    """Return the explicit analysis baseline, never the event-onset window."""
    if session.landmarks is None or session.processed is None:
        return None
    lm = session.landmarks
    if lm.baseline_window_s is not None:
        start_s, end_s = map(float, lm.baseline_window_s)
    else:
        start_s = 0.0
        end_s = lm.zscore_baseline_end_time
        if end_s is None:
            end_s = lm.heating_start_time
    if end_s is None or not np.isfinite(end_s):
        return None
    t = session.processed.time
    start_s = max(float(t[0]), float(start_s))
    end_s = min(float(t[-1]), float(end_s))
    return (start_s, end_s) if end_s > start_s else None


@_style.styled
def plot_baseline(
    session: Session,
    output_dir: str,
    color: Optional[str] = None,
) -> Optional[str]:
    """Write a zoomed-out view of the session's actual baseline window."""
    bounds = _baseline_window_s(session)
    if bounds is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    start_s, end_s = bounds
    proc = session.processed
    time_s = proc.time
    mask = (time_s >= start_s) & (time_s <= end_s)
    if not np.any(mask):
        return None

    if color is None:
        color = _cohort_color(session)

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    axes[0].plot(
        time_s[mask], proc.photometry.dff_zscore[mask],
        color=color, linewidth=0.4,
    )
    _style.add_zero_y(axes[0], ensure_visible=True)
    axes[0].set_ylabel(DFF_LABEL)

    axes[1].plot(
        time_s[mask], proc.ecog_filtered[mask],
        color=ECOG_COLOR, linewidth=0.4,
    )
    _style.add_zero_y(axes[1], ensure_visible=True)
    axes[1].set_ylabel("Filt ECoG (µV)")
    axes[1].set_xlabel("Time (s) - baseline")
    axes[1].set_xlim(start_s, end_s)
    _open_qc_axes(axes)

    fig.suptitle(
        f"{session.mouse_id} S{session.heating_session} - baseline"
    )
    stem = (
        f"{session.mouse_id}_S{session.heating_session}_"
        f"{_cohort_filename_tag(session)}_baseline"
    )
    path = os.path.join(output_dir, stem + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    try:
        _write_baseline_html(
            session=session,
            time_s=time_s[mask],
            dff=proc.photometry.dff_zscore[mask],
            ecog=proc.ecog_filtered[mask],
            color=color,
            x_range=(start_s, end_s),
            html_path=os.path.join(output_dir, stem + ".html"),
        )
    except Exception:
        pass
    return path


@_style.styled
def plot_sanity_check(
    session: Session,
    output_dir: str,
    transients: Optional[List[TransientEvent]] = None,
) -> str:
    """Display v1: 6 vertically stacked subplots, full trace.

    Subplots:
        1. Raw 470 & 405 photometry signals
        2. Corrected z-ΔF/F (mean stream)
        3. HPF + detected transients marked
        4. Raw ECoG
        5. Filtered ECoG
        6. Temperature

    Returns the saved PNG path.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw = session.raw
    proc = session.processed
    time_s = proc.time

    color = _cohort_color(session)
    if transients is None:
        transients = session.transients

    fig, axes = plt.subplots(6, 1, figsize=(24, 26), sharex=True)
    # Six stacked panels share the global 48pt label size, which makes the
    # ylabels collide with each other under constrained_layout. These are
    # diagnostic plots (not for publication), so shrink them locally.
    _YLABEL_SIZE = 28

    # 1. Raw 470 & 405
    ax = axes[0]
    ax.plot(time_s, raw.signal_470, color="blue", linewidth=0.3, label="470nm")
    ax.plot(time_s, raw.signal_405, color="violet", linewidth=0.3, label="405nm")
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel("Raw (V)", fontsize=_YLABEL_SIZE)
    ax.legend(loc="upper right")

    # 2. Corrected z-ΔF/F
    ax = axes[1]
    ax.plot(time_s, proc.photometry.dff_zscore, color=color, linewidth=0.3)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel(DFF_LABEL, fontsize=_YLABEL_SIZE)

    # 3. HPF/detection stream + transients
    _used_hpf = getattr(session.preprocessing_config, 'photometry', None) and session.preprocessing_config.photometry.apply_hpf
    _det_label = f"HPF {DFF_LABEL}" if _used_hpf else DFF_LABEL
    ax = axes[2]
    ax.plot(time_s, proc.photometry.dff_hpf, color=color, linewidth=0.3)
    _style.add_zero_y(ax, ensure_visible=True)
    if transients:
        peak_times = [t.peak_time for t in transients]
        peak_idxs = np.searchsorted(time_s, peak_times).clip(0, len(time_s) - 1)
        ax.plot(
            np.array(peak_times),
            proc.photometry.dff_hpf[peak_idxs],
            "v", color="black", markersize=3,
        )
    ax.set_ylabel(_det_label, fontsize=_YLABEL_SIZE)

    # 4. Raw ECoG
    ax = axes[3]
    ax.plot(time_s, raw.ecog, color=ECOG_COLOR, linewidth=0.3)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel("Raw ECoG (µV)", fontsize=_YLABEL_SIZE)

    # 5. Filtered ECoG
    ax = axes[4]
    ax.plot(time_s, proc.ecog_filtered, color=ECOG_COLOR, linewidth=0.3)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel("Filt ECoG (µV)", fontsize=_YLABEL_SIZE)

    # 6. Temperature
    ax = axes[5]
    ax.plot(time_s, proc.temperature_smooth, color=TEMP_COLOR, linewidth=0.5)
    bl_temp = getattr(session.landmarks, "baseline_temp", None) if session.landmarks else None
    if bl_temp is not None and not np.isnan(bl_temp):
        ax.axhline(bl_temp, color="black", linestyle="--", linewidth=0.6, alpha=0.5,
                   label=f"baseline={bl_temp:.2f}°C")
        ax.legend(loc="lower right")
    ax.set_ylabel("Temp (°C)", fontsize=_YLABEL_SIZE)
    ax.set_xlabel("Time (s)")

    _open_qc_axes(axes)
    _draw_landmark_lines(axes, session.landmarks, offset=0.0)

    fig.suptitle(f"{session.mouse_id} S{session.heating_session} - sanity check")

    path = os.path.join(
        output_dir,
        f"{session.mouse_id}_S{session.heating_session}_{_cohort_filename_tag(session)}_full_trace.png",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Companion interactive HTML - same panels and landmark overlays as the PNG.
    try:
        _write_sanity_check_html(
            session=session,
            color=color,
            detection_label=_det_label,
            transients=transients,
            html_path=os.path.join(
                output_dir,
                f"{session.mouse_id}_S{session.heating_session}_{_cohort_filename_tag(session)}_full_trace.html",
            ),
        )
    except Exception:
        # HTML is best-effort; never block the PNG output if plotly is broken.
        pass

    plot_baseline(session, output_dir, color=color)
    return path


@_style.styled
def plot_zoomed(
    session: Session,
    output_dir: str,
    window_s: float = 150.0,
) -> str:
    """Display v2: 2 subplots centered on seizure onset / equivalent.

    Subplots:
        1. Corrected z-ΔF/F
        2. Filtered ECoG

    x-axis: seconds with 0 = UEO / equivalent.
    Returns the saved PNG path. An interactive .html copy is written alongside.
    """
    os.makedirs(output_dir, exist_ok=True)

    proc = session.processed
    time_s = proc.time
    landmarks = session.landmarks

    # Determine t=0 (UEO or equivalent)
    t_zero = landmarks.ueo_time
    if t_zero is None:
        t_zero = landmarks.equiv_ueo_time
    if t_zero is None:
        t_zero = landmarks.heating_start_time

    rel_time = time_s - t_zero
    mask = (rel_time >= -window_s) & (rel_time <= window_s)

    color = _cohort_color(session)

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    # 1. z-ΔF/F
    ax = axes[0]
    ax.plot(rel_time[mask], proc.photometry.dff_zscore[mask], color=color, linewidth=0.4)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel(DFF_LABEL)
    ax.set_xlim(-window_s, window_s)

    # 2. Filtered ECoG
    ax = axes[1]
    ax.plot(rel_time[mask], proc.ecog_filtered[mask], color=ECOG_COLOR, linewidth=0.4)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel("Filt ECoG (µV)")
    ax.set_xlabel("Time relative to seizure onset (s)")
    ax.set_xlim(-window_s, window_s)

    _open_qc_axes(axes)
    _draw_landmark_lines(axes, landmarks, offset=t_zero)

    fig.suptitle(f"{session.mouse_id} S{session.heating_session} - zoomed")

    path = os.path.join(
        output_dir,
        f"{session.mouse_id}_S{session.heating_session}_{_cohort_filename_tag(session)}_peri_ictal.png",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Companion interactive HTML via plotly
    try:
        _write_peri_ictal_html(
            session=session,
            rel_time=rel_time[mask],
            dff=proc.photometry.dff_zscore[mask],
            ecog=proc.ecog_filtered[mask],
            color=color,
            landmarks=landmarks,
            t_zero=t_zero,
            window_s=window_s,
            html_path=os.path.join(
                output_dir,
                f"{session.mouse_id}_S{session.heating_session}_{_cohort_filename_tag(session)}_peri_ictal.html",
            ),
        )
    except Exception:
        # HTML is best-effort; never block the PNG output if plotly is broken.
        pass

    return path


# Opto plots reuse the UEO landmark color but relabel the line so the
# experimenter sees the stim-event language they expect.
_OPTO_LANDMARK_LABELS = {
    **LANDMARK_DISPLAY_LABELS,
    "ueo": "stim on",
    "off": "stim off",
}


def _draw_landmark_lines_opto(axes, landmarks, offset: float = 0.0) -> None:
    """Opto variant of `_draw_landmark_lines`: same colors / line style,
    but the UEO / OFF lines are labelled ``stim on`` / ``stim off``
    rather than ``seizure onset`` / ``seizure offset``."""
    if landmarks is None:
        return
    entries = [
        ("ueo", landmarks.ueo_time or landmarks.equiv_ueo_time),
        ("off", landmarks.off_time or landmarks.equiv_off_time),
    ]
    _draw_labeled_landmarks(
        axes, entries, _OPTO_LANDMARK_LABELS, offset=offset,
    )


@_style.styled
def plot_sanity_check_opto(
    session: Session,
    output_dir: str,
    transients: Optional[List[TransientEvent]] = None,
) -> str:
    """Opto variant of the full-trace sanity check. Four panels (no ECoG, no
    temperature; opto stim added). All photometry traces are black; the 470
    and 405 channels are distinguished by linestyle. Landmark labels read
    ``stim on`` / ``stim off`` instead of ``seizure onset`` / ``offset``.

    Subplots:
        1. Raw 470 (solid) + 405 (dashed), both black
        2. Corrected z-ΔF/F (black)
        3. HPF + detected transients (black, transient marks black)
        4. Opto stim raw trace (black)
    """
    os.makedirs(output_dir, exist_ok=True)

    raw = session.raw
    proc = session.processed
    time_s = proc.time

    if transients is None:
        transients = session.transients

    fig, axes = plt.subplots(4, 1, figsize=(24, 18), sharex=True)
    # Four stacked panels share the global 48pt label size, which makes the
    # ylabels collide with each other under constrained_layout. These are
    # diagnostic plots (not for publication), so shrink them locally.
    _YLABEL_SIZE = 28

    # 1. Raw 470 (solid) + 405 (dashed) — both black, distinguished by style.
    ax = axes[0]
    ax.plot(time_s, raw.signal_470, color="black", linewidth=0.3, label="470nm")
    ax.plot(time_s, raw.signal_405, color="black", linewidth=0.3,
            linestyle="--", label="405nm")
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel("Raw (V)", fontsize=_YLABEL_SIZE)
    ax.legend(loc="upper right")

    # 2. Corrected z-ΔF/F.
    ax = axes[1]
    ax.plot(time_s, proc.photometry.dff_zscore, color="black", linewidth=0.3)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel(DFF_LABEL, fontsize=_YLABEL_SIZE)

    # 3. HPF stream + detected transients.
    _used_hpf = (
        getattr(session.preprocessing_config, "photometry", None)
        and session.preprocessing_config.photometry.apply_hpf
    )
    _det_label = f"HPF {DFF_LABEL}" if _used_hpf else DFF_LABEL
    ax = axes[2]
    ax.plot(time_s, proc.photometry.dff_hpf, color="black", linewidth=0.3)
    _style.add_zero_y(ax, ensure_visible=True)
    if transients:
        peak_times = [t.peak_time for t in transients]
        peak_idxs = np.searchsorted(time_s, peak_times).clip(0, len(time_s) - 1)
        ax.plot(
            np.array(peak_times),
            proc.photometry.dff_hpf[peak_idxs],
            "v", color="black", markersize=3,
        )
    ax.set_ylabel(_det_label, fontsize=_YLABEL_SIZE)

    # 4. Opto stim raw trace (no axis crash if it's None — render an empty
    # panel and label it as missing so the layout stays consistent).
    ax = axes[3]
    if raw.stim_raw is not None and len(raw.stim_raw) == len(time_s):
        ax.plot(time_s, raw.stim_raw, color="black", linewidth=0.5)
        _style.add_zero_y(ax, ensure_visible=True)
        ax.set_ylabel("Opto stim (V)", fontsize=_YLABEL_SIZE)
    else:
        ax.text(0.5, 0.5, "no opto stim channel",
                ha="center", va="center", transform=ax.transAxes,
                color="gray")
        ax.set_ylabel("Opto stim", fontsize=_YLABEL_SIZE)
    ax.set_xlabel("Time (s)")

    _open_qc_axes(axes)
    _draw_landmark_lines_opto(axes, session.landmarks, offset=0.0)

    fig.suptitle(
        f"{session.mouse_id} S{session.heating_session} - opto sanity check"
    )

    path = os.path.join(
        output_dir,
        f"{session.mouse_id}_S{session.heating_session}_"
        f"{_cohort_filename_tag(session)}_full_trace.png",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    plot_baseline(session, output_dir, color="black")
    return path


@_style.styled
def plot_zoomed_opto(
    session: Session,
    output_dir: str,
    window_s: float = 150.0,
) -> str:
    """Opto variant of `plot_zoomed`. Two stacked panels centered on stim
    onset: dF/F on top, the opto stim trace below. All-black. Landmark
    lines say ``stim on`` / ``stim off``."""
    os.makedirs(output_dir, exist_ok=True)

    raw = session.raw
    proc = session.processed
    time_s = proc.time
    landmarks = session.landmarks

    t_zero = (landmarks.ueo_time
              or landmarks.equiv_ueo_time
              or landmarks.heating_start_time)
    if t_zero is None:
        return ""

    rel_time = time_s - t_zero
    mask = (rel_time >= -window_s) & (rel_time <= window_s)

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

    # 1. z-ΔF/F
    ax = axes[0]
    ax.plot(rel_time[mask], proc.photometry.dff_zscore[mask],
            color="black", linewidth=0.4)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel(DFF_LABEL)
    ax.set_xlim(-window_s, window_s)

    # 2. Opto stim trace.
    ax = axes[1]
    if raw.stim_raw is not None and len(raw.stim_raw) == len(time_s):
        ax.plot(rel_time[mask], raw.stim_raw[mask],
                color="black", linewidth=0.4)
        _style.add_zero_y(ax, ensure_visible=True)
        ax.set_ylabel("Opto stim (V)")
    else:
        ax.text(0.5, 0.5, "no opto stim channel",
                ha="center", va="center", transform=ax.transAxes,
                color="gray")
        ax.set_ylabel("Opto stim")
    ax.set_xlabel("Time relative to stim onset (s)")
    ax.set_xlim(-window_s, window_s)

    _open_qc_axes(axes)
    _draw_landmark_lines_opto(axes, landmarks, offset=t_zero)

    fig.suptitle(
        f"{session.mouse_id} S{session.heating_session} - opto zoomed"
    )

    path = os.path.join(
        output_dir,
        f"{session.mouse_id}_S{session.heating_session}_"
        f"{_cohort_filename_tag(session)}_peri_ictal.png",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def _decimate_for_html(arr: np.ndarray, max_points: int = 200_000) -> tuple:
    """Stride-decimate `arr` to keep it under `max_points`. Returns the
    decimated array and the stride used (so a shared time axis can be
    decimated identically).
    """
    n = len(arr)
    if n <= max_points:
        return arr, 1
    step = int(np.ceil(n / max_points))
    return arr[::step], step


def _plotly_zero_line() -> dict:
    """Canonical dark dashed zero-reference style for interactive QC plots."""
    return dict(
        color=_style.ZERO_LINE_COLOR,
        dash="dash",
        width=_style.ZERO_LINEWIDTH,
    )


def _plotly_range_with_zero(*arrays: np.ndarray) -> list:
    """Return a data-driven y-range with visible padding around zero."""
    lo = 0.0
    hi = 0.0
    for values in arrays:
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        if finite.size:
            lo = min(lo, float(np.min(finite)))
            hi = max(hi, float(np.max(finite)))
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), abs(hi), 1.0)
    pad = 0.04 * span
    return [lo - pad, hi + pad]


def _open_qc_plotly_axes(fig) -> None:
    """Show only bottom/left axis lines on interactive quality checks."""
    axis_style = dict(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        mirror=False,
        ticks="outside",
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)


def _plotly_landmark_top_margin(max_level: int, base: int = 110) -> int:
    """Reserve enough pixels for stacked landmark labels and the title."""
    return max(base, 90 + (max_level + 1) * 42)


def _add_landmark_overlays_plotly(
    fig, landmarks, *, offset: float = 0.0,
    x_range: Optional[tuple] = None,
) -> int:
    """Draw landmark vertical lines on every subplot row + a colored text
    label above the figure at each landmark's x-position. Mirrors
    `_draw_landmark_lines` for matplotlib.

    Parameters
    ----------
    fig : plotly Figure
    landmarks : SessionLandmarks
    offset : value to subtract from each landmark time (0 for absolute,
             t_zero for plots aligned to seizure onset).
    x_range : (xmin, xmax) - landmarks outside this range are skipped.
    """
    if landmarks is None:
        return -1
    entries = [
        ("heating_start", landmarks.heating_start_time),
        ("eec", landmarks.eec_time or landmarks.equiv_eec_time),
        ("ueo", landmarks.ueo_time or landmarks.equiv_ueo_time),
        ("behavioral_onset", landmarks.behavioral_onset_time
                              or landmarks.equiv_behavioral_onset_time),
        ("off", landmarks.off_time or landmarks.equiv_off_time),
    ]
    if x_range is None:
        finite_times = [float(t) - offset for _, t in entries if t is not None]
        if not finite_times:
            return -1
        x_range = (min(finite_times), max(finite_times))
    layout = _layout_landmark_labels(entries, x_range, offset)
    for key, x, level in layout:
        lcolor = LANDMARK_COLORS[key]
        # add_vline with row/col unset spans every subplot
        fig.add_vline(x=x, line=dict(color=lcolor, dash="dash", width=1.5),
                      opacity=0.7)
        fig.add_annotation(
            x=x, y=1.0, xref="x", yref="paper",
            yshift=8 + level * 42,
            # Plotly takes "<br>" for line breaks, not "\n" — translate.
            text=LANDMARK_DISPLAY_LABELS[key].replace("\n", "<br>"),
            showarrow=False, font=dict(color=lcolor, size=14),
            xanchor="center", yanchor="bottom",
        )
    return max((level for _, _, level in layout), default=-1)


def _write_baseline_html(
    session: Session,
    time_s: np.ndarray,
    dff: np.ndarray,
    ecog: np.ndarray,
    color: str,
    x_range: tuple,
    html_path: str,
) -> None:
    """Interactive companion to the zoomed-out actual-baseline PNG."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _, step = _decimate_for_html(time_s)
    t_axis = time_s[::step]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scattergl(
            x=t_axis, y=dff[::step], mode="lines",
            line=dict(color=color, width=1),
            name=DFF_LABEL, showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=t_axis, y=ecog[::step], mode="lines",
            line=dict(color=ECOG_COLOR, width=1),
            name="Filt ECoG", showlegend=False,
        ),
        row=2, col=1,
    )
    for row in (1, 2):
        fig.add_hline(
            y=0, line=_plotly_zero_line(), opacity=0.9,
            row=row, col=1,
        )
    fig.update_layout(
        title=f"{session.mouse_id} S{session.heating_session} - baseline",
        height=900, width=1600,
        margin=dict(t=90, l=100, r=40, b=70),
        font=dict(size=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(
        title_text=DFF_LABEL,
        range=_plotly_range_with_zero(dff[::step]),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Filt ECoG (µV)",
        range=_plotly_range_with_zero(ecog[::step]),
        row=2, col=1,
    )
    fig.update_xaxes(
        title_text="Time (s) - baseline", range=list(x_range), row=2, col=1,
    )
    fig.update_xaxes(range=list(x_range), row=1, col=1)
    _open_qc_plotly_axes(fig)

    import plotly.io as pio
    pio.write_html(
        fig, html_path, include_plotlyjs="cdn",
        full_html=True, auto_open=False,
    )


def _write_peri_ictal_html(
    session: Session,
    rel_time: np.ndarray,
    dff: np.ndarray,
    ecog: np.ndarray,
    color: str,
    landmarks,
    t_zero: float,
    window_s: float,
    html_path: str,
) -> None:
    """Interactive companion to plot_zoomed's PNG. Two stacked traces with
    landmark lines + labels rendered above the top panel."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Scattergl(x=rel_time, y=dff, mode="lines",
                     line=dict(color=color, width=1),
                     name="z-ΔF/F", showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(x=rel_time, y=ecog, mode="lines",
                     line=dict(color=ECOG_COLOR, width=1),
                     name="Filt ECoG", showlegend=False),
        row=2, col=1,
    )
    for row in (1, 2):
        fig.add_hline(
            y=0, line=_plotly_zero_line(), opacity=0.9,
            row=row, col=1,
        )

    max_landmark_level = _add_landmark_overlays_plotly(
        fig, landmarks, offset=t_zero, x_range=(-window_s, window_s),
    )

    fig.update_layout(
        title=f"{session.mouse_id} S{session.heating_session} - zoomed",
        height=900, width=1600,
        margin=dict(
            t=_plotly_landmark_top_margin(max_landmark_level),
            l=90, r=40, b=70,
        ),
        font=dict(size=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(
        title_text=DFF_LABEL,
        range=_plotly_range_with_zero(dff),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Filt ECoG (µV)",
        range=_plotly_range_with_zero(ecog),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Time relative to seizure onset (s)",
                     row=2, col=1, range=[-window_s, window_s])
    fig.update_xaxes(range=[-window_s, window_s], row=1, col=1)
    _open_qc_plotly_axes(fig)

    import plotly.io as pio
    pio.write_html(fig, html_path, include_plotlyjs="cdn",
                   full_html=True, auto_open=False)


def _write_sanity_check_html(
    session: Session,
    color: str,
    detection_label: str,
    transients: Optional[List[TransientEvent]],
    html_path: str,
) -> None:
    """Interactive companion to plot_sanity_check's PNG. Six stacked panels
    over the full recording: raw 470/405, z-ΔF/F, HPF detection stream with
    transient markers, raw ECoG, filtered ECoG, temperature.

    Long ECoG traces are stride-decimated so the file stays interactive in a
    browser; photometry channels run at ~130 Hz so they're left intact.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    raw = session.raw
    proc = session.processed
    time_s = proc.time

    # All raw/processed arrays share `time_s` (sync.py interpolates the
    # photometry onto the ECoG timebase), so a single decimation step keeps
    # every trace aligned with the shared x-axis.
    _, step = _decimate_for_html(time_s)
    t_axis = time_s[::step]

    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.025,
    )

    # 1. Raw 470 & 405
    fig.add_trace(go.Scattergl(x=t_axis, y=raw.signal_470[::step],
                               mode="lines", line=dict(color="blue", width=1),
                               name="470 nm"),
                  row=1, col=1)
    fig.add_trace(go.Scattergl(x=t_axis, y=raw.signal_405[::step],
                               mode="lines", line=dict(color="violet", width=1),
                               name="405 nm"),
                  row=1, col=1)
    fig.add_hline(
        y=0, line=_plotly_zero_line(), opacity=0.9, row=1, col=1,
    )

    # 2. Corrected z-ΔF/F
    fig.add_trace(go.Scattergl(x=t_axis, y=proc.photometry.dff_zscore[::step],
                               mode="lines", line=dict(color=color, width=1),
                               name=DFF_LABEL, showlegend=False),
                  row=2, col=1)
    fig.add_hline(
        y=0, line=_plotly_zero_line(), opacity=0.9, row=2, col=1,
    )

    # 3. Detection stream + transient markers
    fig.add_trace(go.Scattergl(x=t_axis, y=proc.photometry.dff_hpf[::step],
                               mode="lines", line=dict(color=color, width=1),
                               name=detection_label, showlegend=False),
                  row=3, col=1)
    fig.add_hline(
        y=0, line=_plotly_zero_line(), opacity=0.9, row=3, col=1,
    )
    if transients:
        peak_times = np.array([tr.peak_time for tr in transients])
        peak_idx = np.searchsorted(time_s, peak_times).clip(0, len(time_s) - 1)
        peak_y = proc.photometry.dff_hpf[peak_idx]
        fig.add_trace(go.Scattergl(
            x=peak_times, y=peak_y, mode="markers",
            marker=dict(color="black", size=6, symbol="triangle-down"),
            name="transients", showlegend=False,
        ), row=3, col=1)

    # 4 & 5. ECoG (raw + filtered)
    fig.add_trace(go.Scattergl(x=t_axis, y=raw.ecog[::step], mode="lines",
                               line=dict(color=ECOG_COLOR, width=1),
                               name="Raw ECoG", showlegend=False),
                  row=4, col=1)
    fig.add_hline(
        y=0, line=_plotly_zero_line(), opacity=0.9, row=4, col=1,
    )
    fig.add_trace(go.Scattergl(x=t_axis, y=proc.ecog_filtered[::step],
                               mode="lines",
                               line=dict(color=ECOG_COLOR, width=1),
                               name="Filt ECoG", showlegend=False),
                  row=5, col=1)
    fig.add_hline(
        y=0, line=_plotly_zero_line(), opacity=0.9, row=5, col=1,
    )

    # 6. Temperature
    fig.add_trace(go.Scattergl(x=t_axis, y=proc.temperature_smooth[::step],
                               mode="lines",
                               line=dict(color=TEMP_COLOR, width=1.4),
                               name="temp"),
                  row=6, col=1)
    bl_temp = getattr(session.landmarks, "baseline_temp", None) if session.landmarks else None
    if bl_temp is not None and not np.isnan(bl_temp):
        fig.add_hline(y=bl_temp, line=dict(color="black", dash="dash", width=1),
                      opacity=0.5, row=6, col=1)

    # Landmark overlays (heating_start, EEC, UEO, behavioral_onset, OFF)
    max_landmark_level = _add_landmark_overlays_plotly(
        fig, session.landmarks, offset=0.0,
        x_range=(float(time_s[0]), float(time_s[-1])),
    )

    fig.update_layout(
        title=(f"{session.mouse_id} S{session.heating_session} "
               f"({_cohort_filename_tag(session)}) - sanity check"),
        height=1500, width=1800,
        margin=dict(
            t=_plotly_landmark_top_margin(max_landmark_level),
            l=110, r=40, b=70,
        ),
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(
        title_text="Raw (V)",
        range=_plotly_range_with_zero(
            raw.signal_470[::step], raw.signal_405[::step],
        ),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text=DFF_LABEL,
        range=_plotly_range_with_zero(proc.photometry.dff_zscore[::step]),
        row=2, col=1,
    )
    fig.update_yaxes(
        title_text=detection_label,
        range=_plotly_range_with_zero(proc.photometry.dff_hpf[::step]),
        row=3, col=1,
    )
    fig.update_yaxes(
        title_text="Raw ECoG (µV)",
        range=_plotly_range_with_zero(raw.ecog[::step]),
        row=4, col=1,
    )
    fig.update_yaxes(
        title_text="Filt ECoG (µV)",
        range=_plotly_range_with_zero(proc.ecog_filtered[::step]),
        row=5, col=1,
    )
    fig.update_yaxes(title_text="Temp (°C)", row=6, col=1)
    fig.update_xaxes(title_text="Time (s)", row=6, col=1)
    _open_qc_plotly_axes(fig)

    import plotly.io as pio
    pio.write_html(fig, html_path, include_plotlyjs="cdn",
                   full_html=True, auto_open=False)


def _write_transient_review_html(
    session: Session,
    t_baseline: np.ndarray,
    baseline_mask: np.ndarray,
    detection_label: str,
    color: str,
    prominences: List[float],
    transients_by_prominence: Dict[float, List[TransientEvent]],
    heat_start: float,
    html_path: str,
) -> None:
    """Interactive companion to plot_transient_review's PNG: raw 470/405,
    corrected ΔF/F, then one panel per prominence threshold with detected
    peaks marked.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    raw = session.raw
    proc = session.processed
    n_rows = 2 + len(prominences)

    # All baseline-masked arrays share `t_baseline` - pick a single
    # decimation step so every panel stays aligned.
    _, step = _decimate_for_html(t_baseline)
    t_axis = t_baseline[::step]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
    )

    # Row 1: raw 470 + 405
    fig.add_trace(go.Scattergl(x=t_axis, y=raw.signal_470[baseline_mask][::step],
                               mode="lines", line=dict(color="blue", width=1),
                               name="470 nm"),
                  row=1, col=1)
    fig.add_trace(go.Scattergl(x=t_axis, y=raw.signal_405[baseline_mask][::step],
                               mode="lines", line=dict(color="violet", width=1),
                               name="405 nm"),
                  row=1, col=1)
    fig.add_hline(
        y=0, line=_plotly_zero_line(), opacity=0.9, row=1, col=1,
    )

    # Row 2: corrected dF/F (not z-scored)
    fig.add_trace(go.Scattergl(x=t_axis,
                               y=proc.photometry.dff[baseline_mask][::step],
                               mode="lines", line=dict(color=color, width=1),
                               name="ΔF/F", showlegend=False),
                  row=2, col=1)
    fig.add_hline(
        y=0, line=_plotly_zero_line(), opacity=0.9, row=2, col=1,
    )

    # Rows 3..: detection stream + markers per prominence threshold
    det_signal = proc.photometry.dff_hpf
    full_time = proc.time
    det_baseline = det_signal[baseline_mask][::step]
    for i, prom in enumerate(prominences):
        row = 3 + i
        fig.add_trace(go.Scattergl(x=t_axis, y=det_baseline, mode="lines",
                                   line=dict(color=color, width=1),
                                   name=f"prom ≥ {prom}", showlegend=False),
                      row=row, col=1)
        fig.add_hline(
            y=0, line=_plotly_zero_line(), opacity=0.9,
            row=row, col=1,
        )

        bl_trs = [tr for tr in transients_by_prominence.get(prom, [])
                  if tr.peak_time <= heat_start]
        if bl_trs:
            peak_times = np.array([tr.peak_time for tr in bl_trs])
            peak_idx = np.searchsorted(full_time, peak_times).clip(0, len(full_time) - 1)
            peak_y = det_signal[peak_idx]
            fig.add_trace(go.Scattergl(
                x=peak_times, y=peak_y, mode="markers",
                marker=dict(color="black", size=7, symbol="triangle-down"),
                name=f"peaks @ {prom}", showlegend=False,
            ), row=row, col=1)

        fig.update_yaxes(
            title_text=f"{detection_label}<br>prom ≥ {prom}",
            range=_plotly_range_with_zero(det_baseline),
            row=row, col=1,
        )

    max_landmark_level = _add_landmark_overlays_plotly(
        fig, session.landmarks, offset=0.0,
        x_range=(0.0, float(heat_start)),
    )

    fig.update_layout(
        title=(f"{session.mouse_id} S{session.heating_session} "
               f"({_cohort_filename_tag(session)}) - transient review (baseline)"),
        height=300 * n_rows, width=1800,
        margin=dict(
            t=_plotly_landmark_top_margin(max_landmark_level),
            l=120, r=40, b=70,
        ),
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(
        title_text="Raw (V)",
        range=_plotly_range_with_zero(
            raw.signal_470[baseline_mask][::step],
            raw.signal_405[baseline_mask][::step],
        ),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Corrected ΔF/F",
        range=_plotly_range_with_zero(
            proc.photometry.dff[baseline_mask][::step],
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1,
                     range=[0.0, float(heat_start)])
    _open_qc_plotly_axes(fig)

    import plotly.io as pio
    pio.write_html(fig, html_path, include_plotlyjs="cdn",
                   full_html=True, auto_open=False)


@_style.styled
def plot_transient_review(
    session: Session,
    output_dir: str,
    prominences: List[float],
    transients_by_prominence: Dict[float, List[TransientEvent]],
) -> str:
    """Display v3: transient prominence review over baseline period.

    Rows:
        0: Raw 470nm + 405nm
        1: Corrected dF/F (NOT z-scored)
        2..N: z-ΔF/F HPF with transient peaks at each prominence level
    x-axis: 0 to heating_start_time (baseline only).
    """
    os.makedirs(output_dir, exist_ok=True)

    raw = session.raw
    proc = session.processed
    time_s = proc.time

    heat_start = session.landmarks.heating_start_time
    mask = time_s <= heat_start
    t = time_s[mask]

    n_rows = 2 + len(prominences)
    fig, axes = plt.subplots(n_rows, 1, figsize=(22, 4.5 * n_rows), sharex=True)

    color = _cohort_color(session)

    # Row 0: raw 470 + 405
    ax = axes[0]
    ax.plot(t, raw.signal_470[mask], color="blue", linewidth=0.3, label="470nm")
    ax.plot(t, raw.signal_405[mask], color="violet", linewidth=0.3, label="405nm")
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel("Raw (V)")
    ax.legend(loc="upper right")

    # Row 1: corrected dF/F (NOT z-scored)
    ax = axes[1]
    ax.plot(t, proc.photometry.dff[mask], color=color, linewidth=0.3)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_ylabel("Corrected ΔF/F")

    # Rows 2..N: detection stream with transient markers at each prominence
    _used_hpf = getattr(session.preprocessing_config, 'photometry', None) and session.preprocessing_config.photometry.apply_hpf
    _det_label = f"HPF {DFF_LABEL}" if _used_hpf else DFF_LABEL
    det_signal = proc.photometry.dff_hpf
    for i, prom in enumerate(prominences):
        ax = axes[2 + i]
        if det_signal is not None:
            ax.plot(t, det_signal[mask], color=color, linewidth=0.3)
        _style.add_zero_y(ax, ensure_visible=True)
        transients = transients_by_prominence.get(prom, [])
        bl_transients = [tr for tr in transients if tr.peak_time <= heat_start]
        if bl_transients and det_signal is not None:
            peak_times = [tr.peak_time for tr in bl_transients]
            peak_idxs = np.searchsorted(time_s, peak_times).clip(0, len(time_s) - 1)
            ax.plot(np.array(peak_times), det_signal[peak_idxs],
                    "v", color="black", markersize=4)
        ax.set_ylabel(f"{_det_label}\nprom ≥ {prom}")

    axes[-1].set_xlabel("Time (s)")
    _open_qc_axes(axes)
    fig.suptitle(
        f"{session.mouse_id} S{session.heating_session} - transient review (baseline)",
    )

    path = os.path.join(
        output_dir,
        f"{session.mouse_id}_S{session.heating_session}_{_cohort_filename_tag(session)}_transients.png",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Companion interactive HTML - mirrors the prominence-stack PNG.
    try:
        _write_transient_review_html(
            session=session,
            t_baseline=t,
            baseline_mask=mask,
            detection_label=_det_label,
            color=color,
            prominences=prominences,
            transients_by_prominence=transients_by_prominence,
            heat_start=heat_start,
            html_path=os.path.join(
                output_dir,
                f"{session.mouse_id}_S{session.heating_session}_{_cohort_filename_tag(session)}_transients.html",
            ),
        )
    except Exception:
        pass

    return path
