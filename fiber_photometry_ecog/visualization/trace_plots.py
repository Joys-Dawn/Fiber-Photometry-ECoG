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
    ECOG_COLOR, TEMP_COLOR, COHORT_COLORS,
    LANDMARK_COLORS, LANDMARK_DISPLAY_LABELS, DFF_LABEL,
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
    xlim = axes[0].get_xlim()
    top_ax = axes[0]
    for key, t in entries:
        if t is None:
            continue
        x = t - offset
        if x < xlim[0] or x > xlim[1]:
            continue
        for ax in axes:
            ax.axvline(
                x, color=LANDMARK_COLORS[key], linestyle="--", linewidth=0.8, alpha=0.7,
            )
        top_ax.text(
            x, 1.02, LANDMARK_DISPLAY_LABELS[key],
            transform=top_ax.get_xaxis_transform(),
            ha="center", va="bottom",
            color=LANDMARK_COLORS[key],
            clip_on=False,
        )


def _cohort_color(session: Session) -> str:
    return COHORT_COLORS[classify_session(session)]


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
    plots.
    """
    return _FILENAME_TAGS[classify_session(session)]


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

    # 1. Raw 470 & 405
    ax = axes[0]
    ax.plot(time_s, raw.signal_470, color="blue", linewidth=0.3, label="470nm")
    ax.plot(time_s, raw.signal_405, color="violet", linewidth=0.3, label="405nm")
    ax.set_ylabel("Raw (V)")
    ax.legend(loc="upper right")

    # 2. Corrected z-ΔF/F
    ax = axes[1]
    ax.plot(time_s, proc.photometry.dff_zscore, color=color, linewidth=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_ylabel(DFF_LABEL)

    # 3. HPF/detection stream + transients
    _used_hpf = getattr(session.preprocessing_config, 'photometry', None) and session.preprocessing_config.photometry.apply_hpf
    _det_label = f"HPF {DFF_LABEL}" if _used_hpf else DFF_LABEL
    ax = axes[2]
    ax.plot(time_s, proc.photometry.dff_hpf, color=color, linewidth=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    if transients:
        peak_times = [t.peak_time for t in transients]
        peak_idxs = np.searchsorted(time_s, peak_times).clip(0, len(time_s) - 1)
        ax.plot(
            np.array(peak_times),
            proc.photometry.dff_hpf[peak_idxs],
            "v", color="black", markersize=3,
        )
    ax.set_ylabel(_det_label)

    # 4. Raw ECoG
    ax = axes[3]
    ax.plot(time_s, raw.ecog, color=ECOG_COLOR, linewidth=0.3)
    ax.set_ylabel("Raw ECoG (µV)")

    # 5. Filtered ECoG
    ax = axes[4]
    ax.plot(time_s, proc.ecog_filtered, color=ECOG_COLOR, linewidth=0.3)
    ax.set_ylabel("Filt ECoG (µV)")

    # 6. Temperature
    ax = axes[5]
    ax.plot(time_s, proc.temperature_smooth, color=TEMP_COLOR, linewidth=0.5)
    bl_temp = getattr(session.landmarks, "baseline_temp", None) if session.landmarks else None
    if bl_temp is not None and not np.isnan(bl_temp):
        ax.axhline(bl_temp, color="black", linestyle="--", linewidth=0.6, alpha=0.5,
                   label=f"baseline={bl_temp:.2f}°C")
        ax.legend(loc="lower right")
    ax.set_ylabel("Temp (°C)")
    ax.set_xlabel("Time (s)")

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
    ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_ylabel(DFF_LABEL)
    ax.set_xlim(-window_s, window_s)

    # 2. Filtered ECoG
    ax = axes[1]
    ax.plot(rel_time[mask], proc.ecog_filtered[mask], color=ECOG_COLOR, linewidth=0.4)
    ax.set_ylabel("Filt ECoG (µV)")
    ax.set_xlabel("Time relative to seizure onset (s)")
    ax.set_xlim(-window_s, window_s)

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


def _add_landmark_overlays_plotly(
    fig, landmarks, *, offset: float = 0.0,
    x_range: Optional[tuple] = None,
) -> None:
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
        return
    entries = [
        ("heating_start", landmarks.heating_start_time),
        ("eec", landmarks.eec_time or landmarks.equiv_eec_time),
        ("ueo", landmarks.ueo_time or landmarks.equiv_ueo_time),
        ("behavioral_onset", landmarks.behavioral_onset_time
                              or landmarks.equiv_behavioral_onset_time),
        ("off", landmarks.off_time or landmarks.equiv_off_time),
    ]
    for key, t in entries:
        if t is None:
            continue
        x = t - offset
        if x_range is not None and (x < x_range[0] or x > x_range[1]):
            continue
        lcolor = LANDMARK_COLORS[key]
        # add_vline with row/col unset spans every subplot
        fig.add_vline(x=x, line=dict(color=lcolor, dash="dash", width=1.5),
                      opacity=0.7)
        fig.add_annotation(
            x=x, y=1.02, xref="x", yref="paper",
            # Plotly takes "<br>" for line breaks, not "\n" — translate.
            text=LANDMARK_DISPLAY_LABELS[key].replace("\n", "<br>"),
            showarrow=False, font=dict(color=lcolor, size=14),
            xanchor="center", yanchor="bottom",
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
    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1),
                  opacity=0.4, row=1, col=1)

    _add_landmark_overlays_plotly(
        fig, landmarks, offset=t_zero, x_range=(-window_s, window_s),
    )

    fig.update_layout(
        title=f"{session.mouse_id} S{session.heating_session} - zoomed",
        height=900, width=1600,
        margin=dict(t=110, l=90, r=40, b=70),
        font=dict(size=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(title_text=DFF_LABEL, row=1, col=1)
    fig.update_yaxes(title_text="Filt ECoG (µV)", row=2, col=1)
    fig.update_xaxes(title_text="Time relative to seizure onset (s)",
                     row=2, col=1, range=[-window_s, window_s])
    fig.update_xaxes(range=[-window_s, window_s], row=1, col=1)

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

    # 2. Corrected z-ΔF/F
    fig.add_trace(go.Scattergl(x=t_axis, y=proc.photometry.dff_zscore[::step],
                               mode="lines", line=dict(color=color, width=1),
                               name=DFF_LABEL, showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1),
                  opacity=0.4, row=2, col=1)

    # 3. Detection stream + transient markers
    fig.add_trace(go.Scattergl(x=t_axis, y=proc.photometry.dff_hpf[::step],
                               mode="lines", line=dict(color=color, width=1),
                               name=detection_label, showlegend=False),
                  row=3, col=1)
    fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1),
                  opacity=0.4, row=3, col=1)
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
    fig.add_trace(go.Scattergl(x=t_axis, y=proc.ecog_filtered[::step],
                               mode="lines",
                               line=dict(color=ECOG_COLOR, width=1),
                               name="Filt ECoG", showlegend=False),
                  row=5, col=1)

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
    _add_landmark_overlays_plotly(
        fig, session.landmarks, offset=0.0,
        x_range=(float(time_s[0]), float(time_s[-1])),
    )

    fig.update_layout(
        title=(f"{session.mouse_id} S{session.heating_session} "
               f"({_cohort_filename_tag(session)}) - sanity check"),
        height=1500, width=1800,
        margin=dict(t=110, l=110, r=40, b=70),
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(title_text="Raw (V)", row=1, col=1)
    fig.update_yaxes(title_text=DFF_LABEL, row=2, col=1)
    fig.update_yaxes(title_text=detection_label, row=3, col=1)
    fig.update_yaxes(title_text="Raw ECoG (µV)", row=4, col=1)
    fig.update_yaxes(title_text="Filt ECoG (µV)", row=5, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=6, col=1)
    fig.update_xaxes(title_text="Time (s)", row=6, col=1)

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

    # Row 2: corrected dF/F (not z-scored)
    fig.add_trace(go.Scattergl(x=t_axis,
                               y=proc.photometry.dff[baseline_mask][::step],
                               mode="lines", line=dict(color=color, width=1),
                               name="ΔF/F", showlegend=False),
                  row=2, col=1)

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
        fig.add_hline(y=0, line=dict(color="black", dash="dash", width=1),
                      opacity=0.4, row=row, col=1)

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

        fig.update_yaxes(title_text=f"{detection_label}<br>prom ≥ {prom}",
                         row=row, col=1)

    _add_landmark_overlays_plotly(
        fig, session.landmarks, offset=0.0,
        x_range=(0.0, float(heat_start)),
    )

    fig.update_layout(
        title=(f"{session.mouse_id} S{session.heating_session} "
               f"({_cohort_filename_tag(session)}) - transient review (baseline)"),
        height=300 * n_rows, width=1800,
        margin=dict(t=110, l=120, r=40, b=70),
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_yaxes(title_text="Raw (V)", row=1, col=1)
    fig.update_yaxes(title_text="Corrected ΔF/F", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1,
                     range=[0.0, float(heat_start)])

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
    ax.set_ylabel("Raw (V)")
    ax.legend(loc="upper right")

    # Row 1: corrected dF/F (NOT z-scored)
    ax = axes[1]
    ax.plot(t, proc.photometry.dff[mask], color=color, linewidth=0.3)
    ax.set_ylabel("Corrected ΔF/F")

    # Rows 2..N: detection stream with transient markers at each prominence
    _used_hpf = getattr(session.preprocessing_config, 'photometry', None) and session.preprocessing_config.photometry.apply_hpf
    _det_label = f"HPF {DFF_LABEL}" if _used_hpf else DFF_LABEL
    det_signal = proc.photometry.dff_hpf
    for i, prom in enumerate(prominences):
        ax = axes[2 + i]
        if det_signal is not None:
            ax.plot(t, det_signal[mask], color=color, linewidth=0.3)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
        transients = transients_by_prominence.get(prom, [])
        bl_transients = [tr for tr in transients if tr.peak_time <= heat_start]
        if bl_transients and det_signal is not None:
            peak_times = [tr.peak_time for tr in bl_transients]
            peak_idxs = np.searchsorted(time_s, peak_times).clip(0, len(time_s) - 1)
            ax.plot(np.array(peak_times), det_signal[peak_idxs],
                    "v", color="black", markersize=4)
        ax.set_ylabel(f"{_det_label}\nprom ≥ {prom}")

    axes[-1].set_xlabel("Time (s)")
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
