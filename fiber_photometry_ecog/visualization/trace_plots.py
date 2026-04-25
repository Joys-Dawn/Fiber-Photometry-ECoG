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

from ..core.data_models import Session, TransientEvent
from .colors import ECOG_COLOR, TEMP_COLOR, COHORT_COLORS

LANDMARK_COLORS = {
    "Heating start": "green",
    "EEC": "orange",
    "UEO": "red",
    "Behavioral onset": "purple",
    "OFF": "blue",
}


def _draw_landmark_lines(
    axes,
    landmarks,
    offset: float = 0.0,
    label_axis: int = 0,
):
    """Draw vertical lines for all non-None landmarks across *axes*.

    Parameters
    ----------
    axes : array of Axes
    landmarks : SessionLandmarks
    offset : value subtracted from each time (0 for absolute, t_zero for relative)
    label_axis : index of the axis that gets the legend labels (others get no label)
    """
    entries = [
        ("Heating start", landmarks.heating_start_time),
        ("EEC", landmarks.eec_time or landmarks.equiv_eec_time),
        ("UEO", landmarks.ueo_time or landmarks.equiv_ueo_time),
        ("Behavioral onset", landmarks.behavioral_onset_time or landmarks.equiv_behavioral_onset_time),
        ("OFF", landmarks.off_time or landmarks.equiv_off_time),
    ]
    drawn = 0
    for name, t in entries:
        if t is None:
            continue
        x = t - offset
        color = LANDMARK_COLORS[name]
        xlim = axes[0].get_xlim()
        if x < xlim[0] or x > xlim[1]:
            continue
        for i, ax in enumerate(axes):
            ax.axvline(
                x, color=color, linestyle="--", linewidth=0.8, alpha=0.7,
                label=name if i == label_axis else None,
            )
        drawn += 1
    if drawn:
        axes[label_axis].legend(loc="upper right", fontsize=6, ncol=min(drawn, 5))


def _cohort_color(session: Session) -> str:
    if session.n_seizures > 0:
        return COHORT_COLORS["seizure"]
    if session.genotype == "Scn1a":
        return COHORT_COLORS["failed_seizure"]
    return COHORT_COLORS["wt"]


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

    fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)

    # 1. Raw 470 & 405
    ax = axes[0]
    ax.plot(time_s, raw.signal_470, color="blue", linewidth=0.3, label="470nm")
    ax.plot(time_s, raw.signal_405, color="violet", linewidth=0.3, label="405nm")
    ax.set_ylabel("Raw (V)")
    ax.legend(loc="upper right", fontsize=7)

    # 2. Corrected z-ΔF/F
    ax = axes[1]
    ax.plot(time_s, proc.photometry.dff_zscore, color=color, linewidth=0.3)
    ax.set_ylabel("z-ΔF/F")

    # 3. HPF/detection stream + transients
    _used_hpf = getattr(session.preprocessing_config, 'photometry', None) and session.preprocessing_config.photometry.apply_hpf
    _det_label = "HPF z-ΔF/F" if _used_hpf else "z-ΔF/F"
    ax = axes[2]
    ax.plot(time_s, proc.photometry.dff_hpf, color=color, linewidth=0.3)
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
    ax.set_ylabel("Temp (°C)")
    ax.set_xlabel("Time (s)")

    _draw_landmark_lines(axes, session.landmarks, offset=0.0, label_axis=5)

    fig.suptitle(f"{session.mouse_id} S{session.heating_session} — sanity check", fontsize=10)
    fig.tight_layout()

    path = os.path.join(
        output_dir, f"{session.mouse_id}_S{session.heating_session}_v1_sanity.png"
    )
    fig.savefig(path, dpi=150)
    plt.close("all")
    return path


def plot_zoomed(
    session: Session,
    output_dir: str,
    window_s: float = 300.0,
) -> str:
    """Display v2: 2 subplots centered on seizure onset / equivalent.

    Subplots:
        1. Corrected z-ΔF/F
        2. Filtered ECoG

    x-axis: seconds with 0 = UEO / equivalent.
    Returns the saved PNG path.
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

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 1. z-ΔF/F
    ax = axes[0]
    ax.plot(rel_time[mask], proc.photometry.dff_zscore[mask], color=color, linewidth=0.4)
    ax.set_ylabel("z-ΔF/F")

    # 2. Filtered ECoG
    ax = axes[1]
    ax.plot(rel_time[mask], proc.ecog_filtered[mask], color=ECOG_COLOR, linewidth=0.4)
    ax.set_ylabel("Filt ECoG (µV)")
    ax.set_xlabel("Time relative to seizure onset (s)")

    _draw_landmark_lines(axes, landmarks, offset=t_zero, label_axis=0)

    fig.suptitle(f"{session.mouse_id} S{session.heating_session} — zoomed", fontsize=10)
    fig.tight_layout()

    path = os.path.join(
        output_dir, f"{session.mouse_id}_S{session.heating_session}_v2_zoomed.png"
    )
    fig.savefig(path, dpi=150)
    plt.close("all")
    return path


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
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.5 * n_rows), sharex=True)

    color = _cohort_color(session)

    # Row 0: raw 470 + 405
    ax = axes[0]
    ax.plot(t, raw.signal_470[mask], color="blue", linewidth=0.3, label="470nm")
    ax.plot(t, raw.signal_405[mask], color="violet", linewidth=0.3, label="405nm")
    ax.set_ylabel("Raw (V)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Raw Signals")

    # Row 1: corrected dF/F (NOT z-scored)
    ax = axes[1]
    ax.plot(t, proc.photometry.dff[mask], color=color, linewidth=0.3)
    ax.set_ylabel("ΔF/F")
    ax.set_title("Corrected ΔF/F")

    # Rows 2..N: detection stream with transient markers at each prominence
    _used_hpf = getattr(session.preprocessing_config, 'photometry', None) and session.preprocessing_config.photometry.apply_hpf
    _det_label = "HPF z-ΔF/F" if _used_hpf else "z-ΔF/F"
    det_signal = proc.photometry.dff_hpf
    for i, prom in enumerate(prominences):
        ax = axes[2 + i]
        if det_signal is not None:
            ax.plot(t, det_signal[mask], color=color, linewidth=0.3)
        transients = transients_by_prominence.get(prom, [])
        bl_transients = [tr for tr in transients if tr.peak_time <= heat_start]
        if bl_transients and det_signal is not None:
            peak_times = [tr.peak_time for tr in bl_transients]
            peak_idxs = np.searchsorted(time_s, peak_times).clip(0, len(time_s) - 1)
            ax.plot(np.array(peak_times), det_signal[peak_idxs],
                    "v", color="black", markersize=4)
        ax.set_ylabel(_det_label)
        ax.set_title(f"{_det_label} — prominence >= {prom}")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{session.mouse_id} S{session.heating_session} — transient review (baseline)",
        fontsize=10,
    )
    fig.tight_layout()

    path = os.path.join(
        output_dir,
        f"{session.mouse_id}_S{session.heating_session}_v3_transients.png",
    )
    fig.savefig(path, dpi=150)
    plt.close("all")
    return path
