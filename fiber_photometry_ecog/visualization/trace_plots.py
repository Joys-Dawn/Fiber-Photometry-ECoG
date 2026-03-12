"""
Per-session trace plots.

Display v1: full sanity check (6 subplots, entire trace).
Display v2: zoomed processed (2 subplots, centered on seizure onset).
"""

import os
from typing import List, Optional

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.data_models import Session, TransientEvent
from .colors import ECOG_COLOR, TEMP_COLOR, COHORT_COLORS


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

    # 3. HPF + transients
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
    ax.set_ylabel("HPF z-ΔF/F")

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

    fig.suptitle(f"{session.mouse_id} — sanity check", fontsize=10)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{session.mouse_id}_v1_sanity.png")
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
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("z-ΔF/F")

    # 2. Filtered ECoG
    ax = axes[1]
    ax.plot(rel_time[mask], proc.ecog_filtered[mask], color=ECOG_COLOR, linewidth=0.4)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Filt ECoG (µV)")
    ax.set_xlabel("Time relative to seizure onset (s)")

    fig.suptitle(f"{session.mouse_id} — zoomed", fontsize=10)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{session.mouse_id}_v2_zoomed.png")
    fig.savefig(path, dpi=150)
    plt.close("all")
    return path
