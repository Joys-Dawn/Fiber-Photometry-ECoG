"""Generate example traces for selected recordings.

Writes PNGs to ``LC_paper/results/examples`` independently of the analysis runners.

Run from the repository root with:

    python -m LC_paper.generate_example_plots
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PAPER_ROOT / "results" / "examples"

from fiber_photometry_ecog.visualization import _style  # noqa: E402
from fiber_photometry_ecog.visualization.colors import (  # noqa: E402
    DFF_LABEL,
    cohort_color,
)
from fiber_photometry_ecog.preprocessing.transient_detection import (  # noqa: E402
    detect_transients,
)
from LC_paper.run_grab_opto import (  # noqa: E402
    CACHE_ROOT as GRAB_OPTO_CACHE_ROOT,
    CONDITION_COLORS,
    CONDITIONS,
    DATA_ROOT as GRAB_OPTO_ROOT,
    EXPECTED_MOUSE_IDS,
    EXPECTED_TRAIN_COUNTS,
    TRAINS_TO_USE,
    RAW_DFF_LABEL,
    _baseline_correct_dff,
    _detect_and_select_trains,
    _preprocess_dff,
    _validate_sessions,
)
from LC_paper.run_threshold_sweep import (  # noqa: E402
    _make_transient_config,
    load_sessions,
    preprocess_and_cache,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


LC_ROOT = Path(r"D:\Ephys_Data\Hyperthermia_seizure\LC_GCaMP")
DG_ROOT = Path(r"D:\Ephys_Data\Hyperthermia_seizure\DG_GRABne")
MUTANT_ROOT = Path(r"D:\Ephys_Data\Hyperthermia_seizure\DG_GRABneMutant")

LC_CACHE_KEY = "hyperthermia_lc_gcamp"
DG_CACHE_KEY = "hyperthermia_dg_grabne"
STRATEGY = "C"

DS_COLOR = cohort_color("DS_sz")
DS_NOSZ_COLOR = cohort_color("DS_no_sz")
WT_COLOR = cohort_color("WT_no_sz")
RAW_405_COLOR = "violet"
RAW_470_COLOR = "blue"
LINE_WIDTH = 1.0
TRANSIENT_LINE_WIDTH = 1.5
YOHIMBINE_LINE_WIDTH = 1.5
FIG_DPI = 200

BASELINE_WINDOW_S = (60.0, 90.0)
HEATING_WINDOW_S = (-200.0, 200.0)

PLOT_RC = dict(_style._PLOT_RC)


def _select_session(sessions, mouse_id: str, session_number: int, date: str):
    matches = [
        session
        for session in sessions
        if str(session.mouse_id) == mouse_id
        and int(session.heating_session) == session_number
        and session.date == date
    ]
    if len(matches) != 1:
        found = [
            (session.mouse_id, session.heating_session, session.date)
            for session in sessions
            if str(session.mouse_id) == mouse_id
        ]
        raise ValueError(
            f"Expected exactly one {mouse_id} S{session_number} on {date}; "
            f"found {matches!r}. Same-mouse recordings: {found!r}"
        )
    return matches[0]


def _prepare_hyperthermia_session(session, cache_key: str, threshold: float):
    zdff, dff_raw, temperature, fs = preprocess_and_cache(
        session,
        STRATEGY,
        cache_key,
    )
    session.transients = detect_transients(
        zdff,
        dff_raw,
        fs,
        _make_transient_config(STRATEGY, threshold),
        temperature,
    )
    return session


def _style_axis(ax, *, zero_y: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if zero_y:
        _style.add_zero_y(ax)


def _limits(
    values,
    *,
    padding: float = 0.08,
    symmetric: bool = False,
    include_zero: bool = False,
):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot calculate plot limits from empty data")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if symmetric:
        extent = max(abs(lo), abs(hi), 1.0)
        return (-extent * (1.0 + padding), extent * (1.0 + padding))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = max(hi - lo, abs(hi), abs(lo), 1.0)
    return (lo - padding * span, hi + padding * span)


def _window(session, center_s: float, relative_window=HEATING_WINDOW_S):
    time = np.asarray(session.processed.time, dtype=float)
    lo = center_s + relative_window[0]
    hi = center_s + relative_window[1]
    mask = (time >= lo) & (time <= hi)
    if not np.any(mask):
        raise ValueError(
            f"{session.mouse_id} S{session.heating_session} has no data in "
            f"the requested window {relative_window} around {center_s:.3f}s"
        )
    return time[mask] - center_s, mask


def _plot_decimated(ax, x, y, *, max_points: int = 150_000, **kwargs):
    stride = max(1, int(np.ceil(len(x) / max_points)))
    return ax.plot(x[::stride], y[::stride], **kwargs)


def _add_event_lines(ax, offset_s: float | None, *, seizure: bool):
    ax.axvline(
        0,
        color=_style.EVENT_LINE_COLOR,
        linestyle="--",
        linewidth=_style.EVENT_LINEWIDTH,
        zorder=4,
    )
    if seizure and offset_s is not None:
        ax.axvline(
            offset_s,
            color=_style.EVENT_LINE_COLOR,
            linestyle="--",
            linewidth=_style.EVENT_LINEWIDTH,
            zorder=4,
        )


def _save(fig, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    _style.finalize_publication_figure(fig)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Wrote %s", path)
    return path


def plot_lc_baseline_transients(ds_session, wt_session) -> list[Path]:
    examples = [
        (ds_session, DS_COLOR, "LC_baseline_transient_4390_S1_DS.png"),
        (wt_session, WT_COLOR, "LC_baseline_transient_1889_S1_WT.png"),
    ]
    start_s, end_s = BASELINE_WINDOW_S
    traces = []
    plotted = []
    for session, color, filename in examples:
        time = np.asarray(session.processed.time, dtype=float)
        signal = np.asarray(session.processed.photometry.dff_zscore, dtype=float)
        mask = (time >= start_s) & (time <= end_s)
        events = [
            event
            for event in session.transients
            if start_s <= event.peak_time <= end_s
        ]
        traces.append(signal[mask])
        plotted.append(
            (session, color, filename, time[mask] - start_s, signal[mask], events)
        )

    shared_ylim = _limits(
        np.concatenate(traces),
        padding=0.08,
        include_zero=True,
    )
    outputs = []
    for session, color, filename, rel_time, signal, events in plotted:
        with plt.rc_context(PLOT_RC):
            fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
            ax.plot(
                rel_time,
                signal,
                color=color,
                linewidth=TRANSIENT_LINE_WIDTH,
                zorder=2,
            )
            peak_x = np.asarray([event.peak_time - start_s for event in events])
            peak_y = np.asarray(
                [
                    signal[
                        min(
                            int(round((event.peak_time - start_s) * session.processed.fs)),
                            len(signal) - 1,
                        )
                    ]
                    for event in events
                ]
            )
            ax.scatter(
                peak_x,
                peak_y,
                color="black",
                s=180,
                zorder=5,
            )
            _style_axis(ax)
            ax.set_ylabel(DFF_LABEL)
            ax.set_ylim(shared_ylim)
            ax.set_xlim(0, end_s - start_s)
            ax.set_xlabel("Time in selected baseline window (s)")
            outputs.append(_save(fig, filename))
    return outputs


def plot_heating_examples(
    sessions,
    *,
    filename_prefix: str,
) -> list[Path]:
    ds_session = sessions[0][0]
    ds_event_s = float(ds_session.landmarks.ueo_time)
    _, ds_mask = _window(ds_session, ds_event_s)
    photo_ylim = _limits(
        ds_session.processed.photometry.dff_zscore[ds_mask],
        padding=0.06,
        include_zero=True,
    )

    outputs = []
    for session, color, filename_suffix in sessions:
        with plt.rc_context(PLOT_RC):
            fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
            event_s = float(session.landmarks.ueo_time)
            offset_s = (
                float(session.landmarks.off_time) - event_s
                if session.landmarks.off_time is not None
                else None
            )
            rel_time, mask = _window(session, event_s)
            seizure = session.n_seizures > 0

            _plot_decimated(
                ax,
                rel_time,
                session.processed.photometry.dff_zscore[mask],
                color=color,
                linewidth=LINE_WIDTH,
                zorder=2,
            )
            _style_axis(ax)
            _add_event_lines(ax, offset_s, seizure=seizure)
            ax.set_xlim(HEATING_WINDOW_S)
            ax.set_ylim(photo_ylim)
            ax.set_ylabel(DFF_LABEL)
            anchor = "seizure onset" if seizure else "event onset"
            ax.set_xlabel(f"Time rel. to {anchor} (s)")
            outputs.append(
                _save(fig, f"{filename_prefix}_{filename_suffix}.png")
            )
    return outputs


def plot_raw_photometry(session, *, filename: str) -> Path:
    event_s = float(session.landmarks.ueo_time)
    time = np.asarray(session.raw.time, dtype=float)
    lo = event_s + HEATING_WINDOW_S[0]
    hi = event_s + HEATING_WINDOW_S[1]
    mask = (time >= lo) & (time <= hi)
    rel_time = time[mask] - event_s
    raw_405 = np.asarray(session.raw.signal_405, dtype=float)[mask]
    raw_470 = np.asarray(session.raw.signal_470, dtype=float)[mask]
    ylim = _limits(np.concatenate((raw_405, raw_470)), padding=0.06)
    offset_s = float(session.landmarks.off_time) - event_s

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
        _plot_decimated(
            ax,
            rel_time,
            raw_405,
            color=RAW_405_COLOR,
            linewidth=0.3,
            zorder=2,
        )
        _plot_decimated(
            ax,
            rel_time,
            raw_470,
            color=RAW_470_COLOR,
            linewidth=0.3,
            zorder=3,
        )
        _style_axis(ax, zero_y=False)
        _add_event_lines(ax, offset_s, seizure=True)
        ax.set_xlim(HEATING_WINDOW_S)
        ax.set_ylim(ylim)
        ax.set_xlabel("Time rel. to seizure onset (s)")
        ax.set_ylabel("Raw signal (V)")
        return _save(fig, filename)


def plot_grab_opto_examples() -> list[Path]:
    sessions = load_sessions(GRAB_OPTO_ROOT, test_mode=False, has_temperature=False)
    _validate_sessions(sessions)

    processed = {}
    for session in sessions:
        detected, selected = _detect_and_select_trains(
            session.raw.stim_raw,
            session.raw.fs,
        )
        expected_count = EXPECTED_TRAIN_COUNTS[str(session.mouse_id)]
        if len(detected) != expected_count or len(selected) != TRAINS_TO_USE:
            raise ValueError(
                f"{session.mouse_id}/{session.cohort}: expected "
                f"{expected_count} detected and {TRAINS_TO_USE} selected trains; "
                f"found {len(detected)} and {len(selected)}"
            )
        dff = _preprocess_dff(
            session,
            GRAB_OPTO_CACHE_ROOT,
            rebuild_cache=False,
        )
        first_onset = selected[0].onset_s
        baseline_corrected_dff = _baseline_correct_dff(
            dff,
            session.raw.fs,
            first_onset,
        )
        processed[(str(session.mouse_id), str(session.cohort))] = (
            session,
            selected,
            baseline_corrected_dff,
        )

    outputs = []
    for mouse_id in EXPECTED_MOUSE_IDS:
        pair = [processed[(mouse_id, condition)] for condition in CONDITIONS]
        windows = []
        for session, trains, corrected_dff in pair:
            first_onset = trains[0].onset_s
            lo = first_onset - 5.0
            hi = trains[-1].onset_s + 10.0
            time = np.asarray(session.raw.time, dtype=float)
            mask = (time >= lo) & (time <= hi)
            windows.append(
                (session, trains, time[mask] - first_onset, corrected_dff[mask])
            )
        shared_ylim = _limits(
            np.concatenate([window[3] for window in windows]),
            padding=0.06,
            include_zero=True,
        )

        for condition, window in zip(CONDITIONS, windows):
            with plt.rc_context(PLOT_RC):
                fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
                session, trains, rel_time, corrected_dff = window
                first_onset = trains[0].onset_s
                for train in trains:
                    onset = train.onset_s - first_onset
                    offset = train.offset_s - first_onset
                    ax.axvspan(onset, offset, color="#b8b8b8", alpha=0.25, zorder=0)
                    ax.axvline(
                        onset,
                        color=_style.EVENT_LINE_COLOR,
                        linestyle="--",
                        linewidth=_style.EVENT_LINEWIDTH,
                        alpha=1.0,
                        zorder=2,
                    )
                _plot_decimated(
                    ax,
                    rel_time,
                    corrected_dff,
                    color=CONDITION_COLORS[condition],
                    linewidth=YOHIMBINE_LINE_WIDTH,
                    zorder=3,
                )
                _style_axis(ax)
                ax.set_ylim(shared_ylim)
                ax.set_ylabel(RAW_DFF_LABEL)
                ax.set_xlim(-5.0, 100.0)
                ax.set_xlabel("Time rel. to first stim onset (s)")
                outputs.append(
                    _save(
                        fig,
                        f"GRAB_opto_{mouse_id}_{condition}_first_10_stims.png",
                    )
                )

    del sessions, processed
    gc.collect()
    return outputs


def main() -> None:
    outputs = []

    log.info("Loading LC recordings")
    lc_sessions = load_sessions(LC_ROOT, test_mode=False, has_temperature=True)
    lc_targets = {
        "baseline_ds": _select_session(lc_sessions, "4390", 1, "2025-07-09"),
        "baseline_wt": _select_session(lc_sessions, "1889", 1, "2024-03-22"),
        "heating_ds": _select_session(lc_sessions, "1779", 1, "2024-02-27"),
        "heating_nosz": _select_session(lc_sessions, "1945", 1, "2024-03-26"),
        "heating_wt": _select_session(lc_sessions, "1889", 2, "2024-03-26"),
    }
    for session in lc_targets.values():
        _prepare_hyperthermia_session(session, LC_CACHE_KEY, 0.015)
    outputs.extend(
        plot_lc_baseline_transients(
            lc_targets["baseline_ds"],
            lc_targets["baseline_wt"],
        )
    )
    outputs.extend(
        plot_heating_examples(
            [
                (lc_targets["heating_ds"], DS_COLOR, "1779_S1_DS_seizure"),
                (
                    lc_targets["heating_nosz"],
                    DS_NOSZ_COLOR,
                    "1945_S1_DS_no_seizure",
                ),
                (lc_targets["heating_wt"], WT_COLOR, "1889_S2_WT"),
            ],
            filename_prefix="LC_heating_session",
        )
    )
    del lc_sessions, lc_targets
    gc.collect()

    log.info("Loading DG recordings")
    dg_sessions = load_sessions(DG_ROOT, test_mode=False, has_temperature=True)
    dg_targets = {
        "heating_ds": _select_session(dg_sessions, "1938", 1, "2024-06-05"),
        "heating_nosz": _select_session(dg_sessions, "2575", 1, "2024-05-31"),
        "heating_wt": _select_session(dg_sessions, "4313", 1, "2025-09-23"),
    }
    for session in dg_targets.values():
        _prepare_hyperthermia_session(session, DG_CACHE_KEY, 0.030)
    outputs.extend(
        plot_heating_examples(
            [
                (dg_targets["heating_ds"], DS_COLOR, "1938_S1_DS_seizure"),
                (
                    dg_targets["heating_nosz"],
                    DS_NOSZ_COLOR,
                    "2575_S1_DS_no_seizure",
                ),
                (dg_targets["heating_wt"], WT_COLOR, "4313_S1_WT"),
            ],
            filename_prefix="DG_heating_session",
        )
    )
    outputs.append(
        plot_raw_photometry(
            dg_targets["heating_ds"],
            filename="DG_raw_405_470_1938_S1.png",
        )
    )
    del dg_sessions, dg_targets
    gc.collect()

    log.info("Loading DG GRAB-NE mutant recordings")
    mutant_sessions = load_sessions(MUTANT_ROOT, test_mode=False, has_temperature=True)
    mutant = _select_session(mutant_sessions, "3681", 1, "2025-04-01")
    outputs.append(
        plot_raw_photometry(
            mutant,
            filename="DG_mutant_raw_405_470_3681_S1.png",
        )
    )
    del mutant_sessions, mutant
    gc.collect()

    log.info("Loading GRAB-opto recordings")
    outputs.extend(plot_grab_opto_examples())

    expected = 20
    if len(outputs) != expected or not all(path.exists() for path in outputs):
        raise RuntimeError(
            f"Expected {expected} example PNGs, created {len(outputs)}"
        )
    log.info("Created %d example PNGs in %s", len(outputs), OUTPUT_DIR)


if __name__ == "__main__":
    main()
