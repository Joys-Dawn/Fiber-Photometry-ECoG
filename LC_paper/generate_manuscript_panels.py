"""Standalone manuscript-panel exports; not part of the app pipeline.

Run with ``python -m LC_paper.generate_manuscript_panels --section dg cpp lc``.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook

from fiber_photometry_ecog.pairing.engine import assign_all_controls
from fiber_photometry_ecog.visualization import _style
from fiber_photometry_ecog.visualization.colors import DFF_LABEL
from LC_paper.generate_example_plots import (
    DG_CACHE_KEY,
    DG_ROOT,
    LC_CACHE_KEY,
    LC_ROOT,
    _limits,
    _plot_decimated,
    _prepare_hyperthermia_session,
)
from LC_paper.run_opto import OPTO_BASE, _preprocess_and_cache
from LC_paper.run_grab_opto import (
    BEFORE,
    CONDITIONS,
    CACHE_ROOT as GRAB_OPTO_CACHE_ROOT,
    DATA_ROOT as GRAB_OPTO_ROOT,
    RAW_DFF_LABEL,
    _baseline_correct_dff,
    _detect_and_select_trains,
    _preprocess_dff,
)
from LC_paper.run_threshold_sweep import load_sessions


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "manuscript_panels"
BRIGHT_GREEN = "green"
RAW_470 = "blue"
RAW_405 = "#d55ad5"
COHORT_COLORS = {"DS_sz": "red", "DS_no_sz": "purple", "WT_no_sz": "blue"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _style.finalize_publication_figure(fig)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def _only(sessions, mouse: str, number: int, cohort: str | None = None,
          date: str | None = None):
    found = [
        s for s in sessions
        if str(s.mouse_id) == mouse and s.heating_session == number
        and (cohort is None or s.cohort == cohort)
        and (date is None or s.date == date)
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one {mouse} S{number} {cohort}; found {len(found)}")
    return found[0]


def _prepare_pairing(sessions, cache_key: str, threshold: float, controls) -> None:
    seized = [s for s in sessions if s.n_seizures > 0]
    for session in seized + list(controls):
        _prepare_hyperthermia_session(session, cache_key, threshold)
    assign_all_controls(seized, list(controls), mode="temperature")


def _event(session) -> tuple[float, float]:
    lm = session.landmarks
    onset = lm.ueo_time if session.n_seizures else lm.equiv_ueo_time
    offset = lm.off_time if session.n_seizures else lm.equiv_off_time
    if onset is None or offset is None:
        raise ValueError(f"No event equivalents for {session.mouse_id} S{session.heating_session}")
    return float(onset), float(offset)


def _axis(ax, ylabel: str, xlabel: str, xlim, ylim=None) -> None:
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)


def dg_options() -> None:
    sessions = load_sessions(DG_ROOT, test_mode=False, has_temperature=True)
    controls = [
        _only(sessions, "2575", 1, date="2024-05-31"),
        _only(sessions, "4313", 1, date="2025-09-23"),
    ]
    _prepare_pairing(sessions, DG_CACHE_KEY, 0.030, controls)
    candidates = [("2638", 2), ("4359", 1), ("2638", 3)]
    for mouse, number in candidates:
        seizure = _only(sessions, mouse, number)
        if seizure.n_seizures <= 0:
            raise ValueError(f"Candidate {mouse} S{number} is not a seizure session")
        onset, _ = _event(seizure)
        time = np.asarray(seizure.processed.time)
        signal = np.asarray(seizure.processed.photometry.dff_zscore)
        mask = (time >= onset - 200) & (time <= onset + 200)
        ylim = _limits(signal[mask], padding=0.06, include_zero=True)
        for session, color, label in (
            (seizure, "red", f"{mouse}_S{number}_DS_seizure"),
            (controls[0], "purple", "2575_S1_DS_no_seizure"),
            (controls[1], "blue", "4313_S1_WT"),
        ):
            event, off = _event(session)
            time = np.asarray(session.processed.time)
            mask = (time >= event - 200) & (time <= event + 200)
            rel = time[mask] - event
            with plt.rc_context(_style._PLOT_RC):
                fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
                _plot_decimated(
                    ax, rel, session.processed.photometry.dff_zscore[mask],
                    color=color, linewidth=1.0, zorder=3,
                )
                _style.add_zero_y(ax)
                ax.axvline(0, color="black", linestyle="--", linewidth=2.5)
                if session.n_seizures:
                    ax.axvline(off - event, color="black", linestyle="--", linewidth=2.5)
                anchor = "seizure" if session.n_seizures else "event"
                _axis(ax, DFF_LABEL, f"Time rel. to {anchor} onset (s)", (-200, 200), ylim)
                _save(fig, OUTPUT / "figure2_options" / f"{mouse}_S{number}" / f"{label}.png")
    del sessions
    gc.collect()


def yohimbine_example() -> None:
    sessions = load_sessions(GRAB_OPTO_ROOT, test_mode=False, has_temperature=False)
    windows = []
    for condition in CONDITIONS:
        matches = [s for s in sessions if str(s.mouse_id) == "1938" and s.cohort == condition]
        if len(matches) != 1:
            raise ValueError(f"Expected one 1938 {condition} example, found {len(matches)}")
        session = matches[0]
        _, trains = _detect_and_select_trains(session.raw.stim_raw, session.raw.fs)
        dff = _preprocess_dff(session, GRAB_OPTO_CACHE_ROOT, rebuild_cache=False)
        first_onset = trains[0].onset_s
        corrected = _baseline_correct_dff(dff, session.raw.fs, first_onset)
        time = np.asarray(session.raw.time)
        mask = (time >= first_onset - 5.0) & (time <= trains[-1].onset_s + 10.0)
        windows.append((condition, trains, time[mask] - first_onset, corrected[mask]))

    with plt.rc_context(_style._PLOT_RC):
        fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
        for condition, trains, rel_time, corrected in windows:
            _plot_decimated(
                ax, rel_time, corrected, color=BRIGHT_GREEN, linewidth=1.5,
                linestyle="-" if condition == BEFORE else "--", zorder=3,
            )
        _style.add_zero_y(ax)
        _axis(ax, RAW_DFF_LABEL, "Time rel. to first stim onset (s)", (-5.0, 100.0),
              _limits(np.concatenate([window[3] for window in windows]),
                      padding=0.06, include_zero=True))
        _save(fig, OUTPUT / "figure3_sensor_traces" /
              "C_D_1938_pre_post_Yohimbine_first_10_stims.png")
    del sessions, windows
    gc.collect()


def cpp_individuals() -> None:
    sessions = load_sessions(
        OPTO_BASE / "Hotspot_GANE" / "CPP_treatment",
        test_mode=False,
        has_temperature=False,
    )
    selected = [s for s in sessions if s.cohort in {"10HZ_DG", "10HZ_DG_CPP"}]
    for session in selected:
        _preprocess_and_cache(session, "C", "CPP_treatment", zscore=False)

    pairs: dict[tuple[str, int], dict[str, object]] = {}
    for session in selected:
        key = (str(session.mouse_id), int(session.heating_session))
        pairs.setdefault(key, {})[session.cohort] = session
    if len(pairs) != 8 or any(set(pair) != {"10HZ_DG", "10HZ_DG_CPP"} for pair in pairs.values()):
        raise ValueError(f"Expected 8 complete CPP pairs, found {len(pairs)}")

    traces = []
    for (mouse, number), pair in sorted(pairs.items()):
        for condition in ("10HZ_DG", "10HZ_DG_CPP"):
            session = pair[condition]
            onset = float(session.landmarks.ueo_time)
            time = np.asarray(session.processed.time)
            mask = (time >= onset - 40) & (time <= onset + 40)
            if not np.any(mask):
                raise ValueError(f"Empty CPP window for {mouse} S{number} {condition}")
            traces.append((mouse, number, condition, time[mask] - onset,
                           np.asarray(session.processed.photometry.dff_zscore)[mask]))
    pair_limits = {}
    for mouse, number in pairs:
        pair_limits[(mouse, number)] = _limits(
            np.concatenate([item[4] for item in traces
                            if (item[0], item[1]) == (mouse, number)]),
            padding=0.06, include_zero=True,
        )
    for mouse, number, condition, rel, signal in traces:
        with plt.rc_context(_style._PLOT_RC):
            fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
            _plot_decimated(ax, rel, signal, color=BRIGHT_GREEN, linewidth=2.5, zorder=3)
            _style.add_zero_y(ax)
            _style.add_zero_x(ax)
            _axis(ax, "ΔF/F", "Time rel. to stim onset (s)", (-40, 40),
                  pair_limits[(mouse, number)])
            suffix = "pre" if condition == "10HZ_DG" else "post_CPP"
            _save(fig, OUTPUT / "figure5_CPP_individual" / f"{mouse}_S{number}_{suffix}.png")
    del sessions
    gc.collect()


def cpp_control_values() -> None:
    sheet = load_workbook(ROOT / "Live imaging data for new plots.xlsx", read_only=True, data_only=True).active
    groups = []
    for column, label in ((11, "DMSO + NE"), (12, "CPP + NE")):
        values = np.asarray(
            [sheet.cell(row, column).value for row in range(2, sheet.max_row + 1)
             if isinstance(sheet.cell(row, column).value, (float, int))], dtype=float,
        )
        if values.size == 0:
            raise ValueError(f"No spreadsheet values for {label}")
        groups.append((label, values))
    with plt.rc_context(_style._PLOT_RC):
        fig, ax = plt.subplots(figsize=(12.0, _style.FIG_SCATTER[1]))
        for position, (label, values) in enumerate(groups):
            spread = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(position + spread, values, color="black", s=140, zorder=3)
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            ax.errorbar(position, mean, yerr=sem, color="black", capsize=10, linewidth=3, zorder=4)
            ax.hlines(mean, position - 0.15, position + 0.15,
                      color="black", linewidth=3, zorder=4)
        ax.set_xticks((0, 1), ("DMSO\n+ NE", "CPP\n+ NE"))
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("Mean intensity\n(fold change)")
        _style.add_zero_y(ax, ensure_visible=True)
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, OUTPUT / "figure5_CPP_control" / "DMSO_NE_vs_CPP_NE.png")


def drug_ne_values() -> None:
    sheet = load_workbook(ROOT / "Live imaging data for new plots.xlsx", read_only=True, data_only=True).active
    comparisons = (
        ("DMSO", 2, 3, (0.0, 0.8)),
        ("Yoh", 6, 7, (1.6, 2.4)),
    )
    with plt.rc_context(_style._PLOT_RC):
        fig, ax = plt.subplots(figsize=(16.0, 10.0))
        for drug, column_a, column_b, (x_a, x_b) in comparisons:
            a = np.asarray([sheet.cell(row, column_a).value for row in range(2, 8)], dtype=float)
            b = np.asarray([sheet.cell(row, column_b).value for row in range(2, 8)], dtype=float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError(f"{drug}: expected six complete drug/NE pairs")
            offsets = np.linspace(-0.06, 0.06, len(a))
            for offset, value_a, value_b in zip(offsets, a, b):
                ax.plot((x_a + offset, x_b + offset), (value_a, value_b),
                        color="#555555", linewidth=2.5, zorder=2)
            ax.scatter(x_a + offsets, a, marker="o", color="black", s=150, zorder=3)
            ax.scatter(x_b + offsets, b, marker="o", color="black", s=150, zorder=3)
        ax.set_xticks((0.0, 0.8, 1.6, 2.4),
                      ("DMSO", "DMSO\n+ NE", "Yoh", "Yoh\n+ NE"))
        ax.set_xlim(-0.4, 2.8)
        ax.set_ylabel("Mean intensity\n(a.u.)")
        _style.add_zero_y(ax, ensure_visible=True)
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, OUTPUT / "figure3_sensor_traces" / "D_DMSO_Yoh_drug_vs_drug_NE.png")


def lc_supplement() -> None:
    sessions = load_sessions(LC_ROOT, test_mode=False, has_temperature=True)
    examples = [
        (_only(sessions, "1779", 1, date="2024-02-27"), "DS_sz"),
        (_only(sessions, "1945", 1, date="2024-03-26"), "DS_no_sz"),
        (_only(sessions, "1889", 2, date="2024-03-26"), "WT_no_sz"),
    ]
    _prepare_pairing(sessions, LC_CACHE_KEY, 0.015, [s for s, _ in examples[1:]])
    for session, cohort in examples:
        event, off = _event(session)
        heating = float(session.landmarks.heating_start_time)
        name = f"{session.mouse_id}_S{session.heating_session}_{cohort}"
        proc = session.processed
        time = np.asarray(proc.time)
        raw_time = np.asarray(session.raw.time)
        for zoom, left, right in (
            ("zoomed_out", heating - 200, off + 200),
            ("zoomed_in", event - 200, event + 200),
        ):
            right = min(right, float(time[-1]), float(raw_time[-1]))
            mask = (time >= left) & (time <= right)
            raw_mask = (raw_time >= left) & (raw_time <= right)
            if not np.any(mask) or not np.any(raw_mask):
                raise ValueError(f"Empty LC {zoom} {name}")
            zoomed_out = zoom == "zoomed_out"
            nrows = 4 if zoomed_out else 3
            with plt.rc_context({
                **_style._PLOT_RC,
                "figure.constrained_layout.h_pad": 0.20,
                "figure.constrained_layout.hspace": 0.08,
            }):
                fig, axes = plt.subplots(
                    nrows, 1, sharex=True, figsize=(18.0, 4.6 * nrows + 1.4),
                )
                raw_ax, dff_ax, ecog_ax = axes[:3]
                if zoomed_out:
                    _plot_decimated(
                        axes[3], time[mask] - event,
                        np.asarray(proc.temperature_smooth)[mask],
                        color="black", linewidth=1.0, zorder=3,
                    )
                    axes[3].set_ylabel("Temp (°C)")
                _plot_decimated(
                    ecog_ax, time[mask] - event, np.asarray(proc.ecog_filtered)[mask],
                    color="black", linewidth=1.0, zorder=3,
                )
                ecog_ax.set_ylabel("Filt ECoG\n(µV)")
                _style.add_zero_y(ecog_ax)
                for signal, color in ((session.raw.signal_470, RAW_470),
                                      (session.raw.signal_405, RAW_405)):
                    _plot_decimated(raw_ax, raw_time[raw_mask] - event,
                                    np.asarray(signal)[raw_mask],
                                    color=color, linewidth=0.8, zorder=3)
                raw_ax.set_ylabel("Raw signal\n(V)")
                _plot_decimated(
                    dff_ax, time[mask] - event,
                    np.asarray(proc.photometry.dff_zscore)[mask],
                    color=COHORT_COLORS[cohort], linewidth=1.0, zorder=3,
                )
                dff_ax.set_ylabel(DFF_LABEL)
                _style.add_zero_y(dff_ax)
                for ax in axes:
                    ax.set_xlim(left - event, right - event)
                    ax.spines[["top", "right"]].set_visible(False)
                    for marker in (heating, event, off):
                        if left <= marker <= right:
                            ax.axvline(marker - event, color="black", linestyle="--",
                                       linewidth=2.5, zorder=4)
                anchor = "seizure" if session.n_seizures else "event"
                axes[-1].set_xlabel(f"Time rel. to {anchor} onset (s)")
                fig.align_ylabels(axes)
                _save(fig, OUTPUT / "LC_supplement" / name / f"{zoom}.png")
    del sessions
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", nargs="+", choices=("dg", "yoh", "cpp", "drug", "control", "lc"),
                        default=("dg", "yoh", "cpp", "drug", "control", "lc"))
    args = parser.parse_args()
    functions = {"dg": dg_options, "yoh": yohimbine_example, "cpp": cpp_individuals,
                 "drug": drug_ne_values, "control": cpp_control_values,
                 "lc": lc_supplement}
    for section in args.section:
        print(f"Generating {section} panels", flush=True)
        functions[section]()


if __name__ == "__main__":
    main()
