"""Standalone repeated-train GRAB-NE optogenetics analysis.

This runner is intentionally dataset-specific. It analyzes
``D:/Ephys_Data/OPTO_stim/GRAB_opto`` without adding repeated-train behavior
to the GUI pipeline.

Protocol encoded here:
* Channel 23 contains 1 s trains of 20 pulses at 20 Hz.
* Use the first 10 trains from every recording (two mice have 30).
* Subtract the full pre-first-train mean from Strategy-C/IRLS dF/F.
* Measure peak and signed AUC over [0, 10 s) after each train onset.
* Treat trains as technical replicates and mice as the paired statistical unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fiber_photometry_ecog.core.config import PhotometryConfig
from fiber_photometry_ecog.preprocessing.photometry import IRLSStrategy
from fiber_photometry_ecog.preprocessing.stim_detection import (
    detect_stim_bursts,
)
from fiber_photometry_ecog.statistics.paired import compare_paired
from fiber_photometry_ecog.visualization import _style
from fiber_photometry_ecog.visualization.group_plots import _legend_above
from run_threshold_sweep import load_sessions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


DATA_ROOT = Path("D:/Ephys_Data/OPTO_stim/GRAB_opto")
OUTPUT_ROOT = (
    Path("results")
    / "results_GRAB_opto_repeated"
    / "preprocessing_IRLS"
)
CACHE_ROOT = Path("opto_cache") / "GRAB_opto_repeated" / "C"

BEFORE = "before_Yohimbine"
AFTER = "after_Yohimbine"
CONDITIONS = (BEFORE, AFTER)
CONDITION_LABELS = {
    BEFORE: "Before Yohimbine",
    AFTER: "After Yohimbine",
}
CONDITION_COLORS = {
    BEFORE: "#0072B2",
    AFTER: "#D55E00",
}

EXPECTED_MOUSE_IDS = ("1936", "1938", "4900", "4937", "4996")
EXPECTED_TRAIN_COUNTS = {
    "1936": 30,
    "1938": 30,
    "4900": 10,
    "4937": 10,
    "4996": 10,
}
TRAINS_TO_USE = 10
EXPECTED_PULSES_PER_TRAIN = 20
EXPECTED_TRAIN_INTERVAL_S = 10.0
TRAIN_INTERVAL_TOLERANCE_S = 0.02
TRAIN_DURATION_RANGE_S = (0.90, 1.05)

RESPONSE_WINDOW_S = 10.0
TRACE_PRE_S = 5.0
CACHE_VERSION = 1
RAW_DFF_LABEL = "ΔF/F"


@dataclass(frozen=True)
class StimTrain:
    """One detected channel-23 pulse train."""

    index: int
    onset_s: float
    offset_s: float
    supra_threshold_samples: int
    pulse_count: int

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s


@dataclass
class SessionAnalysis:
    """Repeated-train measurements for one mouse and condition."""

    mouse_id: str
    date: str | None
    condition: str
    fs: float
    baseline_end_s: float
    baseline_mean_raw_dff: float
    baseline_sd_raw_dff: float
    detected_trains: tuple[StimTrain, ...]
    selected_trains: tuple[StimTrain, ...]
    trace_time_s: np.ndarray
    event_traces: np.ndarray
    evoked_peaks: np.ndarray
    evoked_aucs: np.ndarray

    @property
    def mouse_mean_trace(self) -> np.ndarray:
        return np.mean(self.event_traces, axis=0)


def _condition_order(value: str) -> int:
    try:
        return CONDITIONS.index(value)
    except ValueError:
        return len(CONDITIONS)


def _safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _validate_sessions(sessions: list) -> None:
    """Require the fixed five-mouse paired design and no silent exclusions."""
    if len(sessions) != 10:
        raise ValueError(
            f"GRAB_opto expected 10 sessions (5 paired mice), found {len(sessions)}"
        )

    seen: dict[tuple[str, str], int] = {}
    for session in sessions:
        key = (str(session.mouse_id), str(session.cohort))
        seen[key] = seen.get(key, 0) + 1

    actual_mice = tuple(sorted({str(s.mouse_id) for s in sessions}))
    if actual_mice != tuple(sorted(EXPECTED_MOUSE_IDS)):
        raise ValueError(
            f"GRAB_opto mouse IDs changed: expected {EXPECTED_MOUSE_IDS}, "
            f"found {actual_mice}"
        )

    problems = []
    for mouse_id in EXPECTED_MOUSE_IDS:
        for condition in CONDITIONS:
            count = seen.get((mouse_id, condition), 0)
            if count != 1:
                problems.append(f"{mouse_id}/{condition}: {count} sessions")
    unexpected = [
        f"{mouse}/{condition}"
        for mouse, condition in seen
        if mouse not in EXPECTED_MOUSE_IDS or condition not in CONDITIONS
    ]
    if problems or unexpected:
        detail = problems + [f"unexpected {item}" for item in unexpected]
        raise ValueError("GRAB_opto pairing validation failed: " + "; ".join(detail))


def _detect_and_select_trains(
    stim_signal: np.ndarray,
    fs: float,
) -> tuple[tuple[StimTrain, ...], tuple[StimTrain, ...]]:
    """Detect, protocol-validate, and select the first 10 pulse trains."""
    stim = np.asarray(stim_signal, dtype=float)
    bursts = detect_stim_bursts(stim, fs)

    abs_stim = np.abs(stim)
    peak = float(np.nanmax(abs_stim))
    threshold = 0.5 * peak
    above = np.isfinite(abs_stim) & (abs_stim > threshold)
    pulse_starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])

    trains = []
    for index, (onset_s, offset_s, sample_count) in enumerate(bursts, start=1):
        lo = max(0, int(math.floor(onset_s * fs)) - 1)
        hi = min(len(stim), int(math.ceil(offset_s * fs)) + 2)
        pulse_count = int(
            np.count_nonzero((pulse_starts >= lo) & (pulse_starts < hi))
        )
        trains.append(
            StimTrain(
                index=index,
                onset_s=float(onset_s),
                offset_s=float(offset_s),
                supra_threshold_samples=int(sample_count),
                pulse_count=pulse_count,
            )
        )

    if len(trains) not in set(EXPECTED_TRAIN_COUNTS.values()):
        raise ValueError(
            f"Expected 10 or 30 channel-23 trains, detected {len(trains)}"
        )

    malformed = [
        train
        for train in trains
        if train.pulse_count != EXPECTED_PULSES_PER_TRAIN
        or not (
            TRAIN_DURATION_RANGE_S[0]
            <= train.duration_s
            <= TRAIN_DURATION_RANGE_S[1]
        )
    ]
    if malformed:
        summary = [
            {
                "index": train.index,
                "duration_s": round(train.duration_s, 4),
                "pulse_count": train.pulse_count,
            }
            for train in malformed
        ]
        raise ValueError(
            "Malformed channel-23 train(s); expected 20 pulses and "
            f"duration in {TRAIN_DURATION_RANGE_S}: {summary}"
        )

    onset_gaps = np.diff([train.onset_s for train in trains])
    if onset_gaps.size and not np.allclose(
        onset_gaps,
        EXPECTED_TRAIN_INTERVAL_S,
        atol=TRAIN_INTERVAL_TOLERANCE_S,
        rtol=0.0,
    ):
        raise ValueError(
            "Channel-23 train spacing changed; expected 10 s, found "
            f"range {float(np.min(onset_gaps)):.4f}-"
            f"{float(np.max(onset_gaps)):.4f} s"
        )

    detected = tuple(trains)
    selected = detected[:TRAINS_TO_USE]
    if len(selected) != TRAINS_TO_USE:
        raise ValueError(
            f"Expected to select {TRAINS_TO_USE} trains, found {len(selected)}"
        )
    return detected, selected


def _signal_fingerprint(session) -> str:
    """Hash synchronized inputs so replaced data invalidate the IRLS cache."""
    raw = session.raw
    digest = hashlib.sha256()
    digest.update(f"cache-v{CACHE_VERSION}|fs={raw.fs:.12g}".encode("ascii"))
    for signal in (raw.signal_470, raw.signal_405):
        array = np.ascontiguousarray(signal, dtype="<f8")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _cache_path(session, cache_root: Path) -> Path:
    parts = (
        _safe_component(session.mouse_id),
        _safe_component(session.date or "undated"),
        _safe_component(session.cohort),
    )
    return cache_root / ("_".join(parts) + ".npz")


def _preprocess_dff(
    session,
    cache_root: Path,
    *,
    rebuild_cache: bool,
) -> np.ndarray:
    """Run Strategy C or load a cache verified against synchronized inputs."""
    raw = session.raw
    fingerprint = _signal_fingerprint(session)
    cache_path = _cache_path(session, cache_root)

    if cache_path.exists() and not rebuild_cache:
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                cache_ok = (
                    int(cached["cache_version"].item()) == CACHE_VERSION
                    and str(cached["fingerprint"].item()) == fingerprint
                    and float(cached["fs"].item()) == float(raw.fs)
                    and int(cached["n_samples"].item()) == len(raw.signal_470)
                )
                if cache_ok:
                    log.info("    IRLS cache hit: %s", cache_path.name)
                    return np.asarray(cached["dff"], dtype=float)
        except (KeyError, OSError, ValueError) as exc:
            log.warning("    Ignoring unreadable IRLS cache %s: %s", cache_path, exc)

    config = PhotometryConfig(strategy="C")
    result = IRLSStrategy().preprocess(
        raw.signal_470,
        raw.signal_405,
        raw.fs,
        config,
    )
    dff = np.asarray(result.dff, dtype=float)
    if dff.shape != np.asarray(raw.signal_470).shape or not np.all(np.isfinite(dff)):
        raise ValueError(
            f"IRLS produced invalid dF/F for {session.mouse_id}/{session.cohort}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_version=np.array(CACHE_VERSION, dtype=np.int64),
        fingerprint=np.array(fingerprint),
        fs=np.array(raw.fs, dtype=np.float64),
        n_samples=np.array(len(raw.signal_470), dtype=np.int64),
        dff=dff,
    )
    log.info("    IRLS cache wrote: %s", cache_path.name)
    return dff


def _baseline_correct_dff(
    dff: np.ndarray,
    fs: float,
    baseline_end_s: float,
) -> np.ndarray:
    """Subtract the full pre-first-train mean without z-scoring."""
    baseline_end_i = min(int(baseline_end_s * fs), len(dff))
    return np.asarray(dff, dtype=float) - float(np.mean(dff[:baseline_end_i]))


def _extract_event_metrics(
    dff: np.ndarray,
    fs: float,
    trains: tuple[StimTrain, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract [-5, 10 s) traces and [0, 10 s) peak/signed AUC values."""
    pre_n = int(round(TRACE_PRE_S * fs))
    response_n = int(round(RESPONSE_WINDOW_S * fs))
    offsets = np.arange(-pre_n, response_n, dtype=np.int64)
    trace_time_s = offsets.astype(float) / fs

    traces = []
    peaks = []
    aucs = []
    for train in trains:
        onset_i = int(round(train.onset_s * fs))
        lo = onset_i - pre_n
        hi = onset_i + response_n
        if lo < 0 or hi > len(dff):
            raise ValueError(
                f"Train {train.index} at {train.onset_s:.3f}s lacks the full "
                f"[-{TRACE_PRE_S:g}, {RESPONSE_WINDOW_S:g}) s analysis window"
            )
        trace = np.asarray(dff[lo:hi], dtype=float)
        if trace.size != offsets.size or not np.all(np.isfinite(trace)):
            raise ValueError(
                f"Train {train.index} contains missing/non-finite photometry values"
            )
        response = trace[pre_n:]
        traces.append(trace)
        peaks.append(float(np.max(response)))
        aucs.append(float(np.trapezoid(response, dx=1.0 / fs)))

    return (
        trace_time_s,
        np.vstack(traces),
        np.asarray(peaks, dtype=float),
        np.asarray(aucs, dtype=float),
    )


def _analyze_session(
    session,
    cache_root: Path,
    *,
    rebuild_cache: bool,
) -> SessionAnalysis:
    raw = session.raw
    if raw is None or raw.stim_raw is None:
        raise ValueError(
            f"Session {session.mouse_id}/{session.cohort} has no channel-23 signal"
        )
    if raw.fs is None or raw.fs <= 0:
        raise ValueError(f"Session {session.mouse_id}/{session.cohort} has invalid fs")

    detected, selected = _detect_and_select_trains(raw.stim_raw, raw.fs)
    expected_count = EXPECTED_TRAIN_COUNTS[str(session.mouse_id)]
    if len(detected) != expected_count:
        raise ValueError(
            f"{session.mouse_id}/{session.cohort}: expected {expected_count} "
            f"trains, detected {len(detected)}"
        )

    dff = _preprocess_dff(
        session,
        cache_root,
        rebuild_cache=rebuild_cache,
    )
    baseline_end_s = selected[0].onset_s
    baseline_end_i = int(baseline_end_s * raw.fs)
    baseline = dff[:baseline_end_i]
    if baseline.size < int(30 * raw.fs):
        raise ValueError(
            f"{session.mouse_id}/{session.cohort}: only "
            f"{baseline.size / raw.fs:.1f}s of pre-stim baseline"
        )
    baseline_mean = float(np.mean(baseline))
    baseline_sd = float(np.std(baseline))
    if not np.isfinite(baseline_sd) or baseline_sd <= 0:
        raise ValueError(
            f"{session.mouse_id}/{session.cohort}: invalid pre-stim baseline SD"
        )

    baseline_corrected_dff = _baseline_correct_dff(dff, raw.fs, baseline_end_s)
    trace_time, event_traces, peaks, aucs = _extract_event_metrics(
        baseline_corrected_dff,
        raw.fs,
        selected,
    )
    return SessionAnalysis(
        mouse_id=str(session.mouse_id),
        date=session.date,
        condition=str(session.cohort),
        fs=float(raw.fs),
        baseline_end_s=baseline_end_s,
        baseline_mean_raw_dff=baseline_mean,
        baseline_sd_raw_dff=baseline_sd,
        detected_trains=detected,
        selected_trains=selected,
        trace_time_s=trace_time,
        event_traces=event_traces,
        evoked_peaks=peaks,
        evoked_aucs=aucs,
    )


def _train_level_frame(analyses: list[SessionAnalysis]) -> pd.DataFrame:
    rows = []
    for analysis in analyses:
        for train, peak, auc in zip(
            analysis.selected_trains,
            analysis.evoked_peaks,
            analysis.evoked_aucs,
        ):
            rows.append(
                {
                    "mouse_id": analysis.mouse_id,
                    "date": analysis.date,
                    "condition": analysis.condition,
                    "condition_label": CONDITION_LABELS[analysis.condition],
                    "technical_replicate": train.index,
                    "train_onset_s": train.onset_s,
                    "train_offset_s": train.offset_s,
                    "train_duration_s": train.duration_s,
                    "pulse_count": train.pulse_count,
                    "baseline_end_s": analysis.baseline_end_s,
                    "baseline_mean_raw_dff": analysis.baseline_mean_raw_dff,
                    "baseline_sd_raw_dff": analysis.baseline_sd_raw_dff,
                    "evoked_peak_dff": peak,
                    "evoked_auc_dff_s": auc,
                }
            )
    frame = pd.DataFrame(rows)
    frame["_condition_order"] = frame["condition"].map(_condition_order)
    frame = frame.sort_values(
        ["mouse_id", "_condition_order", "technical_replicate"]
    ).drop(columns="_condition_order")
    return frame.reset_index(drop=True)


def _mouse_level_frame(train_values: pd.DataFrame) -> pd.DataFrame:
    """Average technical replicates so one row is one mouse/condition."""
    grouped = train_values.groupby(
        ["mouse_id", "date", "condition", "condition_label"],
        sort=False,
        dropna=False,
    )
    rows = []
    for keys, group in grouped:
        mouse_id, date, condition, condition_label = keys
        if len(group) != TRAINS_TO_USE:
            raise ValueError(
                f"{mouse_id}/{condition}: expected {TRAINS_TO_USE} technical "
                f"replicates, found {len(group)}"
            )
        peak = group["evoked_peak_dff"].to_numpy(dtype=float)
        auc = group["evoked_auc_dff_s"].to_numpy(dtype=float)
        rows.append(
            {
                "mouse_id": str(mouse_id),
                "date": date,
                "condition": condition,
                "condition_label": condition_label,
                "n_technical_replicates": len(group),
                "evoked_peak_mean_dff": float(np.mean(peak)),
                "evoked_peak_technical_sem": float(
                    np.std(peak, ddof=1) / np.sqrt(len(peak))
                ),
                "evoked_auc_mean_dff_s": float(np.mean(auc)),
                "evoked_auc_technical_sem": float(
                    np.std(auc, ddof=1) / np.sqrt(len(auc))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["_condition_order"] = frame["condition"].map(_condition_order)
    return frame.sort_values(
        ["mouse_id", "_condition_order"]
    ).drop(columns="_condition_order").reset_index(drop=True)


def _qc_frame(analyses: list[SessionAnalysis]) -> pd.DataFrame:
    rows = []
    for analysis in analyses:
        selected_indices = {train.index for train in analysis.selected_trains}
        for train in analysis.detected_trains:
            rows.append(
                {
                    "mouse_id": analysis.mouse_id,
                    "date": analysis.date,
                    "condition": analysis.condition,
                    "condition_label": CONDITION_LABELS[analysis.condition],
                    "detected_train_index": train.index,
                    "train_onset_s": train.onset_s,
                    "train_offset_s": train.offset_s,
                    "train_duration_s": train.duration_s,
                    "pulse_count": train.pulse_count,
                    "supra_threshold_samples": train.supra_threshold_samples,
                    "selected_for_analysis": train.index in selected_indices,
                }
            )
    frame = pd.DataFrame(rows)
    frame["_condition_order"] = frame["condition"].map(_condition_order)
    return frame.sort_values(
        ["mouse_id", "_condition_order", "detected_train_index"]
    ).drop(columns="_condition_order").reset_index(drop=True)


def _paired_statistics(mouse_values: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("Evoked peak (0-10 s)", "evoked_peak_mean_dff"),
        ("Evoked AUC (0-10 s)", "evoked_auc_mean_dff_s"),
    )
    rows = []
    for measurement, column in specs:
        before = mouse_values.loc[
            mouse_values["condition"] == BEFORE, ["mouse_id", column]
        ].set_index("mouse_id")[column].to_dict()
        after = mouse_values.loc[
            mouse_values["condition"] == AFTER, ["mouse_id", column]
        ].set_index("mouse_id")[column].to_dict()
        result = compare_paired(
            measurement,
            CONDITION_LABELS[BEFORE],
            before,
            CONDITION_LABELS[AFTER],
            after,
        )
        common = sorted(set(before) & set(after))
        differences = np.asarray([after[m] - before[m] for m in common], dtype=float)
        descriptives = {item.name: item for item in result.descriptives}
        before_desc = descriptives[CONDITION_LABELS[BEFORE]]
        after_desc = descriptives[CONDITION_LABELS[AFTER]]
        statistic_direction = (
            "before - after"
            if result.test_name == "Paired t-test"
            else "unsigned rank sum"
            if result.test_name == "Wilcoxon signed-rank"
            else "not applicable"
        )
        rows.append(
            {
                "measurement": measurement,
                "test": result.test_name,
                "test_statistic": result.statistic,
                "test_statistic_direction": statistic_direction,
                "p_value_raw": result.p_value,
                "n_paired_mice": result.n_pairs,
                "before_mean": before_desc.mean,
                "before_sem_between_mice": before_desc.sem,
                "after_mean": after_desc.mean,
                "after_sem_between_mice": after_desc.sem,
                "paired_mean_difference_after_minus_before": float(
                    np.mean(differences)
                ),
                "experimental_unit": "mouse",
                "technical_replicates_per_mouse_condition": TRAINS_TO_USE,
                "multiple_comparison_adjustment": "none; two prespecified outcomes",
            }
        )
    return pd.DataFrame(rows)


def _stable_jitter(mouse_id: str, width: float = 0.055) -> float:
    checksum = zlib.crc32(str(mouse_id).encode("utf-8"))
    return ((checksum / 0xFFFFFFFF) * 2.0 - 1.0) * width


@_style.styled
def _plot_stim_aligned(
    analyses: list[SessionAnalysis],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
    common_time = analyses[0].trace_time_s
    for condition in CONDITIONS:
        per_mouse = [
            analysis.mouse_mean_trace
            for analysis in analyses
            if analysis.condition == condition
        ]
        if len(per_mouse) != len(EXPECTED_MOUSE_IDS):
            raise ValueError(
                f"{condition}: expected {len(EXPECTED_MOUSE_IDS)} mouse traces, "
                f"found {len(per_mouse)}"
            )
        matrix = np.vstack(per_mouse)
        mean = np.mean(matrix, axis=0)
        sem = np.std(matrix, axis=0, ddof=1) / np.sqrt(len(matrix))
        color = "green"
        label = CONDITION_LABELS[condition]
        linestyle = "-" if condition == BEFORE else "--"
        ax.plot(common_time, mean, color=color, lw=3.0,
                linestyle=linestyle, label=label, zorder=3)
        ax.fill_between(
            common_time,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.25,
            linewidth=0,
            zorder=2,
        )

    median_duration = float(
        np.median(
            [
                train.duration_s
                for analysis in analyses
                for train in analysis.selected_trains
            ]
        )
    )
    ax.axvspan(0, median_duration, color="#b8b8b8", alpha=0.25, zorder=0)
    _style.add_zero_y(ax, ensure_visible=True)
    _style.add_zero_x(ax)
    ax.set_xlim(-TRACE_PRE_S, RESPONSE_WINDOW_S)
    ax.set_xlabel("Time rel. to train onset (s)")
    ax.set_ylabel(RAW_DFF_LABEL)
    ax.set_title("GRAB-NE response to repeated LC stimulation")
    ax.spines[["right", "top"]].set_visible(False)
    _legend_above(ax, ncol=2, y_offset=1.14)
    _style.finalize_publication_figure(fig)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


@_style.styled
def _plot_paired_metric(
    mouse_values: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    pivot = mouse_values.pivot(index="mouse_id", columns="condition", values=column)
    pivot = pivot.loc[sorted(pivot.index)]
    if list(pivot.columns.intersection(CONDITIONS)) != list(CONDITIONS):
        pivot = pivot.reindex(columns=CONDITIONS)
    if pivot[list(CONDITIONS)].isna().any().any():
        raise ValueError(f"Missing paired values for {column}")

    fig, ax = plt.subplots(figsize=(8.0, _style.FIG_SCATTER[1]))
    before_values = pivot[BEFORE].to_numpy(dtype=float)
    after_values = pivot[AFTER].to_numpy(dtype=float)
    jitters = np.asarray([_stable_jitter(mouse) for mouse in pivot.index])

    for jitter, before, after in zip(jitters, before_values, after_values):
        ax.plot(
            [jitter, 1.0 + jitter],
            [before, after],
            color="#666666",
            alpha=0.60,
            lw=2.0,
            zorder=2,
        )
    ax.scatter(
        jitters,
        before_values,
        color=CONDITION_COLORS[BEFORE],
        s=180,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    ax.scatter(
        1.0 + jitters,
        after_values,
        color=CONDITION_COLORS[AFTER],
        s=180,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

    for x, values in ((0, before_values), (1, after_values)):
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            fmt="_",
            color="black",
            markersize=55,
            markeredgewidth=5,
            capsize=12,
            elinewidth=4,
            zorder=5,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Before", "After"],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.45, 1.45)
    ax.set_xlabel("Yohimbine", labelpad=80)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.spines[["right", "top"]].set_visible(False)
    _style.finalize_publication_figure(fig)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


@_style.styled
def _plot_train_qc(
    analyses: list[SessionAnalysis],
    output_path: Path,
) -> Path:
    ordered = sorted(
        analyses,
        key=lambda item: (int(item.mouse_id), _condition_order(item.condition)),
    )
    fig, ax = plt.subplots(figsize=(20.0, 14.0))
    for y, analysis in enumerate(ordered):
        first_onset = analysis.detected_trains[0].onset_s
        selected = analysis.detected_trains[:TRAINS_TO_USE]
        unused = analysis.detected_trains[TRAINS_TO_USE:]
        ax.scatter(
            [train.onset_s - first_onset for train in selected],
            np.full(len(selected), y),
            s=120,
            color=CONDITION_COLORS[analysis.condition],
            edgecolor="black",
            linewidth=0.6,
            label="Used (first 10)" if y == 0 else None,
            zorder=3,
        )
        if unused:
            ax.scatter(
                [train.onset_s - first_onset for train in unused],
                np.full(len(unused), y),
                s=100,
                facecolors="none",
                edgecolors="#777777",
                linewidth=1.5,
                label="Detected but not used" if y == 0 else None,
                zorder=2,
            )

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(
        [
            f"{item.mouse_id} - {CONDITION_LABELS[item.condition]}"
            for item in ordered
        ],
        fontsize=24,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Train onset relative to first train (s)")
    ax.set_ylabel("Session")
    ax.set_title("Channel 23 train detection and first-10 selection")
    ax.spines[["right", "top"]].set_visible(False)
    _legend_above(ax, ncol=2, y_offset=1.10)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_parameters(output_root: Path, data_root: Path) -> Path:
    path = output_root / "analysis_parameters.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(data_root),
        "preprocessing": "Strategy C (IRLS with 405-nm isosbestic correction)",
        "normalization": "subtract the full pre-first-train dF/F baseline mean",
        "stim_channel": 23,
        "expected_protocol": {
            "pulse_frequency_hz": 20,
            "pulse_width_ms": 5,
            "pulses_per_train": EXPECTED_PULSES_PER_TRAIN,
            "train_interval_s": EXPECTED_TRAIN_INTERVAL_S,
        },
        "detected_train_counts_by_mouse_per_condition": EXPECTED_TRAIN_COUNTS,
        "selected_trains": "first 10 per recording",
        "response_window_s": [0.0, RESPONSE_WINDOW_S],
        "response_window_end_inclusive": False,
        "auc": "signed trapezoidal AUC of baseline-corrected dF/F",
        "peak": "maximum baseline-corrected dF/F",
        "technical_replicates": TRAINS_TO_USE,
        "statistical_unit": "mouse",
        "pairing_key": "mouse_id",
        "p_value_adjustment": "none; two prespecified outcomes",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run(
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    cache_root: Path = CACHE_ROOT,
    *,
    rebuild_cache: bool = False,
) -> list[Path]:
    data_root = Path(data_root)
    output_root = Path(output_root)
    cache_root = Path(cache_root)
    if not data_root.exists():
        raise FileNotFoundError(f"GRAB_opto data root not found: {data_root}")

    log.info("Loading repeated-train GRAB_opto sessions from %s", data_root)
    sessions = load_sessions(data_root, test_mode=False, has_temperature=False)
    _validate_sessions(sessions)
    sessions = sorted(
        sessions,
        key=lambda session: (
            int(session.mouse_id),
            _condition_order(str(session.cohort)),
        ),
    )

    analyses = []
    for index, session in enumerate(sessions, start=1):
        log.info(
            "[%d/%d] %s - %s",
            index,
            len(sessions),
            session.mouse_id,
            CONDITION_LABELS[str(session.cohort)],
        )
        analyses.append(
            _analyze_session(
                session,
                cache_root,
                rebuild_cache=rebuild_cache,
            )
        )

    means_dir = output_root / "plots and values_means"
    qc_dir = output_root / "quality check"
    stats_dir = output_root / "statistics"
    for path in (means_dir, qc_dir, stats_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_values = _train_level_frame(analyses)
    mouse_values = _mouse_level_frame(train_values)
    qc_values = _qc_frame(analyses)
    stats = _paired_statistics(mouse_values)

    train_csv = means_dir / "GRAB_opto_train_level_values.csv"
    mouse_csv = means_dir / "GRAB_opto_mouse_level_values.csv"
    qc_csv = qc_dir / "GRAB_opto_stim_train_detection.csv"
    stats_csv = stats_dir / "GRAB_opto_paired_statistics.csv"
    train_values.to_csv(train_csv, index=False)
    mouse_values.to_csv(mouse_csv, index=False)
    qc_values.to_csv(qc_csv, index=False)
    stats.to_csv(stats_csv, index=False)

    outputs = [
        train_csv,
        mouse_csv,
        qc_csv,
        stats_csv,
        _plot_stim_aligned(
            analyses,
            means_dir / "GRAB_opto_stim_aligned_before_vs_after_Yohimbine.png",
        ),
        _plot_paired_metric(
            mouse_values,
            "evoked_auc_mean_dff_s",
            f"Evoked AUC\n({RAW_DFF_LABEL} × s)",
            "Evoked AUC (0-10 s)",
            means_dir / "GRAB_opto_evoked_AUC_0_10s.png",
        ),
        _plot_paired_metric(
            mouse_values,
            "evoked_peak_mean_dff",
            f"Evoked peak\n({RAW_DFF_LABEL})",
            "Evoked peak (0-10 s)",
            means_dir / "GRAB_opto_evoked_peak_0_10s.png",
        ),
        _plot_train_qc(
            analyses,
            qc_dir / "GRAB_opto_stim_train_detection.png",
        ),
        _write_parameters(output_root, data_root),
    ]

    log.info(
        "DONE. Analyzed %d events (%d mice x 2 conditions x %d trains)",
        len(train_values),
        len(EXPECTED_MOUSE_IDS),
        TRAINS_TO_USE,
    )
    log.info("Results: %s", output_root)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone repeated-train GRAB_opto analysis"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Re-run Strategy C even when synchronized-input cache hashes match",
    )
    args = parser.parse_args()
    run(
        data_root=args.data_root,
        output_root=args.output_root,
        cache_root=args.cache_root,
        rebuild_cache=args.rebuild_cache,
    )


if __name__ == "__main__":
    main()
