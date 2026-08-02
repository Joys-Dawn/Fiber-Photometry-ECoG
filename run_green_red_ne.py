"""Standalone qualitative analysis for the dual-color DG NE experiment.

This runner is intentionally dataset-specific. Analog input 1 is green
GRAB-NE and analog input 2 is the red NE sensor (or its mutant control).
Neither channel has an isosbestic signal, so each is independently low-pass
filtered, biexponentially detrended, and baseline z-scored with Strategy D.

Only four event-aligned qualitative summary plots are written. No scalar
values, transient analyses, statistics, or general pipeline configuration are
produced here.

Usage:
    python run_green_red_ne.py
    python run_green_red_ne.py --rebuild-cache
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from run_threshold_sweep import load_sessions
from fiber_photometry_ecog.core.config import PhotometryConfig
from fiber_photometry_ecog.core.data_models import Session
from fiber_photometry_ecog.preprocessing.photometry import (
    preprocess_no_isosbestic,
    z_score_baseline,
)
from fiber_photometry_ecog.visualization import _style, group_plots
from fiber_photometry_ecog.visualization.colors import DFF_LABEL


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


DATA_ROOT = Path("D:/Ephys_Data/DG_greenANDred_NE")
OUTPUT_DIR = Path("results") / "results_DG_greenANDred_NE"
CACHE_DIR = Path("sweep_cache") / "dg_green_and_red_ne"

PRE_EVENT_S = 100.0
POST_EVENT_S = 100.0
EVENT_BASELINE_START_S = 5.0
EVENT_BASELINE_END_S = 1.0
CACHE_VERSION = "dual-color-no-isosbestic-v1"

GREEN_COLOR = "green"
RED_COLOR = "red"


@dataclass(frozen=True)
class GroupSpec:
    """Hardcoded definition of one requested qualitative plot."""

    root_name: str
    cohort: str
    title: str
    plot_label: str
    red_label: str
    expected_mice: int
    expected_sessions: int
    filename: str


@dataclass(frozen=True)
class GroupSummary:
    """Mean and SEM traces for one dual-color experimental group."""

    time_axis: np.ndarray
    green_mean: np.ndarray
    green_sem: np.ndarray
    red_mean: np.ndarray
    red_sem: np.ndarray
    n_mice: int
    n_sessions: int


GROUP_SPECS = (
    GroupSpec(
        root_name="real_red_NE",
        cohort="scn1a_seizure",
        title="DS seizure: GRAB-NE + red NE sensor",
        plot_label="DS seizure",
        red_label="Red NE sensor",
        expected_mice=2,
        expected_sessions=4,
        filename="01_DS_seizure_GRABNE_red_NE_sensor.png",
    ),
    GroupSpec(
        root_name="mutant_red_NE",
        cohort="scn1a_seizure_mut",
        title="DS seizure: GRAB-NE + red NE mutant sensor",
        plot_label="DS seizure",
        red_label="Red NE mutant sensor",
        expected_mice=2,
        expected_sessions=4,
        filename="02_DS_seizure_GRABNE_red_NE_mutant.png",
    ),
    GroupSpec(
        root_name="real_red_NE",
        cohort="wt",
        title="WT heat: GRAB-NE + red NE sensor",
        plot_label="WT heat",
        red_label="Red NE sensor",
        expected_mice=2,
        expected_sessions=2,
        filename="03_WT_heat_GRABNE_red_NE_sensor.png",
    ),
    GroupSpec(
        root_name="mutant_red_NE",
        cohort="wt_mut",
        title="WT heat: GRAB-NE + red NE mutant sensor",
        plot_label="WT heat",
        red_label="Red NE mutant sensor",
        expected_mice=2,
        expected_sessions=3,
        filename="04_WT_heat_GRABNE_red_NE_mutant.png",
    ),
)


def _signal_fingerprint(*signals: np.ndarray) -> str:
    """Hash synchronized source channels so stale caches cannot be reused."""
    digest = hashlib.sha256()
    for signal in signals:
        array = np.ascontiguousarray(signal)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _preprocess_channel(
    signal: np.ndarray,
    fs: float,
    baseline_end_s: float,
) -> np.ndarray:
    """Apply Strategy D to one experimental channel, independently."""
    baseline_end_idx = min(int(round(baseline_end_s * fs)), len(signal))
    if baseline_end_idx < 2:
        raise ValueError(
            f"No-isosbestic preprocessing needs a non-empty baseline; "
            f"got baseline_end_s={baseline_end_s}, fs={fs}."
        )

    config = PhotometryConfig(strategy="D")
    result = preprocess_no_isosbestic(
        np.asarray(signal, dtype=float),
        np.empty(0, dtype=float),
        fs,
        config,
    )
    dff = np.asarray(result.dff, dtype=float)
    baseline_sigma = float(np.std(dff[:baseline_end_idx]))
    if not np.isfinite(baseline_sigma) or baseline_sigma == 0:
        raise ValueError(
            "No-isosbestic preprocessing produced an undefined or zero "
            f"baseline SD ({baseline_sigma})."
        )

    z_dff = z_score_baseline(dff, fs, baseline_end_s)
    if not np.all(np.isfinite(z_dff)):
        raise ValueError("No-isosbestic preprocessing produced non-finite z-dF/F.")
    return z_dff


def _extract_aligned_trace(
    signal: np.ndarray,
    fs: float,
    event_s: float,
    *,
    pre_s: float = PRE_EVENT_S,
    post_s: float = POST_EVENT_S,
    baseline_start_s: float = EVENT_BASELINE_START_S,
    baseline_end_s: float = EVENT_BASELINE_END_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an event trace and subtract its -5 to -1 s mean."""
    if not (0 <= baseline_end_s < baseline_start_s <= pre_s):
        raise ValueError(
            "Event baseline must satisfy "
            "0 <= baseline_end_s < baseline_start_s <= pre_s."
        )

    pre_n = int(round(pre_s * fs))
    post_n = int(round(post_s * fs))
    event_idx = int(round(event_s * fs))
    start = event_idx - pre_n
    stop = event_idx + post_n + 1
    if start < 0 or stop > len(signal):
        raise ValueError(
            f"Event window [{event_s - pre_s:.3f}, "
            f"{event_s + post_s:.3f}] s exceeds recording "
            f"[0, {len(signal) / fs:.3f}] s."
        )

    trace = np.asarray(signal[start:stop], dtype=float).copy()
    baseline_lo = pre_n - int(round(baseline_start_s * fs))
    baseline_hi = pre_n - int(round(baseline_end_s * fs))
    baseline = trace[baseline_lo:baseline_hi]
    if baseline.size == 0 or not np.all(np.isfinite(baseline)):
        raise ValueError("Event-aligned baseline is empty or non-finite.")
    trace -= float(np.mean(baseline))
    time_axis = np.arange(-pre_n, post_n + 1, dtype=float) / fs
    return time_axis, trace


def _cache_path(cache_dir: Path, spec: GroupSpec, session: Session) -> Path:
    date = str(session.date or "unknown-date")
    filename = (
        f"{session.mouse_id}_{date}_S{int(session.heating_session)}_"
        f"{spec.cohort}.npz"
    )
    return cache_dir / spec.root_name / filename


def _load_cached_traces(
    path: Path,
    *,
    fingerprint: str,
    session: Session,
    event_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            valid = (
                str(cached["cache_version"].item()) == CACHE_VERSION
                and str(cached["source_fingerprint"].item()) == fingerprint
                and int(cached["raw_n"].item()) == len(session.raw.signal_470)
                and np.isclose(float(cached["fs"].item()), session.raw.fs)
                and np.isclose(float(cached["event_s"].item()), event_s)
                and np.isclose(
                    float(cached["baseline_end_s"].item()),
                    session.landmarks.heating_start_time,
                )
            )
            if not valid:
                return None
            green_trace = np.asarray(cached["green_trace"], dtype=float)
            red_trace = np.asarray(cached["red_trace"], dtype=float)
    except (KeyError, OSError, ValueError) as exc:
        log.warning("    Ignoring unreadable cache %s: %s", path, exc)
        return None

    expected_n = int(round((PRE_EVENT_S + POST_EVENT_S) * session.raw.fs)) + 1
    if green_trace.shape != (expected_n,) or red_trace.shape != (expected_n,):
        return None
    if not np.all(np.isfinite(green_trace)) or not np.all(np.isfinite(red_trace)):
        return None
    return green_trace, red_trace


def _save_cached_traces(
    path: Path,
    *,
    fingerprint: str,
    session: Session,
    event_s: float,
    green_trace: np.ndarray,
    red_trace: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_version=np.array(CACHE_VERSION),
            source_fingerprint=np.array(fingerprint),
            raw_n=np.array(len(session.raw.signal_470)),
            fs=np.array(session.raw.fs),
            event_s=np.array(event_s),
            baseline_end_s=np.array(session.landmarks.heating_start_time),
            green_trace=np.asarray(green_trace, dtype=np.float64),
            red_trace=np.asarray(red_trace, dtype=np.float64),
        )
    temporary.replace(path)


def _session_event_traces(
    session: Session,
    spec: GroupSpec,
    cache_dir: Path,
    *,
    rebuild_cache: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if session.raw is None:
        raise ValueError(f"{session.mouse_id} S{session.heating_session}: missing raw data.")
    if session.landmarks is None:
        raise ValueError(f"{session.mouse_id} S{session.heating_session}: missing landmarks.")
    baseline_end_s = session.landmarks.heating_start_time
    event_s = session.landmarks.ueo_time
    if baseline_end_s is None or not np.isfinite(baseline_end_s):
        raise ValueError(
            f"{session.mouse_id} S{session.heating_session}: invalid heating start."
        )
    if event_s is None or not np.isfinite(event_s):
        raise ValueError(
            f"{session.mouse_id} S{session.heating_session}: invalid event time."
        )
    if session.raw.fs is None or session.raw.fs <= 0:
        raise ValueError(f"{session.mouse_id} S{session.heating_session}: invalid fs.")
    if len(session.raw.signal_470) != len(session.raw.signal_405):
        raise ValueError(
            f"{session.mouse_id} S{session.heating_session}: green/red length mismatch."
        )

    fingerprint = _signal_fingerprint(
        session.raw.signal_470,
        session.raw.signal_405,
    )
    path = _cache_path(cache_dir, spec, session)
    cached = None
    if not rebuild_cache:
        cached = _load_cached_traces(
            path,
            fingerprint=fingerprint,
            session=session,
            event_s=float(event_s),
        )
    if cached is not None:
        log.info(
            "    %s S%d (%s): loaded dual-color event traces from cache",
            session.mouse_id,
            session.heating_session,
            session.date,
        )
        green_trace, red_trace = cached
        pre_n = int(round(PRE_EVENT_S * session.raw.fs))
        post_n = int(round(POST_EVENT_S * session.raw.fs))
        time_axis = np.arange(-pre_n, post_n + 1, dtype=float) / session.raw.fs
        return time_axis, green_trace, red_trace

    log.info(
        "    %s S%d (%s): preprocessing green channel without isosbestic",
        session.mouse_id,
        session.heating_session,
        session.date,
    )
    green_z = _preprocess_channel(
        session.raw.signal_470,
        session.raw.fs,
        float(baseline_end_s),
    )
    log.info(
        "    %s S%d (%s): preprocessing red channel without isosbestic",
        session.mouse_id,
        session.heating_session,
        session.date,
    )
    red_z = _preprocess_channel(
        session.raw.signal_405,
        session.raw.fs,
        float(baseline_end_s),
    )
    time_axis, green_trace = _extract_aligned_trace(
        green_z,
        session.raw.fs,
        float(event_s),
    )
    red_time, red_trace = _extract_aligned_trace(
        red_z,
        session.raw.fs,
        float(event_s),
    )
    if not np.array_equal(time_axis, red_time):
        raise ValueError("Green and red event time axes do not match.")

    _save_cached_traces(
        path,
        fingerprint=fingerprint,
        session=session,
        event_s=float(event_s),
        green_trace=green_trace,
        red_trace=red_trace,
    )
    return time_axis, green_trace, red_trace


def _validate_group(sessions: list[Session], spec: GroupSpec) -> None:
    mouse_ids = {str(session.mouse_id) for session in sessions}
    identities = {
        (str(session.mouse_id), str(session.date), int(session.heating_session))
        for session in sessions
    }
    if len(sessions) != spec.expected_sessions:
        raise ValueError(
            f"{spec.title}: expected {spec.expected_sessions} sessions, "
            f"found {len(sessions)}."
        )
    if len(mouse_ids) != spec.expected_mice:
        raise ValueError(
            f"{spec.title}: expected {spec.expected_mice} mice, "
            f"found {len(mouse_ids)} ({sorted(mouse_ids)})."
        )
    if len(identities) != len(sessions):
        raise ValueError(f"{spec.title}: duplicate mouse/date/session identity.")


def _mean_sem(traces: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    trace_list = [np.asarray(trace, dtype=float) for trace in traces]
    if not trace_list:
        raise ValueError("Cannot summarize an empty trace collection.")
    lengths = {len(trace) for trace in trace_list}
    if len(lengths) != 1:
        raise ValueError(f"Trace lengths do not match: {sorted(lengths)}")
    matrix = np.vstack(trace_list)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Trace collection contains non-finite values.")
    mean = np.mean(matrix, axis=0)
    if len(matrix) == 1:
        sem = np.zeros(matrix.shape[1], dtype=float)
    else:
        sem = np.std(matrix, axis=0, ddof=1) / np.sqrt(len(matrix))
    return mean, sem


def _shared_y_limits(summaries: Iterable[GroupSummary]) -> tuple[float, float]:
    """Return one padded y-range covering every mean +/- SEM envelope."""
    y_min = np.inf
    y_max = -np.inf
    for summary in summaries:
        for mean, sem in (
            (summary.green_mean, summary.green_sem),
            (summary.red_mean, summary.red_sem),
        ):
            y_min = min(y_min, float(np.min(mean - sem)))
            y_max = max(y_max, float(np.max(mean + sem)))

    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        raise ValueError("Cannot compute shared y-limits from the group summaries.")
    y_min = min(y_min, 0.0)
    y_max = max(y_max, 0.0)
    padding = 0.05 * (y_max - y_min)
    return y_min - padding, y_max + padding


def _summarize_group(
    sessions: list[Session],
    spec: GroupSpec,
    cache_dir: Path,
    *,
    rebuild_cache: bool,
) -> GroupSummary:
    _validate_group(sessions, spec)
    ordered = sorted(
        sessions,
        key=lambda session: (
            str(session.mouse_id),
            str(session.date),
            int(session.heating_session),
        ),
    )
    green_traces = []
    red_traces = []
    time_axis = None
    for session in ordered:
        session_time, green_trace, red_trace = _session_event_traces(
            session,
            spec,
            cache_dir,
            rebuild_cache=rebuild_cache,
        )
        if time_axis is None:
            time_axis = session_time
        elif not np.array_equal(time_axis, session_time):
            raise ValueError(f"{spec.title}: session time axes do not match.")
        green_traces.append(green_trace)
        red_traces.append(red_trace)

    green_mean, green_sem = _mean_sem(green_traces)
    red_mean, red_sem = _mean_sem(red_traces)
    return GroupSummary(
        time_axis=time_axis,
        green_mean=green_mean,
        green_sem=green_sem,
        red_mean=red_mean,
        red_sem=red_sem,
        n_mice=len({str(session.mouse_id) for session in ordered}),
        n_sessions=len(ordered),
    )


@_style.styled
def _plot_group(
    summary: GroupSummary,
    spec: GroupSpec,
    output_path: Path,
    *,
    y_limits: tuple[float, float] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=_style.FIG_WIDE)

    ax.plot(
        summary.time_axis,
        summary.green_mean,
        color=GREEN_COLOR,
        label="Green GRAB-NE",
    )
    ax.fill_between(
        summary.time_axis,
        summary.green_mean - summary.green_sem,
        summary.green_mean + summary.green_sem,
        color=GREEN_COLOR,
        alpha=0.25,
    )
    ax.plot(
        summary.time_axis,
        summary.red_mean,
        color=RED_COLOR,
        label=spec.red_label,
    )
    ax.fill_between(
        summary.time_axis,
        summary.red_mean - summary.red_sem,
        summary.red_mean + summary.red_sem,
        color=RED_COLOR,
        alpha=0.25,
    )

    _style.add_zero_x(ax)
    ax.set_xlim(-PRE_EVENT_S, POST_EVENT_S)
    if y_limits is not None:
        y_lo, y_hi = y_limits
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
            raise ValueError(f"Invalid y-limits: {y_limits}")
        ax.set_ylim(y_lo, y_hi)
    _style.add_zero_y(ax, ensure_visible=True)
    ax.set_xlabel("Time rel. to event onset (s)")
    ax.set_ylabel(DFF_LABEL)
    ax.set_title(f"{spec.plot_label} - aligned to event onset")
    group_plots._legend_above(ax, ncol=2, y_offset=1.20)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run(
    *,
    data_root: Path = DATA_ROOT,
    output_dir: Path = OUTPUT_DIR,
    cache_dir: Path = CACHE_DIR,
    rebuild_cache: bool = False,
) -> list[Path]:
    """Run the four hardcoded dual-color qualitative summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_by_filename: dict[str, GroupSummary] = {}
    paths_by_filename: dict[str, Path] = {}

    root_names = list(dict.fromkeys(spec.root_name for spec in GROUP_SPECS))
    for root_name in root_names:
        experiment_root = data_root / root_name
        if not experiment_root.exists():
            raise FileNotFoundError(f"Missing experiment root: {experiment_root}")

        log.info("Loading dual-color sessions from %s", experiment_root)
        sessions = load_sessions(
            experiment_root,
            test_mode=False,
            has_temperature=False,
        )
        specs = [spec for spec in GROUP_SPECS if spec.root_name == root_name]
        expected_cohorts = {spec.cohort for spec in specs}
        actual_cohorts = {str(session.cohort) for session in sessions}
        if actual_cohorts != expected_cohorts:
            raise ValueError(
                f"{experiment_root}: expected cohorts {sorted(expected_cohorts)}, "
                f"found {sorted(actual_cohorts)}."
            )
        expected_total = sum(spec.expected_sessions for spec in specs)
        if len(sessions) != expected_total:
            raise ValueError(
                f"{experiment_root}: expected {expected_total} total sessions, "
                f"found {len(sessions)}."
            )

        for spec in specs:
            group_sessions = [
                session for session in sessions if str(session.cohort) == spec.cohort
            ]
            log.info(
                "Summarizing %s (%d sessions)",
                spec.title,
                len(group_sessions),
            )
            summary = _summarize_group(
                group_sessions,
                spec,
                cache_dir,
                rebuild_cache=rebuild_cache,
            )
            summaries_by_filename[spec.filename] = summary

    shared_y_limits = _shared_y_limits(summaries_by_filename.values())
    log.info(
        "Using shared axes for all four plots: x=[%.1f, %.1f] s, y=[%.3f, %.3f]",
        -PRE_EVENT_S,
        POST_EVENT_S,
        *shared_y_limits,
    )
    for spec in GROUP_SPECS:
        path = _plot_group(
            summaries_by_filename[spec.filename],
            spec,
            output_dir / spec.filename,
            y_limits=shared_y_limits,
        )
        paths_by_filename[spec.filename] = path
        log.info("Wrote %s", path)

    paths = [paths_by_filename[spec.filename] for spec in GROUP_SPECS]
    if len(paths) != 4 or any(not path.exists() for path in paths):
        raise RuntimeError("Dual-color run did not produce exactly four summary plots.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone qualitative dual-color analysis for DG green/red NE data."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Dataset root (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Plot output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help=f"Preprocessed trace cache (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore and replace any existing dual-color trace caches.",
    )
    args = parser.parse_args()

    paths = run(
        data_root=args.data_root,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
    )
    log.info("DONE. Wrote exactly %d summary plots to %s", len(paths), args.output_dir)


if __name__ == "__main__":
    main()
