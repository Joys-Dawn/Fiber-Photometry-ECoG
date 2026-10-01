"""
Strategy-C pipeline over D:/Ephys_Data/OPTO_stim experiment roots.

Each experiment root holds one data log (.xlsx) and a set of cohort
subfolders (e.g. CPP_treatment/{10HZ_DG, 10HZ_DG_CPP}). Sessions are
grouped by subfolder, which is what shows up as the analytical cohort
in plots and stats via `Session.cohort_override`.

Stim onset/offset are auto-detected from the data log's `opto-stim
channel` column and become the UEO/OFF landmarks; the stim onset also
doubles as the baseline-end marker that downstream z-score and
spike-detection windows depend on.

Usage:
    python run_opto.py                                     # all roots
    python run_opto.py --only CPP_treatment                # subset
    python run_opto.py --output-root results_opto_test     # custom out
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import numpy as np

from run_threshold_sweep import (
    load_sessions,
    run_group_analysis,
    _make_transient_config,
)
from fiber_photometry_ecog.core.config import PhotometryConfig, PreprocessingConfig
from fiber_photometry_ecog.core.data_models import (
    PhotometryResult,
    ProcessedData,
    Session,
)
from fiber_photometry_ecog.core.output_layout import ensure_layout
from fiber_photometry_ecog.preprocessing.ecog import filter_ecog
from fiber_photometry_ecog.preprocessing.photometry import (
    strategy_folder_name,
    z_score_baseline,
)
from fiber_photometry_ecog.analysis.opto_response import (
    BL_END_S as _OPTO_BL_END_S,
    BL_START_S as _OPTO_BL_START_S,
    WINDOW_S as _OPTO_WINDOW_S,
)
from fiber_photometry_ecog.preprocessing.pipeline import preprocess_session
from fiber_photometry_ecog.preprocessing.transient_detection import detect_transients
from fiber_photometry_ecog.statistics import run_paired_comparisons
from fiber_photometry_ecog.statistics.extractor import extract_all
from fiber_photometry_ecog.statistics.ratio import gmr_paired_test, pair_arrays
from fiber_photometry_ecog.visualization import _style
from fiber_photometry_ecog.visualization.colors import (
    COHORT_DISPLAY_LABELS,
    DFF_LABEL,
)
from fiber_photometry_ecog.visualization.trace_plots import (
    plot_sanity_check_opto,
    plot_zoomed_opto,
)


# Hardcoded per-session truncations for opto recordings where the fiber/LED
# stopped delivering light before the recording ended, leaving a long noise-
# floor tail that breaks IRLS dF/F (fitted isosbestic crosses zero).
# Key = (cohort_override, mouse_id, heating_session); value = keep_seconds.
# Only files that have been individually inspected belong here.
TRUNCATE_AT_S = {
    # DG_local_release / 20HZ_DG / 3799 S1: signal collapses to noise floor
    # at t≈366s and stays there until end-of-recording (473s). Pre-collapse
    # covers the entire stim window (279–289s) plus ~80s of post-stim
    # observation, which is enough for opto analysis.
    ("20HZ_DG", "3799", 1): 365.0,
}


# Per-session burst-index override for opto stim detection.
# Key = (cohort_override, mouse_id, heating_session); value = 0-indexed
# burst in time order. The default detector picks the longest cluster,
# which is correct when the session has one real burst plus optional
# artifacts. Sessions with multiple equally-sized bouts (where the
# protocol picks a non-largest one) need an explicit override.
STIM_BURST_OVERRIDES = {
    # DG_local_release / 1HZ_DG / 4361 S1: two 10-pulse 1 Hz bouts at
    # 297-306 s and 325-335 s ("stim_Meiling"). Both bouts have identical
    # sample counts, so the longest-cluster rule arbitrarily picks the
    # first. The protocol uses the 2nd bout.
    ("1HZ_DG", "4361", 1): 1,
}


def _refresh_stim_landmarks(session) -> None:
    """Re-detect stim onset/offset and apply any per-session burst-index
    override.

    Called on both the cache-hit and cache-miss paths. Cache-hit re-uses
    the .npz's stored landmarks, which were written by an older detector
    that included post-stim artifacts in the burst span; the new
    `detect_stim_burst` (longest-cluster rule + gap clustering) discards
    those artifacts. Cache-miss runs the new detector inside
    `preprocess_session`, but `STIM_BURST_OVERRIDES` only applies here —
    `preprocess_session` has no per-session config. Either way, the
    landmarks are correct in memory before downstream analysis runs.
    """
    from fiber_photometry_ecog.preprocessing.stim_detection import (
        detect_stim_burst,
    )
    if (session.raw is None
            or session.raw.stim_raw is None
            or session.raw.fs is None
            or session.raw.fs <= 0):
        return
    key = (
        session.cohort_override,
        str(session.mouse_id),
        int(session.heating_session),
    )
    burst_idx = STIM_BURST_OVERRIDES.get(key)
    try:
        onset, offset = detect_stim_burst(
            session.raw.stim_raw, session.raw.fs, burst_index=burst_idx,
        )
    except ValueError as exc:
        log.warning(
            "    Stim landmarks refresh [%s %s S%d]: %s",
            session.cohort_override, session.mouse_id, session.heating_session,
            exc,
        )
        return

    prev_ueo = session.landmarks.ueo_time
    prev_off = session.landmarks.off_time
    # Threshold the log to landmark changes worth noticing (>0.5 s).
    # Clean cache-hit paths with no change shouldn't spam the log.
    changed = (
        prev_ueo is None or prev_off is None
        or abs(prev_ueo - onset) > 0.5
        or abs(prev_off - offset) > 0.5
    )
    if changed:
        override_note = (
            f"  (burst_index override = {burst_idx})" if burst_idx is not None
            else ""
        )
        log.info(
            "    Stim landmarks refreshed [%s %s S%d]: "
            "ueo %s -> %.2fs   off %s -> %.2fs%s",
            session.cohort_override, session.mouse_id, session.heating_session,
            f"{prev_ueo:.2f}" if prev_ueo is not None else "None", onset,
            f"{prev_off:.2f}" if prev_off is not None else "None", offset,
            override_note,
        )
    session.landmarks.ueo_time = onset
    session.landmarks.off_time = offset


def _truncate_405_dropout(session) -> None:
    """Trim the bad tail off recordings listed in TRUNCATE_AT_S.

    Hardcoded by (cohort_override, mouse_id, heating_session); no detection,
    no heuristic. Other sessions are untouched. Modifies `session.raw` in
    place: clips signal_470/405, ecog, emg, temperature_raw, time, stim_raw
    to the same prefix length.
    """
    key = (session.cohort_override, session.mouse_id, session.heating_session)
    keep_s = TRUNCATE_AT_S.get(key)
    if keep_s is None:
        return
    raw = session.raw
    if raw is None or raw.fs is None or raw.fs <= 0:
        return
    keep_n = int(round(keep_s * raw.fs))
    n = len(raw.signal_405)
    if keep_n >= n:
        return

    def _trim(a):
        if a is None:
            return a
        return a[:keep_n] if len(a) >= keep_n else a

    raw.signal_470 = _trim(raw.signal_470)
    raw.signal_405 = _trim(raw.signal_405)
    raw.ecog = _trim(raw.ecog)
    raw.emg = _trim(raw.emg)
    raw.temperature_raw = _trim(raw.temperature_raw)
    raw.time = _trim(raw.time)
    raw.stim_raw = _trim(raw.stim_raw)

    log.info("    Truncated to %.1fs / %.1fs per TRUNCATE_AT_S",
             keep_s, n / raw.fs)


OPTO_CACHE_DIR = Path("opto_cache")


def _opto_cache_path(cache_label: str, strategy: str,
                     session: Session) -> Path:
    """Per-session cache file path.

    Layout: ``opto_cache/<cache_label>/<strategy>/<mouse>_<date>_S<n>_<cohort>.npz``

    `cache_label` is the experiment subpath leaf (e.g. ``DG_local_release``,
    ``CPP_treatment``) — matches the directory layout the scratch script
    used so the cache it produced is reused without re-fitting.

    The filename includes cohort_override because opto records same-day
    pairs (e.g. 1HZ_DG + 20HZ_DG for one mouse on one date, or pre-CPP
    + post-CPP) that share (mouse, date, S#) and would otherwise collide.
    """
    parts = [str(session.mouse_id)]
    if getattr(session, "date", None):
        parts.append(str(session.date))
    parts.append(f"S{int(session.heating_session)}")
    # Match the scratch cache key exactly: use session.cohort. For opto
    # sessions cohort == cohort_override (both set from the data log's
    # cohort column when stim_channel is present in load_sessions).
    parts.append(str(session.cohort))
    fname = "_".join(parts) + ".npz"
    return OPTO_CACHE_DIR / cache_label / strategy / fname


def _opto_recompute_dff_zscore(strategy: str, dff_raw, fs: float,
                               baseline_end_s, *, zscore: bool):
    """Return the analysis signal opto uses for this strategy.

    Strategy B always returns the first-baseline-window z-score (its
    scratch comparison is the most stable across subsetting). Strategy
    C respects the per-experiment ``zscore`` flag:

      * ``zscore=False`` (CPP_treatment): return RAW dF/F. The
        per-session pre-stim baseline divergence between Strategy B
        and C in CPP_treatment (see
        scratch_preprocessing_zscoring/cpp_pre_stim_baseline.csv) is
        amplified by the null-Z divisor and can flip the direction of
        the per-pair ratio test — raw dF/F and B's first-100s z-score
        agree on the direction.
      * ``zscore=True`` (DG_local_release): re-baseline z-score the raw
        dF/F against the first-baseline-window mean+std so the resulting
        traces compare like-for-like with B and Y-axes can be labelled
        ``z-ΔF/F``.

    Called on cache hit so changes to the z-score formula don't
    require invalidating cached raw dF/F.
    """
    if strategy == "C" and not zscore:
        return dff_raw
    return z_score_baseline(dff_raw, fs, baseline_end_s)


def _preprocess_and_cache(session: Session, strategy: str, cache_label: str,
                          *, zscore: bool):
    """Run preprocess_session (or load cached arrays) and return
    (zdff_hpf, dff_for_detection, temp_smooth, fs). Mirrors the shape
    of the original `_preprocess_no_cache` but stores per-session
    raw dF/F + landmarks + temp_smooth so re-runs are fast.

    The cache key includes cohort_override (see `_opto_cache_path`) so
    same-day frequency / treatment pairs do not collide.

    `zscore` toggles the analysis signal returned by
    ``_opto_recompute_dff_zscore`` — see that helper for the rationale.
    """
    config = PreprocessingConfig(
        photometry=PhotometryConfig(strategy=strategy),
    )
    # apply_hpf is OFF: peak detection runs on the z-scored dF/F directly,
    # prominence/amplitude measured on raw dF/F. Matches the sweep/app default.
    config.photometry.apply_hpf = False
    session.preprocessing_config = config

    cpath = _opto_cache_path(cache_label, strategy, session)
    # Cache files written by older versions are missing off_time /
    # terminal_time; detect that and refit so downstream analyses (e.g.
    # mean_seizure_duration) don't crash on None landmarks.
    cache_ok = False
    if cpath.exists():
        try:
            data = np.load(cpath)
            required = ("dff_raw", "fs", "ueo_time", "off_time",
                        "zscore_baseline_end_time", "terminal_time",
                        "temp_smooth")
            if all(k in data.files for k in required):
                dff_raw = data["dff_raw"]
                fs = float(data["fs"])
                ueo_time = float(data["ueo_time"])
                off_time = float(data["off_time"])
                zbe_time = float(data["zscore_baseline_end_time"])
                terminal_time = float(data["terminal_time"])
                temp_smooth = data["temp_smooth"]
                cache_ok = True
            else:
                log.warning(
                    "    Cache file missing fields %s; refitting",
                    [k for k in required if k not in data.files],
                )
        except Exception as e:
            log.warning("    Cache read failed [%s]: %s; refitting",
                        cpath.name, e)

    if cache_ok:
        session.landmarks.ueo_time = ueo_time
        session.landmarks.off_time = off_time
        session.landmarks.zscore_baseline_end_time = zbe_time
        session.landmarks.terminal_time = terminal_time
        # Opto sessions have no heating, but downstream code (spike
        # detection, baseline-window math) keys on heating_start_time.
        # The pipeline aliases it to zscore_baseline_end_time when
        # stim_raw is present; preserve that on cache hit.
        if session.landmarks.heating_start_time is None:
            session.landmarks.heating_start_time = zbe_time

        # Opto has no thermistor (has_temperature=False → NaN-filled).
        # preprocess_session sets baseline_temp/max_temp/terminal_temp to
        # NaN on cache miss; mirror that on cache hit so cohort
        # characteristics' all-or-none check doesn't trip on cache-hit
        # sessions (baseline_temp=None) mixed with cache-miss (NaN).
        lm = session.landmarks
        if lm.baseline_temp is None:
            lm.baseline_temp = float("nan")
        if lm.max_temp is None:
            lm.max_temp = float("nan")
        if lm.max_temp_time is None:
            lm.max_temp_time = float("nan")
        if lm.terminal_temp is None:
            lm.terminal_temp = float("nan")

        # Re-detect stim landmarks with the current (artifact-resistant)
        # detector + per-session override. Cache entries written by the
        # older first-to-last detector keep their stale ueo/off times in
        # the .npz; this refresh overrides them in memory each load.
        _refresh_stim_landmarks(session)

        dff_zscore = _opto_recompute_dff_zscore(
            strategy, dff_raw, fs, zbe_time, zscore=zscore,
        )

        # ECoG filter is cheap (Butterworth) — recompute on hit so plots
        # have the same filtered trace they'd have on a cache miss.
        ecog_filt = (
            filter_ecog(session.raw.ecog, fs, config.ecog)
            if session.raw is not None and session.raw.ecog is not None
            else None
        )

        session.processed = ProcessedData(
            photometry=PhotometryResult(
                dff=dff_raw,
                dff_zscore=dff_zscore,
                dff_hpf=dff_zscore,
            ),
            ecog_filtered=ecog_filt,
            temperature_smooth=temp_smooth,
            time=session.raw.time if session.raw is not None else None,
            fs=fs,
        )
        log.info("    Strategy %s: loaded from cache", strategy)
        return dff_zscore, dff_raw, temp_smooth, fs

    preprocess_session(session, config)
    proc = session.processed

    # Opto runs Strategy C on raw dF/F for experiments configured with
    # zscore=False (see _opto_recompute_dff_zscore for rationale).
    # preprocess_session sets dff_zscore = baseline z-score for every
    # strategy (same as hyperthermia); override it back to raw here so
    # the fresh-fit path and the cache-hit path return the same signal.
    # zscore=True experiments keep the baseline z-score that
    # preprocess_session produced.
    if (strategy == "C" and not zscore and proc is not None
            and proc.photometry is not None):
        proc.photometry.dff_zscore = proc.photometry.dff
        proc.photometry.dff_hpf = proc.photometry.dff

    # Apply per-session burst-index override BEFORE caching so the .npz
    # holds the corrected landmarks. preprocess_session already ran the
    # current (artifact-resistant) detector but doesn't know about
    # STIM_BURST_OVERRIDES.
    _refresh_stim_landmarks(session)

    cpath.parent.mkdir(parents=True, exist_ok=True)
    # Landmarks downstream analyses depend on:
    #  - ueo_time / off_time: mean_seizure_duration, paired-window math.
    #  - zscore_baseline_end_time: re-applies the strategy-specific z-score.
    #  - terminal_time: postictal / terminal-window analyses.
    # Temp landmarks are skipped: opto sessions are loaded with
    # has_temperature=False, so preprocess_session NaN-fills them — there
    # is nothing to preserve.
    np.savez(
        cpath,
        dff_raw=proc.photometry.dff,
        temp_smooth=proc.temperature_smooth,
        fs=np.array(proc.fs, dtype=np.float64),
        ueo_time=np.array(session.landmarks.ueo_time, dtype=np.float64),
        off_time=np.array(session.landmarks.off_time, dtype=np.float64),
        zscore_baseline_end_time=np.array(
            session.landmarks.zscore_baseline_end_time, dtype=np.float64,
        ),
        terminal_time=np.array(
            session.landmarks.terminal_time, dtype=np.float64,
        ),
    )
    log.info("    Strategy %s: preprocessed and cached", strategy)
    return (
        proc.photometry.dff_zscore,
        proc.photometry.dff,
        proc.temperature_smooth,
        proc.fs,
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


OPTO_BASE = Path("D:/Ephys_Data/OPTO_stim")
RESULTS_ROOT = Path("results")
STRATEGY = "C"
THRESHOLD = 0.030  # Strategy-C default prominence (matches DG_GRABne)

# Experiment schema:
#   experiment_subpath  : path under OPTO_BASE
#   fresh_cache_key     : preprocessing-cache subkey under sweep_cache/
#   paired_pair         : (baseline, treatment) for the paired-stats pass,
#                         or None for multi-group designs that aren't paired
#   zscore              : if True, Strategy C re-baseline-z-scores the cached
#                         raw dF/F before downstream analysis; if False (the
#                         current CPP_treatment behaviour), Strategy C uses
#                         raw dF/F directly so the per-pair ratio test isn't
#                         corrupted by the null-Z divisor
#   dff_label           : y-axis / colorbar label for UEO-aligned, heatmap,
#                         and parameter-detection plots ("ΔF/F" or "z-ΔF/F")
#   excluded_from_scale : cohort keys still rendered, but excluded from
#                         auto-y-lim and auto-heatmap-vlim computation. Used
#                         to keep one high-range arm (Atomoxetine) from
#                         dominating the rest of the comparison
# Stats inside run_group_analysis piggyback on the DS_sz vs WT_no_sz
# comparison which is meaningless for opto cohorts — we always pass
# run_stats=False below and rely on the paired wrapper instead.
EXPERIMENT_ROOTS = [
    ("Hotspot_GANE/CPP_treatment",    "opto_cpp_treatment",
     ("10HZ_DG", "10HZ_DG_CPP"), False, "ΔF/F",   ()),
    ("Hotspot_GANE/DG_local_release", "opto_dg_local_release",
     ("1HZ_DG", "20HZ_DG"),      True,  DFF_LABEL, ()),
    # Atomoxetine arm split out to its own root per Meiling: the raw folder
    # is `DG_local_release _Atomoxetine` (space intentional, matches on disk)
    # and holds only the 20HZ_DG_Atomoxetine cohort. No paired comparison
    # here — solo cohort — so paired_pair is None.
    ("Hotspot_GANE/DG_local_release _Atomoxetine", "opto_dg_local_release_atomoxetine",
     None,                        True,  DFF_LABEL, ()),
]


# Opto stats: the only measurements we want in the paired-stats CSV are
# the new stim-response set (max/min/AUC over 40 s post-stim). Everything
# else here is a seizure-named landmark/window measurement borrowed from
# the hyperthermia pipeline; for opto experiments those windows either
# don't apply or are misleading (no real "ictal" or "OFF" phase). Keep
# the modules running so the trace plots and CSVs still render, but
# suppress these measurements from the stats CSV.
_OPTO_LEGACY_MEASUREMENTS_TO_DROP = {
    # heat-named (already dropped previously)
    "early_heat_mean", "late_heat_mean", "end_late_heat_mean", "heating_mean",
    # seizure-named (no real seizure here)
    "seizure_duration_s",
    "seizure_mean",
    "ictal_mean",
    "delta_preictal_ictal",
    "delta_preictal_ictal_max",
    "delta_preictal_ictal_min",
    "auc_UEO",
    "auc_OFF",
    "auc_EEC",
    "auc_behavioral_onset",
    "auc",                     # spike-triggered AUC (interictal spike concept)
    # postictal-named (no real postictal phase)
    "final_time", "final_mean_dff",
    # baseline z-scored to itself is identically zero — uninformative
    "baseline_mean",
}


# Stim-response measurements that get a paired ratio test in addition to
# the LMM mean-difference test. These are the only opto outcomes for which
# a multiplicative effect (b/a, GMR) is biologically meaningful.
_OPTO_RATIO_MEASUREMENTS = (
    "stim_response_max_40s",
    "stim_response_min_40s",
    "stim_auc_40s",
)

# Per-pair normalized response bar plot. The per-session quantity here
# is ``opto_response.stim_response_max_40s`` -- max of the
# [-5, -1] s baseline-corrected signal over [UEO, UEO + 40 s]. Pulling
# the constants from opto_response keeps the bar plot, the per-session
# metric, and the paired ratio test on a single source of truth.


def _session_stim_response_max_40s(session: Session) -> float:
    """Mirror of ``opto_response._per_session``'s ``stim_response_max_40s``.

    Subtracts the mean of [UEO - BL_START_S, UEO - BL_END_S) from the
    trace, then returns the max over [UEO, UEO + WINDOW_S]. Returns
    NaN if any required state is missing.
    """
    if (session.processed is None or session.processed.photometry is None
            or session.processed.photometry.dff_zscore is None):
        return float("nan")
    if session.processed.fs is None or session.processed.fs <= 0:
        return float("nan")
    if session.landmarks is None or session.landmarks.ueo_time is None:
        return float("nan")
    fs = session.processed.fs
    dff = session.processed.photometry.dff_zscore
    ueo = session.landmarks.ueo_time
    if not np.isfinite(ueo):
        return float("nan")
    ueo_i = int(round(ueo * fs))
    bl_lo = ueo_i - int(round(_OPTO_BL_START_S * fs))
    bl_hi = ueo_i - int(round(_OPTO_BL_END_S * fs))
    if bl_lo < 0 or bl_hi <= bl_lo:
        return float("nan")
    bl_mean = float(np.nanmean(dff[bl_lo:bl_hi]))
    if not np.isfinite(bl_mean):
        return float("nan")
    win_hi = min(len(dff), ueo_i + int(round(_OPTO_WINDOW_S * fs)))
    resp = dff[ueo_i:win_hi]
    if resp.size == 0:
        return float("nan")
    return float(np.nanmax(resp) - bl_mean)


@_style.styled
def _make_normalized_response_bar(
    sessions: list,
    group_a: str,
    group_b: str,
    output_dir: Path,
    label: str,
) -> None:
    """Per-pair `group_b / group_a` ratio of stim_response_max_40s.

    Per-session quantity is the same one computed in
    ``opto_response``: max of the [-5, -1] s baseline-corrected trace
    over [UEO, UEO + 40 s]. Writes the bar PNG plus a CSV of per-pair
    values to `output_dir`. Publication-aesthetic styling: inherits the
    48pt/40pt scope from ``_style.styled`` and uses the same vertical-
    rectangle figure size + bolder mean/SEM markers as the other group
    scatter plots. Per-datapoint session labels are dropped (kept in the
    paired-values CSV for traceability).
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []
    for s in sessions:
        if s.cohort not in (group_a, group_b):
            continue
        rows.append({
            "mouse_id": s.mouse_id,
            "heating_session": int(s.heating_session),
            "cohort": s.cohort,
            "response": _session_stim_response_max_40s(s),
        })
    df = pd.DataFrame(rows).dropna(subset=["response"]).reset_index(drop=True)
    if df.empty:
        log.warning("    Normalized response bar: no usable sessions for "
                    "%s / %s -- skipping plot.", group_a, group_b)
        return

    a = df.loc[df["cohort"] == group_a].rename(
        columns={"response": "a_response"})
    b = df.loc[df["cohort"] == group_b].rename(
        columns={"response": "b_response"})
    pairs = b.merge(
        a[["mouse_id", "heating_session", "a_response"]],
        on=["mouse_id", "heating_session"], how="inner",
    )
    if pairs.empty:
        log.warning("    Normalized response bar: no paired sessions for "
                    "%s vs %s -- skipping plot.", group_a, group_b)
        return

    pairs["session_label"] = (
        pairs["mouse_id"].astype(str)
        + "_s" + pairs["heating_session"].astype(str)
    )
    pairs["b_norm"] = pairs["b_response"] / pairs["a_response"]
    pairs = pairs.sort_values(["mouse_id", "heating_session"]).reset_index(drop=True)

    csv_path = output_dir / f"normalized_response_values_40s_{label}.csv"
    pairs[["session_label", "mouse_id", "heating_session",
           "a_response", "b_response", "b_norm"]].to_csv(csv_path, index=False)
    log.info("    Normalized response bar: wrote %s", csv_path)

    # Publication-aesthetic single-bar figure. No title (filename carries
    # the semantics). Short axis label. No per-mouse color coding — all
    # dots single black, matching the cohort-scatter house style. Wide
    # enough to fit the 48pt x-tick as a single line.
    vals = pairs["b_norm"].to_numpy(dtype=float)
    mean = float(np.mean(vals))
    sem = (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
           if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(9.0, 12.0))

    # Bar + error bar in the house style (thick spines/ticks, muted fill).
    ax.bar([0], [mean], yerr=[sem], capsize=14, width=0.55,
           color="#c9c9c9", edgecolor="black", linewidth=2.0, alpha=0.9, zorder=1,
           error_kw={"elinewidth": 4, "capthick": 4})

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(jitter, vals, color="black", s=180, zorder=3,
               edgecolor="black", linewidths=0.8, alpha=0.9)

    ax.axhline(
        1.0,
        color=_style.ZERO_LINE_COLOR,
        lw=_style.ZERO_LINEWIDTH,
        ls=_style.ZERO_LINESTYLE,
        alpha=0.9,
        zorder=2,
    )
    ax.set_xticks([0])
    ax.set_xticklabels([COHORT_DISPLAY_LABELS.get(group_b, group_b)])
    ax.set_ylabel("Normalized response\n(post / paired baseline)")
    ax.set_xlim(-0.6, 0.6)
    ax.spines[["right", "top"]].set_visible(False)

    _style.finalize_publication_figure(fig)
    png_path = output_dir / f"normalized_response_bar_40s_{label}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("    Normalized response bar: wrote %s", png_path)


def _augment_paired_csv_with_ratio(
    csv_path: Path,
    cohort_module_results: dict,
    group_a: str,
    group_b: str,
) -> None:
    """Read the LMM paired-stats CSV and add per-pair ratio-test columns
    for the configured stim-response measurements.

    Adds columns: Ratio_test, GMR, GMR_CI_lo, GMR_CI_hi,
    Signed_log_mean_diff, Ratio_p_value, Ratio_notes. Rows for
    measurements not in `_OPTO_RATIO_MEASUREMENTS` get blank values in
    those columns.
    """
    import pandas as pd

    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)

    long_rows = extract_all(cohort_module_results)
    # Build a (measurement -> list of {mouse_id, heating_session, cohort, value})
    per_meas: dict = {}
    for row in long_rows:
        if row["cohort"] not in (group_a, group_b):
            continue
        if row["measurement"] not in _OPTO_RATIO_MEASUREMENTS:
            continue
        per_meas.setdefault(row["measurement"], []).append({
            "mouse_id": row["mouse_id"],
            "heating_session": row["heating_session"],
            "cohort": row["cohort"],
            "value": row["value"],
        })

    new_cols = {
        "Ratio_test": [""] * len(df),
        "GMR": [float("nan")] * len(df),
        "GMR_CI_lo": [float("nan")] * len(df),
        "GMR_CI_hi": [float("nan")] * len(df),
        "Signed_log_mean_diff": [float("nan")] * len(df),
        "Ratio_p_value": [float("nan")] * len(df),
        "Ratio_notes": [""] * len(df),
    }

    for idx, meas in enumerate(df["Measurement"].astype(str)):
        rows = per_meas.get(meas)
        if not rows:
            continue
        a_vals, b_vals = pair_arrays(rows, group_a, group_b)
        result = gmr_paired_test(a_vals, b_vals)
        if result is None:
            continue
        new_cols["Ratio_test"][idx] = result.test_name
        new_cols["GMR"][idx] = result.gmr
        new_cols["GMR_CI_lo"][idx] = result.gmr_lo
        new_cols["GMR_CI_hi"][idx] = result.gmr_hi
        new_cols["Signed_log_mean_diff"][idx] = result.signed_log_mean_diff
        new_cols["Ratio_p_value"][idx] = result.p_value
        new_cols["Ratio_notes"][idx] = result.notes

    for col, vals in new_cols.items():
        df[col] = vals
    df.to_csv(csv_path, index=False)


def run_one_root(experiment_subpath: str, ds_key: str,
                 paired_pair: tuple | None,
                 *,
                 zscore: bool,
                 dff_label: str,
                 excluded_from_scale: tuple[str, ...] = (),
                 output_root: Path = RESULTS_ROOT) -> None:
    exp_path = OPTO_BASE / experiment_subpath
    if not exp_path.exists():
        log.warning("Skipping %s -- path does not exist: %s",
                    experiment_subpath, exp_path)
        return

    label = experiment_subpath.replace("/", "_")
    log.info("=" * 70)
    log.info("Opto / %s   strategy=%s   threshold=%g   cache_key=%s",
             experiment_subpath, STRATEGY, THRESHOLD, ds_key)
    log.info("=" * 70)

    # Opto sessions don't use a thermistor — the rig still routes some ADC
    # input to whatever was historically the temperature pin, so the raw
    # trace would be garbage. has_temperature=False NaN-fills temperature
    # so downstream temperature analyses produce empty plots, not nonsense.
    sessions = load_sessions(exp_path, test_mode=False, has_temperature=False)
    if not sessions:
        log.warning("No loadable sessions in %s; skipping group analysis.",
                    experiment_subpath)
        return

    # Opto sessions all had the stim event by definition; treat them as
    # seizure-positive so the existing ictal-mean / pairing logic includes
    # them. Pairing won't run because every opto session is "seized" —
    # there are no controls to pair against.
    for s in sessions:
        if s.raw is not None and s.raw.stim_raw is not None:
            s.n_seizures = max(s.n_seizures, 1)

    results_base = output_root / f"results_opto_{label}"
    layout = ensure_layout(results_base / strategy_folder_name(STRATEGY))
    qc_dir = str(layout["quality_check"])

    # Cache label = the experiment subpath leaf (e.g. "DG_local_release",
    # "CPP_treatment"). Matches the directory layout already populated by
    # the scratch sweep so cached fits are reused, not redone.
    cache_label = Path(experiment_subpath).name

    tc = _make_transient_config(STRATEGY, THRESHOLD)
    for i, session in enumerate(sessions, 1):
        log.info("  [%d/%d] %s S%d -- preprocessing + transient detection",
                 i, len(sessions), session.mouse_id, session.heating_session)
        _truncate_405_dropout(session)
        zdff_hpf, dff_det, temp_smooth, fs = _preprocess_and_cache(
            session, STRATEGY, cache_label, zscore=zscore,
        )
        session.transients = detect_transients(
            zdff_hpf, dff_det, fs, tc, temp_smooth,
        )

        plot_sanity_check_opto(session, qc_dir, session.transients)
        plot_zoomed_opto(session, qc_dir)

    log.info("  Running group analysis -> %s", results_base)
    is_cpp_treatment = ds_key == "opto_cpp_treatment"
    paired_scatter = {"connect_pairs": True}
    cpp_scatter_modules = {
        name: paired_scatter
        for name in (
            "Cohort characteristics", "Pre-ictal mean", "Ictal mean",
            "Postictal", "Spike-triggered averages", "Opto response",
        )
    } if is_cpp_treatment else None
    if cpp_scatter_modules is not None:
        cpp_scatter_modules["Baseline transients"] = {"standalone_width": 12.0}
    cohort_module_results = run_group_analysis(
        sessions, STRATEGY, ds_key, results_base,
        stats_name_prefix=label,
        ueo_y_lim=(-0.1, 0.4) if is_cpp_treatment else None,
        ueo_y_ticks=(-0.1, 0.0, 0.1, 0.2, 0.3, 0.4)
        if is_cpp_treatment else None,
        run_stats=False,
        # CPP manuscript panels use ±40 s; other opto traces retain ±100 s.
        ueo_standard_window_s=(-40.0, 40.0) if is_cpp_treatment else (-100.0, 100.0),
        ueo_trace_colors={
            "10HZ_DG": "green", "10HZ_DG_CPP": "green",
            "1HZ_DG": "green", "20HZ_DG": "green",
            "20HZ_DG_Atomoxetine": "green", "20HZ_LC": "green",
            "20HZ_DG_mcherry": "green",
        },
        ueo_xlabel="Time rel. to stim onset (s)",
        ueo_title_suffix="aligned to stim onset",
        # Opto-mode plot wiring: re-label parameter-detection regions for
        # stim experiments, rename UEO_aligned → stim_onset_aligned,
        # apply the experiment's chosen ΔF/F vs z-ΔF/F y-label, and skip
        # the excluded arm (Atomoxetine) when computing auto-scaling.
        is_opto=True,
        ueo_filename_prefix="stim_onset_aligned",
        ueo_dff_label=dff_label,
        module_scatter_options=cpp_scatter_modules,
        ueo_yscale_exclude=list(excluded_from_scale),
        ueo_heatmap_exclude_from_vlim=list(excluded_from_scale),
        ueo_heatmap_vlim_override=0.4 if is_cpp_treatment else None,
        # Opto UEO (stim-onset) triggered AUC integrates over 40 s post-stim,
        # not the default global_mean_duration (= stim_off − stim_on = 10 s).
        # The 10 s default would only capture the during-stim window; the
        # post-stim response is what quantifies the physiological effect.
        ictal_auc_per_landmark_override={"UEO": 40.0},
    )

    if paired_pair is not None and cohort_module_results:
        group_a, group_b = paired_pair
        log.info("  Running paired stats: %s vs %s", group_a, group_b)
        layout = ensure_layout(results_base / strategy_folder_name(STRATEGY))
        paths = run_paired_comparisons(
            cohort_module_results,
            means_dir=str(layout["means"]),
            transients_dir=str(layout["transients"]),
            name_prefix=label,
            group_a=group_a,
            group_b=group_b,
            exclude_measurements=_OPTO_LEGACY_MEASUREMENTS_TO_DROP,
        )
        for folder, p in paths.items():
            log.info("    Paired stats [%s]: %s", folder, p)

        # Augment the means CSV with per-pair ratio tests for the three
        # stim-response measurements. The transients CSV doesn't carry
        # stim-response rows so it's left untouched.
        means_csv = paths.get("means")
        if means_csv is not None:
            _augment_paired_csv_with_ratio(
                Path(means_csv), cohort_module_results, group_a, group_b,
            )
            log.info("    Paired stats [means]: added ratio-test columns "
                     "for %s", _OPTO_RATIO_MEASUREMENTS)

        # Per-pair normalized response bar:
        # stim_response_max_40s(b) / stim_response_max_40s(a), one bar
        # per paired root. Drops next to the paired-stats CSV.
        _make_normalized_response_bar(
            sessions, group_a, group_b,
            output_dir=Path(layout["means"]),
            label=label,
        )

    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy-C pipeline over OPTO_stim experiment roots"
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Subset of experiment subpaths to run (default: all).",
    )
    parser.add_argument(
        "--output-root", default=str(RESULTS_ROOT),
        help="Top-level results directory (default: results).",
    )
    parser.add_argument(
        "--no-paired", action="store_true",
        help="Skip the paired-stats pass even for configured roots.",
    )
    args = parser.parse_args()

    selected = EXPERIMENT_ROOTS
    if args.only:
        wanted = set(args.only)
        selected = [r for r in EXPERIMENT_ROOTS if r[0] in wanted]
        missing = wanted - {r[0] for r in EXPERIMENT_ROOTS}
        if missing:
            parser.error(
                f"Unknown experiment subpaths: {sorted(missing)}. "
                f"Choose from {[r[0] for r in EXPERIMENT_ROOTS]}."
            )

    output_root = Path(args.output_root)

    log.info("Running %d experiment root(s) under %s -> %s",
             len(selected), OPTO_BASE, output_root)
    for (experiment_subpath, ds_key, paired_pair,
         zscore, dff_label, excluded_from_scale) in selected:
        effective_paired = None if args.no_paired else paired_pair
        try:
            run_one_root(experiment_subpath, ds_key, effective_paired,
                         zscore=zscore,
                         dff_label=dff_label,
                         excluded_from_scale=excluded_from_scale,
                         output_root=output_root)
        except Exception as e:
            log.error("FATAL in %s: %s", experiment_subpath, e)
            traceback.print_exc()

    log.info("DONE. Results in %s/results_opto_*", output_root)


if __name__ == "__main__":
    main()
