"""
Threshold sweep: preprocess all sessions with strategies A, B, C,
then run transient detection at multiple thresholds for each strategy.

Preprocessed signals (dff, zdff_hpf, temperature) are cached per
strategy+session so subsequent runs skip the expensive fitting step.

Usage:
    python run_threshold_sweep.py              # full run
    python run_threshold_sweep.py --test       # 1 file per dataset
    python run_threshold_sweep.py --clear-cache  # wipe cache and recompute
"""

import argparse
import csv
import logging
import os
import re
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import numpy as np

from fiber_photometry_ecog.core.config import (
    AnalysisConfig,
    PhotometryConfig,
    PreprocessingConfig,
    TransientConfig,
)
from fiber_photometry_ecog.core.data_models import RawData, Session, SessionLandmarks
from fiber_photometry_ecog.data_loading import (
    read_ppd, read_oep, synchronize,
    scan_experiment_folder, extract_date_from_oep, read_data_log,
)
from fiber_photometry_ecog.preprocessing.pipeline import preprocess_session
from fiber_photometry_ecog.preprocessing.spike_detection import detect_spikes
from fiber_photometry_ecog.preprocessing.transient_detection import detect_transients
from fiber_photometry_ecog.visualization.trace_plots import plot_sanity_check, plot_zoomed, plot_transient_review
from fiber_photometry_ecog.analysis.cohort_characteristics import compute_cohort_characteristics
from fiber_photometry_ecog.analysis.baseline_transients import compute_baseline_transients
from fiber_photometry_ecog.analysis.preictal_mean import compute_preictal_mean
from fiber_photometry_ecog.analysis.preictal_transients import compute_preictal_transients
from fiber_photometry_ecog.analysis.ictal_mean import compute_ictal_mean, compute_wide_ueo_triggered
from fiber_photometry_ecog.analysis.ictal_transients import compute_ictal_transients
from fiber_photometry_ecog.analysis.postictal import compute_postictal
from fiber_photometry_ecog.analysis.spike_triggered import compute_spike_triggered_average
from fiber_photometry_ecog.pairing.engine import assign_all_controls
from fiber_photometry_ecog.visualization.group_plots import (
    plot_cohort_characteristics,
    plot_baseline_transients,
    plot_preictal_mean,
    plot_preictal_mean_diagnostic,
    plot_preictal_transients,
    plot_ictal_mean,
    plot_ictal_transients,
    plot_postictal,
    plot_spike_triggered,
    plot_ueo_per_cohort,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

EPHYS_ROOT = Path("D:/Ephys_Data")

DATASETS = {
    "dg": EPHYS_ROOT / "DG_GRABne",
    "lc": EPHYS_ROOT / "LC_GCaMP",
}

STRATEGIES = ["A", "B", "C"]

THRESHOLDS = {
    "A": [1.0, 1.5, 2.0, 2.5, 3.0],
    "B": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035],
    "C": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035],
}

CACHE_DIR = Path("sweep_cache")

# Mice excluded from spike-triggered analysis: ECoG-to-photometry
# electrical crosstalk visible in both 470 and 405 channels at spike time.
STA_EXCLUDE = {
    "dg": {("2641", 1), ("3154", 1)},       # seizure
    "lc": {("3456", 1)},                      # failed_seizure
}


# ---------------------------------------------------------------------------
# Caching: store preprocessed signals so B/C don't refit every run
# ---------------------------------------------------------------------------

def _cache_path(ds_key: str, strategy: str, session_key: str) -> Path:
    return CACHE_DIR / ds_key / strategy / f"{session_key}.npz"


def _load_cache(ds_key: str, strategy: str, session_key: str):
    """Return (zdff_hpf, dff_for_detection, dff_raw, temp_smooth, fs) or None."""
    path = _cache_path(ds_key, strategy, session_key)
    if path.exists():
        data = np.load(path)
        if "dff_raw" not in data:
            return None
        return (
            data["zdff_hpf"],
            data["dff_for_detection"],
            data["dff_raw"],
            data["temp_smooth"],
            float(data["fs"]),
        )
    return None


def _save_cache(ds_key: str, strategy: str, session_key: str,
                zdff_hpf, dff_for_detection, dff_raw, temp_smooth, fs):
    path = _cache_path(ds_key, strategy, session_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             zdff_hpf=zdff_hpf,
             dff_for_detection=dff_for_detection,
             dff_raw=dff_raw,
             temp_smooth=temp_smooth,
             fs=np.array(fs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _threshold_label(strategy: str, threshold: float) -> str:
    if strategy == "A":
        return f"prom_{threshold:.1f}"
    return f"prom_{threshold:.3f}"


def _make_transient_config(strategy: str, threshold: float) -> TransientConfig:
    if strategy == "A":
        return TransientConfig(
            method="prominence",
            min_prominence=threshold,
            min_height=None,
        )
    return TransientConfig(
        method="wallace",
        min_height=1.0,
        min_prominence=threshold,
    )


def _session_key(session: Session) -> str:
    parts = [session.mouse_id]
    if session.date:
        parts.append(session.date)
    parts.append(f"S{session.heating_session}")
    return "_".join(parts)


def _genotype_from_cohort(cohort: str) -> str:
    c = cohort.lower()
    if "wt" in c:
        return "WT"
    return "Scn1a"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sessions(experiment_dir: Path, test_mode: bool = False):
    """Scan, read data log, and load all raw sessions."""
    exp_str = str(experiment_dir)
    discovered = scan_experiment_folder(exp_str)
    log_lookup = read_data_log(exp_str)

    if not discovered:
        log.warning("No sessions found in %s", experiment_dir)
        return []

    for d in discovered:
        name = d["session_name"]
        m = re.match(r"^(\d+)", name)
        d["mouse_id"] = m.group(1) if m else name.split("_")[0]
        d["date"] = extract_date_from_oep(d["oep_path"]) if d["oep_path"] else None

        if log_lookup and (d["mouse_id"], d["date"]) in log_lookup:
            d.update(log_lookup[(d["mouse_id"], d["date"])])
        else:
            cohort = d["cohort"]
            d["genotype"] = _genotype_from_cohort(cohort)
            d["seizure"] = 1 if "seizure" in cohort.lower() and "no" not in cohort.lower() else 0
            d["sudep"] = False
            d["include"] = True
            d["exclusion_reason"] = None
            d["heating_start"] = None

    loadable = [d for d in discovered
                if d["include"] and d["oep_path"] and d["ppd_path"]
                and d.get("heating_start") is not None]

    if test_mode:
        loadable = loadable[:1]
        log.info("TEST MODE: limited to 1 session")

    sessions = []
    for i, d in enumerate(loadable):
        mouse_id = d["mouse_id"]
        log.info("[%d/%d] Loading %s ...", i + 1, len(loadable), mouse_id)
        try:
            ppd = read_ppd(d["ppd_path"])
            oep = read_oep(
                d["oep_path"],
                ecog_channel=d.get("ecog_channel") or 2,
                emg_channel=d.get("emg_channel") if d.get("emg_channel") is not None else 3,
                temperature_channel=d.get("temperature_channel"),
            )
            sync = synchronize(ppd, oep)

            landmarks = SessionLandmarks(
                heating_start_time=d["heating_start"],
                eec_time=d.get("eec_time"),
                ueo_time=d.get("ueo_time"),
                off_time=d.get("off_time"),
                baseline_window_s=d.get("baseline_window_s"),
            )
            raw = RawData(
                signal_470=sync.signal_470,
                signal_405=sync.signal_405,
                ecog=sync.ecog,
                emg=sync.emg,
                temperature_raw=sync.temperature_raw,
                temp_bit_volts=sync.temp_bit_volts,
                temp_slope=sync.temp_slope,
                temp_intercept=sync.temp_intercept,
                time=sync.time,
                fs=sync.fs,
            )
            session = Session(
                mouse_id=mouse_id,
                genotype=d["genotype"],
                heating_session=d.get("heating_session", 1),
                n_seizures=d.get("seizure", 0),
                sudep=d.get("sudep", False),
                include_session=d["include"],
                exclusion_reason=d.get("exclusion_reason"),
                include_for_baseline=d.get("include_for_baseline", True),
                include_for_transients=d.get("include_for_transients", True),
                transient_prominence=d.get("transient_prominence"),
                landmarks=landmarks,
                raw=raw,
            )
            session.cohort = d["cohort"]
            session.date = d.get("date")
            session.session_name = d.get("session_name")
            sessions.append(session)

        except Exception as e:
            log.error("  FAILED %s: %s", mouse_id, e)
            traceback.print_exc()

    log.info("Loaded %d / %d sessions from %s", len(sessions), len(loadable), experiment_dir.name)
    return sessions


# ---------------------------------------------------------------------------
# Core: preprocess (with cache) + sweep thresholds
# ---------------------------------------------------------------------------

def preprocess_and_cache(session: Session, strategy: str, ds_key: str):
    """Preprocess a session with the given strategy. Uses cache if available.

    Returns (zdff_hpf, dff_for_detection, temp_smooth, fs).
    The session object is also updated in-place with .processed so that
    sanity-check plots can be generated.
    """
    from fiber_photometry_ecog.core.data_models import ProcessedData, PhotometryResult
    from fiber_photometry_ecog.preprocessing.photometry import (
        z_score_baseline, detrend_moving_average,
    )

    sk = _session_key(session)
    cached = _load_cache(ds_key, strategy, sk)
    if cached is not None:
        log.info("    Strategy %s: loaded from cache", strategy)
        _zdff_hpf_cached, _dff_det_cached, dff_raw, temp_smooth, fs = cached

        # Reconstruct minimal session.processed for plotting (no fitting)
        config = PreprocessingConfig(
            photometry=PhotometryConfig(strategy=strategy),
        )
        config.photometry.baseline_end_s = session.landmarks.heating_start_time
        config.photometry.apply_hpf = False
        session.preprocessing_config = config

        dff_zscore = z_score_baseline(
            dff_raw, fs, session.landmarks.heating_start_time
        )

        # No HPF/detrend: peak detection on z-scored dF/F, but
        # prominence/amplitude still measured on raw dF/F so existing
        # B/C thresholds (in raw dF/F units) keep their calibration.
        zdff_hpf = dff_zscore
        dff_for_detection = dff_raw

        # ECoG filtering is cheap — run it for the plots
        from fiber_photometry_ecog.preprocessing.ecog import filter_ecog
        ecog_filt = filter_ecog(session.raw.ecog, fs, config.ecog)

        session.processed = ProcessedData(
            photometry=PhotometryResult(
                dff=dff_raw,
                dff_zscore=dff_zscore,
                dff_hpf=dff_zscore,
            ),
            ecog_filtered=ecog_filt,
            temperature_smooth=temp_smooth,
            time=session.raw.time,
            fs=fs,
        )

        # Populate temperature landmarks needed by analysis modules
        lm = session.landmarks
        bl_end_idx = int(round(lm.heating_start_time * fs))
        lm.baseline_temp = float(np.nanmean(temp_smooth[:bl_end_idx]))
        max_idx = int(np.nanargmax(temp_smooth))
        lm.max_temp = float(temp_smooth[max_idx])
        lm.max_temp_time = max_idx / fs
        valid_temp = temp_smooth[~np.isnan(temp_smooth)]
        lm.terminal_temp = float(valid_temp[-1]) if len(valid_temp) > 0 else np.nan
        lm.terminal_time = float(session.raw.time[-1])
        if lm.eec_time is not None:
            idx = min(int(round(lm.eec_time * fs)), len(temp_smooth) - 1)
            val = float(temp_smooth[idx])
            lm.eec_temp = None if np.isnan(val) else val
        if lm.ueo_time is not None:
            idx = min(int(round(lm.ueo_time * fs)), len(temp_smooth) - 1)
            val = float(temp_smooth[idx])
            lm.ueo_temp = None if np.isnan(val) else val
        if lm.behavioral_onset_time is not None:
            idx = min(int(round(lm.behavioral_onset_time * fs)), len(temp_smooth) - 1)
            val = float(temp_smooth[idx])
            lm.behavioral_onset_temp = None if np.isnan(val) else val

        return zdff_hpf, dff_for_detection, temp_smooth, fs

    # No cache — run full preprocessing
    config = PreprocessingConfig(
        photometry=PhotometryConfig(strategy=strategy),
    )
    config.photometry.baseline_end_s = session.landmarks.heating_start_time
    config.photometry.apply_hpf = False
    session.preprocessing_config = config

    preprocess_session(session, config)

    proc = session.processed
    fs = proc.fs
    temp_smooth = proc.temperature_smooth

    dff_raw = proc.photometry.dff
    dff_zscore = proc.photometry.dff_zscore
    # No HPF/detrend: peak detection on z-scored dF/F, prominence/amplitude
    # still measured on raw dF/F so B/C thresholds keep their calibration.
    zdff_hpf = dff_zscore
    dff_for_detection = dff_raw
    proc.photometry.dff_hpf = dff_zscore

    _save_cache(ds_key, strategy, sk, zdff_hpf, dff_for_detection, dff_raw, temp_smooth, fs)
    log.info("    Strategy %s: preprocessed and cached", strategy)

    return zdff_hpf, dff_for_detection, temp_smooth, fs


def run_sweep_for_session(session, strategy, ds_key, results_base):
    """Preprocess (cached) then sweep all thresholds for one strategy."""
    zdff_hpf, dff_for_detection, temp_smooth, fs = preprocess_and_cache(
        session, strategy, ds_key
    )

    transients_by_prom = {}
    for threshold in THRESHOLDS[strategy]:
        label = _threshold_label(strategy, threshold)
        out_dir = results_base / f"results_{strategy.lower()}" / label
        out_dir.mkdir(parents=True, exist_ok=True)

        tc = _make_transient_config(strategy, threshold)
        transients = detect_transients(zdff_hpf, dff_for_detection, fs, tc, temp_smooth)
        transients_by_prom[threshold] = transients

        # Write CSV
        csv_path = out_dir / f"{session.mouse_id}_S{session.heating_session}_transients.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "peak_time_s", "peak_amplitude", "trough_amplitude",
                "peak_to_trough", "z_peak_amplitude", "z_trough_amplitude",
                "z_peak_to_trough", "half_width_s", "prominence",
                "temperature_at_peak",
            ])
            for t in transients:
                writer.writerow([
                    f"{t.peak_time:.4f}", f"{t.peak_amplitude:.6f}",
                    f"{t.trough_amplitude:.6f}", f"{t.peak_to_trough:.6f}",
                    f"{t.z_peak_amplitude:.6f}" if t.z_peak_amplitude is not None else "",
                    f"{t.z_trough_amplitude:.6f}" if t.z_trough_amplitude is not None else "",
                    f"{t.z_peak_to_trough:.6f}" if t.z_peak_to_trough is not None else "",
                    f"{t.half_width:.4f}", f"{t.prominence:.6f}",
                    f"{t.temperature_at_peak:.2f}" if t.temperature_at_peak is not None else "",
                ])

        # Plots v1 + v2
        session.transients = transients
        plot_sanity_check(session, str(out_dir), transients)
        plot_zoomed(session, str(out_dir))

        log.info("    %s thresh=%s -> %d transients", strategy, label, len(transients))

    # v3 transient review plot (all thresholds on one figure)
    strategy_dir = str(results_base / f"results_{strategy.lower()}")
    try:
        plot_transient_review(
            session, strategy_dir,
            list(THRESHOLDS[strategy]),
            transients_by_prom,
        )
    except Exception as e:
        log.error("    v3 plot failed for %s: %s", session.mouse_id, e)


def _ensure_spikes(session: Session):
    """Run spike detection if not already done (e.g. loaded from cache)."""
    if session.spikes:
        return
    if session.processed is None or session.processed.ecog_filtered is None:
        return
    exclusion_zones = []
    ueo_t = session.landmarks.ueo_time
    if ueo_t is None:
        ueo_t = getattr(session.landmarks, 'equiv_ueo_time', None)
    if ueo_t is not None:
        exclusion_zones.append((ueo_t - 30.0, float('inf')))
    session.spikes = detect_spikes(
        session.processed.ecog_filtered,
        session.processed.fs,
        session.landmarks.heating_start_time,
        exclusion_zones=exclusion_zones if exclusion_zones else None,
    )


def run_group_analysis(sessions, strategy, ds_key, results_base):
    """Run all 8 group analysis modules and generate group plots."""
    out_dir = str(results_base / f"results_{strategy.lower()}" / "group")
    config = AnalysisConfig()

    # Ensure all sessions have spikes detected
    for s in sessions:
        _ensure_spikes(s)

    # Pairing: assign control equivalent landmarks
    seizure_sessions = [s for s in sessions if s.n_seizures > 0]
    control_sessions = [s for s in sessions if s.n_seizures == 0]
    if seizure_sessions and control_sessions:
        try:
            assign_all_controls(seizure_sessions, control_sessions, mode="temperature")
            log.info("  Pairing: %d seizure, %d control", len(seizure_sessions), len(control_sessions))
        except Exception as e:
            log.error("  Pairing failed: %s", e)

    # Group by cohort
    cohort_groups = {}
    for s in sessions:
        if s.n_seizures > 0:
            key = "seizure"
        elif s.genotype == "Scn1a":
            key = "failed_seizure"
        else:
            key = "wt"
        cohort_groups.setdefault(key, []).append(s)

    # Global mean seizure duration: used as the AUC integration window for the
    # ictal-mean triggered averages. Per spec D22, the AUC should reflect the
    # actual seizure length rather than a fixed 60s. Computed across the
    # seizure cohort only (UEO→OFF), then applied to ALL cohorts so that the
    # AUC is comparable across cohorts.
    sz_durations = []
    for s in cohort_groups.get("seizure", []):
        lm = s.landmarks
        if (lm is not None and lm.ueo_time is not None
                and lm.off_time is not None and lm.off_time > lm.ueo_time):
            sz_durations.append(lm.off_time - lm.ueo_time)
    if sz_durations:
        global_mean_duration = float(np.mean(sz_durations))
        log.info("  Mean seizure duration (UEO->OFF, n=%d) = %.2fs -- "
                 "using as triggered-AUC window",
                 len(sz_durations), global_mean_duration)
    else:
        global_mean_duration = None
        log.info("  No seizure durations available; falling back to "
                 "config.triggered_auc_end_s = %.1fs", config.triggered_auc_end_s)

    # Per-cohort wide UEO triggered average (±150s)
    try:
        wide_ueo = {}
        for cohort_name, cohort_sessions in cohort_groups.items():
            wide_ueo[cohort_name] = compute_wide_ueo_triggered(
                cohort_sessions, window_s=config.wide_triggered_window_s)
        plot_ueo_per_cohort(wide_ueo, out_dir)
        log.info("  Per-cohort UEO triggered: done")
    except Exception as e:
        log.error("  Per-cohort UEO triggered: FAILED -- %s", e)
        traceback.print_exc()

    modules = [
        ("Cohort characteristics", compute_cohort_characteristics, plot_cohort_characteristics),
        ("Baseline transients", compute_baseline_transients, plot_baseline_transients),
        ("Pre-ictal mean", compute_preictal_mean, plot_preictal_mean),
        ("Pre-ictal transients", compute_preictal_transients, plot_preictal_transients),
        ("Ictal mean", compute_ictal_mean, plot_ictal_mean),
        ("Ictal transients", compute_ictal_transients, plot_ictal_transients),
        ("Postictal", compute_postictal, plot_postictal),
        ("Spike-triggered averages", None, plot_spike_triggered),
    ]

    for name, compute_fn, plot_fn in modules:
        try:
            results = {}
            for cohort_name, cohort_sessions in cohort_groups.items():
                if name == "Spike-triggered averages":
                    # Same exclusion as transient analyses: ECoG-to-photometry
                    # crosstalk corrupts both transient detection and the
                    # spike-triggered photometry mean (the artifact IS the
                    # signal here).
                    sta_excl = STA_EXCLUDE.get(ds_key, set())
                    sta_sessions = [s for s in cohort_sessions
                                    if s.include_for_transients
                                    and (s.mouse_id, s.heating_session) not in sta_excl]
                    n_dropped = len(cohort_sessions) - len(sta_sessions)
                    if n_dropped:
                        log.info("    STA: excluded %d session(s) from %s", n_dropped, cohort_name)
                    spike_times_list = [
                        np.array([sp.time for sp in s.spikes]) if s.spikes else np.array([])
                        for s in sta_sessions
                    ]
                    results[cohort_name] = compute_spike_triggered_average(
                        sta_sessions, spike_times_list, config)
                elif name == "Ictal mean":
                    results[cohort_name] = compute_ictal_mean(
                        cohort_sessions, config,
                        auc_end_s_override=global_mean_duration,
                    )
                else:
                    results[cohort_name] = compute_fn(cohort_sessions, config)
            plot_fn(results, out_dir)
            log.info("  Group %s: done", name)
        except Exception as e:
            log.error("  Group %s: FAILED -- %s", name, e)
            traceback.print_exc()

    # Pre-ictal mean diagnostic plot per session: bin windows + means + delta
    # annotation overlaid on the trace, so every session's values are
    # auditable by eye.
    try:
        n_done = 0
        for cohort_name, cohort_sessions in cohort_groups.items():
            for s in cohort_sessions:
                lm = s.landmarks
                if lm is None or lm.heating_start_time is None:
                    continue
                if lm.ueo_time is None and getattr(lm, "equiv_ueo_time", None) is None:
                    continue
                p = plot_preictal_mean_diagnostic(s, out_dir)
                if p:
                    n_done += 1
        log.info("  Pre-ictal diagnostic: wrote %d session plots", n_done)
    except Exception as e:
        log.error("  Pre-ictal diagnostic: FAILED -- %s", e)
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Threshold sweep for transient detection")
    parser.add_argument("--test", action="store_true",
                        help="Run on 1 file per dataset only")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete preprocessing cache and recompute")
    parser.add_argument("--refresh-temp", action="store_true",
                        help="Recompute only temperature in cached .npz files (fast)")
    args = parser.parse_args()

    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        log.info("Cleared preprocessing cache at %s", CACHE_DIR)

    if args.refresh_temp:
        from fiber_photometry_ecog.preprocessing.temperature import process_temperature
        from fiber_photometry_ecog.core.config import TemperatureConfig
        log.info("Refreshing temperature in cached .npz files ...")
        for ds_key, ds_path in DATASETS.items():
            sessions = load_sessions(ds_path, test_mode=args.test)
            for session in sessions:
                sk = _session_key(session)
                for strategy in STRATEGIES:
                    path = _cache_path(ds_key, strategy, sk)
                    if not path.exists():
                        continue
                    data = dict(np.load(path))
                    fs = float(data["fs"])
                    raw = session.raw
                    slope = raw.temp_slope if raw.temp_slope is not None else None
                    intercept = raw.temp_intercept if raw.temp_intercept is not None else None
                    tr = process_temperature(
                        raw.temperature_raw, raw.temp_bit_volts, fs,
                        TemperatureConfig(), slope, intercept,
                    )
                    data["temp_smooth"] = tr.temperature_smooth
                    np.savez(path, **data)
                    log.info("  Patched temp: %s / %s / %s", ds_key, strategy, sk)
        log.info("Temperature refresh complete.")

    results_root = Path("results")

    for ds_key, ds_path in DATASETS.items():
        log.info("=" * 60)
        log.info("DATASET: %s (%s)", ds_key.upper(), ds_path)
        log.info("=" * 60)

        sessions = load_sessions(ds_path, test_mode=args.test)
        if not sessions:
            log.warning("No sessions loaded for %s, skipping", ds_key)
            continue

        results_base = results_root / f"results_{ds_key}"

        for strategy in STRATEGIES:
            log.info("-" * 40)
            log.info("Strategy %s", strategy)
            log.info("-" * 40)

            for i, session in enumerate(sessions):
                log.info("  [%d/%d] %s (strategy %s)",
                         i + 1, len(sessions), session.mouse_id, strategy)
                try:
                    run_sweep_for_session(session, strategy, ds_key, results_base)
                except Exception as e:
                    log.error("  FAILED %s strategy %s: %s",
                              session.mouse_id, strategy, e)
                    traceback.print_exc()

            # Group analysis with default threshold for this strategy
            log.info("  Running group analysis for strategy %s ...", strategy)
            try:
                # Ensure all sessions are preprocessed for this strategy
                # (run_sweep_for_session already did this, but sessions may
                # have been modified by sweep iterations — re-apply default
                # threshold transients)
                GROUP_THRESHOLDS = {
                    "A": _make_transient_config("A", 2.5),
                    "B": _make_transient_config("B", 0.03),
                    "C": _make_transient_config("C", 0.03),
                }
                default_tc = GROUP_THRESHOLDS[strategy]
                for session in sessions:
                    zdff_hpf, dff_for_det, temp_smooth, fs = preprocess_and_cache(
                        session, strategy, ds_key)
                    if session.transient_prominence is not None:
                        tc = _make_transient_config(strategy, session.transient_prominence)
                    else:
                        tc = default_tc
                    session.transients = detect_transients(
                        zdff_hpf, dff_for_det, fs, tc, temp_smooth)
                run_group_analysis(sessions, strategy, ds_key, results_base)
            except Exception as e:
                log.error("  Group analysis FAILED for strategy %s: %s", strategy, e)
                traceback.print_exc()

    log.info("DONE. Results in %s", results_root)


if __name__ == "__main__":
    main()
