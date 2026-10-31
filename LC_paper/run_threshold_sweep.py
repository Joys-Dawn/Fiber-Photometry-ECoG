"""
Transient-threshold sweep with strategy C by default; A, B, and D are optional.

Preprocessed signals (dff, zdff_hpf, temperature) are cached per
strategy and session so subsequent runs reuse preprocessing.

Usage:
    python -m LC_paper.run_threshold_sweep
    python -m LC_paper.run_threshold_sweep --test
    python -m LC_paper.run_threshold_sweep --clear-cache
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("agg")
import numpy as np

from fiber_photometry_ecog.core.config import (
    AnalysisConfig,
    PhotometryConfig,
    PreprocessingConfig,
    TransientConfig,
)
from fiber_photometry_ecog.core.cohort import classify_session
from fiber_photometry_ecog.core.data_models import RawData, Session, SessionLandmarks
from fiber_photometry_ecog.data_loading import (
    read_ppd, read_oep, synchronize,
    scan_experiment_folder, extract_date_from_oep, read_data_log,
)
from fiber_photometry_ecog.preprocessing.pipeline import preprocess_session
from fiber_photometry_ecog.preprocessing.photometry import strategy_folder_name
from fiber_photometry_ecog.preprocessing.spike_detection import detect_spikes
from fiber_photometry_ecog.preprocessing.transient_detection import detect_transients
from fiber_photometry_ecog.core.output_layout import ensure_layout
from fiber_photometry_ecog.visualization.trace_plots import plot_sanity_check, plot_zoomed, plot_transient_review
from fiber_photometry_ecog.analysis.cohort_characteristics import compute_cohort_characteristics
from fiber_photometry_ecog.analysis.baseline_transients import compute_baseline_transients
from fiber_photometry_ecog.analysis.preictal_mean import compute_preictal_mean
from fiber_photometry_ecog.analysis.preictal_transients import compute_preictal_transients
from fiber_photometry_ecog.analysis.ictal_mean import (
    TriggeredAverage,
    compute_ictal_mean,
    compute_wide_ueo_triggered,
    mean_seizure_duration,
)
from fiber_photometry_ecog.analysis.ictal_transients import compute_ictal_transients
from fiber_photometry_ecog.analysis.postictal import compute_postictal
from fiber_photometry_ecog.analysis.spike_triggered import compute_spike_triggered_average
from fiber_photometry_ecog.analysis.opto_response import compute_opto_response
from fiber_photometry_ecog.pairing.engine import assign_all_controls
from fiber_photometry_ecog.visualization.group_plots import (
    plot_cohort_characteristics,
    plot_baseline_transients,
    plot_preictal_mean,
    plot_preictal_mean_diagnostic,
    plot_preictal_transients,
    plot_ictal_mean,
    plot_ueo_aligned_heatmaps,
    write_values_means,
    plot_experimental_vs_isosbestic_spike_triggered,
    plot_ictal_transients,
    plot_opto_response,
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

ALL_STRATEGIES = ["A", "B", "C", "D"]
DEFAULT_STRATEGIES = ["C"]  # IRLS only by default; opt in to others via --strategies

THRESHOLDS = {
    "A": [1.0, 1.5, 2.0, 2.5, 3.0],
    "B": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035],
    "C": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035],
    "D": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035],
}

# Default prominence per strategy, used for the final quality-check plots
# and for group analysis when a session has no chosen prominence.
DEFAULT_PROMINENCES = {"A": 2.5, "B": 0.03, "C": 0.03, "D": 0.03}


def _default_prominence(strategy: str) -> float:
    return DEFAULT_PROMINENCES[strategy.upper()]

CACHE_DIR = Path(__file__).resolve().parent / "sweep_cache"

# Mice excluded from spike-triggered analysis: ECoG-to-photometry
# electrical crosstalk visible in both 470 and 405 channels at spike time.
STA_EXCLUDE = {
    "dg": {("2641", 1), ("3154", 1)},       # seizure
    "lc": {("3456", 1)},                      # failed_seizure
}


# ---------------------------------------------------------------------------
# Preprocessed signal cache
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

def load_sessions(experiment_dir: Path, test_mode: bool = False,
                  has_temperature: bool = True):
    """Scan, read data log, and load all raw sessions.

    `has_temperature` is passed to `read_oep`. Set it to False when no
    thermistor was connected, as in PTZ recordings, to represent temperature
    as NaN rather than interpreting an unrecorded ADC input.
    """
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
        elif log_lookup and any(m == d["mouse_id"] for (m, _) in log_lookup):
            # Opto logs may provide one row per mouse rather than per session.
            mouse_key = next(k for k in log_lookup if k[0] == d["mouse_id"])
            d.update(log_lookup[mouse_key])
            d["date"] = d.get("date")  # keep the per-session date
        else:
            cohort = d["cohort"]
            d["genotype"] = _genotype_from_cohort(cohort)
            d["seizure"] = 1 if "seizure" in cohort.lower() and "no" not in cohort.lower() else 0
            d["sudep"] = False
            d["include"] = True
            d["exclusion_reason"] = None
            d["heating_start"] = None

    candidates = [d for d in discovered if d["include"]]
    broken = []
    for d in candidates:
        reasons = []
        if not d["oep_path"]:
            reasons.append("missing oep_path")
        if not d["ppd_path"]:
            reasons.append("missing ppd_path")
        if (d.get("heating_start") is None
                and d.get("stim_channel") is None):
            reasons.append("no heating_start and no stim_channel "
                           "(no landmark anchor)")
        if reasons:
            broken.append((d["mouse_id"], d.get("date"), reasons))
    if broken:
        raise ValueError(
            f"load_sessions: {len(broken)} session(s) marked include=True "
            f"but unloadable -- silent skip disabled. Either fix the data "
            f"or mark the row include=False in the data log. Broken: "
            f"{broken}"
        )
    loadable = candidates

    if test_mode:
        loadable = loadable[:1]
        log.info("TEST MODE: limited to 1 session")

    sessions = []
    for i, d in enumerate(loadable):
        mouse_id = d["mouse_id"]
        log.info("[%d/%d] Loading %s ...", i + 1, len(loadable), mouse_id)
        ppd = read_ppd(d["ppd_path"])
        oep = read_oep(
            d["oep_path"],
            ecog_channel=d.get("ecog_channel") or 2,
            emg_channel=d.get("emg_channel") if d.get("emg_channel") is not None else 3,
            temperature_channel=d.get("temperature_channel"),
            has_temperature=has_temperature,
            stim_channel=d.get("stim_channel"),
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
            stim_raw=sync.stim,
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
        # Opto sessions group by the disk subfolder name (e.g. "10HZ_DG_CPP"),
        # not the DS/WT × seized analytical bucket.
        if d.get("stim_channel") is not None:
            session.cohort_override = d["cohort"]
        session.date = d.get("date")
        session.session_name = d.get("session_name")
        sessions.append(session)

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
        z_score_baseline,
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

        # Recompute baseline z-scores from cached raw dF/F.
        dff_zscore = z_score_baseline(
            dff_raw, fs, session.landmarks.heating_start_time
        )

        # No detrending: detect on z-scores; measure amplitude on fractional dF/F.
        zdff_hpf = dff_zscore
        dff_for_detection = dff_raw

        # ECoG filtering is recomputed on cache hits.
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

        # Populate temperature landmarks only when temperature data is available.
        lm = session.landmarks
        baseline_window_s = lm.baseline_window_s
        if baseline_window_s is None and lm.heating_start_time is not None:
            baseline_window_s = (0.0, float(lm.heating_start_time))

        if baseline_window_s is not None:
            bl_start_idx = max(0, int(baseline_window_s[0] * fs))
            bl_end_idx = min(
                len(temp_smooth),
                int(baseline_window_s[1] * fs),
            )
        else:
            bl_start_idx = 0
            bl_end_idx = min(
                len(temp_smooth),
                int(config.temperature.baseline_duration_s * fs),
            )

        if bl_end_idx <= bl_start_idx:
            bl_start_idx = 0
            bl_end_idx = min(
                len(temp_smooth),
                int(config.temperature.baseline_duration_s * fs),
            )

        all_nan_temp = bool(np.all(np.isnan(temp_smooth)))
        if all_nan_temp:
            lm.baseline_temp = float("nan")
            lm.max_temp = float("nan")
            lm.max_temp_time = float("nan")
            lm.terminal_temp = float("nan")
        else:
            lm.baseline_temp = float(np.nanmean(
                temp_smooth[bl_start_idx:bl_end_idx]
            ))
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

    # Cache miss: run full preprocessing.
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
    # No detrending: detect on z-scores; measure amplitude on fractional dF/F.
    zdff_hpf = dff_zscore
    dff_for_detection = dff_raw
    proc.photometry.dff_hpf = dff_zscore

    _save_cache(ds_key, strategy, sk, zdff_hpf, dff_for_detection, dff_raw, temp_smooth, fs)
    log.info("    Strategy %s: preprocessed and cached", strategy)

    return zdff_hpf, dff_for_detection, temp_smooth, fs


def run_sweep_for_session(session, strategy, ds_key, results_base):
    """Preprocess (cached) then sweep all thresholds for one strategy.

    Layout produced:
        preprocessing_<METHOD>/
            quality check/
                <mouse>_S<n>_full_trace.png    (at default prominence)
                <mouse>_S<n>_peri_ictal.png    (at default prominence)
                    <mouse>_S<n>_transients.png    (all thresholds)
                _sweep/<prom_label>/
                    <mouse>_S<n>_transients.csv
                    <mouse>_S<n>_full_trace.png
                    <mouse>_S<n>_peri_ictal.png
    """
    zdff_hpf, dff_for_detection, temp_smooth, fs = preprocess_and_cache(
        session, strategy, ds_key
    )

    strategy_root = results_base / strategy_folder_name(strategy)
    layout = ensure_layout(strategy_root)
    qc_root = layout["quality_check"]
    sweep_root = qc_root / "_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    transients_by_prom = {}
    for threshold in THRESHOLDS[strategy]:
        label = _threshold_label(strategy, threshold)
        out_dir = sweep_root / label
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

        # Per-threshold quality check plots (audit trail)
        session.transients = transients
        plot_sanity_check(session, str(out_dir), transients)
        plot_zoomed(session, str(out_dir))

        log.info("    %s thresh=%s -> %d transients", strategy, label, len(transients))

    # Final per-session quality check plots at the chosen/default prominence.
    chosen_prom = (session.transient_prominence
                   if session.transient_prominence is not None
                   else _default_prominence(strategy))
    chosen_tc = _make_transient_config(strategy, chosen_prom)
    final_transients = detect_transients(zdff_hpf, dff_for_detection, fs,
                                         chosen_tc, temp_smooth)
    session.transients = final_transients
    plot_sanity_check(session, str(qc_root), final_transients)
    plot_zoomed(session, str(qc_root))

    # Overlay all transient thresholds in the quality-check directory.
    try:
        plot_transient_review(
            session, str(qc_root),
            list(THRESHOLDS[strategy]),
            transients_by_prom,
        )
    except Exception as e:
        log.error("    transient-review plot failed for %s: %s", session.mouse_id, e)


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


def run_group_analysis(sessions, strategy, ds_key, results_base,
                        stats_name_prefix="",
                        ueo_y_lim=None, ueo_y_ticks=None, run_stats=False,
                        ueo_standard_window_s=None,
                        ueo_heatmap_vlim_override=None,
                        ueo_heatmap_diagnostics=None,
                        cohort_classifier=None,
                        trace_cohort_classifier=None,
                        module_cohort_classifiers=None,
                        scatter_options=None,
                        module_scatter_options=None,
                        stats_config=None,
                        module_stats_configs=None,
                        ueo_xlabel="Time rel. to seizure onset (s)",
                        ueo_title_suffix="aligned to seizure onset",
                        is_opto=False,
                        ueo_filename_prefix="UEO_aligned",
                        ueo_dff_label=None,
                        ueo_trace_colors=None,
                        ueo_axes_size_inches=None,
                        ueo_yscale_exclude=None,
                        ueo_heatmap_exclude_from_vlim=None,
                        within_cohort_paired_periods=None,
                        ictal_auc_per_landmark_override=None,
                        ictal_auc_use_individual_seizure_durations=False,
                        ictal_auc_clip_to_available=False,
                        ictal_positive_auc_landmarks=None,
                        ictal_extrema_min_duration_s=None):
    """Write group plots/CSVs and return {cohort: {module: group_result}}.

    Classifiers take a Session and return a cohort key. `cohort_classifier`
    defaults to `classify_session`; `trace_cohort_classifier` defaults to it.
    `module_cohort_classifiers` maps module names to scalar-result classifiers.
    `module_scatter_options` and `module_stats_configs` map module names to
    option dicts and StatsConfig objects, replacing `scatter_options` and
    `stats_config` for that module. Module keys are analysis names such as
    "Pre-ictal mean".

    `scatter_options` accepts connect_pairs, pair_line_color, pair_line_alpha,
    and point_colors_by_id: {session_id or (cohort, session_id): color}, where
    session_id is (mouse_id, heating_session). `ueo_trace_colors` is
    {cohort: color}.

    `within_cohort_paired_periods` is a list of dicts requiring module,
    period_a, period_b, period_a_label, and period_b_label. Periods name scalar
    measurements; labels provide output text. These tests require run_stats=True
    and are appended to the mean-signal statistics CSV.

    `ueo_standard_window_s` is (start, end) in seconds for traces and heatmaps;
    `ueo_axes_size_inches` is their standard-panel data-area (width, height).
    `ueo_heatmap_vlim_override` sets symmetric color limits (-vlim, +vlim).
    `ueo_heatmap_diagnostics` is populated with the key "vlim_standard".
    The two scale-exclusion lists omit cohorts from autoscaling, not plotting.

    `ictal_auc_per_landmark_override` is {landmark: seconds} and takes precedence
    over duration-based windows. OFF defaults to 100 s; other landmarks use the
    mean duration. Individual-duration mode uses each seizure's duration and
    the seizure-group mean for controls. `ictal_positive_auc_landmarks` names
    landmarks using positive-only AUC; other ictal options follow
    `compute_ictal_mean`.
    """
    classifier = cohort_classifier or classify_session
    trace_classifier = trace_cohort_classifier or classifier
    module_cohort_classifiers = module_cohort_classifiers or {}
    module_scatter_options = module_scatter_options or {}
    strategy_root = results_base / strategy_folder_name(strategy)
    layout = ensure_layout(strategy_root)
    means_dir = str(layout["means"])
    transients_dir = str(layout["transients"])
    param_dir = str(layout["parameter_detection"])
    config = AnalysisConfig()

    # Ensure all sessions have spikes detected
    for s in sessions:
        _ensure_spikes(s)

    # Pairing: assign control equivalent landmarks
    seizure_sessions = [s for s in sessions if s.n_seizures > 0]
    control_sessions = [s for s in sessions if s.n_seizures == 0]
    if seizure_sessions and control_sessions:
        assign_all_controls(seizure_sessions, control_sessions, mode="temperature")
        log.info("  Pairing: %d seizure, %d control", len(seizure_sessions), len(control_sessions))

    def _group_sessions(active_classifier):
        grouped = {}
        for session in sessions:
            grouped.setdefault(active_classifier(session), []).append(session)
        return grouped

    # Scalar and trace analyses support separate cohort classifiers.
    trace_cohort_groups = _group_sessions(trace_classifier)

    # Mean seizure duration is the default AUC window. Session-specific
    # durations and per-landmark overrides can replace it. Duration estimation
    # requires at least one session with valid onset and offset landmarks.
    auc_end_s_by_session = None
    if ictal_auc_use_individual_seizure_durations:
        seizure_durations = [
            float(s.landmarks.off_time - s.landmarks.ueo_time)
            for s in sessions
            if (s.n_seizures > 0
                and s.landmarks is not None
                and s.landmarks.ueo_time is not None
                and s.landmarks.off_time is not None
                and s.landmarks.off_time > s.landmarks.ueo_time)
        ]
        if not seizure_durations:
            raise ValueError(
                "Individual seizure-duration AUCs require at least one "
                "seizure session with valid UEO and OFF landmarks."
            )
        global_mean_duration = float(np.mean(seizure_durations))
        auc_end_s_by_session = {}
        for session in sessions:
            lm = session.landmarks
            if (session.n_seizures > 0
                    and lm is not None
                    and lm.ueo_time is not None
                    and lm.off_time is not None
                    and lm.off_time > lm.ueo_time):
                window_s = float(lm.off_time - lm.ueo_time)
            else:
                window_s = global_mean_duration
            auc_end_s_by_session[
                (session.mouse_id, session.date, session.heating_session)
            ] = window_s
        log.info(
            "  Triggered AUCs use individual seizure durations; "
            "no-seizure sessions use the %.2fs seizure-group mean",
            global_mean_duration,
        )
    else:
        global_mean_duration = mean_seizure_duration(sessions)
        log.info("  Mean seizure duration (UEO->OFF) = %.2fs -- "
                 "using as triggered-AUC window", global_mean_duration)

    # Compute standard and extended UEO-aligned averages separately.
    # Incomplete windows are NaN-padded; pointwise means and SEM use the
    # available observations, so contributing session counts can vary.
    wide_ueo: Dict[str, TriggeredAverage] = {}
    wide_ueo_extended: Dict[str, TriggeredAverage] = {}
    for cohort_name, cohort_sessions in trace_cohort_groups.items():
        wide_ueo[cohort_name] = compute_wide_ueo_triggered(
            cohort_sessions,
            window_s=config.wide_triggered_window_s,
        )
        wide_ueo_extended[cohort_name] = compute_wide_ueo_triggered(
            cohort_sessions,
            window_s=config.wide_triggered_window_s,
            post_window_s=1000.0,
        )
    plot_ueo_per_cohort(wide_ueo, means_dir,
                        results_extended=wide_ueo_extended,
                        y_lim_override=ueo_y_lim,
                        y_ticks_override=ueo_y_ticks,
                        standard_window_s=ueo_standard_window_s,
                        xlabel=ueo_xlabel,
                        title_suffix=ueo_title_suffix,
                        filename_prefix=ueo_filename_prefix,
                        dff_label=ueo_dff_label,
                        yscale_exclude=ueo_yscale_exclude,
                        trace_colors=ueo_trace_colors,
                        standard_axes_size_inches=ueo_axes_size_inches)
    log.info("  Per-cohort UEO triggered: done")

    # (analysis_name, compute_fn, plot_fn, destination_dir)
    modules = [
        ("Cohort characteristics", compute_cohort_characteristics, plot_cohort_characteristics, means_dir),
        ("Baseline transients", compute_baseline_transients, plot_baseline_transients, transients_dir),
        ("Pre-ictal mean", compute_preictal_mean, plot_preictal_mean, means_dir),
        ("Pre-ictal transients", compute_preictal_transients, plot_preictal_transients, transients_dir),
        ("Ictal mean", compute_ictal_mean, plot_ictal_mean, means_dir),
        ("Ictal transients", compute_ictal_transients, plot_ictal_transients, transients_dir),
        ("Postictal", compute_postictal, plot_postictal, means_dir),
        ("Spike-triggered averages", None, plot_spike_triggered, means_dir),
        ("Opto response", compute_opto_response, plot_opto_response, means_dir),
    ]

    sta_cohort_sessions: Dict[str, list] = {}
    sta_spike_times: Dict[str, list] = {}
    # Captured for the optional stats pass. Shape matches what the stats
    # runner expects: {cohort_key: {module_name: GroupResult}}.
    cohort_module_results: Dict[str, Dict[str, Any]] = {}

    def _same_groups(left, right):
        if left.keys() != right.keys():
            return False
        return all(
            [id(s) for s in left[key]] == [id(s) for s in right[key]]
            for key in left
        )

    def _compute_module(name, compute_fn, groups):
        computed = {}
        sta_sessions_by_cohort = {}
        sta_spikes_by_cohort = {}
        for cohort_name, cohort_sessions in groups.items():
            if name == "Spike-triggered averages":
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
                sta_sessions_by_cohort[cohort_name] = sta_sessions
                sta_spikes_by_cohort[cohort_name] = spike_times_list
                computed[cohort_name] = compute_spike_triggered_average(
                    sta_sessions, spike_times_list, config)
            elif name == "Ictal mean":
                # OFF defaults to 100 s; explicit landmark overrides take
                # precedence over that default and the mean-duration window.
                per_lm = {"OFF": 100.0}
                if ictal_auc_per_landmark_override:
                    per_lm.update(ictal_auc_per_landmark_override)
                computed[cohort_name] = compute_ictal_mean(
                    cohort_sessions, auc_end_s=global_mean_duration,
                    config=config,
                    auc_end_s_per_landmark=per_lm,
                    auc_end_s_by_session=auc_end_s_by_session,
                    clip_auc_to_available=ictal_auc_clip_to_available,
                    positive_auc_landmarks=ictal_positive_auc_landmarks,
                    extrema_min_duration_s=ictal_extrema_min_duration_s,
                )
            else:
                computed[cohort_name] = compute_fn(cohort_sessions, config)
        return computed, sta_sessions_by_cohort, sta_spikes_by_cohort

    trace_modules = {
        "Pre-ictal mean", "Pre-ictal transients", "Ictal mean",
        "Ictal transients", "Postictal", "Spike-triggered averages",
    }
    scatter_modules = {
        "Cohort characteristics", "Baseline transients", "Pre-ictal mean",
        "Ictal mean", "Postictal", "Spike-triggered averages",
        "Opto response",
    }

    for name, compute_fn, plot_fn, dest_dir in modules:
        module_classifier = module_cohort_classifiers.get(name, classifier)
        module_groups = _group_sessions(module_classifier)
        results, scalar_sta_sessions, scalar_sta_spikes = _compute_module(
            name, compute_fn, module_groups
        )
        trace_results = None
        if name in trace_modules:
            if _same_groups(module_groups, trace_cohort_groups):
                trace_results = results
                trace_sta_sessions = scalar_sta_sessions
                trace_sta_spikes = scalar_sta_spikes
            else:
                trace_results, trace_sta_sessions, trace_sta_spikes = (
                    _compute_module(name, compute_fn, trace_cohort_groups)
                )
            if name == "Spike-triggered averages":
                sta_cohort_sessions = trace_sta_sessions
                sta_spike_times = trace_sta_spikes

        accepts_prefix = name in (
            "Cohort characteristics", "Pre-ictal mean", "Ictal mean",
            "Postictal", "Spike-triggered averages",
        )
        plot_kwargs = {}
        if accepts_prefix:
            plot_kwargs["name_prefix"] = stats_name_prefix
        if name in scatter_modules:
            plot_kwargs["scatter_options"] = module_scatter_options.get(
                name, scatter_options
            )
        if name in trace_modules:
            plot_kwargs["trace_results"] = trace_results
        if name == "Ictal mean":
            # Opto renames UEO → "Stim on" for the AUC standalone filename
            # slug (AUC_stim_on_*.png), the trace-panel title, and the
            # values_auc CSV column ("Stim on AUC (40s)").
            plot_kwargs["landmark_labels"] = (
                {"UEO": "Stim on"} if is_opto else None
            )
        if name in {"Ictal mean", "Spike-triggered averages", "Opto response"}:
            plot_kwargs["dff_label"] = ueo_dff_label
        plot_fn(results, dest_dir, **plot_kwargs)
        log.info("  Group %s: done", name)
        for cohort_name, group_result in results.items():
            cohort_module_results.setdefault(cohort_name, {})[name] = group_result

    # Consolidated per-session means CSV: pulls Pre-ictal / Ictal / Postictal
    # session results from cohort_module_results into one wide values_means_<prefix>.csv.
    vm_path = write_values_means(cohort_module_results, means_dir,
                                 name_prefix=stats_name_prefix)
    if vm_path:
        log.info("  values_means CSV: %s", vm_path)

    if trace_cohort_groups:
        if ueo_standard_window_s is not None:
            hm_lo, hm_hi = ueo_standard_window_s
            heatmap_pre = abs(float(hm_lo))
            heatmap_post = float(hm_hi)
        else:
            heatmap_pre = 150.0
            heatmap_post = 150.0
        paths = plot_ueo_aligned_heatmaps(
            trace_cohort_groups, means_dir,
            pre_s=heatmap_pre, post_s=heatmap_post,
            vlim_override=ueo_heatmap_vlim_override,
            diagnostics_out=ueo_heatmap_diagnostics,
            filename_prefix=ueo_filename_prefix,
            xlabel=ueo_xlabel,
            dff_label=ueo_dff_label,
            exclude_from_vlim=ueo_heatmap_exclude_from_vlim,
            standard_axes_size_inches=ueo_axes_size_inches,
        )
        for c, p in paths.items():
            log.info("  UEO heatmap [%s]: %s", c, p)

    if sta_cohort_sessions:
        p = plot_experimental_vs_isosbestic_spike_triggered(
            sta_cohort_sessions, sta_spike_times, means_dir
        )
        if p:
            log.info("  470/405 spike overlay: %s", p)

    # Per-session diagnostics show measurement windows, means, and extrema.
    n_done = 0
    for cohort_name, cohort_sessions in trace_cohort_groups.items():
        for s in cohort_sessions:
            lm = s.landmarks
            if lm is None or lm.heating_start_time is None:
                continue
            if lm.ueo_time is None and getattr(lm, "equiv_ueo_time", None) is None:
                continue
            p = plot_preictal_mean_diagnostic(
                s, param_dir,
                is_opto=is_opto,
                dff_label=ueo_dff_label,
                extrema_min_duration_s=ictal_extrema_min_duration_s,
            )
            if p:
                n_done += 1
    log.info("  Parameter detection: wrote %d session plots", n_done)

    # Optional unpaired comparisons use the supplied statistics configuration
    # and write tables beside the mean-signal and transient plots.
    if run_stats:
        from fiber_photometry_ecog.statistics import (
            run_group_comparisons,
            run_within_cohort_paired_periods,
        )
        # Build the extra within-cohort paired rows (if any) before invoking
        # the between-cohort runner. They share the CSV so all stats for the
        # dataset live in one file.
        additional_rows_by_folder: dict[str, list] = {}
        for spec in (within_cohort_paired_periods or []):
            extra = run_within_cohort_paired_periods(
                cohort_module_results,
                module=spec["module"],
                period_a=spec["period_a"],
                period_b=spec["period_b"],
                period_a_label=spec["period_a_label"],
                period_b_label=spec["period_b_label"],
            )
            # Configured paired-period tests use mean-signal measurements.
            additional_rows_by_folder.setdefault("means", []).extend(extra)
            log.info("  Within-cohort paired [%s: %s vs %s]: %d cohort rows",
                     spec["module"], spec["period_a_label"],
                     spec["period_b_label"], len(extra))
        out_paths = run_group_comparisons(
            cohort_module_results,
            means_dir=means_dir,
            transients_dir=transients_dir,
            name_prefix=stats_name_prefix,
            config=stats_config,
            module_configs=module_stats_configs,
            additional_rows_by_folder=additional_rows_by_folder or None,
        )
        for folder, p in out_paths.items():
            log.info("  Stats [%s]: %s", folder, p)

    # Return module results for additional caller-specific comparisons.
    return cohort_module_results


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
    parser.add_argument(
        "--strategies", default=",".join(DEFAULT_STRATEGIES),
        help=(f"Comma-separated subset of {ALL_STRATEGIES} to run. "
              f"Default: {DEFAULT_STRATEGIES} (IRLS only)."),
    )
    args = parser.parse_args()

    strategies = [s.strip().upper() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in ALL_STRATEGIES]
    if unknown:
        parser.error(f"Unknown strategies: {unknown}. Choose from {ALL_STRATEGIES}.")
    log.info("Running strategies: %s", strategies)

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
                for strategy in strategies:
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

    results_root = Path(__file__).resolve().parent / "results"

    for ds_key, ds_path in DATASETS.items():
        log.info("=" * 60)
        log.info("DATASET: %s (%s)", ds_key.upper(), ds_path)
        log.info("=" * 60)

        sessions = load_sessions(ds_path, test_mode=args.test)
        if not sessions:
            log.warning("No sessions loaded for %s, skipping", ds_key)
            continue

        results_base = results_root / f"results_{ds_key}"

        for strategy in strategies:
            log.info("-" * 40)
            log.info("Strategy %s", strategy)
            log.info("-" * 40)

            for i, session in enumerate(sessions):
                log.info("  [%d/%d] %s (strategy %s)",
                         i + 1, len(sessions), session.mouse_id, strategy)
                run_sweep_for_session(session, strategy, ds_key, results_base)

            # Restore the default threshold after the sweep.
            log.info("  Running group analysis for strategy %s ...", strategy)
            default_tc = _make_transient_config(strategy, _default_prominence(strategy))
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

    log.info("DONE. Results in %s", results_root)


if __name__ == "__main__":
    main()
