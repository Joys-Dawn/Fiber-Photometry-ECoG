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

# Default prominence per strategy — used for the final quality-check plots
# and for group analysis when a session has no chosen prominence.
DEFAULT_PROMINENCES = {"A": 2.5, "B": 0.03, "C": 0.03, "D": 0.03}


def _default_prominence(strategy: str) -> float:
    return DEFAULT_PROMINENCES[strategy.upper()]

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

def load_sessions(experiment_dir: Path, test_mode: bool = False,
                  has_temperature: bool = True):
    """Scan, read data log, and load all raw sessions.

    `has_temperature` is passed through to `read_oep` — set False for the
    PTZ-evoked paradigm (no thermistor connected), and downstream temperature
    analyses will produce NaN values rather than reading garbage from a
    floating ADC1 input.
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
            # Opto logs commonly have one row per mouse (covering baseline and
            # paired post-treatment sessions on different dates). Fall back to
            # any matching mouse row so the per-mouse settings (stim_channel,
            # include flag, exclusion reason) carry across to later dates.
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

    # Opto sessions have no heating phase; the stim onset gets auto-detected
    # later by preprocess_session and used as the baseline-end marker.
    #
    # Sessions are split into:
    #   - excluded_by_log: include flag = False (the only legitimate "drop")
    #   - loadable:        include=True AND has files AND has landmark anchor
    #   - broken:          include=True but missing files or landmarks -> CRASH.
    # We refuse to silently drop sessions marked include=True.
    excluded_by_log = [d for d in discovered if not d["include"]]
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

        # Match pipeline.preprocess_session: every strategy uses baseline-
        # window z-score (against the pre-heating window for hyperthermia,
        # the first 100 s for opto). The cache stores raw dF/F, so changing
        # the z-score formula doesn't invalidate cached files -- only the
        # recompute logic here needs to mirror the pipeline.
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

        # Populate temperature landmarks needed by analysis modules. Guard
        # the no-temperature case (PTZ paradigm: temp_smooth is all-NaN);
        # nanmean / nanargmax raise on all-NaN slices.
        lm = session.landmarks
        bl_end_idx = int(round(lm.heating_start_time * fs))
        all_nan_temp = bool(np.all(np.isnan(temp_smooth)))
        if all_nan_temp:
            lm.baseline_temp = float("nan")
            lm.max_temp = float("nan")
            lm.max_temp_time = float("nan")
            lm.terminal_temp = float("nan")
        else:
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
    """Preprocess (cached) then sweep all thresholds for one strategy.

    Layout produced:
        preprocessing_<METHOD>/
            quality check/
                <mouse>_S<n>_full_trace.png    (at default prominence)
                <mouse>_S<n>_peri_ictal.png    (at default prominence)
                <mouse>_S<n>_transients.png    (v3, all thresholds overlay)
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

    # v3 transient review (all thresholds overlay) — directly in quality check/
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
                        ueo_y_lim=None, run_stats=False,
                        ueo_standard_window_s=None,
                        ueo_heatmap_vlim_override=None,
                        ueo_heatmap_diagnostics=None,
                        cohort_classifier=None,
                        ueo_xlabel="Time rel. to seizure onset (s)",
                        ueo_title_suffix="aligned to seizure onset",
                        is_opto=False,
                        ueo_filename_prefix="UEO_aligned",
                        ueo_dff_label=None,
                        ueo_yscale_exclude=None,
                        ueo_heatmap_exclude_from_vlim=None,
                        within_cohort_paired_periods=None,
                        ictal_auc_per_landmark_override=None):
    """Run all 8 group analysis modules and generate group plots.

    `ueo_y_lim` (tuple of (lo, hi)) hard-clamps the y-axis range on the
    UEO-aligned trace plots so cohorts from different subfolders can be
    compared directly. Pass None for autoscale (default).

    `ueo_standard_window_s` / `ueo_xlabel` / `ueo_title_suffix` retitle and
    re-window the per-cohort UEO-aligned trace plots — used by the opto
    pipeline to render the stim-aligned variant (-100..+100 s, "stim onset"
    language). Defaults preserve the seizure-language behavior used by
    every non-opto pipeline. `ueo_standard_window_s` also drives the
    standard-window pre_s/post_s on `plot_ueo_aligned_heatmaps` so the
    trace plot and the matching heatmap share an x-axis.

    `ueo_heatmap_vlim_override`: forwards to `plot_ueo_aligned_heatmaps`
    for cross-subfolder color-scale matching (e.g. the mutant subfolder
    inheriting DG_GRABne's standard-window vlim). `ueo_heatmap_diagnostics`,
    if a dict, captures the auto-computed vlim from this run so a downstream
    run can reuse it as the override.

    `cohort_classifier`: override the default genotype × seized cohort
    classifier (used by the PTZ pooled run to drop every session into a
    single "PTZ_pooled" bucket). Defaults to `classify_session`.

    `is_opto=True` switches the per-session parameter-detection diagnostic
    to opto-mode labels (pre-stim baseline / stim on / stim off / post-stim
    detection window) and forces the max/min search window to 40 s post-stim
    instead of [stim_on, stim_off]. `ueo_filename_prefix` swaps the
    "UEO_aligned" prefix on the UEO trace + heatmap outputs ("stim_onset_aligned"
    for opto). `ueo_dff_label` overrides the y/colorbar label (opto raw-ΔF/F
    experiments pass "ΔF/F"). `ueo_yscale_exclude` /
    `ueo_heatmap_exclude_from_vlim` are cohort lists that get rendered but
    don't contribute to auto-y-lim / auto-vlim (e.g. Atomoxetine arm of
    DG_local_release whose dynamic range would dominate otherwise).

    `run_stats` controls the optional stats pass: when True, after all
    modules finish, the per-cohort module results are handed to
    statistics.run_group_comparisons, which writes one Stats_groups.csv
    per output folder (means / transients). Default StatsConfig compares
    DS_sz vs WT_no_sz with raw p-values — no MCC.

    `within_cohort_paired_periods`: optional list of paired-period specs
    that get appended to the same stats CSV. Each spec is a dict with
    keys ``module`` / ``period_a`` / ``period_b`` / ``period_a_label`` /
    ``period_b_label`` (e.g. ``{"module": "Pre-ictal mean",
    "period_a": "heating_mean", "period_b": "ictal_mean",
    "period_a_label": "Heating", "period_b_label": "Ictal"}``). Requires
    ``run_stats=True``. Currently used by run_hyperthermia.py to add
    within-cohort Heating-vs-Ictal paired stats to DG_GRABne.
    """
    classifier = cohort_classifier or classify_session
    strategy_root = results_base / strategy_folder_name(strategy)
    layout = ensure_layout(strategy_root)
    means_dir = str(layout["means"])
    transients_dir = str(layout["transients"])
    param_dir = str(layout["parameter_detection"])
    qc_dir = str(layout["quality_check"])
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

    # Group by cohort (genotype × seized, unless overridden).
    cohort_groups = {}
    for s in sessions:
        cohort_groups.setdefault(classifier(s), []).append(s)

    # Global mean seizure duration: used as the AUC integration window for the
    # ictal-mean triggered averages. Per spec D22, the AUC reflects actual
    # seizure length rather than a fixed window. The helper crashes if no
    # session has both ueo and off set - that's intentional, these analyses
    # are seizure-centric so it should never happen.
    global_mean_duration = mean_seizure_duration(sessions)
    log.info("  Mean seizure duration (UEO->OFF) = %.2fs -- "
             "using as triggered-AUC window", global_mean_duration)

    # Per-cohort wide UEO triggered average — TWO computations, one per
    # window. Both strict: a session only contributes to a window's mean if
    # its recording actually spans that window. This keeps the standard
    # ±150s mean from being hijacked by the 1-2 long-tailed sessions, and
    # the extended -150..+1000s mean honestly reflects whatever N can span
    # 1000s. The heatmap path (below) shows ALL sessions via NaN-padding.
    wide_ueo: Dict[str, "TriggeredAverage"] = {}
    wide_ueo_extended: Dict[str, "TriggeredAverage"] = {}
    for cohort_name, cohort_sessions in cohort_groups.items():
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
                        standard_window_s=ueo_standard_window_s,
                        xlabel=ueo_xlabel,
                        title_suffix=ueo_title_suffix,
                        filename_prefix=ueo_filename_prefix,
                        dff_label=ueo_dff_label,
                        yscale_exclude=ueo_yscale_exclude)
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
        # Opto-specific: max/min/AUC over 40 s post-stim. The compute_fn
        # checks for stim_raw on each session and returns an empty result
        # for non-opto sessions, so the module is a no-op outside the
        # opto pipeline.
        ("Opto response", compute_opto_response, plot_opto_response, means_dir),
    ]

    ictal_results_by_cohort = {}
    sta_cohort_sessions: Dict[str, list] = {}
    sta_spike_times: Dict[str, list] = {}
    # Captured for the optional stats pass. Shape matches what the stats
    # runner expects: {cohort_key: {module_name: GroupResult}}.
    cohort_module_results: Dict[str, Dict[str, "Any"]] = {
        c: {} for c in cohort_groups
    }
    for name, compute_fn, plot_fn, dest_dir in modules:
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
                sta_cohort_sessions[cohort_name] = sta_sessions
                sta_spike_times[cohort_name] = spike_times_list
                results[cohort_name] = compute_spike_triggered_average(
                    sta_sessions, spike_times_list, config)
            elif name == "Ictal mean":
                # Per spec D5: OFF AUC integrates over a fixed 100 s window
                # rather than cohort mean seizure duration. Other landmarks
                # (EEC/UEO/etc) use the global mean duration by default; opto
                # pipelines override UEO to 40 s post-stim via
                # `ictal_auc_per_landmark_override` because the default
                # (global_mean_duration = stim_off − stim_on = 10 s) would
                # only span the during-stim window, not the post-stim
                # response we actually want to quantify.
                per_lm = {"OFF": 100.0}
                if ictal_auc_per_landmark_override:
                    per_lm.update(ictal_auc_per_landmark_override)
                results[cohort_name] = compute_ictal_mean(
                    cohort_sessions, auc_end_s=global_mean_duration,
                    config=config,
                    auc_end_s_per_landmark=per_lm,
                )
            else:
                results[cohort_name] = compute_fn(cohort_sessions, config)
        # Plot functions whose CSV outputs carry the experiment name take
        # `name_prefix`; the older ones (transients modules, opto_response)
        # ignore extra kwargs by being called positionally only. Switch to
        # kwarg per-function to keep the call site uniform.
        accepts_prefix = name in (
            "Cohort characteristics", "Pre-ictal mean", "Ictal mean",
            "Postictal", "Spike-triggered averages",
        )
        if name == "Ictal mean":
            # Opto renames UEO → "Stim on" for the AUC standalone filename
            # slug (AUC_stim_on_*.png), the trace-panel title, and the
            # values_auc CSV column ("Stim on AUC (40s)").
            landmark_labels = {"UEO": "Stim on"} if is_opto else None
            plot_fn(results, dest_dir,
                    name_prefix=stats_name_prefix,
                    landmark_labels=landmark_labels)
        elif accepts_prefix:
            plot_fn(results, dest_dir, name_prefix=stats_name_prefix)
        else:
            plot_fn(results, dest_dir)
        log.info("  Group %s: done", name)
        if name == "Ictal mean":
            ictal_results_by_cohort = results
        for cohort_name, group_result in results.items():
            cohort_module_results.setdefault(cohort_name, {})[name] = group_result

    # Consolidated per-session means CSV: pulls Pre-ictal / Ictal / Postictal
    # session results from cohort_module_results into one wide values_means_<prefix>.csv.
    vm_path = write_values_means(cohort_module_results, means_dir,
                                 name_prefix=stats_name_prefix)
    if vm_path:
        log.info("  values_means CSV: %s", vm_path)

    # UEO-aligned heatmaps split by cohort (Phase 4e). Takes the cohort
    # session list directly so the heatmap can NaN-pad short-tailed rows
    # for display, without contaminating the strict mean/SEM aggregation
    # that drives the trace plot. Standard-window pre_s/post_s mirror the
    # trace plot so the two visuals share an x-axis.
    if cohort_groups:
        if ueo_standard_window_s is not None:
            hm_lo, hm_hi = ueo_standard_window_s
            heatmap_pre = abs(float(hm_lo))
            heatmap_post = float(hm_hi)
        else:
            heatmap_pre = 150.0
            heatmap_post = 150.0
        paths = plot_ueo_aligned_heatmaps(
            cohort_groups, means_dir,
            pre_s=heatmap_pre, post_s=heatmap_post,
            vlim_override=ueo_heatmap_vlim_override,
            diagnostics_out=ueo_heatmap_diagnostics,
            filename_prefix=ueo_filename_prefix,
            xlabel=ueo_xlabel,
            dff_label=ueo_dff_label,
            exclude_from_vlim=ueo_heatmap_exclude_from_vlim,
        )
        for c, p in paths.items():
            log.info("  UEO heatmap [%s]: %s", c, p)

    # Experimental (470) vs isosbestic (405) overlay
    # (AUC_interictal_spikes_experimental_vs_isosbestic). Lives under
    # plots_and_values_means/ alongside the other interictal-spikes AUC
    # outputs — mirrors app.py's routing.
    if sta_cohort_sessions:
        p = plot_experimental_vs_isosbestic_spike_triggered(
            sta_cohort_sessions, sta_spike_times, means_dir
        )
        if p:
            log.info("  470/405 spike overlay: %s", p)

    # Parameter-detection diagnostic per session: bin windows + means + delta
    # annotation overlaid on the trace, so every session's values are
    # auditable by eye.
    n_done = 0
    for cohort_name, cohort_sessions in cohort_groups.items():
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
            )
            if p:
                n_done += 1
    log.info("  Parameter detection: wrote %d session plots", n_done)

    # Optional 2-way unpaired stats (DS_sz vs WT_no_sz by default), per the
    # StatsConfig defaults. Writes Stats_groups.csv into means/ and
    # transients/ alongside the plots. Only enabled for datasets where the
    # stats are meaningful (e.g. plain DG / LC hyperthermia) — see callers.
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
            # `_extract_*` folder is looked up from the extractor registry
            # for the given module — hard-coded here because both existing
            # specs (Pre-ictal mean / Ictal mean) land in "means" and that's
            # the only folder relevant for within-cohort period tests today.
            additional_rows_by_folder.setdefault("means", []).extend(extra)
            log.info("  Within-cohort paired [%s: %s vs %s]: %d cohort rows",
                     spec["module"], spec["period_a_label"],
                     spec["period_b_label"], len(extra))
        out_paths = run_group_comparisons(
            cohort_module_results,
            means_dir=means_dir,
            transients_dir=transients_dir,
            name_prefix=stats_name_prefix,
            additional_rows_by_folder=additional_rows_by_folder or None,
        )
        for folder, p in out_paths.items():
            log.info("  Stats [%s]: %s", folder, p)

    # Expose for callers (e.g. opto driver) that want to run extra stats
    # passes — paired CPP vs baseline comparisons, etc. — using the same
    # extractor pipeline as the standard independent-groups stats.
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

        for strategy in strategies:
            log.info("-" * 40)
            log.info("Strategy %s", strategy)
            log.info("-" * 40)

            for i, session in enumerate(sessions):
                log.info("  [%d/%d] %s (strategy %s)",
                         i + 1, len(sessions), session.mouse_id, strategy)
                run_sweep_for_session(session, strategy, ds_key, results_base)

            # Group analysis with default threshold for this strategy.
            # Re-apply default-prominence transient detection (sweep
            # iterations above may have left the session with a different
            # threshold's transient list).
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
