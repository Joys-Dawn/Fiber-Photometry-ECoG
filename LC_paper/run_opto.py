"""Run strategy-C preprocessing and analysis for optogenetic experiments."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import numpy as np

from LC_paper.run_threshold_sweep import (
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


# Recording cutoffs in seconds, keyed by (cohort, mouse_id, heating_session).
TRUNCATE_AT_S = {
    # Exclude signal loss near 366 s, after the 279-289 s stimulation bout.
    ("20HZ_DG", "3799", 1): 365.0,
}


# Zero-based stimulation-bout index, keyed by (cohort, mouse_id, heating_session).
STIM_BURST_OVERRIDES = {
    # Use the second bout (325-335 s), not the first (297-306 s).
    ("1HZ_DG", "4361", 1): 1,
}


def _refresh_stim_landmarks(session) -> None:
    """Re-detect stimulation landmarks and apply session-specific overrides.

    Runs on cache hits and misses so STIM_BURST_OVERRIDES applies before
    downstream analysis. Burst detection clusters supra-threshold samples
    by their intervening gaps.
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
    # Log missing landmarks or changes greater than 0.5 s.
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
    """Truncate recordings listed in TRUNCATE_AT_S.

    Keys are (cohort_override, mouse_id, heating_session). Clips all raw
    arrays in place to the same prefix length; other sessions are unchanged.
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


OPTO_CACHE_DIR = Path(__file__).resolve().parent / "opto_cache"


def _opto_cache_path(cache_label: str, strategy: str,
                     session: Session) -> Path:
    """Per-session cache file path.

    Layout: ``opto_cache/<cache_label>/<strategy>/<mouse>_<date>_S<n>_<cohort>.npz``

    `cache_label` is the experiment subpath leaf, such as ``CPP_treatment``.
    The cohort distinguishes same-day frequency or treatment conditions
    that share mouse ID, date, and session number.
    """
    parts = [str(session.mouse_id)]
    if getattr(session, "date", None):
        parts.append(str(session.date))
    parts.append(f"S{int(session.heating_session)}")
    # Opto cohort and cohort_override both come from the data log.
    parts.append(str(session.cohort))
    fname = "_".join(parts) + ".npz"
    return OPTO_CACHE_DIR / cache_label / strategy / fname


def _opto_recompute_dff_zscore(strategy: str, dff_raw, fs: float,
                               baseline_end_s, *, zscore: bool):
    """Return raw or baseline-z-scored dF/F for optogenetic analysis.

    Strategy C with ``zscore=False`` returns raw dF/F. Otherwise, z-score
    using the mean and standard deviation from recording onset to
    ``baseline_end_s``. Cached raw dF/F is normalized on each load.
    """
    if strategy == "C" and not zscore:
        return dff_raw
    return z_score_baseline(dff_raw, fs, baseline_end_s)


def _preprocess_and_cache(session: Session, strategy: str, cache_label: str,
                          *, zscore: bool):
    """Preprocess or load cached arrays; return analysis and detection signals.

    Returns (zdff_hpf, dff_for_detection, temp_smooth, fs). Raw dF/F,
    landmarks, and smoothed temperature are cached per session and cohort.
    ``_opto_recompute_dff_zscore`` applies the strategy and ``zscore`` flag.
    """
    config = PreprocessingConfig(
        photometry=PhotometryConfig(strategy=strategy),
    )
    # apply_hpf is OFF: peak detection runs on the z-scored dF/F directly,
    # prominence/amplitude measured on raw dF/F. Matches the sweep/app default.
    config.photometry.apply_hpf = False
    session.preprocessing_config = config

    cpath = _opto_cache_path(cache_label, strategy, session)
    # Recompute caches missing arrays or landmarks required for analysis.
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
        if session.landmarks.heating_start_time is None:
            session.landmarks.heating_start_time = zbe_time

        lm = session.landmarks
        if lm.baseline_temp is None:
            lm.baseline_temp = float("nan")
        if lm.max_temp is None:
            lm.max_temp = float("nan")
        if lm.max_temp_time is None:
            lm.max_temp_time = float("nan")
        if lm.terminal_temp is None:
            lm.terminal_temp = float("nan")

        # Refresh stimulation landmarks rather than using cached values.
        _refresh_stim_landmarks(session)

        dff_zscore = _opto_recompute_dff_zscore(
            strategy, dff_raw, fs, zbe_time, zscore=zscore,
        )

        # ECoG filtering is recomputed on cache hits.
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

    if (strategy == "C" and not zscore and proc is not None
            and proc.photometry is not None):
        proc.photometry.dff_zscore = proc.photometry.dff
        proc.photometry.dff_hpf = proc.photometry.dff

    _refresh_stim_landmarks(session)

    cpath.parent.mkdir(parents=True, exist_ok=True)
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
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
STRATEGY = "C"
THRESHOLD = 0.030  # Strategy-C default prominence (matches DG_GRABne)

# Tuple fields: experiment path, cache key, paired cohorts, z-score flag,
# dF/F label, cohorts excluded from automatic plot scaling.
EXPERIMENT_ROOTS = [
    ("Hotspot_GANE/CPP_treatment",    "opto_cpp_treatment",
     ("10HZ_DG", "10HZ_DG_CPP"), False, "ΔF/F",   ()),
    ("Hotspot_GANE/DG_local_release", "opto_dg_local_release",
     ("1HZ_DG", "20HZ_DG"),      True,  DFF_LABEL, ()),
    # The Atomoxetine directory name contains a space before the underscore.
    # Its single cohort has no paired comparison.
    ("Hotspot_GANE/DG_local_release _Atomoxetine", "opto_dg_local_release_atomoxetine",
     None,                        True,  DFF_LABEL, ()),
]


# Paired statistics retain stimulation-response extrema and AUC over 0-40 s.
# Exclude heating, seizure, and postictal measurements from the statistics
# table without suppressing their shared plotting modules.
_OPTO_LEGACY_MEASUREMENTS_TO_DROP = {
    # Heating measurements
    "early_heat_mean", "late_heat_mean", "end_late_heat_mean", "heating_mean",
    # Seizure measurements
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
    # Postictal measurements
    "final_time", "final_mean_dff",
    # Baseline mean is zero when z-scored to the same baseline.
    "baseline_mean",
}


# Stimulation-response measurements with an additional paired ratio test.
_OPTO_RATIO_MEASUREMENTS = (
    "stim_response_max_40s",
    "stim_response_min_40s",
    "stim_auc_40s",
)



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
    values to `output_dir`. Session identifiers are included in the CSV.
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

    # Single-bar plot with black session markers and mean +/- SEM.
    vals = pairs["b_norm"].to_numpy(dtype=float)
    mean = float(np.mean(vals))
    sem = (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
           if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(9.0, 12.0))

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
        raise FileNotFoundError(f"Dataset directory does not exist: {exp_path}")

    label = experiment_subpath.replace("/", "_")
    log.info("=" * 70)
    log.info("Opto / %s   strategy=%s   threshold=%g   cache_key=%s",
             experiment_subpath, STRATEGY, THRESHOLD, ds_key)
    log.info("=" * 70)

    # Temperature was not recorded; has_temperature=False fills it with NaN.
    sessions = load_sessions(exp_path, test_mode=False, has_temperature=False)
    if not sessions:
        raise ValueError(f"No loadable sessions in {exp_path}")

    # Use the event-positive flag for stimulated sessions in shared analysis.
    # Seizure-control pairing has no control sessions in this dataset.
    for s in sessions:
        if s.raw is not None and s.raw.stim_raw is not None:
            s.n_seizures = max(s.n_seizures, 1)

    results_base = output_root / f"results_opto_{label}"
    layout = ensure_layout(results_base / strategy_folder_name(STRATEGY))
    qc_dir = str(layout["quality_check"])

    # Cache labels use the final component of the experiment path.
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
        is_opto=True,
        ueo_filename_prefix="stim_onset_aligned",
        ueo_dff_label=dff_label,
        module_scatter_options=cpp_scatter_modules,
        ueo_yscale_exclude=list(excluded_from_scale),
        ueo_heatmap_exclude_from_vlim=list(excluded_from_scale),
        ueo_heatmap_vlim_override=0.4 if is_cpp_treatment else None,
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

        # Export the per-pair response ratio beside the paired statistics.
        _make_normalized_response_bar(
            sessions, group_a, group_b,
            output_dir=Path(layout["means"]),
            label=label,
        )

    return sessions


def main() -> None:
    global OPTO_BASE, OPTO_CACHE_DIR

    parser = argparse.ArgumentParser(
        description="Strategy-C pipeline over OPTO_stim experiment roots"
    )
    parser.add_argument(
        "--data-root", type=Path, default=OPTO_BASE,
        help="Parent directory containing the Hotspot_GANE experiments.",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=OPTO_CACHE_DIR,
        help="Preprocessing cache directory (default: opto_cache).",
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
    OPTO_BASE = args.data_root
    OPTO_CACHE_DIR = args.cache_root

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
    failed = []
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
            failed.append(experiment_subpath)

    if failed:
        log.error("Failed experiments: %s", ", ".join(failed))
        raise SystemExit(1)
    log.info("DONE. Results in %s/results_opto_*", output_root)


if __name__ == "__main__":
    main()
