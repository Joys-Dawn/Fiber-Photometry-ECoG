"""
Single-threshold strategy-C analysis of hyperthermia and PTZ datasets.

Thresholds:
    LC_GCaMP             -> 0.015
    everything else      -> 0.030

Default outputs are LC_paper/results/results_hyperthermia_<subfolder>/, with
separate results_PTZ_GRABne folders for the PTZ analyses.

Preprocessing caches are reused per dataset and strategy.

Usage:
    python -m LC_paper.run_hyperthermia
    python -m LC_paper.run_hyperthermia --only LC_GCaMP DG_GRABne
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("agg")

from LC_paper import run_threshold_sweep as sweep

from LC_paper.run_threshold_sweep import (
    load_sessions,
    preprocess_and_cache,
    run_group_analysis,
    _make_transient_config,
)
from fiber_photometry_ecog.preprocessing.transient_detection import detect_transients
from fiber_photometry_ecog.preprocessing.photometry import strategy_folder_name
from fiber_photometry_ecog.core.output_layout import ensure_layout
from fiber_photometry_ecog.core.cohort import classify_session, COHORT_KEYS
from fiber_photometry_ecog.statistics import StatsConfig
from fiber_photometry_ecog.visualization.colors import cohort_color
from fiber_photometry_ecog.visualization.trace_plots import (
    plot_sanity_check,
    plot_zoomed,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


HYPER_ROOT = Path("D:/Ephys_Data/Hyperthermia_seizure")
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
STRATEGY = "C"

# Default UEO-aligned trace and heatmap window; dataset overrides follow.
UEO_X_WINDOW = (-100.0, 100.0)

# Both split and pooled PTZ plots use a +/-50 s display window.
PTZ_UEO_X_WINDOW = (-50.0, 50.0)

# Per-subfolder UEO x-axis override. Missing entries fall back to
# UEO_X_WINDOW. Keyed by the SUBFOLDERS `subfolder` column so both the
# split and pooled PTZ runs inherit the same window.
SUBFOLDER_UEO_X_WINDOWS = {
    "PTZ_GRABne": PTZ_UEO_X_WINDOW,
}

# Manuscript trace panels use tighter ranges to remove unused positive space.
DG_UEO_Y_LIM = (-30.0, 40.0)
MUTANT_UEO_Y_LIM = (-30.0, 25.0)
GRAB_NE_BRIGHT_GREEN = "green"
FIGURE6_AXES_SIZE_INCHES = (14.0, 6.0)

# Reference subfolder whose auto-computed heatmap vlim is captured and
# reused on its paired mutant subfolder so the color scales match.
HEATMAP_VLIM_PAIRS = {
    "DG_GRABneMutant": "DG_GRABne",
}

# Cohort key for all sessions when `pooled=True`.
PTZ_POOLED_KEY = "PTZ_pooled"

# DG and LC publication scalar views compare seizure outcome, with both
# no-seizure genotypes pooled. Qualitative traces stay in the canonical
# genotype-by-outcome groups via `trace_cohort_classifier` below.
SEIZURE_OUTCOME_SUBFOLDERS = {"DG_GRABne", "LC_GCaMP"}

ICTAL_EXTREMA_MIN_DURATION_S = {
    "DG_GRABne": 0.200,
    "LC_GCaMP": 0.200,
}

# LC_GCaMP: 100 s. DG_GRABne: individual seizure durations;
# DG_GRABne controls use the seizure-group mean duration.
LC_AUC_WINDOWS_S = {
    "EEC": 100.0,
    "UEO": 100.0,
    "behavioral_onset": 100.0,
    "max_temp": 100.0,
}
INDIVIDUAL_SEIZURE_AUC_SUBFOLDERS = {"DG_GRABne"}
DG_POSITIVE_ONLY_AUC_LANDMARKS = {"EEC", "UEO"}


def _seizure_outcome_classifier(session) -> str:
    return "Seizure" if session.n_seizures > 0 else "No seizure"


def _genotype_classifier(session) -> str:
    genotype = (session.genotype or "").strip().lower()
    return "DS" if genotype.startswith("scn1a") else "WT"


def _source_point_colors(sessions) -> dict:
    """Color pooled points by their original biological source group."""
    colors = {}
    for session in sessions:
        key = (session.mouse_id, session.heating_session)
        source = classify_session(session)
        plotted_group = _seizure_outcome_classifier(session)
        colors[(plotted_group, key)] = (
            "red" if session.n_seizures > 0 else cohort_color(source)
        )
    return colors


def _genotype_point_colors(sessions) -> dict:
    return {
        (
            _genotype_classifier(session),
            (session.mouse_id, session.heating_session),
        ): (
            "red" if _genotype_classifier(session) == "DS" else "blue"
        )
        for session in sessions
    }

# DG within-cohort heating-versus-ictal comparisons are appended to
# stats_<name>.csv. The distribution of paired differences determines
# whether to use a paired t-test or Wilcoxon signed-rank test.
SUBFOLDER_WITHIN_COHORT_PAIRED_PERIODS: dict[str, list] = {
    "DG_GRABne": [
        {
            "module": "Pre-ictal mean",
            "period_a": "heating_mean",
            "period_b": "ictal_mean",
            "period_a_label": "Heating",
            "period_b_label": "Ictal",
        },
    ],
}

# Tuple fields: subfolder, cache key, prominence threshold, temperature flag,
# y-limits, statistics flag, output subdirectory, pooled-cohort flag.
SUBFOLDERS = [
    ("DG_GRABne",       "hyperthermia_dg_grabne",        0.030, True,  DG_UEO_Y_LIM, True,  None,                           False),
    ("DG_GRABneMutant", "hyperthermia_dg_grabne_mutant", 0.030, True,  MUTANT_UEO_Y_LIM, False, None,                           False),
    ("DG_GRABne_NRI",   "hyperthermia_dg_grabne_nri",    0.030, True,  None,         False, None,                           False),
    ("FCD_GRABne",      "hyperthermia_fcd_grabne",       0.030, True,  None,         False, None,                           False),
    ("LC_GCaMP",        "hyperthermia_lc_gcamp",         0.015, True,  None,         True,  None,                           False),
    # Split and pooled PTZ analyses share a preprocessing cache via ds_key.
    ("PTZ_GRABne",      "hyperthermia_ptz_grabne",       0.030, False, None,         False, "results_PTZ_GRABne",           False),
    ("PTZ_GRABne",      "hyperthermia_ptz_grabne",       0.030, False, None,         False, "results_PTZ_GRABne_pooled",    True),
    ("PVT_GRABne",      "hyperthermia_pvt_grabne",       0.030, True,  None,         False, None,                           False),
]


def run_one_subfolder(subfolder: str, ds_key: str, threshold: float,
                       has_temperature: bool,
                       ueo_y_lim=None,
                       run_stats: bool = False,
                       output_root: Path = RESULTS_ROOT,
                       keep_cohorts: set | None = None,
                       output_subdir: str | None = None,
                       pooled: bool = False,
                       ueo_heatmap_vlim_override: float | None = None,
                       ueo_heatmap_diagnostics: dict | None = None,
                       ueo_x_window: tuple = UEO_X_WINDOW,
                       within_cohort_paired_periods=None) -> None:
    ds_path = HYPER_ROOT / subfolder
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {ds_path}")

    log.info("=" * 70)
    log.info("Hyperthermia / %s   strategy=%s   threshold=%g   cache_key=%s   "
             "has_temperature=%s   pooled=%s",
             subfolder, STRATEGY, threshold, ds_key, has_temperature, pooled)
    log.info("=" * 70)

    sessions = load_sessions(ds_path, test_mode=False,
                             has_temperature=has_temperature)
    if not sessions:
        raise ValueError(f"No loadable sessions in {ds_path}")

    if keep_cohorts is not None:
        before = len(sessions)
        sessions = [s for s in sessions if classify_session(s) in keep_cohorts]
        log.info("  Cohort filter %s: %d -> %d sessions",
                 sorted(keep_cohorts), before, len(sessions))
        if not sessions:
            raise ValueError(
                f"No sessions in {ds_path} match cohorts {sorted(keep_cohorts)}"
            )

    # PTZ uses output names without the hyperthermia prefix.
    folder_name = output_subdir or f"results_hyperthermia_{subfolder}"
    results_base = output_root / folder_name
    layout = ensure_layout(results_base / strategy_folder_name(STRATEGY))
    qc_dir = str(layout["quality_check"])

    # Preprocess + detect transients at the chosen threshold for every session,
    # then write the per-session full-trace and peri-ictal quality-check plots.
    tc = _make_transient_config(STRATEGY, threshold)
    for i, session in enumerate(sessions, 1):
        log.info("  [%d/%d] %s S%d -- preprocessing + transient detection",
                 i, len(sessions), session.mouse_id, session.heating_session)
        zdff_hpf, dff_det, temp_smooth, fs = preprocess_and_cache(
            session, STRATEGY, ds_key,
        )
        session.transients = detect_transients(
            zdff_hpf, dff_det, fs, tc, temp_smooth,
        )

        # Full-trace and peri-ictal QC; threshold-review plots are sweep-only.
        plot_sanity_check(session, qc_dir, session.transients)
        plot_zoomed(session, qc_dir)

    # Pooled PTZ means and SEM combine all genotypes and seizure outcomes.
    cohort_classifier = (lambda _s: PTZ_POOLED_KEY) if pooled else None
    trace_cohort_classifier = None
    module_cohort_classifiers = None
    scatter_options = None
    module_scatter_options = None
    stats_config = None
    module_stats_configs = None

    if subfolder in SEIZURE_OUTCOME_SUBFOLDERS:
        cohort_classifier = _seizure_outcome_classifier
        trace_cohort_classifier = classify_session
        scatter_options = {
            "point_colors_by_id": _source_point_colors(sessions),
        }
        stats_config = StatsConfig(
            cohorts_to_compare=["Seizure", "No seizure"],
        )

        if subfolder == "LC_GCaMP":
            # Baseline transients compare genotype, pooling seizure outcomes.
            module_cohort_classifiers = {
                "Baseline transients": _genotype_classifier,
            }
            module_scatter_options = {
                "Baseline transients": {
                    "point_colors_by_id": _genotype_point_colors(sessions),
                },
            }
            module_stats_configs = {
                "Baseline transients": StatsConfig(
                    cohorts_to_compare=["DS", "WT"],
                ),
            }

    if output_subdir:
        prefix = output_subdir.removeprefix("results_")
    else:
        prefix = subfolder

    # Group measurements, statistics, and plots use the dataset output tree.
    log.info("  Running group analysis -> %s   run_stats=%s",
             results_base, run_stats)
    extrema_min_duration_s = ICTAL_EXTREMA_MIN_DURATION_S.get(subfolder)
    if extrema_min_duration_s is not None:
        log.info(
            "  Ictal extrema require %.0f ms at/above or at/below the value",
            extrema_min_duration_s * 1000,
        )
    run_group_analysis(sessions, STRATEGY, ds_key, results_base,
                        stats_name_prefix=prefix,
                        ueo_y_lim=ueo_y_lim, run_stats=run_stats,
                        ueo_trace_colors=(
                            {"DS_sz": GRAB_NE_BRIGHT_GREEN}
                            if subfolder == "DG_GRABneMutant" else
                            {"DS_sz": GRAB_NE_BRIGHT_GREEN,
                             "FCD_sz": GRAB_NE_BRIGHT_GREEN,
                             "WT_sz": GRAB_NE_BRIGHT_GREEN,
                             "PTZ_pooled": GRAB_NE_BRIGHT_GREEN}
                            if subfolder in {"FCD_GRABne", "PVT_GRABne", "PTZ_GRABne"}
                            else None
                        ),
                        ueo_standard_window_s=ueo_x_window,
                        ueo_axes_size_inches=(
                            FIGURE6_AXES_SIZE_INCHES
                            if subfolder in {"PVT_GRABne", "FCD_GRABne", "PTZ_GRABne"}
                            else None
                        ),
                        ueo_heatmap_vlim_override=ueo_heatmap_vlim_override,
                        ueo_heatmap_diagnostics=ueo_heatmap_diagnostics,
                        cohort_classifier=cohort_classifier,
                        trace_cohort_classifier=trace_cohort_classifier,
                        module_cohort_classifiers=module_cohort_classifiers,
                        scatter_options=scatter_options,
                        module_scatter_options=module_scatter_options,
                        stats_config=stats_config,
                        module_stats_configs=module_stats_configs,
                        within_cohort_paired_periods=within_cohort_paired_periods,
                        ictal_auc_per_landmark_override=(
                            LC_AUC_WINDOWS_S if subfolder == "LC_GCaMP" else None
                        ),
                        ictal_auc_use_individual_seizure_durations=(
                            subfolder in INDIVIDUAL_SEIZURE_AUC_SUBFOLDERS
                        ),
                        ictal_auc_clip_to_available=(subfolder == "LC_GCaMP"),
                        ictal_positive_auc_landmarks=(
                            DG_POSITIVE_ONLY_AUC_LANDMARKS
                            if subfolder == "DG_GRABne" else None
                        ),
                        ictal_extrema_min_duration_s=extrema_min_duration_s)


def main() -> None:
    global HYPER_ROOT

    parser = argparse.ArgumentParser(
        description="Strategy-C single-threshold run over Hyperthermia_seizure"
    )
    parser.add_argument(
        "--data-root", type=Path, default=HYPER_ROOT,
        help="Parent directory containing LC_GCaMP, DG_GRABne, and other datasets.",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=sweep.CACHE_DIR,
        help="Preprocessing cache directory (default: sweep_cache).",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Subset of subfolder names to run (default: all).",
    )
    parser.add_argument(
        "--output-root", default=str(RESULTS_ROOT),
        help="Top-level results directory (default: results).",
    )
    parser.add_argument(
        "--cohorts", default=None,
        help=(
            "Comma-separated cohort keys to keep (e.g. 'DS_sz,WT_no_sz,WT_sz'). "
            "Default: keep all cohorts. Use --no-stats for a plot-only subset."
        ),
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Force-disable the per-subfolder stats pass even when configured.",
    )
    args = parser.parse_args()
    HYPER_ROOT = args.data_root
    sweep.CACHE_DIR = args.cache_root

    selected = SUBFOLDERS
    if args.only:
        wanted = set(args.only)
        selected = [s for s in SUBFOLDERS if s[0] in wanted]
        missing = wanted - {s[0] for s in SUBFOLDERS}
        if missing:
            parser.error(f"Unknown subfolders: {sorted(missing)}. "
                         f"Choose from {[s[0] for s in SUBFOLDERS]}.")

    output_root = Path(args.output_root)

    keep_cohorts: set | None = None
    if args.cohorts:
        keep_cohorts = {c.strip() for c in args.cohorts.split(",") if c.strip()}
        unknown = keep_cohorts - set(COHORT_KEYS)
        if unknown:
            parser.error(f"Unknown cohort keys: {sorted(unknown)}. "
                         f"Choose from {list(COHORT_KEYS)}.")
        log.info("Cohort filter active: keep %s", sorted(keep_cohorts))

    # Dataset flags enable LC/DG statistics unless --no-stats is set.
    # Cohort filtering does not disable the configured comparisons.
    force_no_stats = args.no_stats

    log.info("Running %d subfolder(s) under %s -> %s",
             len(selected), HYPER_ROOT, output_root)

    # Captured heatmap vlim per source subfolder, so a downstream subfolder
    # (e.g. DG_GRABneMutant) can match the reference subfolder's color scale.
    captured_vlim: dict[str, float] = {}
    failed = []

    for (subfolder, ds_key, threshold, has_temperature,
         ueo_y_lim, run_stats, output_subdir, pooled) in selected:
        effective_stats = False if force_no_stats else run_stats
        # Inherit a previously captured vlim from the paired reference
        # subfolder, when one is configured and has already run.
        ref = HEATMAP_VLIM_PAIRS.get(subfolder)
        vlim_override = captured_vlim.get(ref) if ref else None
        # Per-subfolder UEO x-axis window override (e.g. PTZ -> ±50s).
        ueo_x_window = SUBFOLDER_UEO_X_WINDOWS.get(subfolder, UEO_X_WINDOW)
        # Per-subfolder within-cohort paired-period stats (currently
        # DG_GRABne only). Gets appended to the same stats_<name>.csv.
        within_cohort_paired = SUBFOLDER_WITHIN_COHORT_PAIRED_PERIODS.get(subfolder)
        # Always capture this subfolder's vlim so a later subfolder can reuse it.
        diagnostics: dict[str, float] = {}
        try:
            run_one_subfolder(subfolder, ds_key, threshold, has_temperature,
                              ueo_y_lim=ueo_y_lim,
                              run_stats=effective_stats,
                              output_root=output_root,
                              keep_cohorts=keep_cohorts,
                              output_subdir=output_subdir,
                              pooled=pooled,
                              ueo_heatmap_vlim_override=vlim_override,
                              ueo_heatmap_diagnostics=diagnostics,
                              ueo_x_window=ueo_x_window,
                              within_cohort_paired_periods=within_cohort_paired)
        except Exception as e:
            log.error("FATAL in %s: %s", subfolder, e)
            traceback.print_exc()
            failed.append(subfolder)
            continue
        if "vlim_standard" in diagnostics:
            captured_vlim[subfolder] = diagnostics["vlim_standard"]
            log.info("  Captured heatmap vlim for %s: %.4f",
                     subfolder, diagnostics["vlim_standard"])

    if failed:
        log.error("Failed datasets: %s", ", ".join(failed))
        raise SystemExit(1)
    log.info("DONE. Results in %s/results_*", output_root)


if __name__ == "__main__":
    main()
