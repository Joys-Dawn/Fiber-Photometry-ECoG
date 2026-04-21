"""
End-to-end integration test of the full pipeline.

Uses real Chandni data: one session per cohort.
Tests: load → sync → preprocess (all 3 strategies) → pair → analyze → plot.
"""
import logging
import shutil
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Resolve repo root (two levels up from this file: tests/ -> fiber_photometry_ecog/ -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from fiber_photometry_ecog.data_loading import read_ppd, read_oep, synchronize
from fiber_photometry_ecog.preprocessing import preprocess_session
from fiber_photometry_ecog.core.config import (
    PreprocessingConfig,
    AnalysisConfig,
)
from fiber_photometry_ecog.core.data_models import (
    Session,
    SessionLandmarks,
    RawData,
)
from fiber_photometry_ecog.pairing.engine import assign_all_controls
from fiber_photometry_ecog.analysis import (
    compute_cohort_characteristics,
    compute_baseline_transients,
    compute_preictal_mean,
    compute_preictal_transients,
    compute_ictal_mean,
    compute_ictal_transients,
    compute_postictal,
    compute_spike_triggered_average,
)
from fiber_photometry_ecog.visualization import (
    plot_sanity_check,
    plot_zoomed,
    plot_cohort_characteristics,
    plot_baseline_transients,
    plot_preictal_mean,
    plot_preictal_transients,
    plot_ictal_mean,
    plot_ictal_transients,
    plot_postictal,
    plot_spike_triggered,
)

# --- Test sessions (one per cohort) ---
# Landmark times are approximate — real values would come from user marking on GUI.
# These are plausible for a ~15-20 min heating experiment with seizure around 42°C.
_DATA = _REPO_ROOT / "Chandni_data"

TEST_SESSIONS = [
    {
        "cohort": "seizure",
        "mouse_id": "3331",
        "genotype": "Scn1a",
        "n_seizures": 1,
        "heating_start": 60.0,
        "eec": 600.0,
        "ueo": 620.0,
        "off": 640.0,
        "ppd": str(_DATA / "Scn1a_seizure/3331_session1/3331-2024-10-22-215742.ppd"),
        "oep": str(_DATA / "Scn1a_seizure/3331_session1/2024-10-22_21-57-43_3331"),
    },
    {
        "cohort": "failed_seizure",
        "mouse_id": "3339",
        "genotype": "Scn1a",
        "n_seizures": 0,
        "heating_start": 60.0,
        "ppd": str(_DATA / "Scn1a_no_seizure/3339_session1/3339-2024-10-22-231128.ppd"),
        "oep": str(_DATA / "Scn1a_no_seizure/3339_session1/2024-10-22_23-11-27_3339"),
    },
    {
        "cohort": "wt",
        "mouse_id": "3330",
        "genotype": "WT",
        "n_seizures": 0,
        "heating_start": 60.0,
        "ppd": str(_DATA / "WT/3330_session1/3330-2024-10-22-212300.ppd"),
        "oep": str(_DATA / "WT/3330_session1/2024-10-22_21-13-14_3330"),
    },
]

OUTPUT_DIR = _REPO_ROOT / "fiber_photometry_ecog" / "tests" / "test_e2e_output"


def load_and_sync(info: dict) -> tuple:
    """Step 1: Load PPD + OEP and synchronize."""
    logger.info(f"Loading {info['mouse_id']}...")

    ppd = read_ppd(info["ppd"])
    logger.info(f"  PPD: {len(ppd.signal_470)} samples, fs={ppd.fs} Hz")

    oep = read_oep(info["oep"], ecog_channel=3, emg_channel=None)
    logger.info(f"  OEP: {len(oep.ecog)} samples, fs={oep.fs} Hz")

    sync = synchronize(ppd, oep)
    logger.info(
        f"  Sync: {sync.n_matched} pulses, "
        f"drift={sync.drift_ppm:.1f} ppm, residual={sync.residual_ms:.3f} ms"
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

    landmarks = SessionLandmarks(
        heating_start_time=info["heating_start"],
        eec_time=info.get("eec"),
        ueo_time=info.get("ueo"),
        off_time=info.get("off"),
    )

    session = Session(
        mouse_id=info["mouse_id"],
        genotype=info["genotype"],
        n_seizures=info["n_seizures"],
        landmarks=landmarks,
        raw=raw,
    )

    return session, info["cohort"]


def preprocess(session: Session, strategy_name: str, config: PreprocessingConfig) -> None:
    """Step 2: Run all preprocessing on a session."""
    config.photometry.strategy = strategy_name
    preprocess_session(session, config)
    mouse = session.mouse_id
    logger.info(f"  {mouse} preprocessed (strategy {strategy_name}): "
                f"{len(session.transients)} transients, {len(session.spikes)} spikes")


def run_analysis(cohorts: dict, config: AnalysisConfig, output_dir: Path) -> None:
    """Step 3: Run all 8 analysis modules + plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pairing
    seizure_sessions = cohorts.get("seizure", [])
    control_sessions = cohorts.get("failed_seizure", []) + cohorts.get("wt", [])

    # Pairing requires seizure landmarks (EEC/UEO/OFF) which are user-entered.
    # Skip if landmarks are missing.
    seizure_has_landmarks = any(
        s.landmarks and s.landmarks.eec_time is not None
        for s in seizure_sessions
    )
    if seizure_sessions and control_sessions and seizure_has_landmarks:
        logger.info("Running pairing (temperature mode)...")
        assign_all_controls(seizure_sessions, control_sessions, mode="temperature")
        logger.info("  Pairing complete")
    else:
        logger.info("Skipping pairing (no seizure landmarks set — would be entered via GUI)")

    # Analysis modules
    all_sessions = {name: sessions for name, sessions in cohorts.items() if sessions}

    modules = [
        ("1. Cohort characteristics", lambda: _run_cohort_chars(all_sessions, output_dir)),
        ("2. Baseline transients", lambda: _run_baseline_trans(all_sessions, config, output_dir)),
        ("3. Pre-ictal mean", lambda: _run_preictal_mean(all_sessions, config, output_dir)),
        ("4. Pre-ictal transients", lambda: _run_preictal_trans(all_sessions, config, output_dir)),
        ("5. Spike-triggered averages", lambda: _run_sta(all_sessions, config, output_dir)),
        ("6. Ictal mean", lambda: _run_ictal_mean(all_sessions, config, output_dir)),
        ("7. Ictal transients", lambda: _run_ictal_trans(all_sessions, config, output_dir)),
        ("8. Postictal", lambda: _run_postictal(all_sessions, config, output_dir)),
    ]

    for name, func in modules:
        logger.info(f"{name}...")
        try:
            func()
            logger.info("  Done")
        except Exception as e:
            logger.error(f"  FAILED: {e}")


def _run_cohort_chars(all_sessions, output_dir):
    results = {n: compute_cohort_characteristics(s) for n, s in all_sessions.items()}
    plot_cohort_characteristics(results, str(output_dir))

def _run_baseline_trans(all_sessions, config, output_dir):
    results = {n: compute_baseline_transients(s, config) for n, s in all_sessions.items()}
    plot_baseline_transients(results, str(output_dir))

def _run_preictal_mean(all_sessions, config, output_dir):
    results = {n: compute_preictal_mean(s, config) for n, s in all_sessions.items()}
    plot_preictal_mean(results, str(output_dir))

def _run_preictal_trans(all_sessions, config, output_dir):
    results = {n: compute_preictal_transients(s, config) for n, s in all_sessions.items()}
    plot_preictal_transients(results, str(output_dir))

def _run_sta(all_sessions, config, output_dir):
    sta_results = {}
    for name, sessions in all_sessions.items():
        spike_times_list = [np.array([sp.time for sp in s.spikes]) for s in sessions]
        sta_results[name] = compute_spike_triggered_average(sessions, spike_times_list, config)
    plot_spike_triggered(sta_results, str(output_dir))

def _run_ictal_mean(all_sessions, config, output_dir):
    results = {n: compute_ictal_mean(s, config) for n, s in all_sessions.items()}
    plot_ictal_mean(results, str(output_dir))

def _run_ictal_trans(all_sessions, config, output_dir):
    results = {n: compute_ictal_transients(s, config) for n, s in all_sessions.items()}
    plot_ictal_transients(results, str(output_dir))

def _run_postictal(all_sessions, config, output_dir):
    results = {n: compute_postictal(s, config) for n, s in all_sessions.items()}
    plot_postictal(results, str(output_dir))


def run_e2e(strategy_name: str = "A"):
    """Run full end-to-end pipeline."""
    logger.info(f"{'='*60}")
    logger.info(f"E2E TEST — Strategy {strategy_name}")
    logger.info(f"{'='*60}")

    output = OUTPUT_DIR / f"strategy_{strategy_name}"
    if output.exists():
        shutil.rmtree(output)

    config = PreprocessingConfig()
    analysis_config = AnalysisConfig()

    # Step 1: Load all sessions
    cohorts: dict[str, list] = {}
    for info in TEST_SESSIONS:
        session, cohort = load_and_sync(info)
        cohorts.setdefault(cohort, []).append(session)

    # Step 2: Preprocess all sessions
    logger.info(f"\n--- Preprocessing (Strategy {strategy_name}) ---")
    for cohort_name, sessions in cohorts.items():
        for s in sessions:
            preprocess(s, strategy_name, config)

    # Step 2b: Sanity check plots
    logger.info("\nGenerating sanity check plots...")
    for cohort_name, sessions in cohorts.items():
        for s in sessions:
            plot_sanity_check(s, str(output))
            plot_zoomed(s, str(output))
    logger.info("  Done")

    # Step 3: Analysis + plots
    logger.info("\n--- Analysis ---")
    run_analysis(cohorts, analysis_config, output)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"E2E COMPLETE — Strategy {strategy_name}")
    logger.info(f"Output: {output}")
    generated = list(output.rglob("*.png")) + list(output.rglob("*.csv"))
    logger.info(f"Generated: {len(generated)} files")
    for f in sorted(generated):
        logger.info(f"  {f.name}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    for strategy in ["A", "B", "C"]:
        run_e2e(strategy)
    logger.info("ALL STRATEGIES COMPLETE")
