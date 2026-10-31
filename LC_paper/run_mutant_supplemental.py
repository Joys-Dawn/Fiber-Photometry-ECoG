"""
Supplemental: UEO-aligned mean plots for DG_GRABneMutant under
preprocessing strategies A and B (the alternatives to the default C).

For each strategy, writes:
  LC_paper/supplemental/dg_grabne_mutant/strategy_<X>/
    summary_UEO_aligned_traces.png           (zoomed in,  -150..+150 s)
    summary_UEO_aligned_traces_extended.png  (zoomed out, -150..+1000 s)
    UEO_aligned_traces_<cohort>.png          (per-cohort, both windows)

Uses load_sessions and preprocess_and_cache from the threshold sweep runner.
No statistics or transient detection are performed.

Usage:
    python -m LC_paper.run_mutant_supplemental
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("agg")

from LC_paper.run_threshold_sweep import load_sessions, preprocess_and_cache
from fiber_photometry_ecog.analysis import compute_wide_ueo_triggered
from fiber_photometry_ecog.core.cohort import classify_session
from fiber_photometry_ecog.core.config import AnalysisConfig
from fiber_photometry_ecog.core.data_models import Session
from fiber_photometry_ecog.visualization.group_plots import plot_ueo_per_cohort


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


DATASET = Path("D:/Ephys_Data/Hyperthermia_seizure/DG_GRABneMutant")
OUT_ROOT = Path(__file__).resolve().parent / "supplemental" / "dg_grabne_mutant"
STRATEGIES = ("A", "B")


def run_one_strategy(sessions: List[Session], strategy: str,
                     config: AnalysisConfig) -> Path:
    """Preprocess `sessions` under `strategy` and render UEO mean plots."""
    log.info("=" * 70)
    log.info("Strategy %s — %d sessions", strategy, len(sessions))
    log.info("=" * 70)

    # Use a strategy-tagged cache key so A and B don't collide with the
    # existing Strategy C cache for this dataset.
    cache_key = f"supplemental_mutant_{strategy}"
    for i, s in enumerate(sessions, 1):
        log.info("  [%d/%d] preprocessing %s S%d (strategy %s)",
                 i, len(sessions), s.mouse_id, s.heating_session, strategy)
        try:
            preprocess_and_cache(s, strategy, cache_key)
        except Exception as e:
            log.error("    FAILED %s S%d: %s",
                      s.mouse_id, s.heating_session, e)
            raise

    # Group mutant-sensor recordings by data-log genotype and seizure outcome.
    cohort_groups: Dict[str, List[Session]] = {}
    for s in sessions:
        cohort_groups.setdefault(classify_session(s), []).append(s)
    log.info("  Cohorts: %s",
             {k: len(v) for k, v in cohort_groups.items()})

    standard = {
        c: compute_wide_ueo_triggered(
            sessions=ss,
            window_s=config.wide_triggered_window_s,
        )
        for c, ss in cohort_groups.items()
    }
    extended = {
        c: compute_wide_ueo_triggered(
            sessions=ss,
            window_s=config.wide_triggered_window_s,
            post_window_s=1000.0,
        )
        for c, ss in cohort_groups.items()
    }

    out_dir = OUT_ROOT / f"strategy_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = plot_ueo_per_cohort(
        standard, str(out_dir), results_extended=extended,
    )
    log.info("  Wrote summary: %s", summary_path)
    return out_dir


def main() -> int:
    if not DATASET.exists():
        log.error("Dataset folder not found: %s", DATASET)
        return 1

    sessions = load_sessions(DATASET, test_mode=False, has_temperature=True)
    if not sessions:
        log.error("No sessions loaded from %s", DATASET)
        return 1

    config = AnalysisConfig()
    for strategy in STRATEGIES:
        try:
            run_one_strategy(sessions, strategy, config)
        except Exception as e:
            log.error("Strategy %s FAILED: %s", strategy, e)

    log.info("DONE — outputs in %s", OUT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
