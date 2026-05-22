"""
Output folder layout for per-strategy results.

Single source of truth for the four subfolders inside each
`preprocessing_<METHOD>/` results folder, so plot writers and the sweep
driver agree on where files land.

Structure:
    preprocessing_<METHOD>/
        quality check/              <- v1/v2/v3 per-session + isosbestic QC
        parameter detection/        <- per-session diagnostic plots
        plots and values_means/     <- cohort-level mean signal plots + CSVs
        plots and values_transients/<- cohort-level transient plots + CSVs

Spacing in the folder names matches the user-facing spec.
"""

from pathlib import Path

SUBFOLDER_QUALITY_CHECK = "quality check"
SUBFOLDER_PARAMETER_DETECTION = "parameter detection"
SUBFOLDER_MEANS = "plots and values_means"
SUBFOLDER_TRANSIENTS = "plots and values_transients"

ALL_SUBFOLDERS = (
    SUBFOLDER_QUALITY_CHECK,
    SUBFOLDER_PARAMETER_DETECTION,
    SUBFOLDER_MEANS,
    SUBFOLDER_TRANSIENTS,
)


def ensure_layout(strategy_root: Path) -> dict:
    """Create the four standard subfolders under *strategy_root* and return their paths.

    Returns
    -------
    dict with keys "quality_check", "parameter_detection", "means", "transients"
    mapped to Path objects.
    """
    root = Path(strategy_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "quality_check": root / SUBFOLDER_QUALITY_CHECK,
        "parameter_detection": root / SUBFOLDER_PARAMETER_DETECTION,
        "means": root / SUBFOLDER_MEANS,
        "transients": root / SUBFOLDER_TRANSIENTS,
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
