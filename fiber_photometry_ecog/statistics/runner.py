"""
Top-level statistics driver.

`run_group_comparisons` takes the per-cohort module results that app.py
already builds, extracts per-session scalars, runs an independent-group
test per measurement, and writes one CSV per output folder
(means / transients).

Raw p-values only - no multiple-comparison correction.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .config import PairedStatsConfig, StatsConfig
from .extractor import extract_all, group_values
from .independent import IndependentTestResult, compare_independent_groups
from .lmm_paired import compare_paired_lmm
from .paired import PairedTestResult

logger = logging.getLogger(__name__)


def _csv_name(name_prefix: str) -> str:
    """File name for the stats CSV: ``<prefix>_stats.csv``.
    Lets callers open multiple groups side-by-side in Excel without
    filename collisions."""
    return f"{name_prefix}_stats.csv"


def _row_for_result(
    result: IndependentTestResult,
    module: str,
    folder: str,
    cohorts_to_compare: List[str],
    label_for: Dict[str, str],
) -> Dict[str, Any]:
    """Convert one IndependentTestResult into a flat row for the CSV."""
    row: Dict[str, Any] = {
        "Module": module,
        "Folder": folder,
        "Measurement": result.measurement,
        "Test": result.test_name,
        "Statistic": result.statistic,
        "p_value": result.p_value,
    }
    # Descriptives keyed by display label of each cohort.
    by_name = {d.name: d for d in result.descriptives}
    for cohort_key in cohorts_to_compare:
        label = label_for[cohort_key]
        d = by_name.get(cohort_key)
        if d is None:
            row[f"{label}_n"] = 0
            row[f"{label}_mean"] = math.nan
            row[f"{label}_sem"] = math.nan
        else:
            row[f"{label}_n"] = d.n
            row[f"{label}_mean"] = d.mean
            row[f"{label}_sem"] = d.sem

    # Pairwise post-hocs (only populated when N>2 groups). Column per pair.
    for ph in result.posthocs:
        label1 = label_for.get(ph.group1, ph.group1)
        label2 = label_for.get(ph.group2, ph.group2)
        pair_key = f"{label1}_vs_{label2}"
        row[f"posthoc_{pair_key}_p"] = ph.p_value
        row[f"posthoc_{pair_key}_test"] = ph.test_name
    return row


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    # Drop rows where no test could be run (Test=="None") — those are
    # measurements with too few finite observations per group to fit any
    # test, and reporting them just adds noise to the CSV. The per-cohort
    # descriptives for those rows are already empty/zero so nothing
    # informative is lost.
    rows = [r for r in rows if r.get("Test") != "None"]
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_path, index=False)
        return out_path
    df = pd.DataFrame(rows)
    # Stable column ordering: base info, descriptives, p-values, post-hocs.
    base_cols = ["Module", "Measurement", "Test", "Statistic", "p_value"]
    base_cols = [c for c in base_cols if c in df.columns]

    # Per-cohort descriptives, ordered mean -> sem -> n within each cohort,
    # cohorts in alphabetical order. (Order matters in the rendered CSV;
    # see comment in `_row_for_result`.)
    suffix_order = ("_mean", "_sem", "_n")
    by_prefix: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        for suffix in suffix_order:
            if col.endswith(suffix):
                prefix = col[: -len(suffix)]
                by_prefix.setdefault(prefix, {})[suffix] = col
                break
    descriptive_cols: List[str] = []
    for prefix in sorted(by_prefix):
        for suffix in suffix_order:
            col = by_prefix[prefix].get(suffix)
            if col is not None:
                descriptive_cols.append(col)
    posthoc_cols = sorted(c for c in df.columns if c.startswith("posthoc_"))
    other_cols = [c for c in df.columns
                  if c not in base_cols + descriptive_cols + posthoc_cols + ["Folder"]]
    ordered = base_cols + descriptive_cols + posthoc_cols + other_cols
    df = df[ordered]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def run_group_comparisons(
    cohort_results: Dict[str, Dict[str, Any]],
    means_dir: str | Path,
    transients_dir: str | Path,
    name_prefix: str,
    config: Optional[StatsConfig] = None,
) -> Dict[str, Path]:
    """Run all per-measurement comparisons across the configured cohorts.

    Parameters
    ----------
    cohort_results : per-cohort module results, the same dict the app already
        builds. Shape: ``{cohort_key: {module_name: group_result}}``.
    means_dir, transients_dir : output folders (`plots and values_means/`
        and `plots and values_transients/`). Stats CSVs land here, alongside
        the existing per-module plots.
    config : optional `StatsConfig`. Defaults to the DS sz vs WT 2-way
        comparison. Raw p-values only - no multiple-comparison correction.

    Returns
    -------
    dict mapping "means" / "transients" to the written CSV path (only
    folders that received any rows are included). Returns an empty dict if
    no measurements could be tested.
    """
    if config is None:
        config = StatsConfig()

    cohorts_to_compare = list(config.cohorts_to_compare)
    available = [c for c in cohorts_to_compare if c in cohort_results]
    missing = [c for c in cohorts_to_compare if c not in cohort_results]
    if missing:
        logger.warning(
            "Stats: cohorts not present in results, dropping: %s", missing
        )
    if len(available) < 2:
        logger.warning(
            "Stats: need at least 2 cohorts to compare; got %s - skipping",
            available,
        )
        return {}
    cohorts_to_compare = available
    label_for = {c: config.label(c) for c in cohorts_to_compare}

    long_rows = extract_all(cohort_results)
    grouped = group_values(long_rows, cohorts_to_compare)
    if not grouped:
        logger.warning("Stats: no measurements available to compare")
        return {}

    # Run a test per measurement.
    rows_by_folder: Dict[str, List[Dict[str, Any]]] = {"means": [], "transients": []}
    # We need to know the originating module name for each measurement -
    # walk the registry alongside grouped values.
    for (folder, module, measurement), values_by_group in grouped.items():
        # Map values to display labels for nicer descriptives output.
        relabelled = {label_for[c]: values_by_group[c] for c in cohorts_to_compare}
        result = compare_independent_groups(
            measurement=measurement,
            values_by_group=relabelled,
        )
        # Restore cohort-key view for downstream column generation: the row
        # writer iterates by cohort_key but pulls descriptives via label.
        # `descriptives[i].name` is already a label - flip the map so the
        # row writer can find each cohort's stats by key.
        # We achieve this by storing the result with descriptives keyed by
        # label, and translating in _row_for_result via label_for keys.
        # (Replace descriptive .name field back to cohort_key for lookup.)
        label_to_key = {v: k for k, v in label_for.items()}
        for d in result.descriptives:
            d.name = label_to_key.get(d.name, d.name)
        for ph in result.posthocs:
            ph.group1 = label_to_key.get(ph.group1, ph.group1)
            ph.group2 = label_to_key.get(ph.group2, ph.group2)

        row = _row_for_result(
            result, module=module, folder=folder,
            cohorts_to_compare=cohorts_to_compare,
            label_for=label_for,
        )
        rows_by_folder.setdefault(folder, []).append(row)

    out_paths: Dict[str, Path] = {}
    folder_to_dir = {"means": Path(means_dir), "transients": Path(transients_dir)}
    csv_name = _csv_name(name_prefix)
    for folder, rows in rows_by_folder.items():
        if not rows:
            continue
        out_path = folder_to_dir[folder] / csv_name
        out_paths[folder] = _write_csv(rows, out_path)
        logger.info(
            "Stats: wrote %d measurements to %s", len(rows), out_paths[folder]
        )

    return out_paths


def _paired_row(
    result: PairedTestResult,
    module: str,
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    """Convert one PairedTestResult into a flat row for the paired-stats CSV."""
    row: Dict[str, Any] = {
        "Module": module,
        "Measurement": result.measurement,
        "Test": result.test_name,
        "Statistic": result.statistic,
        "p_value": result.p_value,
        "n_pairs": result.n_pairs,
    }
    by_name = {d.name: d for d in result.descriptives}
    for label in (label_a, label_b):
        d = by_name.get(label)
        if d is None:
            row[f"{label}_n"] = 0
            row[f"{label}_mean"] = math.nan
            row[f"{label}_sem"] = math.nan
        else:
            row[f"{label}_n"] = d.n
            row[f"{label}_mean"] = d.mean
            row[f"{label}_sem"] = d.sem
    return row


def run_paired_comparisons(
    cohort_results: Dict[str, Dict[str, Any]],
    means_dir: str | Path,
    transients_dir: str | Path,
    name_prefix: str,
    group_a: str,
    group_b: str,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    config: Optional[PairedStatsConfig] = None,
    exclude_measurements: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """Run within-subject LMM comparisons between two cohorts.

    Subjects are identified by ``mouse_id`` (the extractor key). One row
    per session is fed to ``compare_paired_lmm``; if the LMM is singular
    or fails to converge the function falls back to an independent-groups
    test on session-level data (Welch's / Mann-Whitney U) with a loud
    warning. Per-mouse averaging is never performed.

    Parameters
    ----------
    cohort_results : same dict structure as `run_group_comparisons`,
        ``{cohort_key: {module_name: group_result}}``.
    means_dir, transients_dir : output folders. Paired CSVs land here with
        the suffix ``_paired_stats.csv``.
    name_prefix : filename prefix (typically the opto experiment root name).
    group_a, group_b : cohort keys to compare (e.g. ``"10HZ_DG"`` and
        ``"10HZ_DG_CPP"``). Subjects appearing in only one cohort are
        dropped from the test for that measurement.
    label_a, label_b : optional display labels for the descriptive columns;
        default to the cohort keys themselves.
    config : `PairedStatsConfig` (currently only ``alpha`` is used).
    exclude_measurements : measurement names to suppress from the CSV.
        Used by opto pipelines to drop the heat-named pre-ictal windows
        (``early_heat_mean``, ``late_heat_mean``, ``end_late_heat_mean``,
        ``heating_mean``) that map onto an arbitrary slice of pre-stim
        baseline rather than a real heating phase.

    Returns
    -------
    dict mapping "means" / "transients" to the written CSV path.
    """
    label_a = label_a or group_a
    label_b = label_b or group_b
    if config is None:
        config = PairedStatsConfig()
    excluded: set = set(exclude_measurements) if exclude_measurements else set()

    missing = [c for c in (group_a, group_b) if c not in cohort_results]
    if missing:
        logger.warning(
            "Paired stats: cohorts not present in results: %s - skipping",
            missing,
        )
        return {}

    long_rows = extract_all(cohort_results)
    # Pivot to {(folder, module, measurement): list of (mouse_id, cohort, value)}
    per_meas: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in long_rows:
        if row["cohort"] not in (group_a, group_b):
            continue
        if row["measurement"] in excluded:
            continue
        k = (row["folder"], row["module"], row["measurement"])
        per_meas.setdefault(k, []).append({
            "mouse_id": row["mouse_id"],
            "heating_session": row["heating_session"],
            "cohort": row["cohort"],
            "value": row["value"],
        })

    rows_by_folder: Dict[str, List[Dict[str, Any]]] = {"means": [], "transients": []}
    for (folder, module, measurement), entries in per_meas.items():
        df_long = pd.DataFrame(entries)
        result = compare_paired_lmm(
            measurement=measurement,
            group_a_name=label_a,
            group_b_name=label_b,
            df_long=df_long,
            group_a=group_a,
            group_b=group_b,
        )
        row = _paired_row(result, module=module, label_a=label_a, label_b=label_b)
        rows_by_folder.setdefault(folder, []).append(row)

    out_paths: Dict[str, Path] = {}
    folder_to_dir = {"means": Path(means_dir), "transients": Path(transients_dir)}
    csv_name = f"{name_prefix}_paired_stats.csv"
    for folder, rows in rows_by_folder.items():
        if not rows:
            continue
        out_path = folder_to_dir[folder] / csv_name
        out_paths[folder] = _write_csv(rows, out_path)
        logger.info(
            "Paired stats: wrote %d measurements to %s", len(rows), out_paths[folder],
        )

    return out_paths
