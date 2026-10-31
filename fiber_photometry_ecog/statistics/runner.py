"""
Top-level statistics driver.

`run_group_comparisons` extracts per-session scalars from cohort results,
runs an independent-group
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
from .paired import PairedTestResult, compare_paired

logger = logging.getLogger(__name__)


def _csv_name(name_prefix: str) -> str:
    """Return the statistics filename, optionally qualified by a prefix."""
    return f"stats_{name_prefix}.csv"


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
    # Omit rows for which no statistical test was performed.
    source_df = pd.DataFrame(rows)
    rows = [r for r in rows if r.get("Test") != "None"]
    # Preserve CSV headers when no test rows remain.
    df = pd.DataFrame(rows) if rows else source_df.iloc[0:0].copy()
    # Stable column ordering: base info, descriptives, p-values, post-hocs.
    base_cols = ["Module", "Measurement", "Test", "Statistic", "p_value"]
    base_cols = [c for c in base_cols if c in df.columns]

    # Alphabetical cohorts, with mean, SEM, and n for each.
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
    module_configs: Optional[Dict[str, StatsConfig]] = None,
    additional_rows_by_folder: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Path]:
    """Run all per-measurement comparisons across the configured cohorts.

    Parameters
    ----------
    cohort_results : ``{cohort_key: {module_name: group_result}}``.
    means_dir, transients_dir : output folders (`plots and values_means/`
        and `plots and values_transients/`).
    config : optional `StatsConfig`. Defaults to the DS sz vs WT 2-way
        comparison. Raw p-values only - no multiple-comparison correction.
    module_configs : optional ``{module_name: StatsConfig}`` overrides.

    Returns
    -------
    dict mapping "means" / "transients" to the written CSV path (only
    folders that received any rows are included). Returns an empty dict if
    no measurements could be tested.
    """
    if config is None:
        config = StatsConfig()

    long_rows = extract_all(cohort_results)
    if not long_rows:
        logger.warning("Stats: no measurements available to compare")
        return {}

    # Run a test per measurement.
    rows_by_folder: Dict[str, List[Dict[str, Any]]] = {"means": [], "transients": []}
    module_configs = module_configs or {}
    modules = list(dict.fromkeys(row["module"] for row in long_rows))
    for module in modules:
        active_config = module_configs.get(module, config)
        requested = list(active_config.cohorts_to_compare)
        module_rows = [row for row in long_rows if row["module"] == module]
        present = {row["cohort"] for row in module_rows}
        cohorts_to_compare = [c for c in requested if c in present]
        missing = [c for c in requested if c not in present]
        if missing:
            logger.warning(
                "Stats [%s]: cohorts not present, dropping: %s", module, missing
            )
        if len(cohorts_to_compare) < 2:
            logger.warning(
                "Stats [%s]: need at least 2 cohorts; got %s - skipping",
                module, cohorts_to_compare,
            )
            continue

        label_for = {
            c: active_config.label(c) for c in cohorts_to_compare
        }
        grouped = group_values(module_rows, cohorts_to_compare)
        for (folder, _, measurement), values_by_group in grouped.items():
            relabelled = {
                label_for[c]: values_by_group[c] for c in cohorts_to_compare
            }
            result = compare_independent_groups(
                measurement=measurement,
                values_by_group=relabelled,
            )
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

    if additional_rows_by_folder:
        for folder, extras in additional_rows_by_folder.items():
            if extras:
                rows_by_folder.setdefault(folder, []).extend(extras)

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


def run_within_cohort_paired_periods(
    cohort_results: Dict[str, Dict[str, Any]],
    module: str,
    period_a: str,
    period_b: str,
    period_a_label: str,
    period_b_label: str,
    cohorts: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Compare period_a and period_b within cohorts, paired by mouse and session."""
    long_rows = extract_all(cohort_results)
    if cohorts is None:
        cohorts = list(cohort_results.keys())
    cohorts = list(cohorts)

    # per_cohort_values: {cohort: {"a": {subject: value}, "b": {...}}}
    per_cohort_values: Dict[str, Dict[str, Dict[str, float]]] = {
        c: {"a": {}, "b": {}} for c in cohorts
    }
    for row in long_rows:
        if row["module"] != module:
            continue
        if row["cohort"] not in per_cohort_values:
            continue
        meas = row["measurement"]
        if meas not in (period_a, period_b):
            continue
        subject = f"{row['mouse_id']}_S{row['heating_session']}"
        slot = "a" if meas == period_a else "b"
        per_cohort_values[row["cohort"]][slot][subject] = float(row["value"])

    rows: List[Dict[str, Any]] = []
    for cohort in cohorts:
        vals_a = per_cohort_values[cohort]["a"]
        vals_b = per_cohort_values[cohort]["b"]
        if not vals_a or not vals_b:
            continue
        result = compare_paired(
            measurement=f"{cohort}: {period_a_label} vs {period_b_label}",
            group_a_name=period_a_label,
            values_a_by_subject=vals_a,
            group_b_name=period_b_label,
            values_b_by_subject=vals_b,
        )
        row = _paired_row(result, module=module,
                          label_a=period_a_label, label_b=period_b_label)
        row["Cohort"] = cohort
        rows.append(row)
    return rows


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

    ``compare_paired_lmm`` fits mouse and session-pair random intercepts,
    using one observation per session and condition. If R is unavailable
    or the mixed-model fit is unusable, it warns and uses a paired t-test
    or Wilcoxon signed-rank test on matched session pairs. The fallback
    does not model within-mouse correlation across sessions. Observations
    are not averaged per mouse.

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
