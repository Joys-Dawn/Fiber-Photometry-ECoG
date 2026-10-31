"""
Extract per-session scalar measurements from the per-cohort analysis
results so they can be fed to the stats engine.

Each entry in `_SCALAR_REGISTRY` describes one analysis module:
    key            : module name (matches the keys used by app.py)
    folder         : "means" or "transients" statistics output directory
    extractor      : function(group_result) -> List[(mouse_id, h_session,
                                                     measurement_name, value)]
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import math
import numpy as np


# (mouse_id, heating_session, measurement_name, value)
SessionScalar = Tuple[str, int, str, float]


def _safe_float(x: Any) -> float:
    """Convert to float; return NaN for missing or nonnumeric values."""
    if x is None:
        return math.nan
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _extract_cohort_characteristics(group_result) -> List[SessionScalar]:
    out: List[SessionScalar] = []
    for sr in group_result.session_results:
        out.append((sr.mouse_id, sr.heating_session,
                    "baseline_temp", _safe_float(sr.baseline_temp)))
        out.append((sr.mouse_id, sr.heating_session,
                    "seizure_threshold_temp", _safe_float(sr.seizure_threshold_temp)))
        out.append((sr.mouse_id, sr.heating_session,
                    "seizure_duration_s", _safe_float(sr.seizure_duration_s)))
    return out


def _extract_baseline_transients(group_result) -> List[SessionScalar]:
    out: List[SessionScalar] = []
    for sr in group_result.session_results:
        out.append((sr.mouse_id, sr.heating_session,
                    "frequency_hz", _safe_float(sr.frequency_hz)))
        out.append((sr.mouse_id, sr.heating_session,
                    "mean_amplitude", _safe_float(sr.mean_amplitude)))
        out.append((sr.mouse_id, sr.heating_session,
                    "mean_half_width_s", _safe_float(sr.mean_half_width_s)))
    return out


def _extract_preictal_mean(group_result) -> List[SessionScalar]:
    out: List[SessionScalar] = []
    for sr in group_result.session_results:
        out.append((sr.mouse_id, sr.heating_session,
                    "baseline_mean", _safe_float(sr.baseline_mean)))
        out.append((sr.mouse_id, sr.heating_session,
                    "early_heat_mean", _safe_float(sr.early_heat_mean)))
        out.append((sr.mouse_id, sr.heating_session,
                    "late_heat_mean", _safe_float(sr.late_heat_mean)))
        out.append((sr.mouse_id, sr.heating_session,
                    "end_late_heat_mean", _safe_float(sr.end_late_heat_mean)))
        out.append((sr.mouse_id, sr.heating_session,
                    "heating_mean", _safe_float(sr.heating_mean)))
        out.append((sr.mouse_id, sr.heating_session,
                    "ictal_mean", _safe_float(sr.ictal_mean)))
    return out


def _extract_ictal_mean(group_result) -> List[SessionScalar]:
    out: List[SessionScalar] = []
    for sr in group_result.session_results:
        out.append((sr.mouse_id, sr.heating_session,
                    "seizure_mean", _safe_float(sr.seizure_mean)))
        out.append((sr.mouse_id, sr.heating_session,
                    "baseline_mean", _safe_float(sr.baseline_mean)))
        out.append((sr.mouse_id, sr.heating_session,
                    "delta_preictal_ictal", _safe_float(sr.delta_preictal_ictal)))
        out.append((sr.mouse_id, sr.heating_session,
                    "delta_preictal_ictal_max", _safe_float(sr.delta_preictal_ictal_max)))
        out.append((sr.mouse_id, sr.heating_session,
                    "delta_preictal_ictal_min", _safe_float(sr.delta_preictal_ictal_min)))

    # per_session_meta and per_session_auc share indices.
    for lm_name, ta in group_result.triggered_averages.items():
        meta = getattr(ta, "per_session_meta", None) or []
        for (mouse_id, h_sess), auc in zip(meta, ta.per_session_auc):
            out.append((mouse_id, h_sess,
                        f"auc_{lm_name}", _safe_float(auc)))
    return out


def _extract_postictal(group_result) -> List[SessionScalar]:
    out: List[SessionScalar] = []
    for sr in group_result.session_results:
        out.append((sr.mouse_id, sr.heating_session,
                    "final_time", _safe_float(sr.final_time)))
        out.append((sr.mouse_id, sr.heating_session,
                    "final_temp", _safe_float(sr.final_temp)))
        out.append((sr.mouse_id, sr.heating_session,
                    "final_mean_dff", _safe_float(sr.final_mean_dff)))
    return out


def _extract_spike_triggered(group_result) -> List[SessionScalar]:
    out: List[SessionScalar] = []
    for sr in group_result.session_results:
        out.append((sr.mouse_id, sr.heating_session,
                    "auc", _safe_float(sr.auc)))
    return out


def _extract_opto_response(group_result) -> List[SessionScalar]:
    out: List[SessionScalar] = []
    for sr in group_result.session_results:
        out.append((sr.mouse_id, sr.heating_session,
                    "stim_response_max_40s",
                    _safe_float(sr.stim_response_max_40s)))
        out.append((sr.mouse_id, sr.heating_session,
                    "stim_response_min_40s",
                    _safe_float(sr.stim_response_min_40s)))
        out.append((sr.mouse_id, sr.heating_session,
                    "stim_auc_40s", _safe_float(sr.stim_auc_40s)))
    return out


@dataclass
class ModuleSpec:
    key: str
    folder: str  # "means" or "transients"
    extractor: Callable[[Any], List[SessionScalar]]


# Ordered: drives reporting order in the CSV.
_SCALAR_REGISTRY: List[ModuleSpec] = [
    ModuleSpec("Cohort characteristics", "means", _extract_cohort_characteristics),
    ModuleSpec("Baseline transients", "transients", _extract_baseline_transients),
    ModuleSpec("Pre-ictal mean", "means", _extract_preictal_mean),
    ModuleSpec("Ictal mean", "means", _extract_ictal_mean),
    ModuleSpec("Postictal", "means", _extract_postictal),
    ModuleSpec("Spike-triggered averages", "means", _extract_spike_triggered),
    ModuleSpec("Opto response", "means", _extract_opto_response),
]


def list_supported_modules() -> List[str]:
    """Names of analysis modules with registered extractors."""
    return [m.key for m in _SCALAR_REGISTRY]


def extract_all(
    cohort_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten per-cohort module results into long-form scalar rows.

    Parameters
    ----------
    cohort_results : {cohort_key: {module_name: group_result}}

    Returns
    -------
    list of dicts, each with keys:
        cohort, module, folder, measurement, mouse_id, heating_session, value
    """
    rows: List[Dict[str, Any]] = []
    for spec in _SCALAR_REGISTRY:
        for cohort_key, modules in cohort_results.items():
            group_result = modules.get(spec.key)
            if group_result is None:
                continue
            try:
                scalars = spec.extractor(group_result)
            except Exception:
                continue
            for mouse_id, h_sess, meas_name, value in scalars:
                rows.append({
                    "cohort": cohort_key,
                    "module": spec.key,
                    "folder": spec.folder,
                    "measurement": meas_name,
                    "mouse_id": mouse_id,
                    "heating_session": h_sess,
                    "value": value,
                })
    return rows


def group_values(
    long_rows: List[Dict[str, Any]],
    cohorts_to_compare: List[str],
) -> Dict[Tuple[str, str, str], Dict[str, np.ndarray]]:
    """Pivot long rows into {(folder, module, measurement): {cohort: np.ndarray}}.

    Retains only `cohorts_to_compare`, in that order.
    """
    selected = set(cohorts_to_compare)
    grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    for row in long_rows:
        if row["cohort"] not in selected:
            continue
        k = (row["folder"], row["module"], row["measurement"])
        grouped.setdefault(k, {c: [] for c in cohorts_to_compare})
        grouped[k][row["cohort"]].append(row["value"])
    return {
        k: {c: np.asarray(v[c], dtype=float) for c in cohorts_to_compare}
        for k, v in grouped.items()
    }
