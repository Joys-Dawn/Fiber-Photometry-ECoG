"""Scan experiment folder hierarchy and read Excel data logs.

These functions discover session directories and parse metadata from
the data log spreadsheet. They are used by the GUI to populate the
session tree but have no GUI dependencies themselves.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _extract_date_from_ppd(ppd_path: str) -> Optional[str]:
    """Extract YYYY-MM-DD from a PPD filename (pyPhotometry convention)."""
    m = _DATE_RE.search(Path(ppd_path).name)
    return m.group(0) if m else None


def _has_record_nodes(p: Path) -> bool:
    try:
        return any(d.name.startswith("Record Node") for d in p.iterdir() if d.is_dir())
    except (OSError, PermissionError):
        return False


def scan_experiment_folder(experiment_dir: str) -> List[Dict]:
    """Scan experiment/cohort/mouse_session/ hierarchy.

    Each session_dir may contain one or more recordings (multiple dated OEP
    folders + matching PPD files). When more than one recording is present,
    they are paired by YYYY-MM-DD date and emitted as separate entries with
    a date-suffixed session_name.

    Returns list of dicts with keys: cohort, session_name, oep_path, ppd_path.
    """
    root = Path(experiment_dir)
    discovered = []
    for cohort_dir in sorted(root.iterdir()):
        if not cohort_dir.is_dir():
            continue
        cohort_name = cohort_dir.name
        for session_dir in sorted(cohort_dir.iterdir()):
            if not session_dir.is_dir():
                continue

            oep_subdirs = [
                sub for sub in sorted(session_dir.iterdir())
                if sub.is_dir() and _has_record_nodes(sub)
            ]
            if not oep_subdirs and _has_record_nodes(session_dir):
                oep_subdirs = [session_dir]

            ppd_files = sorted(session_dir.glob("*.ppd"))

            if len(oep_subdirs) <= 1 and len(ppd_files) <= 1:
                oep_path = str(oep_subdirs[0]) if oep_subdirs else None
                ppd_path = str(ppd_files[0]) if ppd_files else None
                if oep_path or ppd_path:
                    discovered.append({
                        "cohort": cohort_name,
                        "session_name": session_dir.name,
                        "oep_path": oep_path,
                        "ppd_path": ppd_path,
                    })
                continue

            # Multi-recording folder: pair OEP and PPD by YYYY-MM-DD
            oep_by_date: Dict[Optional[str], List[str]] = {}
            for oep in oep_subdirs:
                d = extract_date_from_oep(str(oep))
                oep_by_date.setdefault(d, []).append(str(oep))
            ppd_by_date: Dict[Optional[str], List[str]] = {}
            for ppd in ppd_files:
                d = _extract_date_from_ppd(str(ppd))
                ppd_by_date.setdefault(d, []).append(str(ppd))

            all_dates = sorted(
                set(oep_by_date) | set(ppd_by_date),
                key=lambda x: (x is None, x or ""),
            )
            for date in all_dates:
                oep_list = oep_by_date.get(date, [])
                ppd_list = ppd_by_date.get(date, [])
                oep_path = oep_list[0] if oep_list else None
                ppd_path = ppd_list[0] if ppd_list else None
                if len(oep_list) > 1 or len(ppd_list) > 1:
                    logger.warning(
                        "Multiple OEP/PPD files share date %s in %s; using first.",
                        date, session_dir,
                    )
                if not oep_path and not ppd_path:
                    continue
                name = f"{session_dir.name}_{date}" if date else session_dir.name
                discovered.append({
                    "cohort": cohort_name,
                    "session_name": name,
                    "oep_path": oep_path,
                    "ppd_path": ppd_path,
                })
    return discovered


def extract_date_from_oep(oep_path: str) -> Optional[str]:
    """Extract date (YYYY-MM-DD) from OEP folder name."""
    name = Path(oep_path).name
    # OEP folders are named like 2024-10-22_21-57-43_3331
    if len(name) >= 10 and name[4] == "-" and name[7] == "-":
        return name[:10]
    return None


def _find_col(columns, required, forbidden=None):
    """Find column whose lowercased name contains all `required` substrings
    and none of the `forbidden` ones. Returns the original column name or None.
    """
    forbidden = forbidden or []
    for col in columns:
        cl = str(col).lower().strip()
        if all(s in cl for s in required) and not any(s in cl for s in forbidden):
            return col
    return None


def _parse_time_value(v):
    """Parse a time value to seconds. Accepts numeric (already seconds),
    `'5min'` / `'10 min'` (minutes), numeric strings, or empty/None.
    """
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    if not s or s in ("nan", "none"):
        return None
    if s.endswith("min"):
        return float(s[:-3].strip()) * 60.0
    if s.endswith("sec"):
        return float(s[:-3].strip())
    if s.endswith("s"):
        return float(s[:-1].strip())
    try:
        return float(s)
    except ValueError:
        return None


def _parse_bool(v):
    """Parse Yes/No / true/false / 0/1 → bool. Returns None if blank."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return True
    if s in ("no", "n", "false", "0"):
        return False
    return None


def _parse_date(v):
    """Parse date to 'YYYY-MM-DD'. Accepts datetime, 'YYYY-MM-DD' string, or
    8-digit 'YYYYMMDD' int/string.
    """
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip()
    if not s:
        return None
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    digits = s.split(".")[0]  # drop trailing ".0" from float-coerced ints
    if digits.isdigit() and len(digits) == 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return s[:10]


def _genotype_from_cohort(cohort: str) -> str:
    """Map cohort string → 'Scn1a' or 'WT'."""
    c = cohort.lower()
    if "wt" in c:
        return "WT"
    if "scn1a" in c or "het" in c:
        return "Scn1a"
    return cohort


def read_data_log(experiment_dir: str) -> Optional[dict]:
    """Find and read the Excel data log. Matches the spec log schema.

    Spec columns (recognized case-insensitively, partial-match):
      - cohort, date, mouse ID #, heating session #
      - seizure?, SUDEP?
      - channel(s): ECoG / EMG / temperature
      - heating start time  (numeric seconds or e.g. '5min')
      - baseline temperature (optional; calculated if missing)
      - include?, reason for exclusion (if applicable)
      - EEC/UEO/OFF (1st/last seizure) - time

    Also tolerates Chandni's custom columns (mouse, genotype=H/W, fatal,
    exclude, heatStart) for backward compatibility.

    Returns dict keyed by (mouse_id_str, date_str) → row dict.
    """
    root = Path(experiment_dir)
    xlsx_files = list(root.glob("*.xlsx"))
    if not xlsx_files:
        return None
    log_path = xlsx_files[0]
    logger.info("Reading data log: %s", log_path.name)

    # Handle spec-format logs with a banner row ("Step 1", "Step 2", "Step 3")
    df = pd.read_excel(log_path, header=0)
    banner = any("step" in str(c).lower() or "unnamed" in str(c).lower()
                 for c in df.columns)
    if banner:
        df = pd.read_excel(log_path, header=1)
    df.columns = [str(c).strip() for c in df.columns]

    # Locate columns (by substring match)
    cols = df.columns
    mouse_col = _find_col(cols, ["mouse"])
    date_col = _find_col(cols, ["date"])
    cohort_col = _find_col(cols, ["cohort"])
    genotype_col = _find_col(cols, ["genotype"])
    session_col = _find_col(cols, ["heating session"])
    seizure_col = _find_col(cols, ["seizure"], forbidden=["equivalent", "eec", "ueo", "off"])
    sudep_col = _find_col(cols, ["sudep"]) or _find_col(cols, ["fatal"])
    include_col = _find_col(cols, ["include"])
    exclude_col = _find_col(cols, ["exclude"], forbidden=["include"])
    reason_col = _find_col(cols, ["reason"])
    heatstart_col = _find_col(cols, ["heat", "start"])
    baseline_temp_col = _find_col(cols, ["baseline", "temp"])
    ecog_ch_col = _find_col(cols, ["channel", "ecog"])
    emg_ch_col = _find_col(cols, ["channel", "emg"])
    temp_ch_col = _find_col(cols, ["channel", "temp"])
    eec_col = _find_col(cols, ["eec", "time"], forbidden=["equivalent", "temp"])
    ueo_col = _find_col(cols, ["ueo", "time"], forbidden=["equivalent", "temp"])
    off_col = _find_col(cols, ["off", "time"], forbidden=["equivalent", "temp"])

    if mouse_col is None:
        logger.warning("Data log has no recognizable 'mouse' column: %s", log_path.name)
        return None

    lookup = {}
    for _, row in df.iterrows():
        raw_mouse = row.get(mouse_col)
        if raw_mouse is None or (isinstance(raw_mouse, float) and pd.isna(raw_mouse)):
            continue
        try:
            mouse = str(int(raw_mouse))
        except (ValueError, TypeError):
            mouse = str(raw_mouse).strip()
            if not mouse:
                continue

        date_str = _parse_date(row.get(date_col)) if date_col else None

        # Genotype: prefer explicit genotype column (Chandni), else cohort
        genotype = None
        if genotype_col:
            g = row.get(genotype_col)
            if g == "H":
                genotype = "Scn1a"
            elif g == "W":
                genotype = "WT"
            elif isinstance(g, str) and g.strip():
                genotype = g.strip()
        if genotype is None and cohort_col:
            genotype = _genotype_from_cohort(str(row.get(cohort_col, "")))
        if genotype is None:
            genotype = "Unknown"

        # Seizure count (Yes/No → 1/0, or int)
        seizure = 0
        if seizure_col:
            v = row.get(seizure_col)
            b = _parse_bool(v)
            if b is not None:
                seizure = 1 if b else 0
            elif isinstance(v, (int, float)) and not pd.isna(v):
                seizure = int(v)

        sudep = _parse_bool(row.get(sudep_col)) if sudep_col else False
        sudep = bool(sudep) if sudep is not None else False

        # Include: prefer `include?`, fall back to inverse of `exclude`
        include_val = None
        if include_col:
            include_val = _parse_bool(row.get(include_col))
        if include_val is None and exclude_col:
            ex = _parse_bool(row.get(exclude_col))
            if ex is not None:
                include_val = not ex
        include = True if include_val is None else include_val

        reason = row.get(reason_col) if reason_col else None
        if reason is not None and (isinstance(reason, float) and pd.isna(reason)
                                   or str(reason).strip().lower() in ("nan", "none", "")):
            reason = None
        else:
            reason = str(reason).strip() if reason is not None else None

        heat_start = _parse_time_value(row.get(heatstart_col)) if heatstart_col else None
        eec_time = _parse_time_value(row.get(eec_col)) if eec_col else None
        ueo_time = _parse_time_value(row.get(ueo_col)) if ueo_col else None
        off_time = _parse_time_value(row.get(off_col)) if off_col else None

        baseline_temp = None
        if baseline_temp_col:
            v = row.get(baseline_temp_col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                try:
                    baseline_temp = float(v)
                except (ValueError, TypeError):
                    baseline_temp = None

        def _parse_int(col):
            if not col:
                return None
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        ecog_ch = _parse_int(ecog_ch_col)
        emg_ch = _parse_int(emg_ch_col)
        temp_ch = _parse_int(temp_ch_col)
        heating_session = _parse_int(session_col) or 1

        lookup[(mouse, date_str)] = {
            "genotype": genotype,
            "seizure": seizure,
            "sudep": sudep,
            "include": include,
            "exclusion_reason": reason,
            "heating_start": heat_start,
            "heating_session": heating_session,
            "baseline_temperature": baseline_temp,
            "ecog_channel": ecog_ch,
            "emg_channel": emg_ch,
            "temperature_channel": temp_ch,
            "eec_time": eec_time,
            "ueo_time": ueo_time,
            "off_time": off_time,
        }
    return lookup
