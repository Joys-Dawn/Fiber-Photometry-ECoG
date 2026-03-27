"""Scan experiment folder hierarchy and read Excel data logs.

These functions discover session directories and parse metadata from
the data log spreadsheet. They are used by the GUI to populate the
session tree but have no GUI dependencies themselves.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def scan_experiment_folder(experiment_dir: str) -> List[Dict]:
    """Scan experiment/cohort/mouse_session/ hierarchy.

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
            # Find OEP folder (directory containing Record Node folders)
            oep_path = None
            for sub in session_dir.iterdir():
                if sub.is_dir() and any(
                    d.name.startswith("Record Node") for d in sub.iterdir() if d.is_dir()
                ):
                    oep_path = str(sub)
                    break
            # If no nested OEP folder, check if session_dir itself has Record Nodes
            if oep_path is None and any(
                d.name.startswith("Record Node") for d in session_dir.iterdir() if d.is_dir()
            ):
                oep_path = str(session_dir)

            # Find PPD file
            ppd_files = list(session_dir.glob("*.ppd"))
            ppd_path = str(ppd_files[0]) if ppd_files else None

            if oep_path or ppd_path:
                discovered.append({
                    "cohort": cohort_name,
                    "session_name": session_dir.name,
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


def read_data_log(experiment_dir: str) -> Optional[dict]:
    """Find and read the Excel data log in the experiment folder.

    Returns a dict mapping (mouse_id_str, date_str) -> row dict, or None.
    """
    root = Path(experiment_dir)
    xlsx_files = list(root.glob("*.xlsx"))
    if not xlsx_files:
        return None
    log_path = xlsx_files[0]
    logger.info("Reading data log: %s", log_path.name)
    df = pd.read_excel(log_path)

    # Normalize column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Build lookup by (mouse, date)
    lookup = {}
    for _, row in df.iterrows():
        mouse = str(int(row["mouse"])) if "mouse" in df.columns else None
        if mouse is None:
            continue
        date_raw = row.get("date", "")
        if hasattr(date_raw, "strftime"):
            date_str = date_raw.strftime("%Y-%m-%d")
        else:
            date_str = str(date_raw)[:10]

        genotype_raw = row.get("genotype", "")
        genotype = "Scn1a" if genotype_raw == "H" else "WT" if genotype_raw == "W" else str(genotype_raw)
        seizure = int(row.get("seizure", 0))
        fatal = bool(int(row.get("fatal", 0)))
        exclude = bool(int(row.get("exclude", 0)))
        reason = row.get("reason", None)
        if reason is not None and (isinstance(reason, float) or str(reason) == "nan"):
            reason = None
        heat_start = row.get("heatstart", row.get("heat_start", None))
        if heat_start is not None:
            heat_start = float(heat_start)

        lookup[(mouse, date_str)] = {
            "genotype": genotype,
            "seizure": seizure,
            "sudep": fatal,
            "include": not exclude,
            "exclusion_reason": str(reason) if reason else None,
            "heating_start": heat_start,
        }
    return lookup
