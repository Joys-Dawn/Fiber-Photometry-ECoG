"""
Save and load preprocessed session state.

Sessions are saved as .npz files in a .sessions/<strategy>/ directory inside
the experiment folder. Each strategy gets its own subfolder so multiple
preprocessing results can coexist for the same sessions.

Each file contains all arrays (raw + processed) and a JSON string with
scalar metadata, landmarks, transients, spikes, and preprocessing config.

This keeps the results directory clean — .sessions/ is internal state,
results/ is user-facing output.
"""

import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import PhotometryConfig, PreprocessingConfig
from .data_models import (
    Session,
    SessionLandmarks,
    RawData,
    ProcessedData,
    PhotometryResult,
    TransientEvent,
    SpikeEvent,
)

logger = logging.getLogger(__name__)


def _session_filename(session: Session) -> str:
    """Generate a unique filename for a session.

    Uses session_name (e.g. '3339_session1') which is unique within a
    cohort, plus cohort to avoid any cross-cohort collisions.
    """
    cohort = session.cohort or "unknown"
    sname = session.session_name or session.mouse_id
    return f"{sname}_{cohort}.npz"


def save_session(session: Session, sessions_dir: Path) -> Path:
    """Save a preprocessed session to a .npz file.

    Parameters
    ----------
    session : Session with raw and processed data
    sessions_dir : strategy-specific directory
                   (e.g. experiment/.sessions/A/)

    Returns
    -------
    Path to the saved file.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    raw = session.raw
    if raw is not None:
        arrays["raw_signal_470"] = raw.signal_470
        arrays["raw_signal_405"] = raw.signal_405
        arrays["raw_ecog"] = raw.ecog
        arrays["raw_temperature_raw"] = raw.temperature_raw
        arrays["raw_time"] = raw.time
        if raw.emg is not None:
            arrays["raw_emg"] = raw.emg

    proc = session.processed
    if proc is not None:
        if proc.photometry is not None:
            arrays["proc_dff"] = proc.photometry.dff
            if proc.photometry.dff_zscore is not None:
                arrays["proc_dff_zscore"] = proc.photometry.dff_zscore
            if proc.photometry.dff_hpf is not None:
                arrays["proc_dff_hpf"] = proc.photometry.dff_hpf
        if proc.ecog_filtered is not None:
            arrays["proc_ecog_filtered"] = proc.ecog_filtered
        if proc.temperature_c is not None:
            arrays["proc_temperature_c"] = proc.temperature_c
        if proc.temperature_smooth is not None:
            arrays["proc_temperature_smooth"] = proc.temperature_smooth
        if proc.time is not None:
            arrays["proc_time"] = proc.time

    # Scalar metadata as JSON
    meta = {
        "mouse_id": session.mouse_id,
        "genotype": session.genotype,
        "heating_session": session.heating_session,
        "n_seizures": session.n_seizures,
        "sudep": session.sudep,
        "include_session": session.include_session,
        "exclusion_reason": session.exclusion_reason,
        "experiment_label": session.experiment_label,
        "cohort": session.cohort,
        "date": session.date,
        "session_name": session.session_name,
        "include_for_baseline": session.include_for_baseline,
        "include_for_transients": session.include_for_transients,
        "transient_prominence": session.transient_prominence,
        "raw_fs": raw.fs if raw else None,
        "raw_temp_bit_volts": raw.temp_bit_volts if raw else None,
        "raw_temp_slope": raw.temp_slope if raw else None,
        "raw_temp_intercept": raw.temp_intercept if raw else None,
        "proc_fs": proc.fs if proc else None,
    }

    # Preprocessing config
    meta["preprocessing_config"] = asdict(session.preprocessing_config)

    # Landmarks
    if session.landmarks is not None:
        lm = session.landmarks
        meta["landmarks"] = {f.name: getattr(lm, f.name) for f in fields(lm)}

    # Transients
    meta["transients"] = [
        {"peak_time": t.peak_time, "peak_amplitude": t.peak_amplitude,
         "trough_amplitude": t.trough_amplitude, "peak_to_trough": t.peak_to_trough,
         "half_width": t.half_width, "prominence": t.prominence,
         "temperature_at_peak": t.temperature_at_peak,
         "z_peak_amplitude": t.z_peak_amplitude,
         "z_trough_amplitude": t.z_trough_amplitude,
         "z_peak_to_trough": t.z_peak_to_trough}
        for t in session.transients
    ]

    # Spikes
    spikes = session.spikes
    meta["spikes"] = [
        {"time": s.time, "amplitude": s.amplitude, "width_ms": s.width_ms,
         "prominence": s.prominence, "polarity": s.polarity}
        for s in spikes
    ]

    arrays["_metadata_json"] = np.array(json.dumps(meta))

    path = sessions_dir / _session_filename(session)
    np.savez_compressed(path, **arrays)
    logger.info(f"Saved session {session.mouse_id} to {path.name}")
    return path


def load_session(path: Path) -> Session:
    """Load a session from a .npz file.

    Returns a fully reconstructed Session with raw, processed,
    transients, spikes, and preprocessing config.
    """
    data = np.load(path)
    meta = json.loads(str(data["_metadata_json"]))

    # Reconstruct raw. temp_slope/temp_intercept default to Physitemp constants
    # for backward compatibility with older cached sessions that predate the
    # per-session calibration fields.
    raw = RawData(
        signal_470=data["raw_signal_470"],
        signal_405=data["raw_signal_405"],
        ecog=data["raw_ecog"],
        emg=data["raw_emg"] if "raw_emg" in data else None,
        temperature_raw=data["raw_temperature_raw"],
        temp_bit_volts=meta["raw_temp_bit_volts"],
        temp_slope=meta.get("raw_temp_slope", 0.0981),
        temp_intercept=meta.get("raw_temp_intercept", 8.81),
        time=data["raw_time"],
        fs=meta["raw_fs"],
    )

    # Reconstruct processed
    proc = None
    if "proc_dff" in data:
        phot = PhotometryResult(
            dff=data["proc_dff"],
            dff_zscore=data["proc_dff_zscore"] if "proc_dff_zscore" in data else None,
            dff_hpf=data["proc_dff_hpf"] if "proc_dff_hpf" in data else None,
        )
        proc = ProcessedData(
            photometry=phot,
            ecog_filtered=data["proc_ecog_filtered"] if "proc_ecog_filtered" in data else None,
            temperature_c=data["proc_temperature_c"] if "proc_temperature_c" in data else None,
            temperature_smooth=data["proc_temperature_smooth"] if "proc_temperature_smooth" in data else None,
            time=data["proc_time"] if "proc_time" in data else None,
            fs=meta.get("proc_fs"),
        )

    # Landmarks
    landmarks = None
    if "landmarks" in meta and meta["landmarks"]:
        landmarks = SessionLandmarks(**meta["landmarks"])

    # Transients
    transients = [TransientEvent(**t) for t in meta.get("transients", [])]

    # Spikes
    spikes = [SpikeEvent(**s) for s in meta.get("spikes", [])]

    # Preprocessing config
    preproc_config = PreprocessingConfig()
    config_dict = meta.get("preprocessing_config")
    if config_dict:
        phot_dict = config_dict.get("photometry", {})
        preproc_config = PreprocessingConfig(
            photometry=PhotometryConfig(**phot_dict),
        )

    session = Session(
        mouse_id=meta["mouse_id"],
        genotype=meta["genotype"],
        heating_session=meta.get("heating_session", 1),
        n_seizures=meta.get("n_seizures", 0),
        sudep=meta.get("sudep", False),
        include_session=meta.get("include_session", True),
        exclusion_reason=meta.get("exclusion_reason"),
        include_for_baseline=meta.get("include_for_baseline", True),
        include_for_transients=meta.get("include_for_transients", True),
        transient_prominence=meta.get("transient_prominence"),
        experiment_label=meta.get("experiment_label", ""),
        landmarks=landmarks,
        preprocessing_config=preproc_config,
        raw=raw,
        processed=proc,
        transients=transients,
    )
    session.spikes = spikes
    session.cohort = meta.get("cohort") or ""
    session.date = meta.get("date")
    session.session_name = meta.get("session_name")

    return session


def get_sessions_dir(experiment_dir: str, strategy: Optional[str] = None) -> Path:
    """Return the sessions directory path for an experiment folder.

    Parameters
    ----------
    experiment_dir : root experiment folder
    strategy : if provided, returns .sessions/<strategy>/ subfolder

    Returns
    -------
    Path to .sessions/ or .sessions/<strategy>/
    """
    base = Path(experiment_dir) / ".sessions"
    if strategy:
        return base / strategy
    return base


def find_available_strategies(experiment_dir: str) -> List[str]:
    """Find which strategy subfolders exist under .sessions/.

    Returns sorted list of strategy names (e.g. ['A', 'B', 'C']).
    """
    base = Path(experiment_dir) / ".sessions"
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and any(d.glob("*.npz"))
    )


def find_saved_sessions(experiment_dir: str, strategy: Optional[str] = None) -> List[Path]:
    """Find all saved session .npz files.

    Parameters
    ----------
    experiment_dir : root experiment folder
    strategy : if provided, look only in that strategy's subfolder.
               If None, look in base .sessions/ for backward compat.

    Returns
    -------
    Sorted list of .npz file paths.
    """
    sessions_dir = get_sessions_dir(experiment_dir, strategy)
    if not sessions_dir.exists():
        return []
    return sorted(sessions_dir.glob("*.npz"))
