"""Photometry maximum, minimum, and AUC over 40 s after stimulation onset.

The mean from [-5, -1] s before onset is subtracted before measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import get_signal_and_time, get_ueo_time, time_to_index

log = logging.getLogger(__name__)


WINDOW_S = 40.0
BL_START_S = 5.0
BL_END_S = 1.0


@dataclass
class OptoResponseSessionResult:
    mouse_id: str
    heating_session: int
    stim_response_max_40s: float
    stim_response_min_40s: float
    stim_auc_40s: float


@dataclass
class OptoResponseGroupResult:
    session_results: List[OptoResponseSessionResult] = field(default_factory=list)
    window_s: float = WINDOW_S


def _per_session(s: Session) -> Optional[OptoResponseSessionResult]:
    if s.raw is None or s.raw.stim_raw is None:
        return None
    stim_t = get_ueo_time(s)
    if stim_t is None or not np.isfinite(stim_t):
        return None

    signal, time, fs = get_signal_and_time(s)
    if signal is None or fs is None or fs <= 0:
        return None

    stim_idx = time_to_index(stim_t, fs)
    post_end = stim_idx + int(round(WINDOW_S * fs)) + 1
    if stim_idx >= len(signal):
        log.warning(
            "Opto response [%s S%d]: stim index past end of signal "
            "(stim at t=%.2fs, fs=%.1fHz, signal len=%d). Skipping.",
            s.mouse_id, s.heating_session, stim_t, fs, len(signal),
        )
        return None

    bl_lo = stim_idx - int(round(BL_START_S * fs))
    bl_hi = stim_idx - int(round(BL_END_S * fs))
    if bl_lo < 0 or bl_hi <= bl_lo:
        log.warning(
            "Opto response [%s S%d]: insufficient pre-stim data for "
            "[-5,-1]s baseline (stim at t=%.2fs). Skipping.",
            s.mouse_id, s.heating_session, stim_t,
        )
        return None
    bl_window = signal[bl_lo:bl_hi]
    bl_mean = float(np.nanmean(bl_window))
    if not np.isfinite(bl_mean):
        return None
    signal = signal - bl_mean

    post = signal[stim_idx:post_end]
    if post.size == 0:
        log.warning(
            "Opto response [%s S%d]: post-stim window is empty "
            "(stim at t=%.2fs, fs=%.1fHz, signal len=%d). Skipping.",
            s.mouse_id, s.heating_session, stim_t, fs, len(signal),
        )
        return None

    max_val = float(np.nanmax(post))
    min_val = float(np.nanmin(post))
    auc = float(np.trapezoid(post, dx=1.0 / fs))

    return OptoResponseSessionResult(
        mouse_id=s.mouse_id,
        heating_session=s.heating_session,
        stim_response_max_40s=max_val,
        stim_response_min_40s=min_val,
        stim_auc_40s=auc,
    )


def compute_opto_response(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> OptoResponseGroupResult:
    """Compute per-session opto response magnitudes for a cohort."""
    out = OptoResponseGroupResult()
    for s in sessions:
        result = _per_session(s)
        if result is not None:
            out.session_results.append(result)
    return out
