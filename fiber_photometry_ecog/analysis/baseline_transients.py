"""
4.2 Baseline / Interictal Transients.

Filter transients to baseline period (start → heating onset).
Compute: frequency (count / duration), mean amplitude, mean half-width.
Per session + group summaries.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import filter_transients_by_time, compute_sem


@dataclass
class BaselineTransientSessionResult:
    mouse_id: str
    heating_session: int
    n_transients: int
    duration_s: float               # baseline period duration
    frequency_hz: float             # count / duration
    mean_amplitude: float           # mean peak_to_trough
    mean_half_width_s: float        # seconds


@dataclass
class BaselineTransientGroupResult:
    session_results: List[BaselineTransientSessionResult]
    frequency_mean: float
    frequency_sem: float
    amplitude_mean: float
    amplitude_sem: float
    half_width_mean: float
    half_width_sem: float


def compute_baseline_transients(
    sessions: List[Session],
    config: AnalysisConfig | None = None,
) -> BaselineTransientGroupResult:
    """Compute baseline transient metrics for a group of sessions."""
    sessions = [s for s in sessions if s.include_for_baseline]

    session_results = []
    freqs = []
    amps = []
    widths = []

    for s in sessions:
        lm = s.landmarks
        t_start = 0.0
        t_end = lm.heating_start_time
        duration = t_end - t_start

        bl_transients = filter_transients_by_time(s.transients, t_start, t_end)
        n = len(bl_transients)
        freq = n / duration if duration > 0 else 0.0

        if n > 0:
            mean_amp = float(np.mean([
                t.z_peak_to_trough if t.z_peak_to_trough is not None else t.peak_to_trough
                for t in bl_transients]))
            mean_hw = float(np.mean([t.half_width for t in bl_transients]))
        else:
            mean_amp = 0.0
            mean_hw = 0.0

        session_results.append(BaselineTransientSessionResult(
            mouse_id=s.mouse_id,
            heating_session=s.heating_session,
            n_transients=n,
            duration_s=duration,
            frequency_hz=freq,
            mean_amplitude=mean_amp,
            mean_half_width_s=mean_hw,
        ))

        freqs.append(freq)
        amps.append(mean_amp)
        widths.append(mean_hw)

    f_arr = np.array(freqs)
    a_arr = np.array(amps)
    w_arr = np.array(widths)

    return BaselineTransientGroupResult(
        session_results=session_results,
        frequency_mean=float(np.mean(f_arr)),
        frequency_sem=compute_sem(f_arr),
        amplitude_mean=float(np.mean(a_arr)),
        amplitude_sem=compute_sem(a_arr),
        half_width_mean=float(np.mean(w_arr)),
        half_width_sem=compute_sem(w_arr),
    )
