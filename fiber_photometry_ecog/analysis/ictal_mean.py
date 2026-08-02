"""
4.5 Ictal Mean Signal.

Mean z-ΔF/F within seizure period vs baseline.
Δ z-ΔF/F between pre-ictal and ictal windows.
Event-triggered averages (±30s) for each landmark.
AUC via trapezoidal integration on each triggered average.
Multi-seizure handling: EEC/UEO from 1st seizure, OFF from last seizure.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from ..core.config import AnalysisConfig
from ..core.data_models import Session
from ._helpers import (
    get_eec_time,
    get_ueo_time,
    get_behavioral_onset_time,
    get_off_time,
    get_signal_and_time,
    time_to_index,
    compute_sem,
)

log = logging.getLogger(__name__)


# Landmarks that may legitimately be absent on individual sessions (e.g.,
# behavioral_onset isn't scored on every session). For these, the triggered
# average drops sessions that lack the landmark but LOGS the per-session
# exclusion so it's visible. Every other landmark is all-or-none-or-crash.
_OPTIONAL_LANDMARKS = {"behavioral_onset"}


# Landmarks whose AUC integration window is fixed at 100 s post-event per spec
# (instead of using the cohort's mean seizure duration). The post-event trace
# window auto-extends to fit this so the integration is fully visible.
FIXED_AUC_WINDOW_S = {"OFF": 100.0}


@dataclass
class TriggeredAverage:
    """Event-triggered average for one landmark."""
    time_axis: np.ndarray       # seconds relative to event (asymmetric allowed)
    mean_trace: np.ndarray      # mean z-ΔF/F across sessions
    sem_trace: np.ndarray       # SEM across sessions
    auc: float                  # trapezoidal AUC of mean trace
    per_session_auc: List[float]    # AUC for each session's triggered trace
    per_session_traces: List[np.ndarray]  # individual session traces
    auc_window_s: float = 0.0   # integration window (0 to this many seconds post-event)
    # (mouse_id, heating_session) for each per_session_trace, in the same order.
    # Lets downstream plots (heatmaps) keep labels aligned with the trace matrix
    # even when sessions get filtered out (missing landmark, trace boundary).
    per_session_meta: List[tuple] = None


@dataclass
class IctalMeanSessionResult:
    mouse_id: str
    heating_session: int
    seizure_mean: float         # mean z-ΔF/F during seizure/equivalent period
    baseline_mean: float        # mean z-ΔF/F during baseline
    delta_preictal_ictal: float # signed peak deflection (whichever |max| or |min| is larger)
    # Per spec: report ictal max and min separately so the experimenter can
    # pick whichever direction matters for their indicator (some sensors
    # report positive transients, others negative).
    delta_preictal_ictal_max: float = np.nan  # ictal_max - late_heat_mean
    delta_preictal_ictal_min: float = np.nan  # ictal_min - late_heat_mean


@dataclass
class IctalMeanGroupResult:
    session_results: List[IctalMeanSessionResult]
    seizure_mean: float
    seizure_sem: float
    baseline_mean: float
    baseline_sem: float
    delta_mean: float
    delta_sem: float
    triggered_averages: Dict[str, TriggeredAverage]  # keyed by landmark name
    auc_window_s: float = 0.0    # integration window used for triggered AUCs
    mean_seizure_duration_s: float = np.nan  # cohort mean (UEO→OFF) when known
    delta_max_mean: float = np.nan   # group mean of delta_preictal_ictal_max
    delta_max_sem: float = np.nan
    delta_min_mean: float = np.nan   # group mean of delta_preictal_ictal_min
    delta_min_sem: float = np.nan


@dataclass(frozen=True)
class SustainedExtrema:
    """Duration-qualified maximum and minimum within one signal window.

    ``maximum`` is the highest threshold for which the signal remains at or
    above that threshold for ``duration_samples`` consecutive samples.
    ``minimum`` is the lowest threshold for which the signal remains at or
    below that threshold for the same duration. Window indices are half-open;
    point indices identify the sample that limits each selected threshold.
    """

    maximum: float
    minimum: float
    maximum_index: int
    minimum_index: int
    maximum_window_start: int
    minimum_window_start: int
    duration_samples: int


def compute_sustained_extrema(
    signal: np.ndarray,
    fs: float,
    min_duration_s: float | None = None,
) -> SustainedExtrema:
    """Return sample-wise or duration-qualified extrema without smoothing.

    A duration-qualified maximum is the maximum of every sliding window's
    minimum (maximin). The corresponding minimum is the minimum of every
    sliding window's maximum (minimax). This directly implements the rule
    that the signal must remain above/below the reported value for the full
    requested duration.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Extrema require a non-empty one-dimensional signal.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Extrema require finite signal values.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Extrema require a positive sampling rate; got {fs}.")

    if min_duration_s is None:
        duration_samples = 1
    else:
        if not np.isfinite(min_duration_s) or min_duration_s <= 0:
            raise ValueError(
                "Extrema minimum duration must be positive; "
                f"got {min_duration_s}."
            )
        duration_samples = max(1, int(round(min_duration_s * fs)))

    if duration_samples > values.size:
        raise ValueError(
            f"Extrema window has {values.size} samples, fewer than the "
            f"required {duration_samples} samples."
        )

    if duration_samples == 1:
        maximum_index = int(np.argmax(values))
        minimum_index = int(np.argmin(values))
        return SustainedExtrema(
            maximum=float(values[maximum_index]),
            minimum=float(values[minimum_index]),
            maximum_index=maximum_index,
            minimum_index=minimum_index,
            maximum_window_start=maximum_index,
            minimum_window_start=minimum_index,
            duration_samples=1,
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        values, duration_samples,
    )
    window_minima = np.min(windows, axis=1)
    window_maxima = np.max(windows, axis=1)
    maximum_window_start = int(np.argmax(window_minima))
    minimum_window_start = int(np.argmin(window_maxima))

    maximum_window = values[
        maximum_window_start:maximum_window_start + duration_samples
    ]
    minimum_window = values[
        minimum_window_start:minimum_window_start + duration_samples
    ]
    maximum_index = maximum_window_start + int(np.argmin(maximum_window))
    minimum_index = minimum_window_start + int(np.argmax(minimum_window))

    return SustainedExtrema(
        maximum=float(window_minima[maximum_window_start]),
        minimum=float(window_maxima[minimum_window_start]),
        maximum_index=maximum_index,
        minimum_index=minimum_index,
        maximum_window_start=maximum_window_start,
        minimum_window_start=minimum_window_start,
        duration_samples=duration_samples,
    )


def _extract_triggered_trace(
    signal: np.ndarray,
    fs: float,
    event_idx: int,
    pre_samples: int,
    post_samples: int,
    nan_pad: bool = False,
) -> Optional[np.ndarray]:
    """Extract an (asymmetric) window of signal around event_idx.

    With `nan_pad=False` (default), returns None if the window extends
    beyond the signal boundaries. With `nan_pad=True`, returns a
    full-length array (pre_samples + post_samples + 1) padded with NaN
    wherever the signal doesn't reach - so downstream nanmean aggregation
    can still include the session over whatever range it does cover.
    """
    start = event_idx - pre_samples
    end = event_idx + post_samples + 1
    if not nan_pad:
        if start < 0 or end > len(signal):
            return None
        return signal[start:end].copy()

    total = pre_samples + post_samples + 1
    out = np.full(total, np.nan, dtype=float)
    sig_start = max(0, start)
    sig_end = min(len(signal), end)
    if sig_end <= sig_start:
        return None
    out_start = sig_start - start
    out_end = out_start + (sig_end - sig_start)
    out[out_start:out_end] = signal[sig_start:sig_end]
    return out


def _compute_triggered_average(
    sessions: List[Session],
    landmark_func,
    pre_window_s: float,
    post_window_s: float,
    bl_start_s: float,
    bl_end_s: float,
    auc_end_s: float,
    nan_pad: bool = True,
    landmark_name: str = "",
) -> TriggeredAverage:
    """Compute event-triggered average across sessions for a given landmark.

    Window is asymmetric: pre_window_s seconds before the event, post_window_s
    seconds after. The AUC is integrated from 0 to auc_end_s post-event and
    requires post_window_s >= auc_end_s.

    All-or-none-or-crash policy on the landmark:
      - All sessions have a finite landmark time: compute as normal.
      - No sessions have it: return an empty placeholder (the paradigm
        legitimately doesn't use this landmark, e.g. opto has no max_temp).
      - Mixed: crash. Asymmetric per-cohort exclusion is the silent-drop
        bug we refuse to ship. Exception: landmarks in `_OPTIONAL_LANDMARKS`
        (currently just `behavioral_onset`) drop the missing sessions but
        emit an INFO log naming them so the drop is visible.

    `nan_pad=True` (default) lets sessions whose RECORDING doesn't fully
    span the window contribute over whatever range they do cover (NaN-
    padded elsewhere), with cohort mean/SEM using nanmean/nanstd. Short
    recordings no longer blank out the cohort trace.
    """
    # Landmark availability check — done UP FRONT so the all-or-none-or-
    # crash decision is visible, not hidden inside the per-session loop.
    is_optional = landmark_name in _OPTIONAL_LANDMARKS
    valid_sessions: List[Session] = []
    missing: List[tuple] = []
    for s in sessions:
        t = landmark_func(s)
        if t is None or not np.isfinite(t):
            missing.append((s.mouse_id, s.heating_session))
        else:
            valid_sessions.append(s)

    if missing and valid_sessions:
        if is_optional:
            log.info(
                "Triggered avg [%s]: %d/%d session(s) missing landmark, "
                "dropping from this metric only: %s",
                landmark_name or "?", len(missing), len(sessions), missing,
            )
        else:
            raise ValueError(
                f"Triggered avg [{landmark_name or '?'}]: asymmetric landmark "
                f"availability ({len(valid_sessions)}/{len(sessions)} have it). "
                f"Sessions missing: {missing}. Either every session has the "
                f"landmark or none should; the silent-skip path is disabled."
            )

    if not valid_sessions:
        return TriggeredAverage(
            time_axis=np.array([0.0]),
            mean_trace=np.array([np.nan]),
            sem_trace=np.array([np.nan]),
            auc=np.nan,
            per_session_auc=[],
            per_session_traces=[],
            auc_window_s=auc_end_s,
            per_session_meta=[],
        )

    traces = []
    aucs = []
    meta = []

    for s in valid_sessions:
        t_event = landmark_func(s)
        signal, time, fs = get_signal_and_time(s)
        pre_samples = int(round(pre_window_s * fs))
        post_samples = int(round(post_window_s * fs))
        event_idx = time_to_index(t_event, fs)
        trace = _extract_triggered_trace(
            signal, fs, event_idx, pre_samples, post_samples,
            nan_pad=nan_pad)
        if trace is None:
            # nan_pad=True only returns None if the entire window lies
            # outside the recording — should not happen for valid landmarks
            # with non-zero recordings. Crash loudly instead of silently
            # dropping the session.
            raise ValueError(
                f"Triggered avg [{landmark_name or '?'}]: session "
                f"{s.mouse_id} S{s.heating_session} produced an empty "
                f"trace at event_idx={event_idx}, signal_len={len(signal)} "
                f"(pre_samples={pre_samples}, post_samples={post_samples})."
            )

        bl_lo = pre_samples - int(round(bl_start_s * fs))
        bl_hi = pre_samples - int(round(bl_end_s * fs))
        bl_window = trace[bl_lo:bl_hi]
        if nan_pad:
            if np.any(~np.isnan(bl_window)):
                trace = trace - float(np.nanmean(bl_window))
        else:
            trace = trace - np.mean(bl_window)
        traces.append(trace)

        auc_end_idx = pre_samples + int(round(auc_end_s * fs)) + 1
        auc_window = trace[pre_samples:auc_end_idx]
        if nan_pad and np.any(np.isnan(auc_window)):
            auc = float("nan")
        else:
            auc = float(np.trapezoid(auc_window, dx=1.0 / fs))
        aucs.append(auc)
        meta.append((s.mouse_id, s.heating_session))

    trace_mat = np.array(traces)
    if nan_pad:
        # nanmean/nanstd let sessions that don't span the full window
        # contribute over whatever range they do cover. SEM uses the
        # per-bin valid-N so the error bar widens where fewer sessions
        # contribute, instead of going zero or NaN.
        mean_trace = np.nanmean(trace_mat, axis=0)
        n_per_bin = np.sum(~np.isnan(trace_mat), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            std_trace = np.nanstd(trace_mat, axis=0, ddof=1)
            sem_trace = np.where(n_per_bin > 1,
                                 std_trace / np.sqrt(n_per_bin),
                                 0.0)
    else:
        mean_trace = np.mean(trace_mat, axis=0)
        sem_trace = (np.std(trace_mat, axis=0, ddof=1) / np.sqrt(len(traces))
                     if len(traces) > 1 else np.zeros(trace_mat.shape[1]))

    fs = valid_sessions[0].processed.fs
    pre_samples = int(round(pre_window_s * fs))
    n_samples = trace_mat.shape[1]
    time_axis = np.linspace(-pre_window_s, post_window_s, n_samples)
    auc_end_idx = pre_samples + int(round(auc_end_s * fs)) + 1
    auc_window_mean = mean_trace[pre_samples:auc_end_idx]
    if nan_pad and np.any(np.isnan(auc_window_mean)):
        auc_mean = float("nan")
    else:
        auc_mean = float(np.trapezoid(auc_window_mean, dx=1.0 / fs))

    return TriggeredAverage(
        time_axis=time_axis,
        mean_trace=mean_trace,
        sem_trace=sem_trace,
        auc=auc_mean,
        per_session_auc=aucs,
        per_session_traces=traces,
        auc_window_s=auc_end_s,
        per_session_meta=meta,
    )


def mean_seizure_duration(sessions: List[Session]) -> float:
    """Mean UEO->OFF duration across all sessions with both landmarks set.

    Driver-level helper so every cohort's ictal-mean run uses the SAME
    AUC integration window. Includes any session whose `ueo_time` AND
    `off_time` are populated, which covers seized hyperthermia/PTZ
    sessions and opto stim bursts alike. Hyperthermia controls have only
    `equiv_ueo_time` (set by pairing) and are correctly skipped.

    Crashes if no session qualifies - the caller is asking for an AUC
    window with no events to derive it from.
    """
    durations: List[float] = []
    for s in sessions:
        lm = s.landmarks
        if (lm is not None
                and lm.ueo_time is not None and lm.off_time is not None
                and lm.off_time > lm.ueo_time):
            durations.append(lm.off_time - lm.ueo_time)
    if not durations:
        raise ValueError(
            "mean_seizure_duration: no sessions with valid "
            "ueo_time and off_time landmarks - cannot compute AUC window."
        )
    return float(np.mean(durations))


def compute_ictal_mean(
    sessions: List[Session],
    auc_end_s: float,
    config: AnalysisConfig | None = None,
    auc_end_s_per_landmark: Optional[Dict[str, float]] = None,
    extrema_min_duration_s: float | None = None,
) -> IctalMeanGroupResult:
    """Compute ictal mean signal metrics for a group of sessions.

    Parameters
    ----------
    auc_end_s : triggered-AUC integration window in seconds. Required.
        Callers MUST pass the global mean seizure duration computed via
        `mean_seizure_duration(all_sessions)` so every cohort's AUC uses
        the same window. There is no fallback - if you don't have a
        sensible value, you shouldn't be calling this function.
    auc_end_s_per_landmark : optional dict mapping landmark name (e.g. "OFF")
        to an integration window in seconds, overriding the global default for
        that landmark. Also extends the post-event trace window so the full
        AUC interval is visible. Per spec, OFF defaults to 100 s.
    extrema_min_duration_s : optional duration requirement for ictal extrema.
        When omitted, max/min remain sample-wise. When provided, max is the
        highest value sustained at-or-above for this duration and min is the
        lowest value sustained at-or-below for this duration.
    """
    if config is None:
        config = AnalysisConfig()

    # Merge spec defaults with caller overrides (caller wins).
    per_lm = dict(FIXED_AUC_WINDOW_S)
    if auc_end_s_per_landmark:
        per_lm.update(auc_end_s_per_landmark)

    sessions = [s for s in sessions if s.include_for_baseline]

    session_results = []
    sz_means = []
    bl_means = []
    deltas = []
    deltas_max = []
    deltas_min = []

    for s in sessions:
        signal, time, fs = get_signal_and_time(s)
        lm = s.landmarks

        # Baseline mean: start to heating
        i_heat = time_to_index(lm.heating_start_time, fs)
        bl_mean = float(np.mean(signal[:i_heat]))

        # Seizure/equivalent period mean
        ueo_t = get_ueo_time(s)
        off_t = get_off_time(s)

        i_ueo = time_to_index(ueo_t, fs)
        i_off = time_to_index(off_t, fs)
        ictal_window = signal[i_ueo:i_off]
        sz_mean = float(np.mean(ictal_window))

        # Late heat mean: fixed 10s window before UEO
        end_late_n = min(int(10.0 * fs), i_ueo - i_heat)
        late_heat_mean = float(np.mean(signal[i_ueo - end_late_n:i_ueo]))

        # Per spec: report both ictal max and ictal min separately so the
        # experimenter can choose either direction. Also keep the legacy
        # |peak|-based delta for backward compatibility.
        if extrema_min_duration_s is None:
            ictal_max = float(np.max(ictal_window))
            ictal_min = float(np.min(ictal_window))
        else:
            extrema = compute_sustained_extrema(
                ictal_window,
                fs,
                extrema_min_duration_s,
            )
            ictal_max = extrema.maximum
            ictal_min = extrema.minimum
        delta_max = ictal_max - late_heat_mean
        delta_min = ictal_min - late_heat_mean
        if abs(delta_max) >= abs(delta_min):
            delta = delta_max
        else:
            delta = delta_min

        session_results.append(IctalMeanSessionResult(
            mouse_id=s.mouse_id,
            heating_session=s.heating_session,
            seizure_mean=sz_mean,
            baseline_mean=bl_mean,
            delta_preictal_ictal=delta,
            delta_preictal_ictal_max=delta_max,
            delta_preictal_ictal_min=delta_min,
        ))
        sz_means.append(sz_mean)
        bl_means.append(bl_mean)
        deltas.append(delta)
        deltas_max.append(delta_max)
        deltas_min.append(delta_min)
    # Triggered averages for each landmark.
    window_s = config.triggered_window_s
    default_auc_end_s = float(auc_end_s)

    landmarks = {
        "EEC": get_eec_time,
        "UEO": get_ueo_time,
        "behavioral_onset": get_behavioral_onset_time,
        "OFF": get_off_time,
        "max_temp": lambda s: s.landmarks.max_temp_time if s.landmarks else None,
    }

    triggered = {}
    for name, func in landmarks.items():
        auc_end_s = per_lm.get(name, default_auc_end_s)
        # Post-window must fit the AUC interval; add a small buffer for plots.
        post_window_s = max(window_s, auc_end_s + 5.0)
        triggered[name] = _compute_triggered_average(
            sessions, func, window_s, post_window_s,
            config.triggered_baseline_start_s,
            config.triggered_baseline_end_s,
            auc_end_s,
            landmark_name=name,
        )

    sz_arr = np.array(sz_means)
    bl_arr = np.array(bl_means)
    d_arr = np.array(deltas)
    d_max_arr = np.array(deltas_max) if deltas_max else np.array([np.nan])
    d_min_arr = np.array(deltas_min) if deltas_min else np.array([np.nan])

    return IctalMeanGroupResult(
        session_results=session_results,
        seizure_mean=float(np.mean(sz_arr)),
        seizure_sem=compute_sem(sz_arr),
        baseline_mean=float(np.mean(bl_arr)),
        baseline_sem=compute_sem(bl_arr),
        delta_mean=float(np.mean(d_arr)),
        delta_sem=compute_sem(d_arr),
        triggered_averages=triggered,
        auc_window_s=default_auc_end_s,
        mean_seizure_duration_s=default_auc_end_s,
        delta_max_mean=float(np.nanmean(d_max_arr)) if deltas_max else np.nan,
        delta_max_sem=compute_sem(d_max_arr) if deltas_max else np.nan,
        delta_min_mean=float(np.nanmean(d_min_arr)) if deltas_min else np.nan,
        delta_min_sem=compute_sem(d_min_arr) if deltas_min else np.nan,
    )


def compute_wide_ueo_triggered(
    sessions: List[Session],
    window_s: float = 150.0,
    bl_start_s: float = 5.0,
    bl_end_s: float = 1.0,
    auc_end_s: float = 60.0,
    post_window_s: Optional[float] = None,
) -> TriggeredAverage:
    """Wide-window UEO triggered average for per-cohort visualization.

    `window_s` controls the pre-event window; `post_window_s` (when given)
    overrides the post-event extent. Callers needing the extended
    UEO-aligned heatmaps/traces pass post_window_s=1000 so the trace data
    actually spans the +1000s tail the plot expects.

    NaN-pads sessions that don't fully span the requested post window
    (e.g. opto recordings that are ~500s total but the caller asks for a
    +1000s tail) so the cohort mean/SEM still reflect the available
    coverage instead of blanking the entire trace.
    """
    sessions = [s for s in sessions if s.include_for_baseline]
    post = window_s if post_window_s is None else post_window_s
    return _compute_triggered_average(
        sessions, get_ueo_time, window_s, post,
        bl_start_s, bl_end_s, auc_end_s,
        nan_pad=True,
        landmark_name="UEO",
    )
