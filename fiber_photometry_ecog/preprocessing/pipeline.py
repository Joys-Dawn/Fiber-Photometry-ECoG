"""
Core preprocessing pipeline - no GUI dependencies.

Single entry point: preprocess_session(session, config) applies temperature,
ECoG, photometry, transient detection, and spike detection to a Session in-place.
"""

import numpy as np

from ..core.config import PreprocessingConfig, TRANSIENT_CONFIGS
from ..core.data_models import ProcessedData, Session
from .ecog import filter_ecog
from .stim_detection import detect_stim_burst
from .temperature import process_temperature
from .transient_detection import detect_transients
from .spike_detection import detect_spikes
from .photometry import (
    ChandniStrategy,
    MeilingStrategy,
    IRLSStrategy,
    NoIsosbesticStrategy,
    z_score_baseline,
    highpass_filter,
    detrend_moving_average,
)

STRATEGY_MAP = {
    "A": ChandniStrategy,
    "B": MeilingStrategy,
    "C": IRLSStrategy,
    "D": NoIsosbesticStrategy,
}


def preprocess_session(session: Session, config: PreprocessingConfig) -> None:
    """Run the full preprocessing pipeline on a single session (in-place).

    Parameters
    ----------
    session : Session with raw data and landmarks already set.
    config : PreprocessingConfig (photometry.strategy selects A/B/C/D).

    Raises
    ------
    ValueError
        If the photometry strategy fails (e.g. bad signal quality).
    """
    raw = session.raw
    fs = raw.fs
    strategy_name = config.photometry.strategy

    # --- Opto-stim landmark detection ---
    # Stimulation burst onset/offset supply the UEO/OFF landmarks used by
    # shared analysis. Detect them before processing temperature landmarks.
    if raw.stim_raw is not None:
        onset_s, offset_s = detect_stim_burst(raw.stim_raw, fs)
        session.landmarks.ueo_time = onset_s
        session.landmarks.off_time = offset_s
        # Default normalization uses the first 100 s, capped at stimulation onset.
        if session.landmarks.zscore_baseline_end_time is None:
            session.landmarks.zscore_baseline_end_time = min(100.0, onset_s)
        # Use the same default baseline endpoint for downstream analyses.
        if session.landmarks.heating_start_time is None:
            session.landmarks.heating_start_time = session.landmarks.zscore_baseline_end_time

    # Default z-score baseline ends at heating start or PTZ administration.
    if session.landmarks.zscore_baseline_end_time is None:
        session.landmarks.zscore_baseline_end_time = session.landmarks.heating_start_time

    temperature_baseline_window_s = session.landmarks.baseline_window_s
    if (temperature_baseline_window_s is None
            and session.landmarks.heating_start_time is not None):
        temperature_baseline_window_s = (
            0.0,
            float(session.landmarks.heating_start_time),
        )

    temp_result = process_temperature(
        raw.temperature_raw, raw.temp_bit_volts, fs, config.temperature,
        slope=raw.temp_slope, intercept=raw.temp_intercept,
        baseline_window_s=temperature_baseline_window_s)

    session.landmarks.baseline_temp = temp_result.baseline_temp
    session.landmarks.max_temp = temp_result.max_temp
    session.landmarks.max_temp_time = temp_result.max_temp_time
    session.landmarks.terminal_temp = temp_result.terminal_temp
    session.landmarks.terminal_time = raw.time[-1]

    # Temperature at seizure landmarks (needed by analysis modules + pairing).
    # If the landmark falls in a dropout NaN region, leave the landmark unset.
    if session.landmarks.eec_time is not None:
        idx = min(int(round(session.landmarks.eec_time * fs)), len(temp_result.temperature_smooth) - 1)
        val = float(temp_result.temperature_smooth[idx])
        session.landmarks.eec_temp = None if np.isnan(val) else val
    if session.landmarks.ueo_time is not None:
        idx = min(int(round(session.landmarks.ueo_time * fs)), len(temp_result.temperature_smooth) - 1)
        val = float(temp_result.temperature_smooth[idx])
        session.landmarks.ueo_temp = None if np.isnan(val) else val
    if session.landmarks.behavioral_onset_time is not None:
        idx = min(int(round(session.landmarks.behavioral_onset_time * fs)), len(temp_result.temperature_smooth) - 1)
        val = float(temp_result.temperature_smooth[idx])
        session.landmarks.behavioral_onset_temp = None if np.isnan(val) else val

    # --- ECoG ---
    ecog_filt = filter_ecog(raw.ecog, fs, config.ecog)

    # --- Photometry ---
    strategy = STRATEGY_MAP[strategy_name]()
    phot_result = strategy.preprocess(raw.signal_470, raw.signal_405, fs, config.photometry)

    zscore_baseline_end_s = session.landmarks.zscore_baseline_end_time

    phot_result.dff_zscore = z_score_baseline(
        phot_result.dff, fs, zscore_baseline_end_s)

    # Transient detection uses optional detrending followed by baseline z-scoring.
    if not config.photometry.apply_hpf:
        phot_result.dff_hpf = phot_result.dff_zscore
        dff_for_detection = phot_result.dff
    elif strategy_name == "A":
        # Strategy A: HPF then baseline z-score (PASTa/Donka 2025)
        dff_hpf_raw = highpass_filter(phot_result.dff, fs)
        phot_result.dff_hpf = z_score_baseline(
            dff_hpf_raw, fs, zscore_baseline_end_s)
        # Strategy A measures amplitude and trough on raw dF/F.
        dff_for_detection = phot_result.dff
    else:
        # Strategies B/C/D: moving-average detrending (Wallace 2025).
        dff_detrended = detrend_moving_average(
            phot_result.dff, fs, config.photometry.detrend_window_s)
        phot_result.dff_hpf = z_score_baseline(
            dff_detrended, fs, zscore_baseline_end_s)
        # Prominence and amplitude use detrended fractional dF/F.
        dff_for_detection = dff_detrended

    # --- Store processed data ---
    session.processed = ProcessedData(
        photometry=phot_result,
        ecog_filtered=ecog_filt,
        temperature_c=temp_result.temperature_c,
        temperature_smooth=temp_result.temperature_smooth,
        time=raw.time,
        fs=fs,
    )

    # --- Transient detection ---
    session.transients = detect_transients(
        phot_result.dff_hpf, dff_for_detection, fs,
        TRANSIENT_CONFIGS[strategy_name],
        temp_result.temperature_smooth)

    # --- Spike detection (pre-ictal only: cut 30s before seizure/equivalent) ---
    exclusion_zones = []
    ueo_t = session.landmarks.ueo_time
    if ueo_t is None:
        ueo_t = getattr(session.landmarks, 'equiv_ueo_time', None)
    if ueo_t is not None:
        exclusion_zones.append((ueo_t - 30.0, float('inf')))
    session.spikes = detect_spikes(
        ecog_filt, fs, session.landmarks.heating_start_time, config.spike_detection,
        exclusion_zones=exclusion_zones if exclusion_zones else None)
