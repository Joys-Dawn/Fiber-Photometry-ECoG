"""
TTL barcode synchronization between pyPhotometry and Open Ephys.

Follows Chandni's sync_signals.m approach:
1. Detect rising edges in both PPD and OEP digital/TTL channels
2. Match pulses across systems (nearest-neighbour within tolerance)
3. Linear regression (polyfit) on matched pulse times -> drift correction
4. Warp photometry timeline using the linear mapping
5. PCHIP-interpolate photometry signals onto the ECoG timebase
6. Trim to overlapping region

Improvement over single-edge alignment: uses ALL matched pulses to fit a
linear model, which corrects for clock drift (not just a fixed offset).
"""

from dataclasses import dataclass
from typing import Optional

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import decimate

logger = logging.getLogger(__name__)

from .ppd_reader import PPDData
from .oep_reader import OEPData


@dataclass
class SyncResult:
    signal_470: np.ndarray     # photometry signal, upsampled & aligned
    signal_405: np.ndarray     # isosbestic signal, upsampled & aligned
    ecog: np.ndarray           # ECoG signal, trimmed to overlap
    emg: Optional[np.ndarray]  # EMG signal, trimmed to overlap (None if not recorded)
    temperature_raw: np.ndarray # temperature signal, trimmed to overlap
    temp_bit_volts: float      # bit_volts factor for temperature conversion
    time: np.ndarray           # common time vector (seconds)
    fs: float                  # common sampling rate (ECoG rate)
    n_matched: int             # number of matched TTL pulses
    drift_ppm: float           # clock drift in parts per million
    scaling: float             # linear mapping slope
    offset: float              # linear mapping offset (seconds)
    residual_ms: float         # mean residual after linear fit (ms)


def _detect_rising_edges_ppd(ppd: PPDData) -> np.ndarray:
    """Return rising-edge times (seconds) from PPD digital channel."""
    return ppd.pulse_times


def _detect_rising_edges_oep(oep: OEPData) -> np.ndarray:
    """Return rising-edge times (seconds) from OEP TTL events.

    OEP TTL states: +1 = rising edge, -1 = falling edge.
    Uses sample_numbers / fs for reliable timing (OEP timestamps can be
    unreliable absolute values from the recording system's internal clock).
    """
    if len(oep.ttl_states) == 0:
        return np.array([])

    rising_mask = oep.ttl_states > 0
    return oep.ttl_sample_numbers[rising_mask].astype(np.float64) / oep.fs


def _match_pulses(
    photo_times: np.ndarray,
    eeg_times: np.ndarray,
    tolerance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Match TTL pulses between photometry and EEG by nearest-neighbour.

    For each EEG pulse, finds the closest photometry pulse within `tolerance`
    seconds. This mirrors Chandni's matching loop in sync_signals.m.

    Returns matched_photo, matched_eeg arrays of corresponding times.
    """
    matched_photo = []
    matched_eeg = []
    used_photo = set()

    for eeg_t in eeg_times:
        diffs = np.abs(photo_times - eeg_t)
        j = int(np.argmin(diffs))
        if diffs[j] <= tolerance and j not in used_photo:
            matched_eeg.append(eeg_t)
            matched_photo.append(photo_times[j])
            used_photo.add(j)

    return np.array(matched_photo), np.array(matched_eeg)


def synchronize(
    ppd: PPDData,
    oep: OEPData,
    pulse_tolerance: float = 1.0,
    min_matched_pulses: int = 3,
) -> SyncResult:
    """Synchronize photometry and ECoG signals using TTL barcodes.

    Parameters
    ----------
    ppd : PPDData from read_ppd
    oep : OEPData from read_oep
    pulse_tolerance : max time difference (seconds) for matching pulses
    min_matched_pulses : reject session if fewer pulses matched

    Returns
    -------
    SyncResult with aligned signals on a common timebase.

    Raises
    ------
    ValueError if too few pulses are matched.
    """
    # --- Step 1: Detect rising edges ---
    photo_edges = _detect_rising_edges_ppd(ppd)
    eeg_edges = _detect_rising_edges_oep(oep)

    if len(photo_edges) == 0:
        raise ValueError("No rising edges found in photometry sync channel")
    if len(eeg_edges) == 0:
        raise ValueError("No rising edges found in EEG TTL channel")

    # --- Step 2: Correct starting time offset (per Chandni's approach) ---
    t_first_eeg = eeg_edges[0]
    t_first_photo = photo_edges[0]
    start_diff = abs(t_first_photo - t_first_eeg)

    if t_first_eeg < t_first_photo:
        eeg_edges_adj = eeg_edges + start_diff
        photo_edges_adj = photo_edges
    else:
        photo_edges_adj = photo_edges + start_diff
        eeg_edges_adj = eeg_edges

    # --- Step 3: Match pulses ---
    matched_photo, matched_eeg = _match_pulses(
        photo_edges_adj, eeg_edges_adj, tolerance=pulse_tolerance
    )

    if len(matched_photo) < min_matched_pulses:
        raise ValueError(
            f"Only {len(matched_photo)} TTL pulses matched "
            f"(minimum {min_matched_pulses} required). "
            f"Check TTL signals or increase pulse_tolerance."
        )

    # --- Step 4: Linear fit for drift correction (polyfit degree 1) ---
    # Maps photometry time -> EEG time: t_eeg = scaling * t_photo + offset
    p = np.polyfit(matched_photo, matched_eeg, 1)
    scaling = p[0]
    offset = p[1]

    # Compute drift in ppm
    drift_ppm = (scaling - 1.0) * 1e6

    # Compute residual
    predicted = np.polyval(p, matched_photo)
    residuals = matched_eeg - predicted
    residual_ms = float(np.mean(np.abs(residuals)) * 1000)

    # --- Step 5: Warp photometry timeline ---
    t_photo_raw = ppd.time  # seconds
    if t_first_eeg < t_first_photo:
        t_photo_adj = t_photo_raw
    else:
        t_photo_adj = t_photo_raw + start_diff

    t_photo_aligned = scaling * t_photo_adj + offset

    # --- Step 6: Determine overlapping time region ---
    # Use sample_numbers for absolute time (not 0-indexed) to match TTL timebase
    t_eeg = oep.sample_numbers.astype(np.float64) / oep.fs
    t_min = max(t_photo_aligned[0], t_eeg[0])
    t_max = min(t_photo_aligned[-1], t_eeg[-1])

    eeg_mask = (t_eeg >= t_min) & (t_eeg <= t_max)
    t_eeg_common = t_eeg[eeg_mask]
    ecog_common = oep.ecog[eeg_mask]
    emg_common = oep.emg[eeg_mask] if oep.emg is not None else None
    # Temperature may come from a different stream with slightly different length
    n_eeg = len(oep.ecog)
    temp = oep.temperature_raw[:n_eeg] if len(oep.temperature_raw) >= n_eeg else np.pad(
        oep.temperature_raw, (0, n_eeg - len(oep.temperature_raw)), mode="edge"
    )
    temp_common = temp[eeg_mask]

    # --- Step 7: Decimate ECoG to 1000 Hz if recorded at higher rate ---
    TARGET_FS = 1000.0
    output_fs = oep.fs

    if oep.fs > TARGET_FS:
        factor = int(oep.fs / TARGET_FS)
        if oep.fs != factor * TARGET_FS:
            raise ValueError(
                f"ECoG fs={oep.fs} is not an integer multiple of {TARGET_FS}. "
                f"Cannot decimate cleanly."
            )
        logger.info(
            "ECoG fs=%.0f > %.0f; decimating by %dx",
            oep.fs, TARGET_FS, factor,
        )
        ecog_common = decimate(ecog_common, factor)
        if emg_common is not None:
            emg_common = decimate(emg_common, factor)
        temp_common = decimate(temp_common, factor)
        t_eeg_common = t_eeg_common[::factor]  # subsample timestamps to match
        output_fs = TARGET_FS

    # --- Step 8: PCHIP interpolate photometry onto ECoG timebase ---
    signal_470_interp = PchipInterpolator(t_photo_aligned, ppd.signal_470)(t_eeg_common)
    signal_405_interp = PchipInterpolator(t_photo_aligned, ppd.signal_405)(t_eeg_common)

    # Common time vector starting from 0
    time_common = t_eeg_common - t_eeg_common[0]

    return SyncResult(
        signal_470=signal_470_interp,
        signal_405=signal_405_interp,
        ecog=ecog_common,
        emg=emg_common,
        temperature_raw=temp_common,
        temp_bit_volts=oep.temp_bit_volts,
        time=time_common,
        fs=output_fs,
        n_matched=len(matched_photo),
        drift_ppm=drift_ppm,
        scaling=scaling,
        offset=offset,
        residual_ms=residual_ms,
    )
