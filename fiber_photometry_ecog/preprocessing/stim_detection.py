"""
Opto-stim burst onset/offset detection from the analog stim channel.

The stimulator command voltage is recorded on one of the Intan ADC inputs.
Most sessions contain a single ~10-second burst of pulses in an otherwise
near-silent trace, but two failure modes show up in practice:

  1. Post-stim artifacts (single-sample spikes 10-400 s after the real
     burst) that a naive first-to-last threshold-crossing algorithm
     pulls into the burst span — silently misplacing ``off_time``.
  2. Multi-bout sessions where the protocol delivered two bouts and the
     experimenter wants the Nth bout used as the analysis landmark
     (e.g. 4361 1Hz_DG, where bout 2 == "stim_Meiling").

Detection is therefore done in two steps:
    * Gap-cluster supra-threshold samples into discrete bursts (runs
      separated by > ``min_gap_s`` are different bursts).
    * Pick the longest cluster by default (artifacts are short — tens
      of samples vs. thousands for a real 10s burst), or the cluster
      at a caller-supplied 0-indexed position in time order.
"""

import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)


def detect_stim_bursts(
    stim_signal: np.ndarray,
    fs: float,
    threshold_fraction: float = 0.5,
    min_gap_s: float = 5.0,
    min_burst_samples: int = 3,
) -> List[Tuple[float, float, int]]:
    """Return every supra-threshold burst cluster in the stim channel.

    Parameters
    ----------
    stim_signal : 1-D array of stim ADC samples (volts).
    fs : sampling rate of `stim_signal` in Hz.
    threshold_fraction : fraction of |peak| used as the crossing threshold.
        Defaults to 0.5; the per-session amplitude is large enough relative
        to the noise floor that the exact fraction doesn't matter.
    min_gap_s : runs of supra-threshold samples separated by a quiet
        interval longer than this (in seconds) are split into separate
        bursts. Defaults to 5 s — wide enough that 1 Hz / 15 ms pulses
        (1 s inter-pulse interval) merge into one burst cluster, but
        narrow enough that two distinct bouts (typically 10-20 s apart)
        stay separate. Real lab data has tail artifacts as close as
        12.7 s after burst end (4361 S1 20HZ_DG_Atomoxetine), so 5 s
        keeps them rejected.
    min_burst_samples : minimum sample count for a cluster to be kept.
        Short isolated noise spikes are filtered out; the real burst
        for a 10 Hz / 15 ms / 10 s protocol has ~3000 supra-threshold
        samples at 1 kHz.

    Returns
    -------
    List of ``(onset_s, offset_s, sample_count)`` tuples in time order.

    Raises
    ------
    ValueError
        If the channel is flat (peak ~ noise floor) or no cluster meets
        ``min_burst_samples``.
    """
    if stim_signal.ndim != 1:
        raise ValueError(
            f"detect_stim_bursts expects a 1-D signal, got shape {stim_signal.shape}"
        )
    if fs <= 0:
        raise ValueError(f"detect_stim_bursts: invalid fs={fs}")

    abs_sig = np.abs(stim_signal)
    # Opto sessions extend the photometry timebase past the OEP recording,
    # which leaves NaNs in stim_raw outside the OEP overlap. nan-aware max
    # keeps detection robust; NaN > threshold evaluates to False already,
    # so the threshold-crossing logic itself needs no further handling.
    peak = float(np.nanmax(abs_sig))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("detect_stim_bursts: stim channel is identically zero")

    threshold = peak * threshold_fraction
    above = abs_sig > threshold
    indices = np.flatnonzero(above)
    if indices.size < min_burst_samples:
        raise ValueError(
            f"detect_stim_bursts: only {indices.size} samples above threshold "
            f"(peak={peak:.4f}V, threshold={threshold:.4f}V); "
            f"likely a flat/noise-only stim channel"
        )

    gap_n = max(1, int(min_gap_s * fs))
    diffs = np.diff(indices)
    gap_breaks = np.flatnonzero(diffs > gap_n)
    run_starts = np.concatenate([[0], gap_breaks + 1])
    run_ends = np.concatenate([gap_breaks, [len(indices) - 1]])

    bursts: List[Tuple[float, float, int]] = []
    for rs, re_ in zip(run_starts, run_ends):
        sample_count = int(re_ - rs + 1)
        if sample_count < min_burst_samples:
            continue
        onset_s = float(indices[rs]) / fs
        offset_s = float(indices[re_]) / fs
        bursts.append((onset_s, offset_s, sample_count))

    if not bursts:
        raise ValueError(
            f"detect_stim_bursts: no burst with at least {min_burst_samples} "
            f"supra-threshold samples"
        )
    return bursts


def detect_stim_burst(
    stim_signal: np.ndarray,
    fs: float,
    threshold_fraction: float = 0.5,
    min_burst_samples: int = 3,
    burst_index: int | None = None,
) -> Tuple[float, float]:
    """Detect the onset and offset of one opto-stim burst.

    Picks the cluster with the most supra-threshold samples by default —
    artifacts are short (tens of samples) while a real 10s burst has
    thousands, so longest-cluster reliably excludes them. When
    ``burst_index`` is given, returns the (0-indexed) cluster in time
    order instead. Used to override the default pick on sessions where
    the protocol selected a non-largest bout (e.g. the 2nd of two
    equally-sized 1 Hz bouts).

    Logs a warning when multiple bursts are present and ``burst_index``
    was not supplied — the caller probably wants to add an override.

    Raises
    ------
    ValueError
        If the channel has no detectable burst, or ``burst_index`` is out
        of range for the discovered bursts.
    """
    bursts = detect_stim_bursts(
        stim_signal, fs, threshold_fraction,
        min_burst_samples=min_burst_samples,
    )

    if burst_index is not None:
        if burst_index < 0 or burst_index >= len(bursts):
            raise ValueError(
                f"detect_stim_burst: burst_index={burst_index} out of range "
                f"(found {len(bursts)} bursts at "
                f"{[(round(b[0], 2), round(b[1], 2)) for b in bursts]})"
            )
        onset, offset, _ = bursts[burst_index]
    else:
        if len(bursts) > 1:
            log.warning(
                "detect_stim_burst: %d bursts detected (picking longest); "
                "windows = %s",
                len(bursts),
                [(round(b[0], 2), round(b[1], 2), b[2]) for b in bursts],
            )
        onset, offset, _ = max(bursts, key=lambda b: b[2])
    return onset, offset
