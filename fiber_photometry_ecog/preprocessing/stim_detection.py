"""
Opto-stim burst onset/offset detection from the analog stim channel.

The stimulator command voltage is recorded on one of the Intan ADC inputs.
A single ~10-second burst per session sits in an otherwise near-silent
trace, so simple threshold crossing on |signal| gives clean onset/offset
times without any model of the pulse train.

Onset is the first sample where |signal| exceeds half the peak amplitude,
offset is the last such sample. The interior structure (pulse rate, pulse
width) is intentionally not analyzed here — it's encoded in the session
folder name (e.g. ``15ms_10hz_10s``) and not needed for landmark
assignment.
"""

import numpy as np


def detect_stim_burst(
    stim_signal: np.ndarray,
    fs: float,
    threshold_fraction: float = 0.5,
    min_burst_samples: int = 3,
) -> tuple[float, float]:
    """Detect the onset and offset of a single opto-stim burst.

    Parameters
    ----------
    stim_signal : 1-D array of stim ADC samples (volts).
    fs : sampling rate of `stim_signal` in Hz.
    threshold_fraction : fraction of |peak| used as the crossing threshold.
        Defaults to 0.5; the per-session amplitude is large enough relative
        to the noise floor that the exact fraction doesn't matter.
    min_burst_samples : minimum number of supra-threshold samples required.
        A passing burst must have at least this many samples above threshold;
        otherwise the call raises ValueError (no real stim was delivered).

    Returns
    -------
    (onset_s, offset_s) : burst onset and offset in seconds from the start
    of `stim_signal`.

    Raises
    ------
    ValueError
        If the channel is flat (peak ~ noise floor) or the supra-threshold
        sample count is below `min_burst_samples`. Opto sessions are
        expected to contain exactly one burst, so absence is a hard error.
    """
    if stim_signal.ndim != 1:
        raise ValueError(
            f"detect_stim_burst expects a 1-D signal, got shape {stim_signal.shape}"
        )
    if fs <= 0:
        raise ValueError(f"detect_stim_burst: invalid fs={fs}")

    abs_sig = np.abs(stim_signal)
    # Opto sessions extend the photometry timebase past the OEP recording,
    # which leaves NaNs in stim_raw outside the OEP overlap. nan-aware max
    # keeps detection robust; NaN > threshold evaluates to False already,
    # so the threshold-crossing logic itself needs no further handling.
    peak = float(np.nanmax(abs_sig))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("detect_stim_burst: stim channel is identically zero")

    threshold = peak * threshold_fraction
    above = abs_sig > threshold
    n_above = int(above.sum())
    if n_above < min_burst_samples:
        raise ValueError(
            f"detect_stim_burst: only {n_above} samples above threshold "
            f"(peak={peak:.4f}V, threshold={threshold:.4f}V); "
            f"likely a flat/noise-only stim channel"
        )

    indices = np.flatnonzero(above)
    onset_s = float(indices[0]) / fs
    offset_s = float(indices[-1]) / fs
    return onset_s, offset_s
