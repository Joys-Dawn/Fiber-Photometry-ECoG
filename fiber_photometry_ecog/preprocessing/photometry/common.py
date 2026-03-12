"""
Shared photometry utilities used across strategies.

- Biexponential fitting (photobleaching correction for strategies B and C)
- z-scoring relative to baseline
- High-pass filter for transient detection stream
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
# Biexponential photobleaching model
# ---------------------------------------------------------------------------

def _biexp_model(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float, offset: float) -> np.ndarray:
    """Biexponential decay: a1*exp(-t/tau1) + a2*exp(-t/tau2) + offset."""
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + offset


def _monoexp_model(t: np.ndarray, a: float, tau: float, off: float) -> np.ndarray:
    """Monoexponential decay: a*exp(-t/tau) + offset."""
    return a * np.exp(-t / tau) + off


def fit_biexponential(
    signal: np.ndarray,
    fs: float,
    crop_s: float = 120.0,
) -> np.ndarray:
    """Fit a biexponential decay to model photobleaching.

    Parameters
    ----------
    signal : 1-D fluorescence signal
    fs : sampling rate (Hz)
    crop_s : seconds to crop from start if initial fit fails (per PASTa, Donka 2025)

    Returns
    -------
    Fitted bleaching curve (same length as input).

    Raises
    ------
    RuntimeError if neither biexponential nor monoexponential fits converge.
    """
    n = len(signal)
    t = np.arange(n) / fs

    # Initial guesses based on signal
    amp = signal[0] - signal[-1]
    p0 = [amp * 0.6, n / (3 * fs), amp * 0.4, n / (10 * fs), signal[-1]]

    try:
        popt, _ = curve_fit(_biexp_model, t, signal, p0=p0, maxfev=10000)
        return _biexp_model(t, *popt)
    except RuntimeError:
        pass

    # Fallback: crop initial transient and retry
    crop_n = int(crop_s * fs)
    if crop_n < n:
        t_crop = t[crop_n:]
        sig_crop = signal[crop_n:]
        amp_c = sig_crop[0] - sig_crop[-1]
        p0_c = [amp_c * 0.6, len(sig_crop) / (3 * fs), amp_c * 0.4, len(sig_crop) / (10 * fs), sig_crop[-1]]
        try:
            popt, _ = curve_fit(_biexp_model, t_crop - t_crop[0], sig_crop, p0=p0_c, maxfev=10000)
            # Extrapolate to full length; clamp pre-crop region to t=0
            # (negative time would cause exponential growth, not decay)
            t_shifted = np.maximum(t - t[crop_n], 0)
            fitted = _biexp_model(t_shifted, *popt)
            return fitted
        except RuntimeError:
            pass

    # Final fallback: monoexponential
    p0_mono = [amp, n / (3 * fs), signal[-1]]
    try:
        popt, _ = curve_fit(_monoexp_model, t, signal, p0=p0_mono, maxfev=10000)
        warnings.warn("Biexponential fit failed; using monoexponential fallback.")
        return _monoexp_model(t, *popt)
    except RuntimeError:
        raise RuntimeError(
            "Photobleaching fit failed (biexp and monoexp). "
            "Check signal quality or try increasing crop_s."
        )


# ---------------------------------------------------------------------------
# Shared low-pass + biexp detrend (used by strategies B and C)
# ---------------------------------------------------------------------------

def lowpass_and_detrend(
    signal_470: np.ndarray,
    signal_405: np.ndarray,
    fs: float,
    cutoff: float,
    crop_s: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Low-pass filter both channels, then biexp detrend.

    Parameters
    ----------
    signal_470 : GCaMP / signal channel
    signal_405 : isosbestic / control channel
    fs : sampling rate (Hz)
    cutoff : low-pass cutoff frequency (Hz)
    crop_s : biexp fit crop window (seconds)

    Returns
    -------
    (detrended_470, detrended_405, expfit_470, expfit_405)
    """
    sos = butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    filt_470 = sosfiltfilt(sos, signal_470)
    filt_405 = sosfiltfilt(sos, signal_405)

    expfit_470 = fit_biexponential(filt_470, fs, crop_s=crop_s)
    expfit_405 = fit_biexponential(filt_405, fs, crop_s=crop_s)

    return filt_470 - expfit_470, filt_405 - expfit_405, expfit_470, expfit_405


# ---------------------------------------------------------------------------
# z-scoring
# ---------------------------------------------------------------------------

def z_score_baseline(
    signal: np.ndarray,
    fs: float,
    baseline_end_s: float,
) -> np.ndarray:
    """z-score signal relative to a baseline window (0 to baseline_end_s).

    Parameters
    ----------
    signal : 1-D signal (e.g., dF/F)
    fs : sampling rate
    baseline_end_s : end of baseline window in seconds from signal start

    Returns
    -------
    z-scored signal. Returns zeros with a warning if baseline std is zero.
    """
    baseline_end_idx = int(baseline_end_s * fs)
    baseline_end_idx = min(baseline_end_idx, len(signal))

    baseline = signal[:baseline_end_idx]
    mu = np.mean(baseline)
    sigma = np.std(baseline)

    return (signal - mu) / sigma


# ---------------------------------------------------------------------------
# High-pass filter for transient detection
# ---------------------------------------------------------------------------

def highpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff: float = 0.01,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    signal : 1-D signal
    fs : sampling rate (Hz)
    cutoff : cutoff frequency (Hz)
    order : filter order

    Returns
    -------
    High-pass filtered signal.
    """
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)
