"""
Shared photometry utilities used across strategies.

- Biexponential fitting (photobleaching correction for strategy B)
- z-scoring relative to baseline
- High-pass filter for transient detection stream
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
# Biexponential photobleaching model (Meiling's parameterization)
# ---------------------------------------------------------------------------

def _biexp_model(
    t: np.ndarray,
    const: float,
    amp_fast: float,
    amp_slow: float,
    tau_slow: float,
    tau_multiplier: float,
) -> np.ndarray:
    """Biexponential decay (Meiling's parameterization).

    tau_fast = tau_slow * tau_multiplier; with 0 <= tau_multiplier <= 1
    this enforces tau_fast <= tau_slow and breaks the label-switching
    ambiguity of two independent exponential components.
    """
    tau_fast = tau_slow * tau_multiplier
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def fit_biexponential(
    signal: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Fit a biexponential photobleaching curve (Meiling's exact recipe).

    Parameters
    ----------
    signal : 1-D fluorescence signal
    fs : sampling rate (Hz)

    Returns
    -------
    Fitted bleaching curve (same length as input).

    Raises
    ------
    RuntimeError if the fit does not converge.
    """
    n = len(signal)
    t = np.arange(n) / fs

    max_sig = np.max(signal)
    p0 = [max_sig / 2, max_sig / 4, max_sig / 4, 3600.0, 0.1]
    bounds = (
        [0.0,     0.0,     0.0,     600.0,   0.0],
        [max_sig, max_sig, max_sig, 36000.0, 1.0],
    )

    popt, _ = curve_fit(
        _biexp_model, t, signal, p0=p0, bounds=bounds, maxfev=1000
    )
    return _biexp_model(t, *popt)


# ---------------------------------------------------------------------------
# Shared low-pass + biexp detrend (used by strategy B)
# ---------------------------------------------------------------------------

def lowpass_and_detrend(
    signal_470: np.ndarray,
    signal_405: np.ndarray,
    fs: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Low-pass filter both channels, then biexp detrend.

    Parameters
    ----------
    signal_470 : GCaMP / signal channel
    signal_405 : isosbestic / control channel
    fs : sampling rate (Hz)
    cutoff : low-pass cutoff frequency (Hz)

    Returns
    -------
    (detrended_470, detrended_405, expfit_470, expfit_405)
    """
    sos = butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    filt_470 = sosfiltfilt(sos, signal_470)
    filt_405 = sosfiltfilt(sos, signal_405)

    expfit_470 = fit_biexponential(filt_470, fs)
    expfit_405 = fit_biexponential(filt_405, fs)

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
    z-scored signal.
    """
    baseline_end_idx = int(baseline_end_s * fs)
    baseline_end_idx = min(baseline_end_idx, len(signal))

    baseline = signal[:baseline_end_idx]
    mu = np.mean(baseline)
    sigma = np.std(baseline)

    return (signal - mu) / sigma


def null_z(signal: np.ndarray) -> np.ndarray:
    """Null-Z normalization: signal / sqrt(mean(signal^2)).

    The Keevers IRLS dF/F is already centered on the IRLS-derived
    baseline of zero, so "z-score relative to zero" only needs a scale
    estimate, not a centering step. Keevers et al. (Neurophotonics 2025)
    recommend this over standard baseline-window z-scoring because the
    latter "reverts the signal mean to 0 (as per OLS), which may not
    reflect the IRLS-derived baseline."

    Reference implementation: github.com/philjrdb/RegressionSim/nullZ.m
        function nullZ_data = nullZ(data)
        nullZ_data = data ./ sqrt(mean(data.^2));

    Using mean (not sum) of squared deviations keeps the scale
    length-invariant, so longer recordings don't produce systematically
    smaller normalized values than shorter ones.

    NaNs in the input propagate through (`np.nanmean` only affects the
    scale estimate; positions of NaN in the output match the input).
    Returns an all-NaN array if the scale is zero or undefined.
    """
    finite = np.isfinite(signal)
    if not finite.any():
        return np.full_like(signal, np.nan, dtype=np.float64)
    rms = float(np.sqrt(np.mean(signal[finite] ** 2)))
    if not np.isfinite(rms) or rms == 0:
        return np.full_like(signal, np.nan, dtype=np.float64)
    out = np.array(signal, dtype=np.float64, copy=True)
    out[finite] = signal[finite] / rms
    return out


# ---------------------------------------------------------------------------
# High-pass filter for transient detection (Strategy A, per Chandni's code)
# ---------------------------------------------------------------------------

def highpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff: float = 0.01,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter.

    Used by Strategy A (Chandni's original pipeline). Chandni's
    detect_transients.m operates on zdff_hpf (z-scored HPF dF/F).

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


# ---------------------------------------------------------------------------
# Detrending for transient detection (Strategy B/C, Wallace 2025 moving-average)
# ---------------------------------------------------------------------------

def detrend_moving_average(
    signal: np.ndarray,
    fs: float,
    window_s: float = 100.0,
) -> np.ndarray:
    """Detrend by subtracting a moving average, per Wallace et al. 2025.

    Wallace 2025: "adjusted for bleaching, drift, and motion artifacts
    through normalization by detrending the 465-nm-excited signal and
    subtracting a moving average smoothed (width=100 s) version of the
    signal."

    Parameters
    ----------
    signal : 1-D signal (dF/F)
    fs : sampling rate (Hz)
    window_s : moving average window in seconds (default 100s per Wallace)

    Returns
    -------
    Detrended signal (signal - moving_average).
    """
    window_samples = int(window_s * fs)
    if window_samples < 1:
        window_samples = 1
    # Ensure odd window for symmetric averaging
    if window_samples % 2 == 0:
        window_samples += 1
    # Uniform 1-D filter (moving average) preserving array length
    kernel = np.ones(window_samples) / window_samples
    # Pad edges to avoid boundary artifacts
    pad = window_samples // 2
    padded = np.pad(signal, pad, mode="edge")
    moving_avg = np.convolve(padded, kernel, mode="valid")
    return signal - moving_avg
