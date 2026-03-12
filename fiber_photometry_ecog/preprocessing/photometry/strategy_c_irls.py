"""
Strategy C — IRLS / Keevers photometry preprocessing.

Reproduces the pipeline from Keevers & Jean-Richard-dit-Bressel 2025
(Neurophotonics 12(2):025003, PMID 40166421).
Reference implementation: https://github.com/philjrdb/RegressionSim (IRLS_dFF.m)

1. Butterworth low-pass (3 Hz) on both channels + mean correction
2. IRLS robust regression with Tukey's bisquare weighting (c=1.4)
   and leverage-adjusted residuals (matching MATLAB robustfit)
   on the filtered signals directly (no biexponential detrending)
3. dF/F = (filtered_signal - fitted_iso) / fitted_iso

The regression captures both shared motion artifacts AND the shared
photobleaching trend. The dF/F division normalizes for photobleaching
because fitted_iso retains the original signal scale.

Keevers 2025: "detrending and/or z-scoring experimental and isosbestic
signals preclude their use in dF/F calculations as the original
scale...is necessary."
"""

import numpy as np
from scipy.signal import butter, buttord, sosfiltfilt

from ...core.config import PhotometryConfig
from ...core.data_models import PhotometryResult


def _tukey_bisquare_weights(
    residuals: np.ndarray, c: float, h: np.ndarray
) -> np.ndarray:
    """Compute Tukey's bisquare weights with leverage adjustment.

    Matches MATLAB robustfit: the scaled residual is
        r = residual / (c * s * sqrt(1 - h))
    where s = MAD / 0.6745 and h = hat-matrix diagonal (leverage).

    w(r) = (1 - r^2)^2  if |r| <= 1
    w(r) = 0             if |r| > 1
    """
    mad = np.median(np.abs(residuals - np.median(residuals)))
    s = mad / 0.6745 if mad > 0 else 1.0
    leverage_adj = np.sqrt(1.0 - h)
    r = residuals / (c * s * leverage_adj)
    weights = np.where(np.abs(r) <= 1.0, (1.0 - r ** 2) ** 2, 0.0)
    return weights


def _irls_regression(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 1.4,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """IRLS robust linear regression: y = slope * x + intercept.

    Matches MATLAB robustfit(x, y, 'bisquare', c, 'on'):
    - Leverage-adjusted residuals via hat-matrix diagonal
    - MAD-based scale estimate (MAD / 0.6745)
    - Tukey bisquare weight function

    Parameters
    ----------
    x : predictor (filtered isosbestic)
    y : response (filtered signal)
    c : Tukey bisquare tuning constant
    max_iter : maximum iterations
    tol : convergence tolerance on parameter change

    Returns
    -------
    (slope, intercept) tuple.
    """
    # Design matrix: [x, 1]
    A = np.column_stack([x, np.ones(len(x))])

    # Hat-matrix diagonal h_i = A_i @ (A'A)^{-1} @ A_i'
    # Only need the 2x2 inverse, then vectorised dot product for diagonal
    ATA_inv = np.linalg.inv(A.T @ A)
    h = np.sum((A @ ATA_inv) * A, axis=1)

    # Initialize with OLS
    params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    for _ in range(max_iter):
        residuals = y - A @ params
        weights = _tukey_bisquare_weights(residuals, c, h)

        # Weighted least squares (element-wise, avoids N*N dense matrix)
        Aw = A * weights[:, None]
        try:
            new_params = np.linalg.solve(Aw.T @ A, Aw.T @ y)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(new_params - params)) < tol:
            params = new_params
            break
        params = new_params

    return float(params[0]), float(params[1])


class IRLSStrategy:
    """Strategy C: IRLS robust regression per Keevers 2025."""

    def preprocess(
        self,
        signal_470: np.ndarray,
        signal_405: np.ndarray,
        fs: float,
        config: PhotometryConfig | None = None,
    ) -> PhotometryResult:
        """Apply IRLS-based photometry preprocessing (Keevers 2025).

        Pipeline: low-pass filter → IRLS regression → dF/F.
        No biexponential detrending — the regression captures shared
        bleaching via the isosbestic channel, and dF/F normalizes it.

        Parameters
        ----------
        signal_470 : GCaMP / signal channel
        signal_405 : isosbestic / control channel
        fs : sampling rate (Hz)
        config : photometry parameters (uses defaults if None)

        Returns
        -------
        PhotometryResult with dff field populated.
        """
        if config is None:
            config = PhotometryConfig()

        cutoff = config.lowpass_cutoff_c
        c = config.irls_tuning_c
        max_iter = config.irls_max_iter
        irls_tol = config.irls_tol

        # --- Step 1: Low-pass filter both channels + mean correction ---
        # Matches MATLAB: lowpass(signal, cutoff, fs, ImpulseResponse="iir",
        # Steepness=0.95) which uses a minimum-order Butterworth with:
        #   - passband ripple = 0.1 dB (fixed)
        #   - stopband attenuation = 60 dB (default)
        #   - transition width W = (0.99 - 0.98*steepness) * (f_nyq - fpass)
        # Then corrects: filtered += mean(raw) - mean(filtered)
        steepness = 0.95
        f_nyq = fs / 2
        transition_w = (0.99 - 0.98 * steepness) * (f_nyq - cutoff)
        fstop = cutoff + transition_w
        order, wn = buttord(
            cutoff / f_nyq, fstop / f_nyq,
            gpass=0.1, gstop=60,
        )
        sos = butter(order, wn, btype="lowpass", output="sos")
        filt_470 = sosfiltfilt(sos, signal_470)
        filt_470 += np.mean(signal_470) - np.mean(filt_470)
        filt_405 = sosfiltfilt(sos, signal_405)
        filt_405 += np.mean(signal_405) - np.mean(filt_405)

        # --- Step 2: IRLS robust regression on filtered signals ---
        # Fit: filt_470 = slope * filt_405 + intercept
        slope, intercept = _irls_regression(
            filt_405, filt_470,
            c=c, max_iter=max_iter, tol=irls_tol,
        )
        fitted_iso = slope * filt_405 + intercept

        # --- Step 3: dF/F = (filtered_signal - fitted_iso) / fitted_iso ---
        dff = (filt_470 - fitted_iso) / fitted_iso

        return PhotometryResult(dff=dff)


# Module-level convenience for backward compatibility
def preprocess_irls(
    signal_470: np.ndarray,
    signal_405: np.ndarray,
    fs: float,
    config: PhotometryConfig | None = None,
) -> PhotometryResult:
    """Convenience wrapper around IRLSStrategy.preprocess."""
    return IRLSStrategy().preprocess(signal_470, signal_405, fs, config)
