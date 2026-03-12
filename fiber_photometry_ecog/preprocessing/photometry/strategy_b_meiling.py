"""
Strategy B — Meiling's photometry preprocessing.

1. Butterworth low-pass (4th order, 10 Hz) on both channels
2. Biexponential fit on each channel -> subtract (photobleaching correction)
3. OLS linear regression: detrended_iso -> detrended_signal -> motion estimate
4. dF/F = (detrended_signal - motion_estimate) / expfit

Limitation: OLS produces "downshifted" corrected signals (Keevers 2025)
because it treats calcium transients as part of the isosbestic fit.
"""

import numpy as np

from ...core.config import PhotometryConfig
from ...core.data_models import PhotometryResult
from .common import lowpass_and_detrend


class MeilingStrategy:
    """Strategy B: Low-pass + biexp detrend + OLS motion correction."""

    def preprocess(
        self,
        signal_470: np.ndarray,
        signal_405: np.ndarray,
        fs: float,
        config: PhotometryConfig | None = None,
    ) -> PhotometryResult:
        """Apply Meiling's low-pass + biexp detrend + OLS correction.

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

        cutoff = config.lowpass_cutoff_b
        crop_s = config.biexp_crop_s

        # --- Steps 1-2: Low-pass filter + biexponential detrend ---
        detrended_470, detrended_405, expfit_470, _ = lowpass_and_detrend(
            signal_470, signal_405, fs, cutoff, crop_s
        )

        # --- Step 3: OLS motion correction ---
        # Fit: detrended_470 = slope * detrended_405 + intercept
        A = np.column_stack([detrended_405, np.ones(len(detrended_405))])
        params, _, _, _ = np.linalg.lstsq(A, detrended_470, rcond=None)
        slope, intercept = params
        motion_estimate = slope * detrended_405 + intercept

        # --- Step 4: dF/F = (detrended_signal - motion_estimate) / expfit ---
        corrected = detrended_470 - motion_estimate
        dff = corrected / expfit_470

        return PhotometryResult(dff=dff)


# Module-level convenience for backward compatibility
def preprocess_meiling(
    signal_470: np.ndarray,
    signal_405: np.ndarray,
    fs: float,
    config: PhotometryConfig | None = None,
) -> PhotometryResult:
    """Convenience wrapper around MeilingStrategy.preprocess."""
    return MeilingStrategy().preprocess(signal_470, signal_405, fs, config)
