"""
Strategy D — Photometry preprocessing without isosbestic correction.

For sensors where no isosbestic wavelength is known or available (e.g. some
non-GCaMP indicators). The 405 channel is ignored entirely; the 470 channel
is low-pass filtered then biexponentially detrended for photobleaching,
producing a dF/F derived only from the experimental signal.

dff_corrected = dff_detrended = (filt_470 - biexp_fit_470) / biexp_fit_470
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ...core.config import PhotometryConfig
from ...core.data_models import PhotometryResult
from .common import fit_biexponential


class NoIsosbesticStrategy:
    """Strategy D: biexponential detrend on 470 only, no isosbestic correction."""

    def preprocess(
        self,
        signal_470: np.ndarray,
        signal_405: np.ndarray,
        fs: float,
        config: PhotometryConfig | None = None,
        **kwargs,
    ) -> PhotometryResult:
        """Detrend 470 against its own biexponential photobleaching fit.

        The 405 channel is accepted for API compatibility but unused.

        Parameters
        ----------
        signal_470 : experimental fluorescence channel
        signal_405 : isosbestic channel — IGNORED
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

        sos = butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
        filt_470 = sosfiltfilt(sos, signal_470)

        expfit_470 = fit_biexponential(filt_470, fs, crop_s=crop_s)

        if np.any(expfit_470 <= 0):
            raise ValueError(
                "Biexponential fit of 470 crosses zero — dF/F is undefined. "
                f"min(expfit_470)={expfit_470.min():.6f}"
            )

        dff = (filt_470 - expfit_470) / expfit_470

        return PhotometryResult(dff=dff)


def preprocess_no_isosbestic(
    signal_470: np.ndarray,
    signal_405: np.ndarray,
    fs: float,
    config: PhotometryConfig | None = None,
) -> PhotometryResult:
    """Convenience wrapper around NoIsosbesticStrategy.preprocess."""
    return NoIsosbesticStrategy().preprocess(signal_470, signal_405, fs, config)
