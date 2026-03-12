"""
Strategy A — Chandni's photometry preprocessing.

1. Gaussian smoothing (sigma=75 samples) on both 470 and 405 channels
   via filtfilt with Gaussian kernel (zero-phase, applied forward+backward)
2. dF/F = (smoothed_470 - smoothed_405) / smoothed_405

Simple approach: no explicit photobleaching model or motion regression.
Gaussian acts as low-pass; isosbestic subtraction handles shared noise.
"""

import numpy as np
from scipy.signal import filtfilt

from ...core.config import PhotometryConfig
from ...core.data_models import PhotometryResult


def _gaussian_kernel(sigma: int) -> np.ndarray:
    """Create a normalized Gaussian FIR kernel for use with filtfilt.

    Kernel spans +/-3*sigma (covers 99.7% of the distribution).
    """
    half_width = 3 * sigma
    x = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


class ChandniStrategy:
    """Strategy A: Gaussian smoothing + simple dF/F."""

    def preprocess(
        self,
        signal_470: np.ndarray,
        signal_405: np.ndarray,
        fs: float,
        config: PhotometryConfig | None = None,
    ) -> PhotometryResult:
        """Apply Chandni's Gaussian smoothing + simple dF/F.

        Uses filtfilt with a Gaussian kernel for zero-phase smoothing
        (forward + backward pass), per spec.

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

        sigma = config.gaussian_sigma
        kernel = _gaussian_kernel(sigma)

        smoothed_470 = filtfilt(kernel, [1.0], signal_470)
        smoothed_405 = filtfilt(kernel, [1.0], signal_405)

        dff = (smoothed_470 - smoothed_405) / smoothed_405

        return PhotometryResult(dff=dff)


# Module-level convenience for backward compatibility
def preprocess_chandni(
    signal_470: np.ndarray,
    signal_405: np.ndarray,
    fs: float,
    config: PhotometryConfig | None = None,
) -> PhotometryResult:
    """Convenience wrapper around ChandniStrategy.preprocess."""
    return ChandniStrategy().preprocess(signal_470, signal_405, fs, config)
