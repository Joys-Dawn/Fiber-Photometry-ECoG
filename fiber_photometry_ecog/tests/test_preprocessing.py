"""
Tests for Phase 2: Preprocessing Pipeline.

Tests ECoG filtering, all three photometry strategies, transient detection,
temperature processing, shared utilities, and Protocol compliance.
"""

import numpy as np
import pytest

from fiber_photometry_ecog.core.config import (
    ECoGConfig,
    TransientConfig,
    TemperatureConfig,
)
from fiber_photometry_ecog.core.data_models import (
    PhotometryResult,
)
from fiber_photometry_ecog.preprocessing.ecog import filter_ecog
from fiber_photometry_ecog.preprocessing.photometry.common import (
    fit_biexponential,
    z_score_baseline,
    highpass_filter,
    detrend_moving_average,
)
from fiber_photometry_ecog.preprocessing.photometry.strategy_a_chandni import (
    ChandniStrategy,
    preprocess_chandni,
)
from fiber_photometry_ecog.preprocessing.photometry.strategy_b_meiling import (
    MeilingStrategy,
    preprocess_meiling,
)
from fiber_photometry_ecog.preprocessing.photometry.strategy_c_irls import (
    IRLSStrategy,
    preprocess_irls,
)
from fiber_photometry_ecog.preprocessing.photometry import (
    PhotometryStrategy,
)
from fiber_photometry_ecog.preprocessing.transient_detection import (
    detect_transients,
)
from fiber_photometry_ecog.preprocessing.temperature import (
    process_temperature,
    temp_at_time,
    detect_heating_start,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sine(
    freq: float, fs: float, duration: float, amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(fs * duration)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


def make_bleaching_signal(
    fs: float, duration: float,
    tau: float = 200.0, amp: float = 2.0, baseline: float = 1.0,
) -> np.ndarray:
    """Simulate photobleaching: exponential decay + baseline."""
    t = np.arange(int(fs * duration)) / fs
    return amp * np.exp(-t / tau) + baseline


def make_photometry_pair(
    fs: float = 130.0,
    duration: float = 300.0,
    bleach_tau: float = 200.0,
    transient_times: list | None = None,
    transient_amp: float = 0.5,
    transient_width_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic 470 + 405 signals with optional transients.

    Returns (signal_470, signal_405, time).
    """
    n = int(fs * duration)
    t = np.arange(n) / fs

    # Shared bleaching
    bleach = 2.0 * np.exp(-t / bleach_tau) + 1.0

    # Shared motion artifact (slow oscillation)
    motion = 0.05 * np.sin(2 * np.pi * 0.3 * t)

    signal_405 = bleach + motion + 0.005 * np.random.randn(n)
    signal_470 = bleach + motion + 0.005 * np.random.randn(n)

    # Add calcium transients to 470 only
    if transient_times is not None:
        for tt in transient_times:
            sigma = transient_width_s * fs / 2.355  # FWHM -> sigma
            idx = np.arange(n)
            center = tt * fs
            transient = transient_amp * np.exp(
                -0.5 * ((idx - center) / sigma) ** 2
            )
            signal_470 += transient

    return signal_470, signal_405, t


# ---------------------------------------------------------------------------
# ECoG Filtering Tests
# ---------------------------------------------------------------------------

class TestECoGFilter:
    def test_removes_dc(self):
        """Bandpass should remove DC offset."""
        fs = 1000.0
        ecog = np.ones(10000) * 100.0 + make_sine(10.0, fs, 10.0, 50.0)
        filtered = filter_ecog(ecog, fs)
        assert abs(np.mean(filtered)) < 1.0

    def test_preserves_passband(self):
        """10 Hz signal should pass through 1-70 Hz bandpass."""
        fs = 1000.0
        signal = make_sine(10.0, fs, 10.0, 100.0)
        filtered = filter_ecog(signal, fs)
        n_trim = int(fs)
        ratio = (
            np.std(filtered[n_trim:-n_trim])
            / np.std(signal[n_trim:-n_trim])
        )
        assert 0.8 < ratio < 1.2

    def test_attenuates_high_freq(self):
        """200 Hz signal should be heavily attenuated by 70 Hz low-pass."""
        fs = 1000.0
        signal = make_sine(200.0, fs, 10.0, 100.0)
        filtered = filter_ecog(signal, fs)
        n_trim = int(fs)
        assert np.std(filtered[n_trim:-n_trim]) < 5.0

    def test_attenuates_60hz(self):
        """60 Hz line noise should be notched out."""
        fs = 1000.0
        signal_10hz = make_sine(10.0, fs, 10.0, 100.0)
        noise_60hz = make_sine(60.0, fs, 10.0, 50.0)
        signal = signal_10hz + noise_60hz
        filtered = filter_ecog(signal, fs)
        n = len(filtered)
        fft = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(n, 1 / fs)
        idx_60 = np.argmin(np.abs(freqs - 60))
        power_60 = np.abs(fft[idx_60])
        idx_10 = np.argmin(np.abs(freqs - 10))
        power_10 = np.abs(fft[idx_10])
        assert power_60 < power_10 * 0.1

    def test_custom_config(self):
        """Custom config parameters should be respected."""
        fs = 1000.0
        config = ECoGConfig(bandpass_low=5.0, bandpass_high=50.0)
        ecog = make_sine(2.0, fs, 10.0, 100.0)
        filtered = filter_ecog(ecog, fs, config)
        n_trim = int(fs)
        assert np.std(filtered[n_trim:-n_trim]) < 20.0

    def test_zero_phase(self):
        """sosfiltfilt should produce zero-phase output (no time delay)."""
        fs = 1000.0
        n = 5000
        t = np.arange(n) / fs
        signal = np.sin(2 * np.pi * 10 * t)
        filtered = filter_ecog(signal, fs)
        peaks_orig = (
            np.where(
                (signal[1:-1] > signal[:-2])
                & (signal[1:-1] > signal[2:])
            )[0] + 1
        )
        peaks_filt = (
            np.where(
                (filtered[1:-1] > filtered[:-2])
                & (filtered[1:-1] > filtered[2:])
            )[0] + 1
        )
        n_compare = min(5, len(peaks_orig), len(peaks_filt))
        if n_compare > 0:
            for i in range(1, n_compare):
                assert abs(peaks_orig[i] - peaks_filt[i]) <= 1


# ---------------------------------------------------------------------------
# Biexponential Fit Tests
# ---------------------------------------------------------------------------

class TestBiexponentialFit:
    def test_recovers_known_decay(self):
        """Fit should recover a known exponential decay."""
        fs = 130.0
        duration = 300.0
        n = int(fs * duration)
        t = np.arange(n) / fs
        true_signal = (
            2.0 * np.exp(-t / 150.0) + 0.5 * np.exp(-t / 50.0) + 1.0
        )
        noisy = true_signal + 0.01 * np.random.randn(n)

        fitted = fit_biexponential(noisy, fs)
        residuals = noisy - fitted
        assert np.std(residuals) < 0.05

    def test_monoexp_fallback(self):
        """If biexp fails, should fall back to monoexponential."""
        fs = 130.0
        n = int(fs * 60.0)
        t = np.arange(n) / fs
        signal = 1.0 * np.exp(-t / 30.0) + 0.5
        signal += 0.001 * np.random.randn(n)
        fitted = fit_biexponential(signal, fs)
        assert len(fitted) == n

    def test_fit_output_is_finite_and_bounded(self):
        """Fitted biexponential must not contain inf, nan, or wild values.

        Regression test: the crop fallback previously extrapolated with
        negative time, causing exp(positive) = exponential growth.
        With the np.maximum clamp fix, all paths produce bounded output.
        """
        fs = 130.0
        n = int(fs * 300)
        t = np.arange(n) / fs

        # Standard bleaching signal
        signal = 2.0 * np.exp(-t / 150.0) + 0.5 * np.exp(-t / 50.0) + 1.0
        signal += 0.01 * np.random.randn(n)

        fitted = fit_biexponential(signal, fs, crop_s=120.0)

        assert np.all(np.isfinite(fitted))
        # Fitted values should be in a reasonable range for fluorescence
        assert np.max(fitted) < 100.0
        assert np.min(fitted) > -10.0


# ---------------------------------------------------------------------------
# z-score Tests
# ---------------------------------------------------------------------------

class TestZScore:
    def test_baseline_zero_mean_unit_std(self):
        """After z-scoring, baseline should have mean~0 and std~1."""
        np.random.seed(42)
        fs = 130.0
        signal = np.random.randn(int(fs * 120)) * 2.0 + 5.0
        z = z_score_baseline(signal, fs, baseline_end_s=60.0)
        baseline = z[:int(60 * fs)]
        assert abs(np.mean(baseline)) < 0.1
        assert abs(np.std(baseline) - 1.0) < 0.1

    def test_preserves_relative_changes(self):
        """A step change should appear in z-scored output."""
        fs = 130.0
        n = int(fs * 120)
        signal = np.ones(n) * 5.0
        signal[:int(60 * fs)] += np.random.randn(int(60 * fs)) * 0.5
        signal[int(60 * fs):] += 2.0
        z = z_score_baseline(signal, fs, baseline_end_s=60.0)
        post = z[int(60 * fs):]
        assert np.mean(post) > 2.0


# ---------------------------------------------------------------------------
# High-pass Filter Tests
# ---------------------------------------------------------------------------

class TestHighpassFilter:
    def test_removes_dc(self):
        """HPF should remove DC component."""
        fs = 130.0
        signal = np.ones(int(fs * 60)) * 10.0
        filtered = highpass_filter(signal, fs, cutoff=0.01)
        assert abs(np.mean(filtered)) < 0.5

    def test_preserves_fast_signal(self):
        """Signals well above cutoff should pass through."""
        fs = 130.0
        signal = make_sine(1.0, fs, 60.0, 5.0)
        filtered = highpass_filter(signal, fs, cutoff=0.01)
        n_trim = int(fs * 2)
        ratio = (
            np.std(filtered[n_trim:-n_trim])
            / np.std(signal[n_trim:-n_trim])
        )
        assert 0.8 < ratio < 1.2


class TestDetrendMovingAverage:
    def test_removes_dc(self):
        """Moving-average subtraction should remove DC component."""
        fs = 130.0
        signal = np.ones(int(fs * 60)) * 10.0
        detrended = detrend_moving_average(signal, fs, window_s=100.0)
        assert abs(np.mean(detrended)) < 0.5

    def test_preserves_fast_signal(self):
        """Signals much faster than window should pass through."""
        fs = 130.0
        signal = make_sine(1.0, fs, 60.0, 5.0)
        detrended = detrend_moving_average(signal, fs, window_s=100.0)
        n_trim = int(fs * 2)
        ratio = (
            np.std(detrended[n_trim:-n_trim])
            / np.std(signal[n_trim:-n_trim])
        )
        assert 0.8 < ratio < 1.2



# ---------------------------------------------------------------------------
# PhotometryStrategy Protocol Tests
# ---------------------------------------------------------------------------

class TestPhotometryProtocol:
    def test_chandni_implements_protocol(self):
        """ChandniStrategy must satisfy PhotometryStrategy Protocol."""
        assert isinstance(ChandniStrategy(), PhotometryStrategy)

    def test_meiling_implements_protocol(self):
        """MeilingStrategy must satisfy PhotometryStrategy Protocol."""
        assert isinstance(MeilingStrategy(), PhotometryStrategy)

    def test_irls_implements_protocol(self):
        """IRLSStrategy must satisfy PhotometryStrategy Protocol."""
        assert isinstance(IRLSStrategy(), PhotometryStrategy)

    def test_protocol_method_works(self):
        """Calling preprocess via Protocol interface should work."""
        np.random.seed(42)
        s470, s405, _ = make_photometry_pair(duration=120.0)
        for strategy_cls in [ChandniStrategy, MeilingStrategy, IRLSStrategy]:
            strategy: PhotometryStrategy = strategy_cls()
            result = strategy.preprocess(s470, s405, 130.0)
            assert isinstance(result, PhotometryResult)
            assert len(result.dff) == len(s470)


# ---------------------------------------------------------------------------
# Photometry Strategy Tests
# ---------------------------------------------------------------------------

class TestStrategyA:
    def test_returns_photometry_result(self):
        np.random.seed(42)
        s470, s405, t = make_photometry_pair(duration=120.0)
        result = preprocess_chandni(s470, s405, 130.0)
        assert isinstance(result, PhotometryResult)
        assert len(result.dff) == len(s470)

    def test_class_matches_function(self):
        """Class-based and function-based APIs produce identical output."""
        np.random.seed(42)
        s470, s405, _ = make_photometry_pair(duration=120.0)
        result_fn = preprocess_chandni(s470, s405, 130.0)
        np.random.seed(42)
        s470, s405, _ = make_photometry_pair(duration=120.0)
        result_cls = ChandniStrategy().preprocess(s470, s405, 130.0)
        np.testing.assert_array_equal(result_fn.dff, result_cls.dff)

    def test_constant_signals_give_near_zero_dff(self):
        """Identical constant signals should give dF/F near 0."""
        n = 10000
        s470 = np.ones(n) * 5.0
        s405 = np.ones(n) * 5.0
        result = preprocess_chandni(s470, s405, 130.0)
        assert np.max(np.abs(result.dff)) < 0.01

    def test_signal_above_iso_gives_positive_dff(self):
        """When 470 > 405, dF/F should be positive."""
        n = 10000
        s470 = np.ones(n) * 6.0
        s405 = np.ones(n) * 5.0
        result = preprocess_chandni(s470, s405, 130.0)
        assert np.mean(result.dff) > 0

    def test_smoothing_reduces_noise(self):
        """Gaussian filtfilt smoothing should reduce high-frequency noise."""
        np.random.seed(42)
        n = 10000
        s470 = np.ones(n) * 5.0 + np.random.randn(n) * 0.5
        s405 = np.ones(n) * 5.0 + np.random.randn(n) * 0.5
        result = preprocess_chandni(s470, s405, 130.0)
        assert np.std(result.dff) < 0.5

    def test_uses_filtfilt_not_single_pass(self):
        """filtfilt applies the kernel twice; verify stronger smoothing
        than a single convolution pass."""
        np.random.seed(42)
        n = 10000
        s470 = np.ones(n) * 5.0 + np.random.randn(n) * 0.5
        s405 = np.ones(n) * 5.0
        result = preprocess_chandni(s470, s405, 130.0)
        # filtfilt (forward+backward) should produce very smooth output;
        # std of dF/F should be very low with sigma=75 applied twice
        assert np.std(result.dff) < 0.1


class TestStrategyB:
    def test_returns_photometry_result(self):
        np.random.seed(42)
        s470, s405, t = make_photometry_pair(duration=300.0)
        result = preprocess_meiling(s470, s405, 130.0)
        assert isinstance(result, PhotometryResult)
        assert len(result.dff) == len(s470)

    def test_class_matches_function(self):
        """Class-based and function-based APIs produce identical output."""
        np.random.seed(42)
        s470, s405, _ = make_photometry_pair(duration=300.0)
        result_fn = preprocess_meiling(s470, s405, 130.0)
        np.random.seed(42)
        s470, s405, _ = make_photometry_pair(duration=300.0)
        result_cls = MeilingStrategy().preprocess(s470, s405, 130.0)
        np.testing.assert_array_equal(result_fn.dff, result_cls.dff)

    def test_removes_shared_motion(self):
        """Motion artifact present in both channels should be corrected."""
        np.random.seed(42)
        fs = 130.0
        n = int(fs * 300)
        t = np.arange(n) / fs
        bleach = 2.0 * np.exp(-t / 200) + 1.0
        motion = 0.3 * np.sin(2 * np.pi * 0.2 * t)
        s405 = bleach + motion
        s470 = bleach + motion
        result = preprocess_meiling(s470, s405, fs)
        assert np.std(result.dff) < 0.1

    def test_preserves_transient(self):
        """Calcium transient in 470 only should survive correction."""
        np.random.seed(42)
        s470, s405, t = make_photometry_pair(
            duration=300.0,
            transient_times=[150.0],
            transient_amp=1.0,
            transient_width_s=2.0,
        )
        result = preprocess_meiling(s470, s405, 130.0)
        peak_region = result.dff[int(148 * 130):int(152 * 130)]
        baseline_region = result.dff[:int(50 * 130)]
        assert np.max(peak_region) > (
            np.mean(baseline_region) + 2 * np.std(baseline_region)
        )


class TestStrategyC:
    def test_returns_photometry_result(self):
        np.random.seed(42)
        s470, s405, t = make_photometry_pair(duration=300.0)
        result = preprocess_irls(s470, s405, 130.0)
        assert isinstance(result, PhotometryResult)
        assert len(result.dff) == len(s470)

    def test_class_matches_function(self):
        """Class-based and function-based APIs produce identical output."""
        np.random.seed(42)
        s470, s405, _ = make_photometry_pair(duration=300.0)
        result_fn = preprocess_irls(s470, s405, 130.0)
        np.random.seed(42)
        s470, s405, _ = make_photometry_pair(duration=300.0)
        result_cls = IRLSStrategy().preprocess(s470, s405, 130.0)
        np.testing.assert_array_equal(result_fn.dff, result_cls.dff)

    def test_removes_shared_motion(self):
        """IRLS should correct shared motion + bleaching artifacts.

        Per Keevers 2025: no biexp detrending. The regression captures
        both bleaching and motion directly. When both channels are
        identical, fitted_iso ≈ filt_470, so dF/F ≈ 0.
        """
        np.random.seed(42)
        fs = 130.0
        n = int(fs * 300)
        t = np.arange(n) / fs
        bleach = 2.0 * np.exp(-t / 200) + 1.0
        motion = 0.3 * np.sin(2 * np.pi * 0.2 * t)
        s405 = bleach + motion
        s470 = bleach + motion
        result = preprocess_irls(s470, s405, fs)
        assert np.std(result.dff) < 0.1

    def test_preserves_transient(self):
        """IRLS should preserve calcium transients."""
        np.random.seed(42)
        s470, s405, t = make_photometry_pair(
            duration=300.0,
            transient_times=[150.0],
            transient_amp=1.0,
            transient_width_s=2.0,
        )
        result = preprocess_irls(s470, s405, 130.0)
        peak_region = result.dff[int(148 * 130):int(152 * 130)]
        baseline_region = result.dff[:int(50 * 130)]
        assert np.max(peak_region) > (
            np.mean(baseline_region) + 2 * np.std(baseline_region)
        )

    def test_irls_regression_downweights_transients(self):
        """IRLS regression slope should be less biased by transients.

        Setup: iso has a slow oscillation (shared motion). Signal = iso
        with true slope 1.0 + transients placed at iso peaks. OLS
        interprets the transient-at-peak correlation as higher slope.
        IRLS should downweight those outliers and recover slope closer
        to the true 1.0.
        """
        from fiber_photometry_ecog.preprocessing.photometry.strategy_c_irls import (
            _irls_regression,
        )

        np.random.seed(42)
        fs = 130.0
        n = int(fs * 600)
        t = np.arange(n) / fs

        # Shared slow motion (iso predictor) — large amplitude
        iso = 0.5 * np.sin(2 * np.pi * 0.05 * t)
        # True coupling: slope=1.0
        sig = iso.copy() + 0.005 * np.random.randn(n)

        # Add LARGE transients at iso PEAKS (where iso is maximal)
        # sin(2*pi*0.05*t) peaks at t = 5, 25, 45, 65, ...
        # (period = 20s, peak at t = period/4 + k*period)
        # This creates strong positive outliers correlated with high iso,
        # biasing OLS slope upward
        peak_times = [5.0, 25.0, 45.0, 65.0, 85.0, 105.0, 125.0,
                      145.0, 165.0, 185.0, 205.0, 225.0, 245.0,
                      265.0, 285.0, 305.0, 325.0, 345.0, 365.0, 385.0]
        for tt in peak_times:
            sigma = 1.0 * fs / 2.355
            center = tt * fs
            idx = np.arange(n)
            sig += 2.0 * np.exp(-0.5 * ((idx - center) / sigma) ** 2)

        # OLS slope (biased upward by transients at peaks)
        A = np.column_stack([iso, np.ones(n)])
        ols_params, _, _, _ = np.linalg.lstsq(A, sig, rcond=None)
        ols_slope = ols_params[0]

        # IRLS slope (robust to transient outliers)
        irls_slope, _ = _irls_regression(iso, sig, c=1.4)

        # Both should be near 1.0, but IRLS should be closer
        assert abs(irls_slope - 1.0) < abs(ols_slope - 1.0)

    def test_no_biexp_detrending(self):
        """Strategy C per Keevers 2025 does NOT biexp-detrend.

        The fitted_iso retains original signal scale (including bleaching),
        so dF/F normalizes for bleaching via the division. This means
        dF/F values are small (signal ≈ fitted_iso for shared components).
        """
        np.random.seed(42)
        s470, s405, _ = make_photometry_pair(duration=300.0)
        result = preprocess_irls(s470, s405, 130.0)
        # dF/F should be small when signals are similar — fitted_iso
        # is on the same scale as filt_470 (both ~1-3 range), so
        # (filt_470 - fitted_iso) / fitted_iso should be near 0
        assert np.mean(np.abs(result.dff)) < 0.5


# ---------------------------------------------------------------------------
# Transient Detection Tests
# ---------------------------------------------------------------------------

class TestTransientDetection:
    def test_detects_known_transients(self):
        """Inject known transients and verify detection."""
        np.random.seed(42)
        fs = 130.0
        duration = 120.0
        n = int(fs * duration)

        dff = np.zeros(n) + 0.01 * np.random.randn(n)
        known_times = [20.0, 40.0, 60.0, 80.0, 100.0]
        for tt in known_times:
            sigma = 0.5 * fs / 2.355
            center = tt * fs
            idx = np.arange(n)
            dff += 2.0 * np.exp(-0.5 * ((idx - center) / sigma) ** 2)

        events = detect_transients(dff, dff, fs)
        assert len(events) >= 4

        detected_times = [e.peak_time for e in events]
        for kt in known_times:
            closest = min(abs(dt - kt) for dt in detected_times)
            assert closest < 2.0

    def test_no_transients_in_flat_signal(self):
        """Flat signal should yield no transients (or very few)."""
        np.random.seed(42)
        fs = 130.0
        n = int(fs * 60)
        dff = 0.01 * np.random.randn(n)
        events = detect_transients(dff, dff, fs)
        assert len(events) <= 1

    def test_transient_properties(self):
        """Check that detected properties match injected values."""
        fs = 130.0
        n = int(fs * 60)
        idx = np.arange(n)

        sigma = 1.0 * fs / 2.355
        center = 30.0 * fs
        dff = 3.0 * np.exp(-0.5 * ((idx - center) / sigma) ** 2)

        events = detect_transients(dff, dff, fs)
        assert len(events) >= 1
        event = events[0]
        assert abs(event.peak_time - 30.0) < 1.0
        assert abs(event.peak_amplitude - 3.0) < 0.5
        assert event.half_width > 0
        assert event.peak_to_trough > 0

    def test_max_width_respected(self):
        """Very wide peaks should be excluded by max_width_s."""
        fs = 130.0
        n = int(fs * 120)
        idx = np.arange(n)
        sigma = 10.0 * fs
        dff = 3.0 * np.exp(-0.5 * ((idx - 60.0 * fs) / sigma) ** 2)

        config = TransientConfig(max_width_s=8.0)
        events = detect_transients(dff, dff, fs, config)
        assert len(events) == 0

    def test_temperature_crossref(self):
        """Temperature at peak should be populated when provided."""
        fs = 130.0
        n = int(fs * 60)
        idx = np.arange(n)
        dff = 0.01 * np.random.randn(n)
        sigma = 0.5 * fs / 2.355
        dff += 3.0 * np.exp(-0.5 * ((idx - 30.0 * fs) / sigma) ** 2)

        temperature = np.linspace(30.0, 40.0, n)
        events = detect_transients(dff, dff, fs, temperature=temperature)
        assert len(events) >= 1
        assert events[0].temperature_at_peak is not None
        assert 33.0 < events[0].temperature_at_peak < 37.0

    def test_invalid_method_raises(self):
        config = TransientConfig(method="invalid")
        with pytest.raises(ValueError, match="Unknown"):
            detect_transients(np.zeros(100), np.zeros(100), 130.0, config)


# ---------------------------------------------------------------------------
# Temperature Processing Tests
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_linear_calibration(self):
        """Known voltage should map to expected temperature."""
        fs = 1000.0
        n = int(fs * 10)
        bit_volts = 0.000152587890625
        raw = 300.0 / (bit_volts * 1000.0)
        temperature_raw = np.ones(n) * raw

        result = process_temperature(temperature_raw, bit_volts, fs)
        expected_temp = 0.0981 * 300.0 + 8.81
        np.testing.assert_allclose(
            result.temperature_c, expected_temp, atol=0.1,
        )

    def test_smoothing_reduces_noise(self):
        """Moving average should reduce noise in the interior."""
        np.random.seed(42)
        fs = 1000.0
        n = int(fs * 10)
        bit_volts = 0.000152587890625
        raw = np.ones(n) * 1000.0 + 50.0 * np.random.randn(n)
        result = process_temperature(raw, bit_volts, fs)
        margin = 500
        assert (
            np.std(result.temperature_smooth[margin:-margin])
            < np.std(result.temperature_c[margin:-margin])
        )

    def test_landmarks(self):
        """Landmarks should be extracted from temperature trace."""
        fs = 1000.0
        n = int(fs * 120)
        bit_volts = 0.000152587890625
        raw = np.linspace(500, 2000, n)
        result = process_temperature(raw, bit_volts, fs)

        assert result.baseline_temp < result.max_temp
        assert result.max_temp >= result.terminal_temp
        assert result.max_temp_time > 0

    def test_baseline_temp_from_first_n_seconds(self):
        """Baseline temp should come from first baseline_duration_s."""
        fs = 1000.0
        n = int(fs * 120)
        bit_volts = 0.000152587890625

        raw = np.ones(n) * 1000.0
        raw[int(60 * fs):] = 2000.0

        config = TemperatureConfig(baseline_duration_s=30.0)
        result = process_temperature(raw, bit_volts, fs, config)

        expected_baseline_temp = (
            0.0981 * (1000.0 * bit_volts * 1000.0) + 8.81
        )
        assert abs(result.baseline_temp - expected_baseline_temp) < 1.0

    def test_temp_at_time(self):
        """temp_at_time should return correct temperature at given time."""
        fs = 1000.0
        temp = np.linspace(30.0, 40.0, int(fs * 10))
        val = temp_at_time(temp, fs, 5.0)
        assert val is not None
        assert abs(val - 35.0) < 0.1

    def test_temp_at_time_out_of_range(self):
        """Out-of-range time should return None."""
        fs = 1000.0
        temp = np.linspace(30.0, 40.0, int(fs * 10))
        assert temp_at_time(temp, fs, -1.0) is None
        assert temp_at_time(temp, fs, 20.0) is None


# ---------------------------------------------------------------------------
# Heating Start Auto-Detection Tests
# ---------------------------------------------------------------------------

class TestHeatingStartDetection:
    def test_detects_ramp_onset(self):
        """Should detect the start of a temperature ramp."""
        fs = 1000.0
        n = int(fs * 120)
        temp = np.ones(n) * 37.0
        # Start heating at t=30s: linear ramp of 0.1 C/s
        ramp_start = int(30 * fs)
        ramp = np.arange(n - ramp_start) * 0.1 / fs
        temp[ramp_start:] += ramp

        onset = detect_heating_start(temp, fs, derivative_threshold=0.05)
        assert onset is not None
        assert abs(onset - 30.0) < 6.0

    def test_no_heating_returns_none(self):
        """Flat temperature should return None."""
        fs = 1000.0
        temp = np.ones(int(fs * 60)) * 37.0
        onset = detect_heating_start(temp, fs)
        assert onset is None

    def test_noisy_baseline_no_false_positive(self):
        """Small noise fluctuations should not trigger detection."""
        np.random.seed(42)
        fs = 1000.0
        n = int(fs * 60)
        temp = 37.0 + 0.01 * np.random.randn(n)
        onset = detect_heating_start(
            temp, fs, derivative_threshold=0.05, sustained_s=5.0,
        )
        assert onset is None

    def test_custom_threshold(self):
        """Higher threshold should detect later onset or not at all."""
        fs = 1000.0
        n = int(fs * 120)
        temp = np.ones(n) * 37.0
        ramp_start = int(30 * fs)
        # Gentle ramp: 0.02 C/s
        ramp = np.arange(n - ramp_start) * 0.02 / fs
        temp[ramp_start:] += ramp

        # Low threshold should detect it
        onset_low = detect_heating_start(
            temp, fs, derivative_threshold=0.01,
        )
        assert onset_low is not None

        # High threshold should NOT detect it (ramp is only 0.02 C/s)
        onset_high = detect_heating_start(
            temp, fs, derivative_threshold=0.05,
        )
        assert onset_high is None


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

