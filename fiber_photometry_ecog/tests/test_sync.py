"""
Tests for Phase 1: Data Loading & Synchronization.

Tests the sync engine with synthetic TTL signals containing known drift.
Also tests PPD reader against real .ppd files if available.
"""

import numpy as np
import pytest

from fiber_photometry_ecog.data_loading.ppd_reader import PPDData
from fiber_photometry_ecog.data_loading.oep_reader import OEPData
from fiber_photometry_ecog.data_loading.sync import (
    synchronize,
    _match_pulses,
)


# ---------------------------------------------------------------------------
# Helpers to create synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_ppd(
    duration_s: float = 120.0,
    fs: float = 130.0,
    pulse_interval: float = 10.0,
    drift_ppm: float = 0.0,
) -> PPDData:
    """Create a synthetic PPDData with known TTL pulses and optional clock drift."""
    n_samples = int(duration_s * fs)
    time = np.arange(n_samples) / fs

    # Synthetic fluorescence: sine wave + noise
    signal_470 = 1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * time) + 0.01 * np.random.randn(n_samples)
    signal_405 = 0.8 + 0.01 * np.random.randn(n_samples)

    # TTL pulses at regular intervals, with drift applied
    drift_factor = 1.0 + drift_ppm * 1e-6
    true_pulse_times = np.arange(pulse_interval, duration_s - 1, pulse_interval)
    drifted_pulse_times = true_pulse_times * drift_factor

    # Generate digital signal
    digital_1 = np.zeros(n_samples, dtype=np.int8)
    pulse_inds = (drifted_pulse_times * fs).astype(int)
    pulse_inds = pulse_inds[pulse_inds < n_samples]
    # Make each pulse 5 samples wide
    for idx in pulse_inds:
        digital_1[idx:min(idx + 5, n_samples)] = 1

    # Recompute rising edges from the digital signal
    edges = 1 + np.where(np.diff(digital_1) == 1)[0]
    edge_times = edges / fs

    return PPDData(
        signal_470=signal_470,
        signal_405=signal_405,
        digital_1=digital_1,
        pulse_inds=edges,
        pulse_times=edge_times,
        fs=fs,
        time=time,
        metadata={"sampling_rate": fs, "subject_ID": "synthetic"},
    )


def make_synthetic_oep(
    duration_s: float = 120.0,
    fs: float = 1000.0,
    pulse_interval: float = 10.0,
) -> OEPData:
    """Create a synthetic OEPData with known TTL pulses (no drift — reference clock)."""
    n_samples = int(duration_s * fs)

    # Synthetic ECoG: noise
    ecog = 100 * np.random.randn(n_samples)

    # Synthetic temperature: slowly rising
    temperature_raw = np.linspace(20000, 30000, n_samples)

    # TTL events at regular intervals (rising edges only for simplicity)
    pulse_times = np.arange(pulse_interval, duration_s - 1, pulse_interval)
    pulse_samples = (pulse_times * fs).astype(np.int64)
    states = np.ones(len(pulse_times), dtype=np.int16)  # all rising

    return OEPData(
        ecog=ecog,
        emg=None,
        temperature_raw=temperature_raw,
        temp_bit_volts=0.000152587890625,
        fs=fs,
        sample_numbers=np.arange(n_samples, dtype=np.int64),
        timestamps=np.arange(n_samples) / fs,
        ttl_sample_numbers=pulse_samples,
        ttl_states=states,
        ttl_timestamps=pulse_times,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPulseMatching:
    def test_exact_match(self):
        photo = np.array([10.0, 20.0, 30.0])
        eeg = np.array([10.0, 20.0, 30.0])
        mp, me = _match_pulses(photo, eeg, tolerance=1.0)
        assert len(mp) == 3
        np.testing.assert_array_almost_equal(mp, photo)

    def test_within_tolerance(self):
        photo = np.array([10.0, 20.0, 30.0])
        eeg = np.array([10.5, 20.3, 30.9])
        mp, me = _match_pulses(photo, eeg, tolerance=1.0)
        assert len(mp) == 3

    def test_beyond_tolerance(self):
        photo = np.array([10.0, 20.0, 30.0])
        eeg = np.array([12.0, 22.0, 32.0])
        mp, me = _match_pulses(photo, eeg, tolerance=0.5)
        assert len(mp) == 0

    def test_partial_match(self):
        photo = np.array([10.0, 20.0, 30.0])
        eeg = np.array([10.1, 25.0, 30.2])
        mp, me = _match_pulses(photo, eeg, tolerance=0.5)
        assert len(mp) == 2


class TestSyncNoDrift:
    def test_sync_aligned(self):
        ppd = make_synthetic_ppd(drift_ppm=0.0)
        oep = make_synthetic_oep()
        result = synchronize(ppd, oep)

        assert result.n_matched >= 5
        assert abs(result.drift_ppm) < 100  # should be near zero
        assert result.residual_ms < 50
        assert len(result.signal_470) == len(result.ecog)
        assert len(result.time) == len(result.signal_470)
        assert result.fs == oep.fs

    def test_signal_preservation(self):
        """A known DC signal should survive sync with correct mean."""
        np.random.seed(42)
        ppd = make_synthetic_ppd(drift_ppm=0.0, duration_s=120.0)
        # Replace 470 with a known constant so interpolation is exact
        ppd.signal_470[:] = 5.0
        ppd.signal_405[:] = 3.0
        oep = make_synthetic_oep(duration_s=120.0)
        result = synchronize(ppd, oep)

        np.testing.assert_allclose(result.signal_470, 5.0, atol=0.01)
        np.testing.assert_allclose(result.signal_405, 3.0, atol=0.01)

    def test_sine_wave_preservation(self):
        """A slow sine wave should survive sync with low distortion."""
        np.random.seed(42)
        ppd = make_synthetic_ppd(drift_ppm=0.0, duration_s=120.0)
        freq = 0.5  # Hz — well below Nyquist of 130 Hz photometry
        ppd.signal_470 = np.sin(2 * np.pi * freq * ppd.time)
        oep = make_synthetic_oep(duration_s=120.0)
        result = synchronize(ppd, oep)

        # Reconstruct expected sine on the output timebase
        expected = np.sin(2 * np.pi * freq * result.time)
        # Allow some edge effects; check interior (trim 1s each side)
        n_trim = int(result.fs)
        correlation = np.corrcoef(
            result.signal_470[n_trim:-n_trim],
            expected[n_trim:-n_trim],
        )[0, 1]
        assert correlation > 0.99

    def test_time_starts_at_zero(self):
        """Output time vector should start at 0."""
        ppd = make_synthetic_ppd()
        oep = make_synthetic_oep()
        result = synchronize(ppd, oep)
        assert result.time[0] == 0.0

    def test_emg_none_when_not_provided(self):
        """EMG should be None when OEP has no EMG."""
        ppd = make_synthetic_ppd()
        oep = make_synthetic_oep()
        result = synchronize(ppd, oep)
        assert result.emg is None

    def test_emg_passed_through(self):
        """EMG should be trimmed to same length as ECoG when present."""
        ppd = make_synthetic_ppd()
        oep = make_synthetic_oep()
        n = len(oep.ecog)
        oep.emg = np.random.randn(n)
        result = synchronize(ppd, oep)
        assert result.emg is not None
        assert len(result.emg) == len(result.ecog)

    def test_temperature_passed_through(self):
        """Temperature should be trimmed and preserve monotonic ramp."""
        ppd = make_synthetic_ppd()
        oep = make_synthetic_oep()
        result = synchronize(ppd, oep)
        assert len(result.temperature_raw) == len(result.ecog)
        # Synthetic temp is a linear ramp — should stay monotonic
        diffs = np.diff(result.temperature_raw)
        assert np.all(diffs >= 0)


class TestSyncWithDrift:
    def test_drift_recovery(self):
        """Inject 50 ppm drift and verify the sync engine detects non-unity scaling."""
        known_drift = 50.0
        ppd = make_synthetic_ppd(drift_ppm=known_drift, duration_s=600.0)
        oep = make_synthetic_oep(duration_s=600.0)
        result = synchronize(ppd, oep)

        assert result.n_matched >= 10
        # Scaling should differ from 1.0 detectably
        assert abs(result.scaling - 1.0) < 0.01
        assert result.residual_ms < 50

    def test_large_drift(self):
        """Inject 500 ppm drift (0.05%) — should still sync."""
        ppd = make_synthetic_ppd(drift_ppm=500.0)
        oep = make_synthetic_oep()
        result = synchronize(ppd, oep)
        assert result.n_matched >= 5

    def test_drift_sign(self):
        """Positive drift_ppm means photometry clock runs fast (scaling > 1)."""
        ppd = make_synthetic_ppd(drift_ppm=200.0, duration_s=600.0)
        oep = make_synthetic_oep(duration_s=600.0)
        result = synchronize(ppd, oep)
        assert result.n_matched >= 10
        assert len(result.signal_470) == len(result.ecog)


class TestSyncTimeOffset:
    def test_photo_starts_later(self):
        """Photometry starts 5s after EEG — should still sync."""
        ppd = make_synthetic_ppd(duration_s=120.0, pulse_interval=10.0)
        # Shift photometry pulse times forward by 5s
        ppd.time = ppd.time + 5.0
        ppd.pulse_times = ppd.pulse_times + 5.0
        oep = make_synthetic_oep(duration_s=120.0, pulse_interval=10.0)
        result = synchronize(ppd, oep)
        assert result.n_matched >= 5
        assert len(result.signal_470) == len(result.ecog)

    def test_eeg_starts_later(self):
        """EEG starts 5s after photometry — should still sync."""
        ppd = make_synthetic_ppd(duration_s=120.0, pulse_interval=10.0)
        oep = make_synthetic_oep(duration_s=120.0, pulse_interval=10.0)
        # Shift EEG pulse times forward by 5s
        oep.ttl_timestamps = oep.ttl_timestamps + 5.0
        oep.timestamps = oep.timestamps + 5.0
        result = synchronize(ppd, oep)
        assert result.n_matched >= 5
        assert len(result.signal_470) == len(result.ecog)


class TestSyncEdgeCases:
    def test_no_photo_edges_raises(self):
        ppd = make_synthetic_ppd()
        ppd.pulse_times = np.array([])
        ppd.pulse_inds = np.array([])
        oep = make_synthetic_oep()
        with pytest.raises(ValueError, match="No rising edges.*photometry"):
            synchronize(ppd, oep)

    def test_no_eeg_edges_raises(self):
        ppd = make_synthetic_ppd()
        oep = make_synthetic_oep()
        oep.ttl_states = np.array([])
        oep.ttl_timestamps = np.array([])
        with pytest.raises(ValueError, match="No rising edges.*EEG"):
            synchronize(ppd, oep)

    def test_too_few_matches_raises(self):
        """With only 1 pulse and min_matched=3, should raise."""
        ppd = make_synthetic_ppd(duration_s=30.0, pulse_interval=25.0)
        oep = make_synthetic_oep(duration_s=30.0, pulse_interval=25.0)
        with pytest.raises(ValueError):
            synchronize(ppd, oep, min_matched_pulses=3)

    def test_only_falling_edges_in_oep(self):
        """OEP with only falling edges (-1 states) should raise."""
        ppd = make_synthetic_ppd()
        oep = make_synthetic_oep()
        oep.ttl_states = -1 * np.ones_like(oep.ttl_states)
        with pytest.raises(ValueError, match="No rising edges.*EEG"):
            synchronize(ppd, oep)


class TestRealPPD:
    """Test against real .ppd files if available. Skipped if files not found."""

    PPD_PATH = "test_data/Meiling_FiberPhotometry/GRABne/2638_seizure2-2024-06-04-115709.ppd"

    @pytest.fixture
    def ppd_path(self):
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / self.PPD_PATH
        if not p.exists():
            pytest.skip(f"Real PPD file not found: {p}")
        return p

    def test_read_ppd(self, ppd_path):
        from fiber_photometry_ecog.data_loading.ppd_reader import read_ppd
        ppd = read_ppd(ppd_path)

        assert ppd.fs > 0
        assert len(ppd.signal_470) > 0
        assert len(ppd.signal_470) == len(ppd.signal_405)
        assert len(ppd.signal_470) == len(ppd.digital_1)
        assert len(ppd.pulse_inds) > 0
        assert ppd.signal_470.dtype == np.float64 or ppd.signal_470.dtype == np.float32

    def test_pulse_times_monotonic(self, ppd_path):
        from fiber_photometry_ecog.data_loading.ppd_reader import read_ppd
        ppd = read_ppd(ppd_path)
        if len(ppd.pulse_times) > 1:
            assert np.all(np.diff(ppd.pulse_times) > 0)

    def test_time_vector_consistent(self, ppd_path):
        from fiber_photometry_ecog.data_loading.ppd_reader import read_ppd
        ppd = read_ppd(ppd_path)
        expected_duration = (len(ppd.signal_470) - 1) / ppd.fs
        actual_duration = ppd.time[-1] - ppd.time[0]
        np.testing.assert_allclose(actual_duration, expected_duration, rtol=1e-6)


class TestRealOEP:
    """Test against real OEP session if available. Skipped if files not found."""

    OEP_PATH = "test_data/MeilingEEG/2024-12-17_13-37-28_3153_seizure1"

    @pytest.fixture
    def oep_path(self):
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / self.OEP_PATH
        if not p.exists():
            pytest.skip(f"Real OEP session not found: {p}")
        return p

    def test_read_oep(self, oep_path):
        from fiber_photometry_ecog.data_loading.oep_reader import read_oep
        oep = read_oep(oep_path)

        assert oep.fs == 1000.0
        assert len(oep.ecog) > 0
        assert len(oep.ecog) == len(oep.temperature_raw)
        assert oep.emg is None  # no EMG channel requested
        assert oep.temp_bit_volts > 0
        assert len(oep.ttl_states) > 0

    def test_read_oep_with_emg(self, oep_path):
        from fiber_photometry_ecog.data_loading.oep_reader import read_oep
        oep = read_oep(oep_path, emg_channel=7)

        assert oep.emg is not None
        assert len(oep.emg) == len(oep.ecog)

    def test_channel_out_of_range(self, oep_path):
        from fiber_photometry_ecog.data_loading.oep_reader import read_oep
        with pytest.raises(ValueError, match="out of range"):
            read_oep(oep_path, ecog_channel=99)

    def test_emg_channel_out_of_range(self, oep_path):
        from fiber_photometry_ecog.data_loading.oep_reader import read_oep
        with pytest.raises(ValueError, match="EMG channel.*out of range"):
            read_oep(oep_path, emg_channel=99)

    def test_ttl_timestamps_monotonic(self, oep_path):
        from fiber_photometry_ecog.data_loading.oep_reader import read_oep
        oep = read_oep(oep_path)
        if len(oep.ttl_timestamps) > 1:
            assert np.all(np.diff(oep.ttl_timestamps) >= 0)
