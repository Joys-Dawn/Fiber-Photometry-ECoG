"""
Tests for Phase 4: Analysis Modules.

Uses synthetic sessions with known signals and landmarks to verify
each analyzer produces correct metrics.
"""

import numpy as np
import pytest

from fiber_photometry_ecog.core.config import AnalysisConfig
from fiber_photometry_ecog.core.data_models import (
    Session,
    SessionLandmarks,
    ProcessedData,
    PhotometryResult,
    TransientEvent,
)
from fiber_photometry_ecog.analysis.cohort_characteristics import compute_cohort_characteristics
from fiber_photometry_ecog.analysis.baseline_transients import compute_baseline_transients
from fiber_photometry_ecog.analysis.preictal_mean import compute_preictal_mean
from fiber_photometry_ecog.analysis.preictal_transients import compute_preictal_transients
from fiber_photometry_ecog.analysis.ictal_mean import compute_ictal_mean
from fiber_photometry_ecog.analysis.ictal_transients import compute_ictal_transients
from fiber_photometry_ecog.analysis.postictal import compute_postictal
from fiber_photometry_ecog.analysis.spike_triggered import compute_spike_triggered_average


# ---------------------------------------------------------------------------
# Helpers to build synthetic sessions
# ---------------------------------------------------------------------------

def _make_session(
    mouse_id: str = "mouse1",
    genotype: str = "Scn1a",
    n_seizures: int = 1,
    fs: float = 1000.0,
    duration_s: float = 300.0,
    heating_start: float = 60.0,
    eec_time: float = 150.0,
    ueo_time: float = 180.0,
    off_time: float = 200.0,
    baseline_temp: float = 36.0,
    ueo_temp: float = 42.0,
    max_temp: float = 42.5,
    max_temp_time: float = 185.0,
    signal_value: float = 0.0,
    transients: list | None = None,
) -> Session:
    """Create a synthetic session with constant signal and linear temperature."""
    n_samples = int(duration_s * fs)
    time = np.arange(n_samples) / fs

    # Constant z-scored signal (or override)
    zdff = np.full(n_samples, signal_value)

    # Linear temperature ramp: baseline_temp at t=0, max_temp at max_temp_time,
    # then cooling back down
    temperature = np.full(n_samples, baseline_temp, dtype=float)
    max_idx = int(max_temp_time * fs)
    if max_idx > 0:
        temperature[:max_idx] = np.linspace(baseline_temp, max_temp, max_idx)
    if max_idx < n_samples:
        terminal_temp = baseline_temp + 2.0
        temperature[max_idx:] = np.linspace(max_temp, terminal_temp, n_samples - max_idx)

    landmarks = SessionLandmarks(
        heating_start_time=heating_start,
        eec_time=eec_time if n_seizures > 0 else None,
        ueo_time=ueo_time if n_seizures > 0 else None,
        off_time=off_time if n_seizures > 0 else None,
        baseline_temp=baseline_temp,
        max_temp=max_temp,
        max_temp_time=max_temp_time,
        terminal_temp=float(temperature[-1]),
        terminal_time=duration_s,
        eec_temp=float(temperature[int(eec_time * fs)]) if n_seizures > 0 else None,
        ueo_temp=ueo_temp if n_seizures > 0 else None,
        equiv_eec_time=eec_time if n_seizures == 0 else None,
        equiv_ueo_time=ueo_time if n_seizures == 0 else None,
        equiv_off_time=off_time if n_seizures == 0 else None,
        equiv_eec_temp=float(temperature[int(eec_time * fs)]) if n_seizures == 0 else None,
        equiv_ueo_temp=ueo_temp if n_seizures == 0 else None,
    )

    # Synthetic ECoG: zero signal by default
    ecog = np.zeros(n_samples)

    processed = ProcessedData(
        photometry=PhotometryResult(
            dff=zdff,
            dff_zscore=zdff,
            dff_hpf=zdff,
        ),
        ecog_filtered=ecog,
        temperature_smooth=temperature,
        time=time,
        fs=fs,
    )

    return Session(
        mouse_id=mouse_id,
        genotype=genotype,
        n_seizures=n_seizures,
        landmarks=landmarks,
        processed=processed,
        transients=transients or [],
    )


def _make_transient(peak_time: float, amplitude: float = 1.0, half_width: float = 0.5, temp: float | None = None) -> TransientEvent:
    return TransientEvent(
        peak_time=peak_time,
        peak_amplitude=amplitude,
        trough_amplitude=0.0,
        peak_to_trough=amplitude,
        half_width=half_width,
        prominence=amplitude,
        temperature_at_peak=temp,
    )


# ---------------------------------------------------------------------------
# 4.1 Cohort Characteristics
# ---------------------------------------------------------------------------

class TestCohortCharacteristics:
    def test_single_session(self):
        s = _make_session(baseline_temp=36.5, ueo_temp=42.0)
        result = compute_cohort_characteristics([s])
        assert len(result.session_results) == 1
        assert result.baseline_temp_mean == pytest.approx(36.5)
        assert result.seizure_threshold_mean == pytest.approx(42.0)

    def test_group_mean_sem(self):
        s1 = _make_session(mouse_id="m1", baseline_temp=36.0, ueo_temp=41.0)
        s2 = _make_session(mouse_id="m2", baseline_temp=37.0, ueo_temp=43.0)
        result = compute_cohort_characteristics([s1, s2])
        assert result.baseline_temp_mean == pytest.approx(36.5)
        assert result.seizure_threshold_mean == pytest.approx(42.0)
        assert result.baseline_temp_sem > 0
        assert result.seizure_threshold_sem > 0

    def test_control_session(self):
        s = _make_session(n_seizures=0, ueo_temp=41.5)
        result = compute_cohort_characteristics([s])
        assert result.seizure_threshold_mean == pytest.approx(41.5)


# ---------------------------------------------------------------------------
# 4.2 Baseline Transients
# ---------------------------------------------------------------------------

class TestBaselineTransients:
    def test_transients_in_baseline(self):
        transients = [
            _make_transient(10.0, amplitude=2.0, half_width=0.3),
            _make_transient(30.0, amplitude=4.0, half_width=0.5),
            _make_transient(50.0, amplitude=3.0, half_width=0.4),
        ]
        s = _make_session(heating_start=60.0, transients=transients)
        result = compute_baseline_transients([s])
        sr = result.session_results[0]
        assert sr.n_transients == 3
        assert sr.duration_s == pytest.approx(60.0)
        assert sr.frequency_hz == pytest.approx(3.0 / 60.0)
        assert sr.mean_amplitude == pytest.approx(3.0)
        assert sr.mean_half_width_s == pytest.approx(0.4)

    def test_no_transients_in_baseline(self):
        transients = [_make_transient(100.0)]  # after heating start
        s = _make_session(heating_start=60.0, transients=transients)
        result = compute_baseline_transients([s])
        assert result.session_results[0].n_transients == 0

    def test_group_summary(self):
        t1 = [_make_transient(10.0, amplitude=2.0, half_width=0.2)]
        t2 = [_make_transient(20.0, amplitude=4.0, half_width=0.6)]
        s1 = _make_session(mouse_id="m1", heating_start=60.0, transients=t1)
        s2 = _make_session(mouse_id="m2", heating_start=60.0, transients=t2)
        result = compute_baseline_transients([s1, s2])
        assert result.amplitude_mean == pytest.approx(3.0)
        assert result.half_width_mean == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 4.3 Pre-Ictal Mean Signal
# ---------------------------------------------------------------------------

class TestPreictalMean:
    def test_time_binned_means(self):
        s = _make_session(
            heating_start=60.0,
            ueo_time=180.0,
            signal_value=5.0,
        )
        # Set distinct signal values for each period
        proc = s.processed
        signal = proc.photometry.dff_zscore
        fs = proc.fs
        signal[:int(60 * fs)] = 1.0      # baseline
        signal[int(60 * fs):int(120 * fs)] = 3.0  # early heat
        signal[int(120 * fs):int(180 * fs)] = 5.0  # late heat
        result = compute_preictal_mean([s])
        sr = result.session_results[0]
        assert sr.baseline_mean == pytest.approx(1.0)
        assert sr.early_heat_mean == pytest.approx(3.0)
        assert sr.late_heat_mean == pytest.approx(5.0)

    def test_temperature_bins_exist(self):
        s = _make_session()
        config = AnalysisConfig(temp_bin_size=1.0)
        result = compute_preictal_mean([s], config)
        assert len(result.temp_bin_centers) > 0
        assert len(result.temp_bin_group_mean) == len(result.temp_bin_centers)

    def test_group_means(self):
        s1 = _make_session(mouse_id="m1", signal_value=2.0)
        s2 = _make_session(mouse_id="m2", signal_value=4.0)
        result = compute_preictal_mean([s1, s2])
        assert result.baseline_mean == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 4.4 Pre-Ictal Transients
# ---------------------------------------------------------------------------

class TestPreictalTransients:
    def test_sliding_window(self):
        # Place transients during heating (60s to 180s)
        transients = [
            _make_transient(80.0, amplitude=2.0, half_width=0.3, temp=38.0),
            _make_transient(100.0, amplitude=3.0, half_width=0.4, temp=39.0),
            _make_transient(140.0, amplitude=4.0, half_width=0.5, temp=40.5),
        ]
        s = _make_session(heating_start=60.0, ueo_time=180.0, transients=transients)
        config = AnalysisConfig(moving_avg_window_s=30.0, moving_avg_step_s=10.0)
        result = compute_preictal_transients([s], config)
        sr = result.session_results[0]
        assert len(sr.moving_avg) > 0

    def test_temperature_binned(self):
        transients = [
            _make_transient(100.0, amplitude=2.0, half_width=0.3, temp=40.0),
        ]
        s = _make_session(
            heating_start=60.0, ueo_time=180.0, ueo_temp=42.0,
            transients=transients,
        )
        config = AnalysisConfig(temp_bin_size=1.0)
        result = compute_preictal_transients([s], config)
        sr = result.session_results[0]
        assert len(sr.temp_bin_frequency) > 0
        # Transient at temp=40.0, ueo_temp=42.0 → rel=-2.0
        # Should land in the bin containing -2.0
        bin_idx = np.searchsorted(sr.temp_bin_centers, -2.0, side='left')
        assert sr.temp_bin_frequency[max(0, bin_idx - 1)] >= 1 or sr.temp_bin_frequency[min(bin_idx, len(sr.temp_bin_frequency) - 1)] >= 1


# ---------------------------------------------------------------------------
# 4.5 Ictal Mean Signal
# ---------------------------------------------------------------------------

class TestIctalMean:
    def test_seizure_vs_baseline_mean(self):
        s = _make_session(
            heating_start=60.0,
            ueo_time=180.0,
            off_time=200.0,
        )
        signal = s.processed.photometry.dff_zscore
        fs = s.processed.fs
        signal[:int(60 * fs)] = 1.0      # baseline
        signal[int(180 * fs):int(200 * fs)] = 10.0  # seizure
        result = compute_ictal_mean([s])
        sr = result.session_results[0]
        assert sr.baseline_mean == pytest.approx(1.0)
        assert sr.seizure_mean == pytest.approx(10.0)

    def test_delta_preictal_ictal(self):
        s = _make_session(
            heating_start=60.0,
            ueo_time=180.0,
            off_time=200.0,
        )
        signal = s.processed.photometry.dff_zscore
        fs = s.processed.fs
        # Late heat: midpoint(60,180)=120 to 180
        signal[int(120 * fs):int(180 * fs)] = 3.0
        signal[int(180 * fs):int(200 * fs)] = 8.0
        result = compute_ictal_mean([s])
        assert result.session_results[0].delta_preictal_ictal == pytest.approx(5.0)

    def test_triggered_averages_computed(self):
        s = _make_session(
            fs=100.0,
            duration_s=300.0,
            ueo_time=150.0,
        )
        config = AnalysisConfig(triggered_window_s=10.0)
        result = compute_ictal_mean([s], config)
        assert "UEO" in result.triggered_averages
        assert "EEC" in result.triggered_averages
        assert "OFF" in result.triggered_averages
        ta = result.triggered_averages["UEO"]
        assert len(ta.mean_trace) > 1
        assert len(ta.per_session_traces) == 1

    def test_triggered_auc(self):
        s = _make_session(
            fs=100.0,
            duration_s=300.0,
            ueo_time=150.0,
            signal_value=1.0,
        )
        config = AnalysisConfig(triggered_window_s=10.0)
        result = compute_ictal_mean([s], config)
        ta = result.triggered_averages["UEO"]
        # Constant signal → baseline-subtracted → AUC = 0.0
        assert ta.auc == pytest.approx(0.0, abs=1e-10)

    def test_max_temp_triggered(self):
        s = _make_session(
            fs=100.0,
            duration_s=300.0,
            max_temp_time=185.0,
        )
        config = AnalysisConfig(triggered_window_s=10.0)
        result = compute_ictal_mean([s], config)
        assert "max_temp" in result.triggered_averages


# ---------------------------------------------------------------------------
# 4.6 Ictal Transients
# ---------------------------------------------------------------------------

class TestIctalTransients:
    def test_psth_bins(self):
        transients = [
            _make_transient(170.0),  # 10s before UEO
            _make_transient(175.0),  # 5s before
            _make_transient(185.0),  # 5s after
            _make_transient(195.0),  # 15s after
        ]
        s = _make_session(ueo_time=180.0, transients=transients)
        config = AnalysisConfig(psth_bin_size_s=10.0, psth_window_s=30.0)
        result = compute_ictal_transients([s], config)
        sr = result.session_results[0]
        assert len(sr.psth_counts) > 0
        assert np.sum(sr.psth_counts) == 4

    def test_moving_avg_around_ueo(self):
        transients = [_make_transient(180.0 + i) for i in range(-20, 20, 5)]
        s = _make_session(ueo_time=180.0, transients=transients)
        config = AnalysisConfig(
            psth_window_s=30.0,
            moving_avg_window_s=10.0,
            moving_avg_step_s=5.0,
        )
        result = compute_ictal_transients([s], config)
        assert len(result.moving_avg_times) > 0
        assert len(result.freq_mean) == len(result.moving_avg_times)


# ---------------------------------------------------------------------------
# 4.7 Postictal Recovery
# ---------------------------------------------------------------------------

class TestPostictal:
    def test_cooling_curve(self):
        s = _make_session(ueo_temp=42.0, max_temp=42.5, max_temp_time=185.0)
        config = AnalysisConfig(temp_bin_size=1.0)
        result = compute_postictal([s], config)
        sr = result.session_results[0]
        assert len(sr.cooling_bin_means) > 0

    def test_final_metrics(self):
        s = _make_session(duration_s=300.0)
        result = compute_postictal([s])
        sr = result.session_results[0]
        assert sr.final_time == pytest.approx(300.0, abs=1.0)
        assert sr.final_temp > 0

    def test_group_final_metrics(self):
        s1 = _make_session(mouse_id="m1", duration_s=300.0)
        s2 = _make_session(mouse_id="m2", duration_s=300.0)
        result = compute_postictal([s1, s2])
        assert len(result.final_times) == 2
        assert len(result.final_temps) == 2
        assert len(result.final_dffs) == 2


# ---------------------------------------------------------------------------
# 4.8 Spike-Triggered Averages
# ---------------------------------------------------------------------------

class TestSpikeTriggered:
    def test_basic_sta_baseline_subtracted(self):
        s = _make_session(fs=100.0, duration_s=300.0, signal_value=1.0)
        spike_times = np.array([50.0, 100.0, 200.0])
        config = AnalysisConfig(spike_triggered_window_s=5.0, spike_triggered_baseline_start_s=5.0)
        result = compute_spike_triggered_average([s], [spike_times], config)
        sr = result.session_results[0]
        assert sr.n_spikes == 3
        # Constant signal → baseline-subtracted → mean trace all 0.0
        np.testing.assert_allclose(sr.mean_trace, 0.0)

    def test_auc_baseline_subtracted(self):
        s = _make_session(fs=100.0, duration_s=300.0, signal_value=2.0)
        spike_times = np.array([150.0])
        config = AnalysisConfig(spike_triggered_window_s=5.0, spike_triggered_baseline_start_s=5.0)
        result = compute_spike_triggered_average([s], [spike_times], config)
        # Constant signal baseline-subtracted → AUC = 0.0
        assert result.group_auc == pytest.approx(0.0, abs=1e-10)

    def test_nonconstant_sta(self):
        """Non-constant signal: a step at the spike should produce non-zero AUC."""
        s = _make_session(fs=100.0, duration_s=300.0, signal_value=0.0)
        # Create a step: signal goes from 0 to 1.0 at t=150s
        step_idx = int(150.0 * 100.0)
        s.processed.photometry.dff_zscore[step_idx:] = 1.0
        spike_times = np.array([150.0])
        config = AnalysisConfig(spike_triggered_window_s=5.0, spike_triggered_baseline_start_s=5.0)
        result = compute_spike_triggered_average([s], [spike_times], config)
        sr = result.session_results[0]
        # Baseline = mean of -5s to -1s before spike = 0.0
        # Before spike: 0-0=0. After spike: 1-0=1.
        assert sr.mean_trace[0] == pytest.approx(0.0)
        assert sr.mean_trace[-1] == pytest.approx(1.0)
        # AUC should be positive (signal rises after spike)
        assert sr.auc > 0

    def test_edge_spikes_excluded(self):
        s = _make_session(fs=100.0, duration_s=300.0)
        # Spike at 1.0s with 5s window → start at -4.0s → out of bounds
        spike_times = np.array([1.0])
        config = AnalysisConfig(spike_triggered_window_s=5.0)
        result = compute_spike_triggered_average([s], [spike_times], config)
        assert result.session_results[0].n_spikes == 0

    def test_no_spikes(self):
        s = _make_session(fs=100.0, duration_s=300.0)
        spike_times = np.array([])
        config = AnalysisConfig(spike_triggered_window_s=5.0)
        result = compute_spike_triggered_average([s], [spike_times], config)
        assert result.session_results[0].n_spikes == 0
        assert np.isnan(result.group_auc)

    def test_group_average_zero_centered(self):
        s1 = _make_session(mouse_id="m1", fs=100.0, duration_s=300.0, signal_value=2.0)
        s2 = _make_session(mouse_id="m2", fs=100.0, duration_s=300.0, signal_value=4.0)
        spikes1 = np.array([150.0])
        spikes2 = np.array([150.0])
        config = AnalysisConfig(spike_triggered_window_s=5.0)
        result = compute_spike_triggered_average([s1, s2], [spikes1, spikes2], config)
        # Both constant signals → zero-centered → group mean = 0.0
        np.testing.assert_allclose(result.group_mean, 0.0)

    def test_eeg_polarity_alignment(self):
        """EEG segments with negative spikes should be flipped upward."""
        s = _make_session(fs=100.0, duration_s=300.0)
        # Create a negative EEG spike slightly before t=150s
        # (offset from center so zero-centering doesn't cancel it)
        spike_idx = int(150.0 * 100.0)
        s.processed.ecog_filtered[spike_idx - 3:spike_idx] = -2.0
        spike_times = np.array([150.0])
        config = AnalysisConfig(spike_triggered_window_s=5.0)
        result = compute_spike_triggered_average([s], [spike_times], config)
        sr = result.session_results[0]
        window_samples = int(5.0 * 100.0)
        center = window_samples
        # The spike region before center should be positive after flipping
        # (original was -2.0, flipped to +2.0, then zero-centered by value at center=0)
        assert sr.eeg_mean_trace[center - 2] > 0

    def test_eeg_group_results_present(self):
        s = _make_session(fs=100.0, duration_s=300.0)
        spike_times = np.array([150.0])
        config = AnalysisConfig(spike_triggered_window_s=5.0)
        result = compute_spike_triggered_average([s], [spike_times], config)
        assert result.eeg_group_mean is not None
        assert result.eeg_group_sem is not None
        assert len(result.eeg_group_mean) == len(result.group_mean)
