"""
Tests for interictal ECoG spike detection.
"""

import numpy as np

from fiber_photometry_ecog.core.config import SpikeDetectionConfig
from fiber_photometry_ecog.preprocessing.spike_detection import detect_spikes


def _make_ecog_with_spikes(
    fs: float = 1000.0,
    duration_s: float = 60.0,
    baseline_end_s: float = 30.0,
    spike_times: list[float] | None = None,
    spike_amplitude: float = 8.0,
    spike_width_ms: float = 150.0,
    noise_std: float = 1.0,
    negative: bool = False,
) -> np.ndarray:
    """Create a synthetic filtered ECoG with injected spikes."""
    rng = np.random.default_rng(42)
    n_samples = int(duration_s * fs)
    signal = rng.normal(0, noise_std, n_samples)

    if spike_times is None:
        spike_times = []

    spike_width_samples = int(spike_width_ms * fs / 1000)
    half_w = spike_width_samples // 2
    sigma = half_w / 2.5  # yields ~80-100ms half-height width for 150ms total

    for t in spike_times:
        center = int(t * fs)
        start = max(0, center - half_w)
        end = min(n_samples, center + half_w)
        x = np.arange(start, end) - center
        spike_shape = spike_amplitude * np.exp(-0.5 * (x / sigma) ** 2)
        if negative:
            spike_shape = -spike_shape
        signal[start:end] += spike_shape

    return signal


class TestSpikeDetection:
    def test_detects_positive_spikes(self):
        signal = _make_ecog_with_spikes(
            spike_times=[35.0, 45.0],
            spike_amplitude=8.0,
        )
        spikes = detect_spikes(signal, fs=1000.0, baseline_end_s=30.0)
        assert len(spikes) >= 2
        times = [s.time for s in spikes]
        assert any(abs(t - 35.0) < 0.05 for t in times)
        assert any(abs(t - 45.0) < 0.05 for t in times)

    def test_detects_negative_spikes(self):
        signal = _make_ecog_with_spikes(
            spike_times=[40.0],
            spike_amplitude=8.0,
            negative=True,
        )
        spikes = detect_spikes(signal, fs=1000.0, baseline_end_s=30.0)
        neg_spikes = [s for s in spikes if s.polarity == "negative"]
        assert len(neg_spikes) >= 1
        assert any(abs(s.time - 40.0) < 0.05 for s in neg_spikes)

    def test_excludes_seizure_zones(self):
        signal = _make_ecog_with_spikes(
            spike_times=[35.0, 45.0],
            spike_amplitude=8.0,
        )
        # Exclude zone around 45.0
        spikes = detect_spikes(
            signal, fs=1000.0, baseline_end_s=30.0,
            exclusion_zones=[(44.0, 46.0)],
        )
        times = [s.time for s in spikes]
        assert any(abs(t - 35.0) < 0.05 for t in times)
        assert not any(abs(t - 45.0) < 0.05 for t in times)

    def test_edge_spikes_excluded(self):
        signal = _make_ecog_with_spikes(
            duration_s=60.0,
            spike_times=[0.05],  # within 0.1s edge margin
            spike_amplitude=8.0,
        )
        config = SpikeDetectionConfig(edge_margin_s=0.1)
        spikes = detect_spikes(signal, fs=1000.0, baseline_end_s=30.0, config=config)
        assert not any(s.time < 0.1 for s in spikes)

    def test_deduplication(self):
        signal = _make_ecog_with_spikes(
            spike_times=[40.0],
            spike_amplitude=8.0,
        )
        # With both positive and negative detection, a large positive spike
        # should not produce a duplicate negative detection
        spikes = detect_spikes(signal, fs=1000.0, baseline_end_s=30.0)
        # All spikes near t=40 should be deduplicated to one
        near_40 = [s for s in spikes if abs(s.time - 40.0) < 0.015]
        assert len(near_40) <= 1

    def test_no_spikes_in_clean_signal(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1.0, 60000)  # 60s at 1kHz, pure noise
        spikes = detect_spikes(signal, fs=1000.0, baseline_end_s=30.0)
        # Threshold is 3.0 * 1.0 = 3.0; with 30k post-baseline samples,
        # might get a few false positives but they must also pass width constraints
        assert len(spikes) < 5

    def test_width_constraints(self):
        signal = _make_ecog_with_spikes(
            spike_times=[40.0],
            spike_amplitude=8.0,
            spike_width_ms=100.0,
        )
        spikes = detect_spikes(signal, fs=1000.0, baseline_end_s=30.0)
        for s in spikes:
            if abs(s.time - 40.0) < 0.05:
                assert 70.0 <= s.width_ms <= 200.0

    def test_configurable_threshold(self):
        signal = _make_ecog_with_spikes(
            spike_times=[40.0],
            spike_amplitude=4.0,  # moderate amplitude
        )
        # Default tmul=3.0 should detect
        spikes_default = detect_spikes(signal, fs=1000.0, baseline_end_s=30.0)
        # High tmul=10.0 should miss it
        config = SpikeDetectionConfig(tmul=10.0)
        spikes_strict = detect_spikes(signal, fs=1000.0, baseline_end_s=30.0, config=config)
        assert len(spikes_default) >= len(spikes_strict)

