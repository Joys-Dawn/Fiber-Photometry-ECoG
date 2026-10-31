import numpy as np
import pytest

from LC_paper import run_threshold_sweep as sweep
from fiber_photometry_ecog.core.data_models import (
    PhotometryResult,
    RawData,
    Session,
    SessionLandmarks,
)
from fiber_photometry_ecog.preprocessing import ecog as ecog_module
from fiber_photometry_ecog.preprocessing import pipeline as pipeline_module


class _DeterministicPhotometryStrategy:
    def preprocess(self, signal_470, signal_405, fs, config):
        return PhotometryResult(dff=np.linspace(0.0, 1.0, len(signal_470)))


def _make_session() -> Session:
    fs = 10.0
    duration_s = 120.0
    heating_start_s = 100.0
    n_samples = int(fs * duration_s)
    time = np.arange(n_samples) / fs

    temperature_raw = np.full(n_samples, 30.0)
    temperature_raw[int(60.0 * fs):] = 40.0

    raw = RawData(
        signal_470=np.linspace(1.0, 2.0, n_samples),
        signal_405=np.ones(n_samples),
        ecog=np.zeros(n_samples),
        emg=None,
        temperature_raw=temperature_raw,
        temp_bit_volts=0.001,
        temp_slope=1.0,
        temp_intercept=0.0,
        time=time,
        fs=fs,
    )
    return Session(
        mouse_id="test_mouse",
        genotype="WT",
        heating_session=1,
        landmarks=SessionLandmarks(heating_start_time=heating_start_s),
        raw=raw,
    )


def test_cache_miss_and_hit_use_complete_preheating_temperature_baseline(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(sweep, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline_module,
        "filter_ecog",
        lambda signal, fs, config: signal.copy(),
    )
    monkeypatch.setattr(
        ecog_module,
        "filter_ecog",
        lambda signal, fs, config: signal.copy(),
    )
    monkeypatch.setitem(
        pipeline_module.STRATEGY_MAP,
        "C",
        _DeterministicPhotometryStrategy,
    )
    monkeypatch.setattr(pipeline_module, "detect_transients", lambda *args: [])
    monkeypatch.setattr(
        pipeline_module,
        "detect_spikes",
        lambda *args, **kwargs: [],
    )

    cache_miss_session = _make_session()
    sweep.preprocess_and_cache(cache_miss_session, "C", "test_dataset")

    fs = cache_miss_session.processed.fs
    temperature = cache_miss_session.processed.temperature_smooth
    heating_start_idx = int(
        cache_miss_session.landmarks.heating_start_time * fs
    )
    expected = float(np.nanmean(temperature[:heating_start_idx]))
    first_60_s = float(np.nanmean(temperature[:int(60.0 * fs)]))

    assert cache_miss_session.landmarks.baseline_temp == pytest.approx(expected)
    assert cache_miss_session.landmarks.baseline_temp != pytest.approx(first_60_s)

    cache_hit_session = _make_session()
    sweep.preprocess_and_cache(cache_hit_session, "C", "test_dataset")

    assert cache_hit_session.landmarks.baseline_temp == pytest.approx(expected)
