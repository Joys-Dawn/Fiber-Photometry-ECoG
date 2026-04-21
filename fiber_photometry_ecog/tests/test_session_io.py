"""Round-trip tests for session save/load."""

import numpy as np
import pytest

from fiber_photometry_ecog.core.config import PreprocessingConfig, PhotometryConfig
from fiber_photometry_ecog.core.data_models import (
    Session,
    SessionLandmarks,
    RawData,
    ProcessedData,
    PhotometryResult,
    TransientEvent,
    SpikeEvent,
)
from fiber_photometry_ecog.core.session_io import save_session, load_session


def _make_session() -> Session:
    """Create a fully populated Session for round-trip testing."""
    n = 1000
    fs = 1000.0
    time = np.linspace(0, n / fs, n)

    raw = RawData(
        signal_470=np.random.randn(n),
        signal_405=np.random.randn(n),
        ecog=np.random.randn(n),
        emg=np.random.randn(n),
        temperature_raw=np.random.randn(n),
        temp_bit_volts=0.195,
        temp_slope=0.0981,
        temp_intercept=8.81,
        time=time,
        fs=fs,
    )

    phot = PhotometryResult(
        dff=np.random.randn(n),
        dff_zscore=np.random.randn(n),
        dff_hpf=np.random.randn(n),
    )
    proc = ProcessedData(
        photometry=phot,
        ecog_filtered=np.random.randn(n),
        temperature_c=np.linspace(33, 42, n),
        temperature_smooth=np.linspace(33, 42, n),
        time=time,
        fs=fs,
    )

    landmarks = SessionLandmarks(
        heating_start_time=10.0,
        eec_time=50.0,
        ueo_time=55.0,
        behavioral_onset_time=56.0,
        off_time=70.0,
        baseline_temp=36.5,
        max_temp=42.1,
        max_temp_time=65.0,
    )

    transients = [
        TransientEvent(
            peak_time=20.0, peak_amplitude=0.5, trough_amplitude=0.1,
            peak_to_trough=0.4, half_width=0.05, prominence=0.3,
            temperature_at_peak=37.5,
        ),
        TransientEvent(
            peak_time=30.0, peak_amplitude=0.8, trough_amplitude=0.2,
            peak_to_trough=0.6, half_width=0.04, prominence=0.5,
        ),
    ]

    spikes = [
        SpikeEvent(time=15.0, amplitude=3.5, width_ms=12.0, prominence=2.8, polarity="positive"),
        SpikeEvent(time=25.0, amplitude=-4.0, width_ms=10.0, prominence=3.1, polarity="negative"),
    ]

    config = PreprocessingConfig(
        photometry=PhotometryConfig(strategy="B"),
    )

    session = Session(
        mouse_id="3339",
        genotype="Scn1a",
        heating_session=1,
        n_seizures=1,
        sudep=False,
        include_session=True,
        exclusion_reason=None,
        experiment_label="GCaMP / mPFC / PV",
        landmarks=landmarks,
        preprocessing_config=config,
        raw=raw,
        processed=proc,
        transients=transients,
        spikes=spikes,
        cohort="seizure",
        date="2024-06-05",
        session_name="3339_session1",
    )
    return session


class TestSessionIoRoundTrip:
    def test_save_load_preserves_metadata(self, tmp_path):
        session = _make_session()
        path = save_session(session, tmp_path)
        loaded = load_session(path)

        assert loaded.mouse_id == session.mouse_id
        assert loaded.genotype == session.genotype
        assert loaded.heating_session == session.heating_session
        assert loaded.n_seizures == session.n_seizures
        assert loaded.sudep == session.sudep
        assert loaded.include_session == session.include_session
        assert loaded.experiment_label == session.experiment_label
        assert loaded.cohort == session.cohort
        assert loaded.date == session.date
        assert loaded.session_name == session.session_name

    def test_save_load_preserves_landmarks(self, tmp_path):
        session = _make_session()
        path = save_session(session, tmp_path)
        loaded = load_session(path)

        assert loaded.landmarks is not None
        assert loaded.landmarks.heating_start_time == session.landmarks.heating_start_time
        assert loaded.landmarks.eec_time == session.landmarks.eec_time
        assert loaded.landmarks.ueo_time == session.landmarks.ueo_time
        assert loaded.landmarks.off_time == session.landmarks.off_time
        assert loaded.landmarks.baseline_temp == session.landmarks.baseline_temp

    def test_save_load_preserves_arrays(self, tmp_path):
        session = _make_session()
        path = save_session(session, tmp_path)
        loaded = load_session(path)

        np.testing.assert_array_equal(loaded.raw.signal_470, session.raw.signal_470)
        np.testing.assert_array_equal(loaded.raw.ecog, session.raw.ecog)
        np.testing.assert_array_equal(loaded.raw.emg, session.raw.emg)
        np.testing.assert_array_equal(loaded.processed.photometry.dff, session.processed.photometry.dff)
        np.testing.assert_array_equal(loaded.processed.ecog_filtered, session.processed.ecog_filtered)

    def test_save_load_preserves_transients(self, tmp_path):
        session = _make_session()
        path = save_session(session, tmp_path)
        loaded = load_session(path)

        assert len(loaded.transients) == 2
        assert loaded.transients[0].peak_time == 20.0
        assert loaded.transients[1].peak_amplitude == 0.8
        assert loaded.transients[0].temperature_at_peak == 37.5
        assert loaded.transients[1].temperature_at_peak is None

    def test_save_load_preserves_spikes(self, tmp_path):
        session = _make_session()
        path = save_session(session, tmp_path)
        loaded = load_session(path)

        assert len(loaded.spikes) == 2
        assert loaded.spikes[0].time == 15.0
        assert loaded.spikes[0].polarity == "positive"
        assert loaded.spikes[1].amplitude == -4.0

    def test_save_load_preserves_config(self, tmp_path):
        session = _make_session()
        path = save_session(session, tmp_path)
        loaded = load_session(path)

        assert loaded.preprocessing_config.photometry.strategy == "B"

    def test_save_load_no_emg(self, tmp_path):
        session = _make_session()
        session.raw.emg = None
        path = save_session(session, tmp_path)
        loaded = load_session(path)

        assert loaded.raw.emg is None

    def test_save_load_no_spikes(self, tmp_path):
        session = _make_session()
        session.spikes = []
        path = save_session(session, tmp_path)
        loaded = load_session(path)

        assert loaded.spikes == []

    def test_no_pickle_needed(self, tmp_path):
        """Verify that saved files load without allow_pickle=True."""
        session = _make_session()
        path = save_session(session, tmp_path)
        # This should not raise — no pickle objects in the file
        data = np.load(path)
        assert "_metadata_json" in data
        data.close()
