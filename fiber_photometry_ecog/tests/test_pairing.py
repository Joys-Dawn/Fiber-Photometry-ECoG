"""
Tests for Phase 3: Control Equivalent Landmark Assignment.
"""

import numpy as np
import pytest

from fiber_photometry_ecog.core.data_models import (
    Session,
    SessionLandmarks,
    ProcessedData,
)
from fiber_photometry_ecog.pairing.engine import (
    compute_seizure_group_means,
    assign_equivalents_temperature,
    assign_equivalents_time,
    assign_all_controls,
    _find_first_time_at_temp,
)


def _make_temp_trace(fs, duration_s, base_temp, max_temp, peak_fraction=0.6):
    """Create a synthetic temperature trace that ramps up then down.

    Linearly ramps from base_temp to max_temp over peak_fraction of the
    duration, then linearly back down.
    """
    n = int(fs * duration_s)
    peak_idx = int(n * peak_fraction)
    ramp_up = np.linspace(base_temp, max_temp, peak_idx)
    ramp_down = np.linspace(max_temp, base_temp, n - peak_idx)
    return np.concatenate([ramp_up, ramp_down])


def _make_seizure_session(
    mouse_id, fs, duration_s, heat_start, eec_time, ueo_time, off_time,
    base_temp=36.0, max_temp=43.0, beh_time=None,
):
    """Create a seizure Session with synthetic temperature data."""
    temp_trace = _make_temp_trace(fs, duration_s, base_temp, max_temp)
    landmarks = SessionLandmarks(
        heating_start_time=heat_start,
        eec_time=eec_time,
        ueo_time=ueo_time,
        behavioral_onset_time=beh_time,
        off_time=off_time,
    )
    processed = ProcessedData(
        temperature_smooth=temp_trace,
        fs=fs,
    )
    return Session(
        mouse_id=mouse_id,
        genotype="Scn1a",
        n_seizures=1,
        landmarks=landmarks,
        processed=processed,
    )


def _make_control_session(mouse_id, fs, duration_s, heat_start, base_temp=36.0, max_temp=43.0):
    """Create a control Session with synthetic temperature data."""
    temp_trace = _make_temp_trace(fs, duration_s, base_temp, max_temp)
    landmarks = SessionLandmarks(heating_start_time=heat_start)
    processed = ProcessedData(
        temperature_smooth=temp_trace,
        fs=fs,
    )
    return Session(
        mouse_id=mouse_id,
        genotype="WT",
        n_seizures=0,
        landmarks=landmarks,
        processed=processed,
    )


# ──────────────────────────────────────────────────────────────
# compute_seizure_group_means
# ──────────────────────────────────────────────────────────────

class TestComputeSeizureGroupMeans:

    def test_single_session(self):
        fs = 100.0
        sess = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                     eec_time=300.0, ueo_time=320.0, off_time=350.0)
        means = compute_seizure_group_means([sess])

        assert means.mean_eec_time == pytest.approx(300.0 - 60.0)
        assert means.mean_ueo_time == pytest.approx(320.0 - 60.0)
        assert means.mean_off_time == pytest.approx(350.0 - 60.0)
        assert means.mean_seizure_duration == pytest.approx(30.0)
        assert means.mean_behavioral_onset_time is None

    def test_two_sessions_averaged(self):
        fs = 100.0
        s1 = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                   eec_time=300.0, ueo_time=320.0, off_time=350.0)
        s2 = _make_seizure_session("M2", fs, 600, heat_start=60.0,
                                   eec_time=280.0, ueo_time=310.0, off_time=330.0)
        means = compute_seizure_group_means([s1, s2])

        assert means.mean_eec_time == pytest.approx((240.0 + 220.0) / 2)
        assert means.mean_ueo_time == pytest.approx((260.0 + 250.0) / 2)
        assert means.mean_seizure_duration == pytest.approx((30.0 + 20.0) / 2)

    def test_behavioral_onset_included(self):
        fs = 100.0
        sess = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                     eec_time=300.0, ueo_time=320.0, off_time=350.0,
                                     beh_time=325.0)
        means = compute_seizure_group_means([sess])
        assert means.mean_behavioral_onset_time == pytest.approx(265.0)

    def test_no_valid_sessions_raises(self):
        with pytest.raises(ValueError, match="No valid seizure sessions"):
            compute_seizure_group_means([])

    def test_session_without_landmarks_skipped(self):
        fs = 100.0
        good = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                     eec_time=300.0, ueo_time=320.0, off_time=350.0)
        bad = Session(mouse_id="M2", genotype="Scn1a", n_seizures=1,
                      landmarks=None)
        means = compute_seizure_group_means([good, bad])
        assert means.mean_eec_time == pytest.approx(240.0)

    def test_session_with_partial_landmarks_skipped(self):
        fs = 100.0
        good = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                     eec_time=300.0, ueo_time=320.0, off_time=350.0)
        partial = Session(
            mouse_id="M2", genotype="Scn1a", n_seizures=1,
            landmarks=SessionLandmarks(heating_start_time=60.0, eec_time=300.0),
        )
        means = compute_seizure_group_means([good, partial])
        assert means.mean_eec_time == pytest.approx(240.0)


# ──────────────────────────────────────────────────────────────
# _find_first_time_at_temp
# ──────────────────────────────────────────────────────────────

class TestFindFirstTimeAtTemp:

    def test_exact_match(self):
        fs = 10.0
        # Linear ramp from 36 to 43 over 100 samples (10 seconds)
        trace = np.linspace(36.0, 43.0, 100)
        t = _find_first_time_at_temp(trace, fs, 40.0)
        # 40.0 is at fraction (40-36)/(43-36) = 4/7 of 100 samples ≈ idx 57
        expected_time = (4.0 / 7.0 * 99) / fs
        assert t == pytest.approx(expected_time, abs=0.2)

    def test_returns_first_occurrence(self):
        fs = 10.0
        # Ramp up then down — target temp appears twice; should return first
        up = np.linspace(36.0, 43.0, 50)
        down = np.linspace(43.0, 36.0, 50)
        trace = np.concatenate([up, down])
        t = _find_first_time_at_temp(trace, fs, 40.0, max_idx=49)
        assert t is not None
        # Should be in the first half
        assert t < 5.0

    def test_never_reaches_temp(self):
        fs = 10.0
        trace = np.linspace(36.0, 39.0, 100)
        t = _find_first_time_at_temp(trace, fs, 42.0)
        # 42 is more than 1°C away from max of 39 → None
        assert t is None

    def test_empty_trace_with_max_idx(self):
        t = _find_first_time_at_temp(np.array([]), 10.0, 40.0, max_idx=0)
        assert t is None

    def test_empty_trace_without_max_idx(self):
        t = _find_first_time_at_temp(np.array([]), 10.0, 40.0)
        assert t is None


# ──────────────────────────────────────────────────────────────
# Temperature-matched assignment
# ──────────────────────────────────────────────────────────────

class TestAssignEquivalentsTemperature:

    def test_assigns_equivalent_times(self):
        fs = 100.0
        seizure = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                        eec_time=300.0, ueo_time=320.0, off_time=350.0)
        control = _make_control_session("C1", fs, 600, heat_start=60.0)

        means = compute_seizure_group_means([seizure])
        assign_equivalents_temperature(control, means)

        assert control.landmarks.equiv_eec_time is not None
        assert control.landmarks.equiv_ueo_time is not None
        assert control.landmarks.equiv_off_time is not None

    def test_off_equals_ueo_plus_duration(self):
        fs = 100.0
        seizure = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                        eec_time=300.0, ueo_time=320.0, off_time=350.0)
        control = _make_control_session("C1", fs, 600, heat_start=60.0)

        means = compute_seizure_group_means([seizure])
        assign_equivalents_temperature(control, means)

        assert control.landmarks.equiv_off_time == pytest.approx(
            control.landmarks.equiv_ueo_time + means.mean_seizure_duration
        )

    def test_eec_before_ueo(self):
        fs = 100.0
        seizure = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                        eec_time=300.0, ueo_time=320.0, off_time=350.0)
        control = _make_control_session("C1", fs, 600, heat_start=60.0)

        means = compute_seizure_group_means([seizure])
        assign_equivalents_temperature(control, means)

        assert control.landmarks.equiv_eec_time < control.landmarks.equiv_ueo_time

    def test_no_processed_data_raises(self):
        control = Session(
            mouse_id="C1", genotype="WT",
            landmarks=SessionLandmarks(heating_start_time=60.0),
        )
        means = compute_seizure_group_means([
            _make_seizure_session("M1", 100.0, 600, 60.0, 300.0, 320.0, 350.0)
        ])
        with pytest.raises(ValueError, match="no processed temperature"):
            assign_equivalents_temperature(control, means)

    def test_no_behavioral_onset_stays_none(self):
        fs = 100.0
        seizure = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                        eec_time=300.0, ueo_time=320.0, off_time=350.0)
        control = _make_control_session("C1", fs, 600, heat_start=60.0)

        means = compute_seizure_group_means([seizure])
        assign_equivalents_temperature(control, means)

        assert control.landmarks.equiv_behavioral_onset_time is None


# ──────────────────────────────────────────────────────────────
# Time-matched assignment
# ──────────────────────────────────────────────────────────────

class TestAssignEquivalentsTime:

    def test_assigns_mean_elapsed_times(self):
        fs = 100.0
        seizure = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                        eec_time=300.0, ueo_time=320.0, off_time=350.0)
        control = _make_control_session("C1", fs, 600, heat_start=70.0)

        means = compute_seizure_group_means([seizure])
        assign_equivalents_time(control, means)

        # Mean EEC time from heat start = 240s, so equiv = 70 + 240 = 310
        assert control.landmarks.equiv_eec_time == pytest.approx(70.0 + 240.0)
        assert control.landmarks.equiv_ueo_time == pytest.approx(70.0 + 260.0)
        assert control.landmarks.equiv_off_time == pytest.approx(70.0 + 290.0)

    def test_no_landmarks_raises(self):
        control = Session(mouse_id="C1", genotype="WT")
        means = compute_seizure_group_means([
            _make_seizure_session("M1", 100.0, 600, 60.0, 300.0, 320.0, 350.0)
        ])
        with pytest.raises(ValueError, match="no landmarks"):
            assign_equivalents_time(control, means)

    def test_behavioral_onset_none_when_absent(self):
        fs = 100.0
        seizure = _make_seizure_session("M1", fs, 600, heat_start=60.0,
                                        eec_time=300.0, ueo_time=320.0, off_time=350.0)
        control = _make_control_session("C1", fs, 600, heat_start=60.0)

        means = compute_seizure_group_means([seizure])
        assign_equivalents_time(control, means)

        assert control.landmarks.equiv_behavioral_onset_time is None


# ──────────────────────────────────────────────────────────────
# assign_all_controls
# ──────────────────────────────────────────────────────────────

class TestAssignAllControls:

    def test_temperature_mode(self):
        fs = 100.0
        seizures = [
            _make_seizure_session("M1", fs, 600, 60.0, 300.0, 320.0, 350.0),
            _make_seizure_session("M2", fs, 600, 60.0, 280.0, 310.0, 330.0),
        ]
        controls = [
            _make_control_session("C1", fs, 600, 60.0),
            _make_control_session("C2", fs, 600, 60.0),
        ]

        means = assign_all_controls(seizures, controls, mode="temperature")

        for c in controls:
            assert c.landmarks.equiv_eec_time is not None
            assert c.landmarks.equiv_ueo_time is not None
            assert c.landmarks.equiv_off_time is not None

        assert means.mean_seizure_duration == pytest.approx(25.0)

    def test_time_mode(self):
        fs = 100.0
        seizures = [
            _make_seizure_session("M1", fs, 600, 60.0, 300.0, 320.0, 350.0),
        ]
        controls = [
            _make_control_session("C1", fs, 600, 70.0),
        ]

        assign_all_controls(seizures, controls, mode="time")

        assert controls[0].landmarks.equiv_eec_time == pytest.approx(70.0 + 240.0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            assign_all_controls([], [], mode="invalid")

    def test_returns_group_means(self):
        fs = 100.0
        seizures = [
            _make_seizure_session("M1", fs, 600, 60.0, 300.0, 320.0, 350.0),
        ]
        means = assign_all_controls(seizures, [], mode="time")
        assert means.mean_eec_time == pytest.approx(240.0)
