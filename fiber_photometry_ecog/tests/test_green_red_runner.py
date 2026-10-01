"""Tests for the standalone dual-color no-isosbestic runner."""

from __future__ import annotations

import matplotlib.image as mpimg
import numpy as np
import pytest

from fiber_photometry_ecog.core.data_models import (
    RawData,
    Session,
    SessionLandmarks,
)
import run_green_red_ne as runner


def _session(
    mouse_id: str,
    date: str,
    heating_session: int,
    cohort: str,
    green: np.ndarray | None = None,
    red: np.ndarray | None = None,
    fs: float = 10.0,
) -> Session:
    if green is None:
        green = np.zeros(2501, dtype=float)
    if red is None:
        red = np.zeros_like(green)
    n = len(green)
    session = Session(
        mouse_id=mouse_id,
        genotype="Scn1a",
        heating_session=heating_session,
        n_seizures=1,
        landmarks=SessionLandmarks(
            heating_start_time=50.0,
            ueo_time=120.0,
            off_time=150.0,
        ),
        raw=RawData(
            signal_470=green,
            signal_405=red,
            ecog=np.zeros(n),
            emg=None,
            temperature_raw=np.zeros(n),
            temp_bit_volts=1.0,
            temp_slope=1.0,
            temp_intercept=0.0,
            time=np.arange(n, dtype=float) / fs,
            fs=fs,
        ),
        cohort=cohort,
        date=date,
    )
    return session


def test_group_specs_encode_the_four_requested_experimental_conditions():
    conditions = [
        (
            spec.root_name,
            spec.cohort,
            spec.expected_mice,
            spec.expected_sessions,
        )
        for spec in runner.GROUP_SPECS
    ]

    assert conditions == [
        ("real_red_NE", "scn1a_seizure", 2, 4),
        ("mutant_red_NE", "scn1a_seizure_mut", 2, 4),
        ("real_red_NE", "wt", 2, 2),
        ("mutant_red_NE", "wt_mut", 2, 3),
    ]


def test_extract_aligned_trace_subtracts_only_the_requested_event_baseline():
    fs = 10.0
    signal = np.arange(500, dtype=float)

    time_axis, trace = runner._extract_aligned_trace(
        signal,
        fs,
        event_s=20.0,
        pre_s=2.0,
        post_s=2.0,
        baseline_start_s=1.0,
        baseline_end_s=0.5,
    )

    np.testing.assert_allclose(time_axis, np.arange(-20, 21) / fs)
    np.testing.assert_allclose(np.mean(trace[10:15]), 0.0)
    assert trace[20] == pytest.approx(8.0)


def test_session_event_traces_keep_green_and_red_channels_independent(
    tmp_path,
    monkeypatch,
):
    fs = 10.0
    n = 2501
    event_idx = int(120.0 * fs)
    green = np.zeros(n, dtype=float)
    red = np.zeros(n, dtype=float)
    green[event_idx:] = 2.0
    red[event_idx:] = -3.0
    session = _session(
        "m1",
        "2026-01-01",
        1,
        "scn1a_seizure",
        green=green,
        red=red,
        fs=fs,
    )
    processed_inputs = []

    def fake_preprocess(signal, _fs, _baseline_end_s):
        processed_inputs.append(np.asarray(signal).copy())
        return np.asarray(signal, dtype=float).copy()

    monkeypatch.setattr(runner, "_preprocess_channel", fake_preprocess)

    _, green_trace, red_trace = runner._session_event_traces(
        session,
        runner.GROUP_SPECS[0],
        tmp_path,
        rebuild_cache=True,
    )

    assert len(processed_inputs) == 2
    np.testing.assert_array_equal(processed_inputs[0], green)
    np.testing.assert_array_equal(processed_inputs[1], red)
    assert green_trace[int(runner.PRE_EVENT_S * fs)] == pytest.approx(2.0)
    assert red_trace[int(runner.PRE_EVENT_S * fs)] == pytest.approx(-3.0)


def test_validate_group_rejects_missing_sessions_instead_of_silent_output():
    spec = runner.GROUP_SPECS[0]
    sessions = [
        _session("m1", "2026-01-01", 1, spec.cohort),
        _session("m1", "2026-01-02", 2, spec.cohort),
        _session("m2", "2026-01-01", 1, spec.cohort),
    ]

    with pytest.raises(ValueError, match="expected 4 sessions, found 3"):
        runner._validate_group(sessions, spec)


def test_shared_y_limits_cover_all_mean_sem_envelopes_with_five_percent_padding():
    time_axis = np.array([-1.0, 1.0])
    summaries = [
        runner.GroupSummary(
            time_axis=time_axis,
            green_mean=np.array([-3.0, 1.0]),
            green_sem=np.array([0.5, 0.5]),
            red_mean=np.array([-1.0, 2.0]),
            red_sem=np.array([0.25, 0.25]),
            n_mice=2,
            n_sessions=2,
        ),
        runner.GroupSummary(
            time_axis=time_axis,
            green_mean=np.array([0.0, 4.0]),
            green_sem=np.array([1.0, 1.0]),
            red_mean=np.array([-2.0, 3.0]),
            red_sem=np.array([0.5, 0.5]),
            n_mice=2,
            n_sessions=2,
        ),
    ]

    y_limits = runner._shared_y_limits(summaries)

    assert y_limits == pytest.approx((-3.925, 5.425))


def test_plot_group_writes_nonblank_dual_color_summary(tmp_path):
    time_axis = np.linspace(-100.0, 100.0, 401)
    summary = runner.GroupSummary(
        time_axis=time_axis,
        green_mean=np.exp(-((time_axis - 5.0) / 15.0) ** 2),
        green_sem=np.full_like(time_axis, 0.08),
        red_mean=-0.7 * np.exp(-((time_axis - 8.0) / 18.0) ** 2),
        red_sem=np.full_like(time_axis, 0.06),
        n_mice=2,
        n_sessions=4,
    )
    output_path = tmp_path / "summary.png"

    result = runner._plot_group(summary, runner.GROUP_SPECS[0], output_path)

    image = mpimg.imread(result)
    assert result == output_path
    assert output_path.stat().st_size > 0
    assert float(np.std(image)) > 0.01


def test_plot_group_matches_app_aligned_trace_style(tmp_path, monkeypatch):
    time_axis = np.linspace(-100.0, 100.0, 401)
    summary = runner.GroupSummary(
        time_axis=time_axis,
        green_mean=np.sin(time_axis / 20.0),
        green_sem=np.full_like(time_axis, 0.08),
        red_mean=np.cos(time_axis / 20.0),
        red_sem=np.full_like(time_axis, 0.06),
        n_mice=2,
        n_sessions=4,
    )
    original_close = runner.plt.close
    monkeypatch.setattr(runner.plt, "close", lambda _fig: None)

    runner._plot_group(
        summary,
        runner.GROUP_SPECS[0],
        tmp_path / "styled_summary.png",
        y_limits=(-10.0, 20.0),
    )

    figure = runner.plt.gcf()
    try:
        ax = figure.axes[0]
        trace_lines = {
            line.get_label(): line
            for line in ax.lines
            if line.get_label() in {"Green GRAB-NE", "Red NE sensor"}
        }
        assert ax.get_title() == ""
        assert ax.get_xlabel() == "Time rel. to seizure onset (s)"
        assert ax.get_ylabel() == runner.DFF_LABEL
        assert ax.get_xlim() == pytest.approx((-100.0, 100.0))
        assert ax.get_ylim() == pytest.approx((-10.0, 20.0))
        assert set(trace_lines) == {"Green GRAB-NE", "Red NE sensor"}
        assert trace_lines["Green GRAB-NE"].get_color() == "green"
        assert trace_lines["Red NE sensor"].get_color() == "red"
        assert all(line.get_linestyle() == "-" for line in trace_lines.values())
        assert [band.get_alpha() for band in ax.collections] == [0.25, 0.25]
        assert ax.get_legend() is None
    finally:
        original_close(figure)
