"""Tests for the standalone repeated-train GRAB_opto runner."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from LC_paper import run_grab_opto as runner


def _stim_signal(n_trains: int, *, pulses_in_first: int = 20) -> tuple[np.ndarray, float]:
    fs = 1000.0
    first_onset_s = 5.0
    signal = np.zeros(int((first_onset_s + n_trains * 10.0 + 2.0) * fs))
    for train_index in range(n_trains):
        onset_s = first_onset_s + 10.0 * train_index
        pulse_count = pulses_in_first if train_index == 0 else 20
        for pulse_index in range(pulse_count):
            pulse_s = onset_s + pulse_index / 20.0
            lo = int(round(pulse_s * fs))
            signal[lo : lo + 5] = 4.0
    return signal, fs


def _train(index: int, onset_s: float) -> runner.StimTrain:
    return runner.StimTrain(
        index=index,
        onset_s=onset_s,
        offset_s=onset_s + 0.954,
        supra_threshold_samples=100,
        pulse_count=20,
    )


def test_detect_and_select_trains_keeps_first_ten_of_thirty():
    signal, fs = _stim_signal(30)

    detected, selected = runner._detect_and_select_trains(signal, fs)

    assert len(detected) == 30
    assert len(selected) == 10
    assert [train.index for train in selected] == list(range(1, 11))
    assert all(train.pulse_count == 20 for train in detected)
    assert selected[0].onset_s == pytest.approx(5.0)
    assert selected[-1].onset_s == pytest.approx(95.0)


def test_detect_and_select_trains_rejects_malformed_pulse_train():
    signal, fs = _stim_signal(10, pulses_in_first=19)

    with pytest.raises(ValueError, match="Malformed channel-23 train"):
        runner._detect_and_select_trains(signal, fs)


def test_extract_event_metrics_uses_nonoverlapping_ten_second_windows():
    fs = 10.0
    signal = np.zeros(300)
    signal[50:150] = 1.0
    signal[150:250] = 2.0
    trains = (_train(1, 5.0), _train(2, 15.0))

    time_s, traces, peaks, aucs = runner._extract_event_metrics(
        signal,
        fs,
        trains,
    )

    assert time_s[0] == pytest.approx(-5.0)
    assert time_s[-1] == pytest.approx(9.9)
    assert traces.shape == (2, 150)
    np.testing.assert_allclose(peaks, [1.0, 2.0])
    np.testing.assert_allclose(aucs, [9.9, 19.8])


def test_baseline_correct_dff_subtracts_mean_without_scaling():
    signal = np.array([1.0, 3.0, 5.0, 7.0])

    corrected = runner._baseline_correct_dff(signal, fs=1.0, baseline_end_s=2.0)

    np.testing.assert_allclose(corrected, [-1.0, 1.0, 3.0, 5.0])


def test_validate_sessions_requires_one_before_and_after_recording_per_mouse():
    sessions = [
        SimpleNamespace(mouse_id=mouse, cohort=condition)
        for mouse in runner.EXPECTED_MOUSE_IDS
        for condition in runner.CONDITIONS
    ]
    runner._validate_sessions(sessions)

    sessions[-1] = SimpleNamespace(
        mouse_id=runner.EXPECTED_MOUSE_IDS[-1],
        cohort=runner.BEFORE,
    )
    with pytest.raises(ValueError, match="pairing validation failed"):
        runner._validate_sessions(sessions)


def test_mouse_level_frame_averages_technical_replicates_within_mouse():
    rows = []
    for mouse_i, mouse in enumerate(runner.EXPECTED_MOUSE_IDS):
        for condition_i, condition in enumerate(runner.CONDITIONS):
            for train in range(1, 11):
                value = mouse_i + condition_i + train
                rows.append(
                    {
                        "mouse_id": mouse,
                        "date": "2026-01-01",
                        "condition": condition,
                        "condition_label": runner.CONDITION_LABELS[condition],
                        "technical_replicate": train,
                        "evoked_peak_dff": value,
                        "evoked_auc_dff_s": 2 * value,
                    }
                )

    result = runner._mouse_level_frame(pd.DataFrame(rows))

    assert len(result) == 10
    assert set(result["n_technical_replicates"]) == {10}
    first = result.iloc[0]
    assert first["mouse_id"] == runner.EXPECTED_MOUSE_IDS[0]
    assert first["condition"] == runner.BEFORE
    assert first["evoked_peak_mean_dff"] == pytest.approx(5.5)
    assert first["evoked_auc_mean_dff_s"] == pytest.approx(11.0)


def test_paired_statistics_uses_five_mouse_means():
    rows = []
    for index, mouse in enumerate(runner.EXPECTED_MOUSE_IDS, start=1):
        for condition, shift in ((runner.BEFORE, 0.0), (runner.AFTER, -0.5)):
            rows.append(
                {
                    "mouse_id": mouse,
                    "condition": condition,
                    "evoked_peak_mean_dff": index + shift,
                    "evoked_auc_mean_dff_s": 2 * index + shift,
                }
            )

    stats = runner._paired_statistics(pd.DataFrame(rows))

    assert len(stats) == 2
    assert set(stats["n_paired_mice"]) == {5}
    assert set(stats["experimental_unit"]) == {"mouse"}
    assert set(stats["technical_replicates_per_mouse_condition"]) == {10}
    np.testing.assert_allclose(
        stats["paired_mean_difference_after_minus_before"],
        [-0.5, -0.5],
    )


def test_plot_paired_metric_writes_nonblank_app_styled_png(tmp_path, monkeypatch):
    rows = []
    for index, mouse in enumerate(runner.EXPECTED_MOUSE_IDS, start=1):
        rows.extend(
            [
                {
                    "mouse_id": mouse,
                    "condition": runner.BEFORE,
                    "evoked_peak_mean_dff": float(index),
                },
                {
                    "mouse_id": mouse,
                    "condition": runner.AFTER,
                    "evoked_peak_mean_dff": float(index) / 2,
                },
            ]
        )
    output_path = tmp_path / "paired.png"
    original_close = runner.plt.close
    monkeypatch.setattr(runner.plt, "close", lambda _fig: None)

    result = runner._plot_paired_metric(
        pd.DataFrame(rows),
        "evoked_peak_mean_dff",
        "Evoked peak",
        "Peak",
        output_path,
    )

    image = mpimg.imread(result)
    assert result == output_path
    assert output_path.stat().st_size > 0
    assert float(np.std(image)) > 0.01
    figure = runner.plt.gcf()
    try:
        ax = figure.axes[0]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "Before",
            "After",
        ]
        assert all(tick.get_rotation() == 45 for tick in ax.get_xticklabels())
        assert ax.get_xlabel() == "Yohimbine"
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        xlabel_box = ax.xaxis.label.get_window_extent(renderer)
        assert all(
            not tick.get_window_extent(renderer).overlaps(xlabel_box)
            for tick in ax.get_xticklabels()
        )
        assert ax.get_title() == ""
        assert ax.get_legend() is None
    finally:
        original_close(figure)
