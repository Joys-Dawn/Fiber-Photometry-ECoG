"""Regression tests for publication plotting and dataset runner policies."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from types import SimpleNamespace

from fiber_photometry_ecog.analysis.baseline_transients import (
    BaselineTransientGroupResult,
    BaselineTransientSessionResult,
)
from fiber_photometry_ecog.analysis.ictal_transients import IctalTransientGroupResult
from fiber_photometry_ecog.analysis.ictal_mean import (
    IctalMeanGroupResult,
    IctalMeanSessionResult,
    TriggeredAverage,
)
from fiber_photometry_ecog.analysis.opto_response import (
    OptoResponseGroupResult,
    OptoResponseSessionResult,
)
from fiber_photometry_ecog.core.data_models import (
    PhotometryResult,
    ProcessedData,
    Session,
    SessionLandmarks,
)
from fiber_photometry_ecog.statistics import StatsConfig, run_group_comparisons
from fiber_photometry_ecog.visualization import _style
from fiber_photometry_ecog.visualization.group_plots import (
    _add_nested_period_labels,
    _grouped_scatter_figsize,
    _grouped_scatter_mean_sem,
    _save_standalone_scatter,
    _scatter_mean_sem,
    _seizure_cohort_key,
    plot_baseline_transients,
    plot_ictal_mean,
    plot_ictal_transients,
    plot_opto_response,
    plot_preictal_mean_diagnostic,
    plot_ueo_aligned_heatmaps,
    plot_ueo_per_cohort,
)
from fiber_photometry_ecog.visualization.trace_plots import (
    _add_landmark_overlays_plotly,
    _draw_landmark_lines,
    _layout_landmark_labels,
    _plotly_range_with_zero,
    plot_baseline,
    plot_zoomed,
)
from LC_paper.run_hyperthermia import (
    DG_POSITIVE_ONLY_AUC_LANDMARKS,
    ICTAL_EXTREMA_MIN_DURATION_S,
    INDIVIDUAL_SEIZURE_AUC_SUBFOLDERS,
    LC_AUC_WINDOWS_S,
    _genotype_classifier,
    _genotype_point_colors,
    _seizure_outcome_classifier,
    _source_point_colors,
)
from LC_paper import run_threshold_sweep as sweep
import fiber_photometry_ecog.visualization.group_plots as group_plots


def test_zero_reference_expands_positive_axis_and_uses_canonical_style():
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [2.0, 3.0])

    _style.add_zero_y(ax, ensure_visible=True)

    lo, hi = ax.get_ylim()
    zero_line = ax.lines[-1]
    assert lo < 0 < hi
    assert zero_line.get_color() == _style.ZERO_LINE_COLOR
    assert zero_line.get_linestyle() == "--"
    assert zero_line.get_linewidth() == _style.ZERO_LINEWIDTH
    plt.close(fig)


def test_vertical_zero_reference_is_black():
    fig, ax = plt.subplots()
    _style.add_zero_x(ax)

    assert ax.lines[-1].get_color() == "black"
    assert ax.lines[-1].get_linestyle() == "--"
    plt.close(fig)


def test_fixed_axes_size_survives_long_labels_colorbars_and_tight_export(tmp_path):
    for with_colorbar in (False, True):
        for long_labels in (False, True):
            with plt.rc_context(_style._PLOT_RC):
                fig, ax = plt.subplots(figsize=_style.FIG_WIDE)
                colorbar = None
                if with_colorbar:
                    im = ax.imshow(np.arange(12).reshape(3, 4), aspect="auto")
                    colorbar = fig.colorbar(im, ax=ax)
                    colorbar.set_label("Signal with a longer label" if long_labels else "Signal")
                else:
                    ax.plot([0, 1], [0, 1])
                ax.set_xlabel("Time relative to seizure onset (s)" if long_labels else "Time")
                ax.set_ylabel("A longer signal label\nwith a second line" if long_labels else "Signal")
                _style.finalize_publication_figure(fig)
                _style.fix_axes_size(ax, (14.0, 6.0), colorbar=colorbar)
                fig.savefig(tmp_path / f"{with_colorbar}_{long_labels}.png",
                            dpi=30, bbox_inches="tight")
                fig.canvas.draw()
                np.testing.assert_allclose(ax.get_window_extent().size / fig.dpi, (14.0, 6.0))
                if colorbar is not None:
                    np.testing.assert_allclose(
                        colorbar.ax.get_window_extent().height / fig.dpi, 6.0,
                    )
                plt.close(fig)


def test_plotly_zero_range_is_data_driven_and_padded():
    positive_range = _plotly_range_with_zero(np.array([2.0, 10.0]))
    negative_range = _plotly_range_with_zero(np.array([-10.0, -2.0]))

    assert positive_range[0] < 0.0
    assert positive_range[1] > 10.0
    assert negative_range[0] < -10.0
    assert negative_range[1] > 0.0


def test_scatter_connects_matching_mouse_sessions_and_preserves_point_colors():
    ids = [("m1", 1), ("m2", 1)]
    cohorts = {
        "10HZ_DG": ([1.0, 2.0], 1.5, 0.5, ids),
        "10HZ_DG_CPP": ([1.5, 2.5], 2.0, 0.5, ids),
    }
    point_colors = {ids[0]: "blue", ids[1]: "purple"}
    fig, ax = plt.subplots()

    _scatter_mean_sem(
        ax,
        cohorts,
        "Response",
        scatter_options={
            "connect_pairs": True,
            "point_colors_by_id": point_colors,
        },
    )

    pair_lines = [line for line in ax.lines if line.get_linewidth() == 1.8]
    assert len(pair_lines) == 2
    for line in pair_lines:
        assert np.diff(line.get_xdata()) == np.array([1.0])

    expected_colors = {
        to_rgba("blue", alpha=0.85),
        to_rgba("purple", alpha=0.85),
    }
    colored_collections = [
        collection for collection in ax.collections
        if len(collection.get_facecolors()) > 0
    ]
    assert len(colored_collections) == 2
    for collection in colored_collections:
        assert set(map(tuple, collection.get_facecolors())) == expected_colors
    plt.close(fig)


def test_grouped_scatter_rotates_period_labels_without_overlap():
    period_data = [
        {
            "No seizure": ([1.0, 2.0], 1.5, 0.5),
            "Seizure": ([2.0, 3.0], 2.5, 0.5),
        }
        for _ in range(3)
    ]
    with plt.rc_context(_style._PLOT_RC):
        fig, ax = plt.subplots(figsize=_grouped_scatter_figsize(2, 3))
        _grouped_scatter_mean_sem(
            ax,
            period_data,
            ["Early Heat", "Late Heat", "Ictal"],
            "Mean z_ΔF/F",
        )
        fig.canvas.draw()

        labels = ax.get_xticklabels()
        assert all(label.get_rotation() == 45 for label in labels)
        assert all(label.get_ha() == "right" for label in labels)
        renderer = fig.canvas.get_renderer()
        boxes = [label.get_window_extent(renderer) for label in labels]
        assert all(not left.overlaps(right) for left, right in zip(boxes, boxes[1:]))
        plt.close(fig)


def test_coincident_landmark_labels_use_distinct_vertical_levels():
    landmarks = SessionLandmarks(
        eec_time=10.0,
        ueo_time=10.0,
        off_time=20.0,
    )
    entries = [
        ("eec", landmarks.eec_time),
        ("ueo", landmarks.ueo_time),
        ("off", landmarks.off_time),
    ]
    layout = _layout_landmark_labels(entries, (0.0, 30.0))

    levels_at_ten = [level for _, x, level in layout if x == 10.0]
    assert len(levels_at_ten) == 2
    assert len(set(levels_at_ten)) == 2

    fig, axes = plt.subplots(2, 1)
    for ax in axes:
        ax.set_xlim(0.0, 30.0)
    _draw_landmark_lines(axes, landmarks)
    assert all(
        line.get_color() == "black"
        for axis in axes
        for line in axis.lines
    )
    assert all(
        line.get_alpha() == 1.0
        for axis in axes
        for line in axis.lines
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    landmark_boxes = [
        text.get_window_extent(renderer) for text in axes[0].texts
    ]
    assert len(landmark_boxes) == 3
    assert not landmark_boxes[0].overlaps(landmark_boxes[1])
    plt.close(fig)


def test_plotly_landmark_lines_are_black_and_labels_keep_their_colors():
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1)
    landmarks = SessionLandmarks(ueo_time=10.0, off_time=20.0)
    _add_landmark_overlays_plotly(
        fig, landmarks, x_range=(0.0, 30.0),
    )

    assert all(shape.line.color == "black" for shape in fig.layout.shapes)
    assert all(shape.opacity == 1.0 for shape in fig.layout.shapes)
    assert {annotation.font.color for annotation in fig.layout.annotations} == {
        "red", "blue",
    }


def test_pooled_hyperthermia_uses_actual_seizure_cohort_for_characteristics():
    results = {"No seizure": object(), "Seizure": object()}

    assert _seizure_cohort_key(results) == "Seizure"


def test_nested_period_and_cohort_labels_do_not_overlap():
    with plt.rc_context(_style._PLOT_RC):
        fig, ax = plt.subplots(figsize=(24.0, 16.0))
        cohort_centers = np.array([0.0, 1.0])
        period_offsets = np.array([-0.35, 0.0, 0.35])
        period_texts = _add_nested_period_labels(
            ax,
            cohort_centers,
            period_offsets,
            ["Early Heat", "Late Heat", "Ictal"],
            ["No seizure", "Seizure"],
        )
        ax.set_xlim(-0.6, 1.6)
        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()
        period_boxes = [text.get_window_extent(renderer) for text in period_texts]
        cohort_boxes = [
            text.get_window_extent(renderer) for text in ax.get_xticklabels()
        ]
        assert all(
            not period_box.overlaps(cohort_box)
            for period_box in period_boxes
            for cohort_box in cohort_boxes
        )
        plt.close(fig)


def test_publication_finalizer_removes_titles_legends_and_sparsifies_ticks():
    with plt.rc_context(_style._PLOT_RC):
        fig, ax = plt.subplots(figsize=(_style.PANEL_W, _style.PANEL_H))
        ax.plot([0.0, 1.0], [0.0, 1.0], label="PTZ pooled")
        ax.set_title("earliest\nelectrographic change")
        ax.legend()
        fig.suptitle("Summary")

        _style.finalize_publication_figure(fig)

        assert ax.get_title() == ""
        assert fig._suptitle.get_text() == ""
        assert ax.get_legend() is None
        assert ax.xaxis.get_major_locator().__class__.__name__ == "MaxNLocator"
        assert ax.yaxis.get_major_locator().__class__.__name__ == "MaxNLocator"
        plt.close(fig)


def test_standalone_scatter_suppresses_title_and_legend(tmp_path, monkeypatch):
    path = tmp_path / "standalone.png"
    checked = False
    original_savefig = Figure.savefig

    def check_figure(figure, *args, **kwargs):
        nonlocal checked
        assert figure.axes[0].get_title() == ""
        assert figure.axes[0].get_legend() is None
        assert figure.get_size_inches()[0] == 12.0
        checked = True
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_figure)
    _save_standalone_scatter(
        {"Seizure": ([1.0, 2.0], 1.5, 0.5, [("m1", 1), ("m2", 1)])},
        "AUC",
        str(path),
        title="AUC - seizure onset",
        scatter_options={"standalone_width": 12.0},
        zero_line=True,
    )
    assert checked
    assert path.exists()
    assert path.stat().st_size > 0


def test_ictal_transient_summary_labels_and_legends_do_not_overlap(
    tmp_path, monkeypatch,
):
    time = np.linspace(-45.0, 45.0, 19)
    bins = np.linspace(-55.0, 55.0, 12)
    result = IctalTransientGroupResult(
        session_results=[],
        psth_bin_centers=bins,
        psth_mean=np.ones_like(bins),
        psth_sem=np.full_like(bins, 0.2),
        per_session_psth_freq=[],
        moving_avg_times=time,
        freq_mean=np.ones_like(time),
        freq_sem=np.full_like(time, 0.1),
        amp_mean=np.full_like(time, 4.0),
        amp_sem=np.ones_like(time),
        hw_mean=np.full_like(time, 0.5),
        hw_sem=np.full_like(time, 0.1),
    )
    checked = False
    original_savefig = Figure.savefig

    def check_summary(figure, path, *args, **kwargs):
        nonlocal checked
        if str(path).endswith("summary_ictal_transients.png"):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            artists = []
            for axis in figure.axes:
                artists.extend([axis.xaxis.label, axis.yaxis.label])
                assert axis.get_title() == ""
                assert axis.get_legend() is None
            boxes = [artist.get_window_extent(renderer) for artist in artists]
            assert all(
                not first.overlaps(second)
                for index, first in enumerate(boxes)
                for second in boxes[index + 1:]
            )
            checked = True
        return original_savefig(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_summary)
    plot_ictal_transients({"PTZ pooled": result}, str(tmp_path))
    assert checked


def test_parameter_detection_callouts_clear_legend(tmp_path, monkeypatch):
    time = np.arange(0.0, 121.0, 0.01)
    dff = 2.0 + 0.2 * np.sin(time)
    post_onset = time >= 80.0
    dff[post_onset] += (time[post_onset] - 80.0) * 2.5
    dff[(time >= 80.0) & (time < 80.2)] = -8.0
    session = Session(
        mouse_id="probe",
        genotype="Scn1a",
        heating_session=1,
        n_seizures=1,
        cohort="DS_sz",
        landmarks=SessionLandmarks(
            heating_start_time=20.0,
            ueo_time=80.0,
            off_time=115.0,
        ),
        processed=ProcessedData(
            photometry=PhotometryResult(
                dff=dff,
                dff_zscore=dff,
                dff_hpf=dff,
            ),
            ecog_filtered=np.zeros_like(dff),
            temperature_c=np.zeros_like(dff),
            temperature_smooth=np.zeros_like(dff),
            time=time,
            fs=100.0,
        ),
    )
    checked = False
    original_savefig = Figure.savefig

    def check_diagnostic(figure, path, *args, **kwargs):
        nonlocal checked
        if str(path).endswith("_parameter_detection.png"):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            zoom_axis = figure.axes[1]
            legend_box = zoom_axis.get_legend().get_window_extent(renderer)
            callout_boxes = [
                annotation.get_window_extent(renderer)
                for annotation in zoom_axis.texts
            ]
            assert len(callout_boxes) == 2
            assert all(
                "200 ms sustained" in annotation.get_text()
                for annotation in zoom_axis.texts
            )
            assert all(not box.overlaps(legend_box) for box in callout_boxes)
            assert not callout_boxes[0].overlaps(callout_boxes[1])
            checked = True
        return original_savefig(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_diagnostic)
    plot_preictal_mean_diagnostic(
        session,
        str(tmp_path),
        extrema_min_duration_s=0.200,
    )
    assert checked


def test_baseline_plot_uses_explicit_window_and_autoscales_y_axes(
    tmp_path, monkeypatch,
):
    time = np.arange(0.0, 12.0, 0.01)
    dff = 10.0 * np.sin(time)
    ecog = 4000.0 * np.sin(2.0 * time)
    session = Session(
        mouse_id="m1",
        genotype="WT",
        heating_session=1,
        landmarks=SessionLandmarks(
            heating_start_time=10.0,
            baseline_window_s=(2.0, 8.0),
        ),
        processed=ProcessedData(
            photometry=PhotometryResult(
                dff=dff,
                dff_zscore=dff,
                dff_hpf=dff,
            ),
            ecog_filtered=ecog,
            time=time,
            fs=100.0,
        ),
    )

    saved_ylims = []
    original_savefig = Figure.savefig

    def capture_ylims(figure, *args, **kwargs):
        saved_ylims.extend(axis.get_ylim() for axis in figure.axes)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_ylims)
    png_path = plot_baseline(session, str(tmp_path), color="black")

    assert png_path is not None
    assert saved_ylims[0][0] < dff[(time >= 2.0) & (time <= 8.0)].min()
    assert saved_ylims[0][1] > dff[(time >= 2.0) & (time <= 8.0)].max()
    assert saved_ylims[1][0] < ecog[(time >= 2.0) & (time <= 8.0)].min()
    assert saved_ylims[1][1] > ecog[(time >= 2.0) & (time <= 8.0)].max()
    assert (tmp_path / "m1_S1_WT_baseline.png").exists()
    html_path = tmp_path / "m1_S1_WT_baseline.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert '"range":[-5.0,5.0]' not in html
    assert '"range":[-3000.0,3000.0]' not in html
    assert '"range":[2.0,8.0]' in html


def _aligned_session(mouse_id: str, n_seizures: int) -> Session:
    time = np.arange(0.0, 20.0, 0.1)
    landmarks = (
        SessionLandmarks(ueo_time=10.0)
        if n_seizures
        else SessionLandmarks(equiv_ueo_time=10.0)
    )
    signal = np.sin(time)
    return Session(
        mouse_id=mouse_id,
        genotype="Scn1a",
        heating_session=1,
        n_seizures=n_seizures,
        landmarks=landmarks,
        processed=ProcessedData(
            photometry=PhotometryResult(
                dff=signal,
                dff_zscore=signal,
                dff_hpf=signal,
            ),
            ecog_filtered=signal,
            time=time,
            fs=10.0,
        ),
    )


def test_zoomed_trace_uses_seizure_or_event_onset_label(
    tmp_path, monkeypatch,
):
    saved_xlabels = {}
    original_savefig = Figure.savefig

    def capture_xlabel(figure, path, *args, **kwargs):
        saved_xlabels[str(path)] = figure.axes[-1].get_xlabel()
        return original_savefig(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_xlabel)
    plot_zoomed(_aligned_session("seizure", 1), str(tmp_path), window_s=5.0)
    plot_zoomed(_aligned_session("control", 0), str(tmp_path), window_s=5.0)

    assert any(
        label == "Time rel. to seizure onset (s)"
        for label in saved_xlabels.values()
    )
    assert any(
        label == "Time rel. to event onset (s)"
        for label in saved_xlabels.values()
    )
    seizure_html = next(tmp_path.glob("seizure_*_peri_ictal.html"))
    control_html = next(tmp_path.glob("control_*_peri_ictal.html"))
    assert "Time rel. to seizure onset (s)" in seizure_html.read_text("utf-8")
    assert "Time rel. to event onset (s)" in control_html.read_text("utf-8")


def _triggered_average() -> TriggeredAverage:
    time = np.linspace(-5.0, 5.0, 11)
    signal = np.linspace(-1.0, 1.0, 11)
    return TriggeredAverage(
        time_axis=time,
        mean_trace=signal,
        sem_trace=np.full_like(signal, 0.1),
        auc=1.0,
        per_session_auc=[1.0],
        per_session_traces=[signal],
        auc_window_s=5.0,
        per_session_meta=[("m1", 1)],
    )


def test_group_traces_use_event_label_for_no_seizure_cohorts(
    tmp_path, monkeypatch,
):
    captured = []

    def capture_labels(figure, path, *args, **kwargs):
        if str(path).endswith("summary_UEO_aligned_traces.png"):
            assert all(axis.get_title() == "" for axis in figure.axes)
            assert all(axis.get_legend() is None for axis in figure.axes)
            captured.extend(axis.get_xlabel() for axis in figure.axes)

    monkeypatch.setattr(Figure, "savefig", capture_labels)
    plot_ueo_per_cohort(
        {"DS_sz": _triggered_average(), "DS_no_sz": _triggered_average()},
        str(tmp_path),
    )

    assert captured == [
        "Time rel. to\nseizure onset (s)",
        "Time rel. to\nevent onset (s)",
    ]


def test_group_traces_honor_explicit_shared_y_ticks(tmp_path, monkeypatch):
    captured = []

    def capture_ticks(figure, path, *args, **kwargs):
        captured.extend(
            tuple(axis.get_yticks())
            for axis in figure.axes
            if axis.axison
        )

    monkeypatch.setattr(Figure, "savefig", capture_ticks)
    expected = (-0.1, 0.0, 0.1, 0.2, 0.3, 0.4)
    plot_ueo_per_cohort(
        {"10HZ_DG": _triggered_average(), "10HZ_DG_CPP": _triggered_average()},
        str(tmp_path),
        y_lim_override=(-0.1, 0.4),
        y_ticks_override=expected,
    )

    assert captured
    assert all(ticks == expected for ticks in captured)


def test_heatmap_session_ticks_match_standard_plot_font_size(
    tmp_path, monkeypatch,
):
    captured_sizes = []
    captured_xlabels = []

    def capture_sizes(figure, path, *args, **kwargs):
        captured_sizes.extend(
            label.get_fontsize()
            for label in figure.axes[0].get_yticklabels()
        )
        captured_xlabels.append(figure.axes[0].get_xlabel())

    monkeypatch.setattr(Figure, "savefig", capture_sizes)
    plot_ueo_aligned_heatmaps(
        {"DS_no_sz": [_aligned_session("control", 0)]},
        str(tmp_path),
        pre_s=6.0,
        post_s=6.0,
    )

    assert captured_sizes
    assert set(captured_sizes) == {float(_style.TICK_SIZE)}
    assert set(captured_xlabels) == {"Time rel. to event onset (s)"}


def _baseline_group(prefix: str, values: list[float]) -> BaselineTransientGroupResult:
    rows = [
        BaselineTransientSessionResult(
            mouse_id=f"{prefix}{index}",
            heating_session=1,
            n_transients=1,
            duration_s=60.0,
            frequency_hz=value,
            mean_amplitude=value,
            mean_half_width_s=value,
        )
        for index, value in enumerate(values)
    ]
    return BaselineTransientGroupResult(
        session_results=rows,
        frequency_mean=float(np.mean(values)),
        frequency_sem=0.0,
        amplitude_mean=float(np.mean(values)),
        amplitude_sem=0.0,
        half_width_mean=float(np.mean(values)),
        half_width_sem=0.0,
    )


def test_baseline_transient_summary_ylabels_stay_inside_canvas(
    tmp_path, monkeypatch,
):
    checked = False
    original_savefig = Figure.savefig

    def check_summary(figure, path, *args, **kwargs):
        nonlocal checked
        if str(path).endswith("summary_baseline_transients.png"):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            canvas = figure.bbox
            boxes = [
                axis.yaxis.label.get_window_extent(renderer)
                for axis in figure.axes
            ]
            assert all(
                box.y0 >= canvas.y0 and box.y1 <= canvas.y1
                for box in boxes
            )
            checked = True
        return original_savefig(figure, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_summary)
    plot_baseline_transients(
        {
            "No seizure": _baseline_group("n", [1.0, 2.0, 3.0]),
            "Seizure": _baseline_group("s", [2.0, 3.0, 4.0]),
        },
        str(tmp_path),
    )
    assert checked


def _opto_group(prefix: str, values: list[float]) -> OptoResponseGroupResult:
    return OptoResponseGroupResult(session_results=[
        OptoResponseSessionResult(
            mouse_id=f"{prefix}{index}",
            heating_session=1,
            stim_response_max_40s=value,
            stim_response_min_40s=-value,
            stim_auc_40s=value,
        )
        for index, value in enumerate(values)
    ])


def test_auc_scatter_labels_include_signal_units(tmp_path, monkeypatch):
    labels = []
    original_scatter = group_plots._scatter_mean_sem

    def capture_scatter(ax, cohorts, ylabel, *args, **kwargs):
        labels.append(ylabel)
        return original_scatter(ax, cohorts, ylabel, *args, **kwargs)

    monkeypatch.setattr(group_plots, "_scatter_mean_sem", capture_scatter)
    monkeypatch.setattr(Figure, "savefig", lambda *_args, **_kwargs: None)

    session_result = IctalMeanSessionResult(
        mouse_id="m1",
        heating_session=1,
        seizure_mean=1.0,
        baseline_mean=0.0,
        delta_preictal_ictal=1.0,
        delta_preictal_ictal_max=2.0,
        delta_preictal_ictal_min=-1.0,
    )
    ictal_result = IctalMeanGroupResult(
        session_results=[session_result],
        seizure_mean=1.0,
        seizure_sem=0.0,
        baseline_mean=0.0,
        baseline_sem=0.0,
        delta_mean=1.0,
        delta_sem=0.0,
        triggered_averages={"UEO": _triggered_average()},
        delta_max_mean=2.0,
        delta_max_sem=0.0,
        delta_min_mean=-1.0,
        delta_min_sem=0.0,
    )
    plot_ictal_mean(
        {"DS_sz": ictal_result}, str(tmp_path), dff_label="ΔF/F",
    )
    plot_opto_response(
        {"10HZ_DG": _opto_group("m", [1.0, 2.0])},
        str(tmp_path),
        dff_label="ΔF/F",
    )

    assert "AUC\n(ΔF/F × s)" in labels
    assert "Stim AUC\n(ΔF/F × s)" in labels


def test_variable_auc_windows_are_explicit_in_values_csv(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Figure, "savefig", lambda *_args, **_kwargs: None)
    triggered = _triggered_average()
    triggered.per_session_auc = [10.0, 20.0]
    triggered.per_session_traces = [
        triggered.mean_trace.copy(), triggered.mean_trace.copy(),
    ]
    triggered.per_session_meta = [("m1", 1), ("m2", 1)]
    triggered.per_session_auc_window_s = [10.0, 20.0]

    session_result = IctalMeanSessionResult(
        mouse_id="m1",
        heating_session=1,
        seizure_mean=1.0,
        baseline_mean=0.0,
        delta_preictal_ictal=1.0,
        delta_preictal_ictal_max=2.0,
        delta_preictal_ictal_min=-1.0,
    )
    result = IctalMeanGroupResult(
        session_results=[session_result],
        seizure_mean=1.0,
        seizure_sem=0.0,
        baseline_mean=0.0,
        baseline_sem=0.0,
        delta_mean=1.0,
        delta_sem=0.0,
        triggered_averages={"UEO": triggered},
        delta_max_mean=2.0,
        delta_max_sem=0.0,
        delta_min_mean=-1.0,
        delta_min_sem=0.0,
    )

    plot_ictal_mean({"Seizure": result}, str(tmp_path), name_prefix="DG")

    values = pd.read_csv(tmp_path / "values_auc_DG.csv")
    assert "UEO AUC (variable window)" in values.columns
    assert values["UEO AUC window (s)"].tolist() == [10.0, 20.0]


def test_positive_only_auc_is_explicit_in_plot_and_values_labels(
    tmp_path, monkeypatch,
):
    labels = []
    original_scatter = group_plots._scatter_mean_sem

    def capture_scatter(ax, cohorts, ylabel, *args, **kwargs):
        labels.append(ylabel)
        return original_scatter(ax, cohorts, ylabel, *args, **kwargs)

    monkeypatch.setattr(group_plots, "_scatter_mean_sem", capture_scatter)
    monkeypatch.setattr(Figure, "savefig", lambda *_args, **_kwargs: None)
    triggered = _triggered_average()
    triggered.auc_positive_only = True
    result = IctalMeanGroupResult(
        session_results=[],
        seizure_mean=0.0,
        seizure_sem=0.0,
        baseline_mean=0.0,
        baseline_sem=0.0,
        delta_mean=0.0,
        delta_sem=0.0,
        triggered_averages={"UEO": triggered},
    )

    plot_ictal_mean({"Seizure": result}, str(tmp_path), name_prefix="DG")

    assert "AUC above 0\n(z_ΔF/F × s)" in labels
    values = pd.read_csv(tmp_path / "values_auc_DG.csv")
    assert "UEO positive-only AUC (5s)" in values.columns


def test_statistics_support_different_comparison_groups_per_module(tmp_path):
    cohort_results = {
        "Seizure": {"Opto response": _opto_group("s", [4.0, 5.0, 6.0])},
        "No seizure": {"Opto response": _opto_group("n", [1.0, 2.0, 3.0])},
        "DS": {"Baseline transients": _baseline_group("d", [4.0, 5.0, 6.0])},
        "WT": {"Baseline transients": _baseline_group("w", [1.0, 2.0, 3.0])},
    }
    means_dir = tmp_path / "means"
    transients_dir = tmp_path / "transients"

    paths = run_group_comparisons(
        cohort_results,
        means_dir=means_dir,
        transients_dir=transients_dir,
        name_prefix="groups",
        config=StatsConfig(cohorts_to_compare=["Seizure", "No seizure"]),
        module_configs={
            "Baseline transients": StatsConfig(
                cohorts_to_compare=["DS", "WT"],
            ),
        },
    )

    means = pd.read_csv(paths["means"])
    transients = pd.read_csv(paths["transients"])
    assert {"Seizure_n", "No seizure_n"}.issubset(means.columns)
    assert {"DS_n", "WT_n"}.issubset(transients.columns)
    assert set(means["Module"]) == {"Opto response"}
    assert set(transients["Module"]) == {"Baseline transients"}


def test_hyperthermia_runner_classifiers_keep_pooling_dataset_specific():
    ds_seizure = Session("ds1", "Scn1a", n_seizures=1)
    ds_control = Session("ds2", "Scn1a", n_seizures=0)
    wt_control = Session("wt1", "WT", n_seizures=0)
    sessions = [ds_seizure, ds_control, wt_control]

    assert [_seizure_outcome_classifier(s) for s in sessions] == [
        "Seizure", "No seizure", "No seizure",
    ]
    assert [_genotype_classifier(s) for s in sessions] == ["DS", "DS", "WT"]
    assert _source_point_colors(sessions) == {
        ("Seizure", ("ds1", 1)): "red",
        ("No seizure", ("ds2", 1)): "purple",
        ("No seizure", ("wt1", 1)): "blue",
    }
    assert _genotype_point_colors(sessions) == {
        ("DS", ("ds1", 1)): "red",
        ("DS", ("ds2", 1)): "red",
        ("WT", ("wt1", 1)): "blue",
    }
    assert ICTAL_EXTREMA_MIN_DURATION_S == {
        "DG_GRABne": 0.200,
        "LC_GCaMP": 0.200,
    }
    assert INDIVIDUAL_SEIZURE_AUC_SUBFOLDERS == {"DG_GRABne"}
    assert DG_POSITIVE_ONLY_AUC_LANDMARKS == {"EEC", "UEO"}
    assert LC_AUC_WINDOWS_S == {
        "EEC": 100.0,
        "UEO": 100.0,
        "behavioral_onset": 100.0,
        "max_temp": 100.0,
    }


def test_group_runner_keeps_scalar_and_trace_groupings_separate(
    tmp_path, monkeypatch,
):
    sessions = [
        Session(
            "ds1", "Scn1a", n_seizures=1,
            landmarks=SessionLandmarks(ueo_time=0.0, off_time=10.0),
        ),
        Session(
            "wt1", "WT", n_seizures=1,
            landmarks=SessionLandmarks(ueo_time=0.0, off_time=10.0),
        ),
    ]
    captures = {}
    ictal_extrema_durations = []
    ictal_auc_windows = []
    ictal_positive_landmarks = []

    def generic_compute(group_sessions, *_args, **_kwargs):
        rows = [
            SimpleNamespace(
                mouse_id=session.mouse_id,
                heating_session=session.heating_session,
            )
            for session in group_sessions
        ]
        return SimpleNamespace(session_results=rows)

    def recorder(name):
        def record(results, _output_dir, **kwargs):
            captures[name] = {
                "scalar": set(results),
                "trace": set(kwargs.get("trace_results") or {}),
            }
            return ""
        return record

    monkeypatch.setattr(sweep, "_ensure_spikes", lambda _session: None)
    monkeypatch.setattr(sweep, "mean_seizure_duration", lambda _sessions: 10.0)
    monkeypatch.setattr(
        sweep,
        "compute_wide_ueo_triggered",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(sweep, "plot_ueo_per_cohort", lambda *_a, **_k: "")
    monkeypatch.setattr(sweep, "plot_ueo_aligned_heatmaps", lambda *_a, **_k: {})
    monkeypatch.setattr(
        sweep, "plot_experimental_vs_isosbestic_spike_triggered",
        lambda *_a, **_k: "",
    )
    monkeypatch.setattr(sweep, "write_values_means", lambda *_a, **_k: None)

    compute_names = (
        "compute_cohort_characteristics",
        "compute_baseline_transients",
        "compute_preictal_mean",
        "compute_preictal_transients",
        "compute_ictal_mean",
        "compute_ictal_transients",
        "compute_postictal",
        "compute_opto_response",
        "compute_spike_triggered_average",
    )
    for name in compute_names:
        monkeypatch.setattr(sweep, name, generic_compute)

    def compute_ictal(group_sessions, *_args, **kwargs):
        ictal_extrema_durations.append(kwargs["extrema_min_duration_s"])
        ictal_auc_windows.append(kwargs["auc_end_s_by_session"])
        ictal_positive_landmarks.append(kwargs["positive_auc_landmarks"])
        return generic_compute(group_sessions)

    monkeypatch.setattr(sweep, "compute_ictal_mean", compute_ictal)

    plot_names = {
        "plot_cohort_characteristics": "Cohort characteristics",
        "plot_baseline_transients": "Baseline transients",
        "plot_preictal_mean": "Pre-ictal mean",
        "plot_preictal_transients": "Pre-ictal transients",
        "plot_ictal_mean": "Ictal mean",
        "plot_ictal_transients": "Ictal transients",
        "plot_postictal": "Postictal",
        "plot_spike_triggered": "Spike-triggered averages",
        "plot_opto_response": "Opto response",
    }
    for function_name, module_name in plot_names.items():
        monkeypatch.setattr(sweep, function_name, recorder(module_name))

    result = sweep.run_group_analysis(
        sessions,
        strategy="C",
        ds_key="test",
        results_base=tmp_path,
        cohort_classifier=_seizure_outcome_classifier,
        trace_cohort_classifier=sweep.classify_session,
        module_cohort_classifiers={
            "Baseline transients": _genotype_classifier,
        },
        ictal_auc_use_individual_seizure_durations=True,
        ictal_positive_auc_landmarks={"EEC", "UEO"},
        ictal_extrema_min_duration_s=0.200,
    )

    assert captures["Baseline transients"] == {
        "scalar": {"DS", "WT"},
        "trace": set(),
    }
    assert captures["Ictal mean"] == {
        "scalar": {"Seizure"},
        "trace": {"DS_sz", "WT_sz"},
    }
    assert set(result) == {"Seizure", "DS", "WT"}
    assert ictal_extrema_durations == [0.200, 0.200, 0.200]
    assert ictal_auc_windows == [
        {("ds1", None, 1): 10.0, ("wt1", None, 1): 10.0},
    ] * 3
    assert ictal_positive_landmarks == [{"EEC", "UEO"}] * 3
