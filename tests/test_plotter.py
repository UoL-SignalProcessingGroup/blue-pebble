"""Tests for plotting helpers and figure construction."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_fake_stonesoup_plotter_modules,
    install_repo_package,
    load_package_module_from_repo,
)


def _load_plotter(monkeypatch):
    """Load ``plotter.py`` with minimal package scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_fake_stonesoup_plotter_modules(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    load_package_module_from_repo(
        "bluepebble/detector/metrics.py",
        "bluepebble.detector.metrics",
    )
    return load_package_module_from_repo("bluepebble/plotter.py", "bluepebble.plotter")


def test_distance_axis_scale_switches_to_kilometres(monkeypatch) -> None:
    """Large coordinate magnitudes should be displayed in kilometres."""
    plotter = _load_plotter(monkeypatch)

    assert plotter._distance_axis_scale(-500.0, 500.0) == (1.0, "m")
    assert plotter._distance_axis_scale(-1500.0, 500.0) == (1e-3, "km")


def test_range_padding_applies_floor_in_display_units(monkeypatch) -> None:
    """Padding should use a 0.5-unit floor in display coordinates."""
    plotter = _load_plotter(monkeypatch)

    assert plotter._range_padding_for_scale(scale=1.0, span=2.0) == pytest.approx(0.5)
    assert plotter._range_padding_for_scale(scale=1e-3, span=20_000.0) == pytest.approx(1000.0)


def test_expand_heatmap_coords_only_expands_outer_cells(monkeypatch) -> None:
    """Heatmap coordinates should be expanded by half a cell at each outer edge."""
    plotter = _load_plotter(monkeypatch)

    expanded = plotter._expand_heatmap_coords(np.array([10.0, 20.0, 30.0]))

    np.testing.assert_allclose(expanded, np.array([5.0, 20.0, 35.0]))


def test_validate_btr_shapes_rejects_shape_mismatches(monkeypatch) -> None:
    """BTR validation should reject incompatible time-bearing-data combinations."""
    plotter = _load_plotter(monkeypatch)

    with pytest.raises(ValueError, match="data shape mismatch"):
        plotter._validate_btr_shapes(
            timesteps=np.array([1, 2]),
            steering_azimuths=np.array([0.0, 1.0, 2.0]),
            data=np.zeros((2, 2)),
        )


def test_validate_btr_shapes_rejects_non_numeric_bearings(monkeypatch) -> None:
    """Steering azimuths must be coercible to floats."""
    plotter = _load_plotter(monkeypatch)

    with pytest.raises(ValueError, match="must contain numeric values"):
        plotter._validate_btr_shapes(
            timesteps=np.array([1, 2]),
            steering_azimuths=np.array(["east", "west"], dtype=object),
            data=None,
        )


def test_validate_spectrogram_params_normalises_and_validates_inputs(monkeypatch) -> None:
    """Spectrogram parameter validation should normalise case and check limits."""
    plotter = _load_plotter(monkeypatch)

    fmt, y_lim = plotter._validate_spectrogram_params(
        sr=48_000,
        n_fft=1024,
        hop_length=256,
        y_lim=(100.0, 2000.0),
        yaxis_format=" khZ ",
    )

    assert fmt == "kHz"
    assert y_lim == (100.0, 2000.0)

    with pytest.raises(ValueError, match="hop_length must be less than or equal to n_fft"):
        plotter._validate_spectrogram_params(
            sr=48_000,
            n_fft=128,
            hop_length=256,
            y_lim=None,
            yaxis_format="Hz",
        )

    with pytest.raises(ValueError, match="y_lim must satisfy low < high"):
        plotter._validate_spectrogram_params(
            sr=48_000,
            n_fft=256,
            hop_length=128,
            y_lim=(5.0, 5.0),
            yaxis_format="Hz",
        )


def test_validate_spectrogram_render_params_checks_modes_and_limits(monkeypatch) -> None:
    """Spectrogram rendering validation should check mode/reference and z limits."""
    plotter = _load_plotter(monkeypatch)

    mode, reference, z_lim = plotter._validate_spectrogram_render_params(
        analysis_mode=" PSD ",
        db_reference=" absolute ",
        z_lim=(-50.0, 20.0),
    )

    assert mode == "psd"
    assert reference == "absolute"
    assert z_lim == (-50.0, 20.0)

    with pytest.raises(ValueError, match="analysis_mode must be one of"):
        plotter._validate_spectrogram_render_params(
            analysis_mode="wavelet",
            db_reference="peak",
            z_lim=None,
        )

    with pytest.raises(ValueError, match="db_reference must be one of"):
        plotter._validate_spectrogram_render_params(
            analysis_mode="stft",
            db_reference="linear",
            z_lim=None,
        )

    with pytest.raises(ValueError, match="z_lim must satisfy low < high"):
        plotter._validate_spectrogram_render_params(
            analysis_mode="stft",
            db_reference="peak",
            z_lim=(1.0, 1.0),
        )


def test_normalise_plotly_figsize_handles_inches_and_pixels(monkeypatch) -> None:
    """Small figure sizes should be treated as inches and larger ones as pixels."""
    plotter = _load_plotter(monkeypatch)

    assert plotter._normalise_plotly_figsize((12, 6)) == (1200, 600)
    assert plotter._normalise_plotly_figsize((640, 480)) == (640, 480)

    with pytest.raises(ValueError, match="figsize values must be positive"):
        plotter._normalise_plotly_figsize((0, 10))


def test_plot_spectrogram_builds_expected_axes(monkeypatch) -> None:
    """Spectrogram plotting should produce a heatmap with the requested axis labelling."""
    plotter = _load_plotter(monkeypatch)
    signal = np.sin(2.0 * np.pi * 440.0 * np.arange(4096) / 48_000.0)

    fig = plotter.plot_spectrogram(
        signal=signal,
        sr=48_000,
        n_fft=512,
        hop_length=128,
        y_lim=(0.0, 4000.0),
        yaxis_format="Hz",
        figsize=(8, 4),
    )

    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    assert fig.layout.xaxis.title.text == "Time (s)"
    assert fig.layout.yaxis.title.text == "Frequency (Hz)"
    assert fig.layout.width == 800
    assert fig.layout.height == 400


def test_plot_spectrogram_rejects_empty_signal(monkeypatch) -> None:
    """Empty signals should be rejected before attempting an STFT."""
    plotter = _load_plotter(monkeypatch)

    with pytest.raises(ValueError, match="signal is empty"):
        plotter.plot_spectrogram(signal=np.array([]), sr=48_000)


def test_plot_spectrogram_validates_row_and_col_arguments(monkeypatch) -> None:
    """Spectrogram plotting should reject inconsistent subplot targeting arguments."""
    plotter = _load_plotter(monkeypatch)
    signal = np.ones(1024)

    with pytest.raises(ValueError, match="can only be used when fig is supplied"):
        plotter.plot_spectrogram(signal=signal, sr=48_000, row=1)

    with pytest.raises(ValueError, match="must both be provided when fig is supplied"):
        plotter.plot_spectrogram(
            signal=signal,
            sr=48_000,
            fig=plotter.go.Figure(),
            row=1,
        )

    with pytest.raises(ValueError, match="row and col must be positive"):
        plotter.plot_spectrogram(
            signal=signal,
            sr=48_000,
            fig=plotter.go.Figure(),
            row=0,
            col=1,
        )


def test_plot_spectrogram_writes_into_subplot_cell(monkeypatch) -> None:
    """Spectrogram plotting should be able to draw into a provided subplot cell."""
    plotter = _load_plotter(monkeypatch)
    signal = np.sin(2.0 * np.pi * 220.0 * np.arange(2048) / 48_000.0)
    fig = plotter.make_subplots(rows=1, cols=2)

    returned_fig = plotter.plot_spectrogram(
        signal=signal,
        sr=48_000,
        n_fft=256,
        hop_length=64,
        yaxis_format="Hz",
        fig=fig,
        row=1,
        col=2,
    )

    assert returned_fig is fig
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    assert fig.layout.xaxis2.title.text == "Time (s)"
    assert fig.layout.yaxis2.title.text == "Frequency (Hz)"


def test_plot_spectrogram_supports_psd_absolute_and_colourbar_controls(monkeypatch) -> None:
    """PSD mode should support absolute dB plotting with explicit z-limits."""
    plotter = _load_plotter(monkeypatch)
    signal = np.sin(2.0 * np.pi * 440.0 * np.arange(4096) / 48_000.0)

    fig = plotter.plot_spectrogram(
        signal=signal,
        sr=48_000,
        n_fft=512,
        hop_length=128,
        yaxis_format="Hz",
        analysis_mode="psd",
        db_reference="absolute",
        z_lim=(-50.0, 20.0),
        showscale=False,
        colorbar_title="dB re 1 uPa^2/Hz",
    )

    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    assert fig.data[0].zmin == -50.0
    assert fig.data[0].zmax == 20.0
    assert fig.data[0].showscale is False


def test_plot_roc_and_pr_include_auc_in_trace_names(monkeypatch) -> None:
    """ROC and PR plots should label traces with their corresponding AUC values."""
    plotter = _load_plotter(monkeypatch)
    result = SimpleNamespace(
        label="Detector A",
        fpr=np.array([1.0, 0.0, 0.5]),
        tpr=np.array([1.0, 0.0, 0.75]),
        recall=np.array([1.0, 0.25, 0.75]),
        precision=np.array([0.5, 1.0, 0.6]),
        auc_roc=0.625,
        auc_pr=0.587,
    )

    roc_fig = plotter.plot_roc([result], show_diagonal=True, figsize=(600, 500))
    pr_fig = plotter.plot_pr([result], figsize=(600, 500))

    assert roc_fig.data[0].name == "Detector A (AUC=0.625)"
    assert roc_fig.data[1].name == "Random"
    np.testing.assert_allclose(roc_fig.data[0].x, np.array([0.0, 0.5, 1.0]))
    assert pr_fig.data[0].name == "Detector A (AUC=0.587)"
    np.testing.assert_allclose(pr_fig.data[0].x, np.array([0.25, 0.75, 1.0]))


def test_plot_roc_pr_builds_side_by_side_subplots(monkeypatch) -> None:
    """Combined ROC/PR plotting should reuse legend groups and optional diagonal."""
    plotter = _load_plotter(monkeypatch)
    result = SimpleNamespace(
        label="Detector B",
        fpr=np.array([0.5, 0.0]),
        tpr=np.array([0.75, 0.0]),
        recall=np.array([1.0, 0.25]),
        precision=np.array([0.5, 1.0]),
        auc_roc=0.5,
        auc_pr=0.375,
    )

    fig = plotter.plot_roc_pr([result], show_diagonal=True, figsize=(1100, 500))

    assert len(fig.data) == 3
    assert fig.data[0].legendgroup == "Detector B"
    assert fig.data[1].legendgroup == "Detector B"
    assert fig.data[1].showlegend is False
    assert fig.data[2].name == "Random"
    assert fig.layout.xaxis.title.text == "False Positive Rate"
    assert fig.layout.xaxis2.title.text == "Recall"


def test_plot_btr_validates_row_and_col_arguments(monkeypatch) -> None:
    """BTR plotting should reject inconsistent subplot targeting arguments."""
    plotter = _load_plotter(monkeypatch)
    timesteps = np.array([datetime(2026, 1, 1, 12, 0, 0)])
    steering = np.array([0.0])
    data = np.zeros((1, 1))

    with pytest.raises(ValueError, match="can only be used when fig is supplied"):
        plotter.plot_btr(
            timesteps=timesteps,
            steering_azimuths=steering,
            data=data,
            row=1,
        )

    with pytest.raises(ValueError, match="must both be provided when fig is supplied"):
        plotter.plot_btr(
            timesteps=timesteps,
            steering_azimuths=steering,
            data=data,
            fig=plotter.go.Figure(),
            row=1,
        )

    with pytest.raises(ValueError, match="row and col must be positive"):
        plotter.plot_btr(
            timesteps=timesteps,
            steering_azimuths=steering,
            data=data,
            fig=plotter.go.Figure(),
            row=0,
            col=1,
        )


def test_plot_world_uses_marker_for_stationary_platform(monkeypatch) -> None:
    """A stationary platform should render as a marker rather than a degenerate line."""
    plotter = _load_plotter(monkeypatch)
    stationary_state = SimpleNamespace(state_vector=np.array([0.0, 0.0, 0.0]))
    platform = SimpleNamespace(states=[stationary_state, stationary_state])
    truth_state = SimpleNamespace(state_vector=np.array([10.0, 0.0, 5.0]))
    truths = [[truth_state, truth_state]]

    fig = plotter.plot_world(truths=truths, platform=platform)

    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "markers"
    assert fig.data[0].name == "Platform"


def test_plot_world_hovertemplate_omits_native_metres_when_scale_is_metres(
    monkeypatch,
) -> None:
    """Hover template should not repeat native-metre fields when display unit is already metres."""
    plotter = _load_plotter(monkeypatch)
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    state_a = SimpleNamespace(state_vector=np.array([0.0, 0.0, 0.0]), timestamp=t0)
    state_b = SimpleNamespace(state_vector=np.array([5.0, 0.0, 3.0]), timestamp=t0)
    platform = SimpleNamespace(states=[state_a, state_b])
    truth_state = SimpleNamespace(state_vector=np.array([10.0, 0.0, 5.0]), timestamp=t0)
    truths = [[truth_state, truth_state]]

    fig = plotter.plot_world(truths=truths, platform=platform)

    platform_trace = next(t for t in fig.data if t.name == "Platform")
    truth_trace = next(t for t in fig.data if t.name == "Truth")

    assert "customdata[1]" not in platform_trace.hovertemplate
    assert "customdata[2]" not in platform_trace.hovertemplate
    assert "customdata[0]" in platform_trace.hovertemplate
    assert "customdata[1]" not in truth_trace.hovertemplate
    assert "customdata[2]" not in truth_trace.hovertemplate
    assert "customdata[0]" in truth_trace.hovertemplate
    assert platform_trace.customdata.shape[1] == 3
    assert truth_trace.customdata.shape[1] == 3


def test_plot_world_hovertemplate_uses_km_unit_when_scale_is_km(
    monkeypatch,
) -> None:
    """Hover template should display coordinates in km when the scene exceeds 1 km."""
    plotter = _load_plotter(monkeypatch)
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    # Coordinates > 1000 m trigger km display scale.
    state_a = SimpleNamespace(state_vector=np.array([0.0, 0.0, 0.0]), timestamp=t0)
    state_b = SimpleNamespace(state_vector=np.array([2000.0, 0.0, 1500.0]), timestamp=t0)
    platform = SimpleNamespace(states=[state_a, state_b])
    truth_state = SimpleNamespace(state_vector=np.array([3000.0, 0.0, 2000.0]), timestamp=t0)
    truths = [[truth_state, truth_state]]

    fig = plotter.plot_world(truths=truths, platform=platform)

    platform_trace = next(t for t in fig.data if t.name == "Platform")
    truth_trace = next(t for t in fig.data if t.name == "Truth")

    assert "km" in platform_trace.hovertemplate
    assert "customdata[0]" in platform_trace.hovertemplate
    assert "customdata[1]" not in platform_trace.hovertemplate
    assert "km" in truth_trace.hovertemplate
    assert "customdata[0]" in truth_trace.hovertemplate
    assert "customdata[1]" not in truth_trace.hovertemplate


def test_plot_world_adds_direction_arrows_for_moving_elements(monkeypatch) -> None:
    """Arrow marker traces should be added at the last position of each moving element."""
    plotter = _load_plotter(monkeypatch)
    # Platform moves due East: dx > 0, dy == 0 → plotly angle == 90°.
    state_a = SimpleNamespace(state_vector=np.array([0.0, 0.0, 0.0]))
    state_b = SimpleNamespace(state_vector=np.array([10.0, 0.0, 0.0]))
    platform = SimpleNamespace(states=[state_a, state_b])
    # Truth moves due North: dx == 0, dy > 0 → plotly angle == 0°.
    truth_a = SimpleNamespace(state_vector=np.array([50.0, 0.0, 0.0]))
    truth_b = SimpleNamespace(state_vector=np.array([50.0, 0.0, 10.0]))
    truths = [[truth_a, truth_b]]

    fig = plotter.plot_world(truths=truths, platform=platform)

    annotations = fig.layout.annotations
    assert len(annotations) == 2, "expected one direction annotation per moving element"

    # Platform moves due East: base at (10, 0), tip at x > 10, y == 0.
    platform_ann = next(a for a in annotations if a.arrowcolor == "black")
    assert float(platform_ann.ax) == pytest.approx(10.0)
    assert float(platform_ann.ay) == pytest.approx(0.0)
    assert float(platform_ann.x) > 10.0
    assert float(platform_ann.y) == pytest.approx(0.0)

    # Truth moves due North: base at (50, 10), tip at x == 50, y > 10.
    truth_ann = next(a for a in annotations if a.arrowcolor != "black")
    assert float(truth_ann.ax) == pytest.approx(50.0)
    assert float(truth_ann.ay) == pytest.approx(10.0)
    assert float(truth_ann.x) == pytest.approx(50.0)
    assert float(truth_ann.y) > 10.0


def test_plot_world_rejects_empty_platform_history(monkeypatch) -> None:
    """World plotting should fail clearly when no platform states are available."""
    plotter = _load_plotter(monkeypatch)
    platform = SimpleNamespace(states=[])

    with pytest.raises(ValueError, match="platform has no recorded states"):
        plotter.plot_world(truths=[], platform=platform)


def test_plot_world_rejects_bathymetry_without_get_grid(monkeypatch) -> None:
    """Bathymetry overlays must provide the expected gridding method."""
    plotter = _load_plotter(monkeypatch)
    stationary_state = SimpleNamespace(state_vector=np.array([0.0, 0.0, 0.0]))
    platform = SimpleNamespace(states=[stationary_state])
    truths = [[SimpleNamespace(state_vector=np.array([10.0, 0.0, 5.0]))]]

    with pytest.raises(ValueError, match="must provide get_grid"):
        plotter.plot_world(truths=truths, platform=platform, bathymetry=object())


def test_plot_btr_inserts_gap_for_wrapped_truth_line(monkeypatch) -> None:
    """Wrapped truth bearings should insert a ``None`` gap instead of drawing across the plot."""
    plotter = _load_plotter(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 12, 1, 0)
    truth = [
        SimpleNamespace(state_vector=np.array([np.deg2rad(179.0)]), timestamp=t0),
        SimpleNamespace(state_vector=np.array([np.deg2rad(-179.0)]), timestamp=t1),
    ]

    fig = plotter.plot_btr(
        timesteps=np.array([t0, t1]),
        steering_azimuths=np.array([-180.0, 180.0]),
        data=np.zeros((2, 2)),
        truths=[truth],
    )

    truth_trace = fig.data[-1]
    assert truth_trace.name == "Truth"
    assert list(truth_trace.x) == [179.0, None, -179.0]


def test_plot_btr_wraps_detection_bearings_and_splits_wrapped_track(monkeypatch) -> None:
    """Detection markers should wrap cleanly and wrapped tracks should insert a gap."""
    plotter = _load_plotter(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 12, 1, 0)
    detections = [
        SimpleNamespace(state_vector=np.array([np.deg2rad(181.0)]), timestamp=t0),
        SimpleNamespace(state_vector=np.array([np.deg2rad(-181.0)]), timestamp=t1),
    ]
    track = [
        SimpleNamespace(state_vector=np.array([np.deg2rad(179.0)]), timestamp=t0),
        SimpleNamespace(state_vector=np.array([np.deg2rad(-179.0)]), timestamp=t1),
    ]

    fig = plotter.plot_btr(
        timesteps=np.array([t0, t1]),
        steering_azimuths=np.array([-180.0, 180.0]),
        data=np.zeros((2, 2)),
        detections=detections,
        tracks=[track],
    )

    detection_trace = fig.data[1]
    track_trace = fig.data[2]
    assert detection_trace.name == "Detection"
    assert list(detection_trace.x) == [-179.0, 179.0]
    assert track_trace.name == "Track"
    assert list(track_trace.x) == [179.0, None, -179.0]


def test_plot_btr_groups_multiple_tracks_and_truths_in_legend(monkeypatch) -> None:
    """Multiple track and truth overlays should share legend groups with one group title."""
    plotter = _load_plotter(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 12, 1, 0)
    tracks = [
        [
            SimpleNamespace(state_vector=np.array([np.deg2rad(10.0)]), timestamp=t0),
            SimpleNamespace(state_vector=np.array([np.deg2rad(20.0)]), timestamp=t1),
        ],
        [
            SimpleNamespace(state_vector=np.array([np.deg2rad(-10.0)]), timestamp=t0),
            SimpleNamespace(state_vector=np.array([np.deg2rad(-20.0)]), timestamp=t1),
        ],
    ]
    truths = [
        [
            SimpleNamespace(state_vector=np.array([np.deg2rad(30.0)]), timestamp=t0),
            SimpleNamespace(state_vector=np.array([np.deg2rad(35.0)]), timestamp=t1),
        ],
        [
            SimpleNamespace(state_vector=np.array([np.deg2rad(-30.0)]), timestamp=t0),
            SimpleNamespace(state_vector=np.array([np.deg2rad(-35.0)]), timestamp=t1),
        ],
    ]

    fig = plotter.plot_btr(
        timesteps=np.array([t0, t1]),
        steering_azimuths=np.array([-180.0, 180.0]),
        data=np.zeros((2, 2)),
        tracks=tracks,
        truths=truths,
    )

    track_one = fig.data[1]
    track_two = fig.data[2]
    truth_one = fig.data[3]
    truth_two = fig.data[4]
    assert track_one.name == "Track 1"
    assert track_two.name == "Track 2"
    assert track_one.legendgroup == "tracks"
    assert track_two.legendgroup == "tracks"
    assert track_one.legendgrouptitle.text == "Tracks"
    assert truth_one.name == "Truth 1"
    assert truth_two.name == "Truth 2"
    assert truth_one.legendgroup == "truths"
    assert truth_two.legendgroup == "truths"
    assert truth_one.legendgrouptitle.text == "Ground Truths"


def test_plot_btr_accepts_custom_color_limits(monkeypatch) -> None:
    """BTR heatmaps should honor explicitly provided color scale bounds."""
    plotter = _load_plotter(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    t1 = datetime(2026, 1, 1, 12, 1, 0)

    fig = plotter.plot_btr(
        timesteps=np.array([t0, t1]),
        steering_azimuths=np.array([-180.0, 180.0]),
        data=np.array([[0.0, 5.0], [10.0, 15.0]]),
        cmin=-10.0,
        cmax=20.0,
    )

    heatmap = fig.data[0]
    assert heatmap.zmin == -10.0
    assert heatmap.zmax == 20.0


def test_plot_btr_rejects_invalid_color_limits(monkeypatch) -> None:
    """BTR color scale bounds should be finite and ordered when both are provided."""
    plotter = _load_plotter(monkeypatch)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    timesteps = np.array([t0])
    steering = np.array([0.0])
    data = np.array([[1.0]])

    with pytest.raises(ValueError, match="cmin must be less than cmax"):
        plotter.plot_btr(
            timesteps=timesteps,
            steering_azimuths=steering,
            data=data,
            cmin=1.0,
            cmax=1.0,
        )

    with pytest.raises(ValueError, match="cmax must be finite"):
        plotter.plot_btr(
            timesteps=timesteps,
            steering_azimuths=steering,
            data=data,
            cmax=np.inf,
        )
