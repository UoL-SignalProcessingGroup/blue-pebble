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
    platform = SimpleNamespace(
        platform_history=[
            SimpleNamespace(host=SimpleNamespace(state=stationary_state)),
            SimpleNamespace(host=SimpleNamespace(state=stationary_state)),
        ]
    )
    truth_state = SimpleNamespace(state_vector=np.array([10.0, 0.0, 5.0]))
    truths = [[truth_state, truth_state]]

    fig = plotter.plot_world(truths=truths, platform=platform)

    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "markers"
    assert fig.data[0].name == "Platform"


def test_plot_world_rejects_empty_platform_history(monkeypatch) -> None:
    """World plotting should fail clearly when no platform states are available."""
    plotter = _load_plotter(monkeypatch)
    platform = SimpleNamespace(platform_history=[])

    with pytest.raises(ValueError, match="platform.platform_history is empty"):
        plotter.plot_world(truths=[], platform=platform)


def test_plot_world_rejects_bathymetry_without_get_grid(monkeypatch) -> None:
    """Bathymetry overlays must provide the expected gridding method."""
    plotter = _load_plotter(monkeypatch)
    stationary_state = SimpleNamespace(state_vector=np.array([0.0, 0.0, 0.0]))
    platform = SimpleNamespace(
        platform_history=[SimpleNamespace(host=SimpleNamespace(state=stationary_state))]
    )
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
