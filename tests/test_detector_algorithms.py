"""Tests for detector algorithm primitives."""

from __future__ import annotations

import numpy as np
import pytest

from .support import install_fake_stonesoup, install_repo_package, load_package_module_from_repo


def _load_detector_algorithms(monkeypatch):
    """Load detector algorithms with minimal package scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    return load_package_module_from_repo(
        "bluepebble/detector/algorithms.py",
        "bluepebble.detector.algorithms",
    )


def test_threshold_detector_uses_strict_greater_than(monkeypatch) -> None:
    """Values equal to the threshold should not count as detections."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.ThresholdDetector(threshold=1.0)

    detections = detector.detect(np.array([0.5, 1.0, 1.5]))

    np.testing.assert_array_equal(detections, np.array([[2.0, 1.5]]))


def test_threshold_detector_returns_empty_array_when_nothing_detected(monkeypatch) -> None:
    """No threshold crossings should produce the standard empty detection shape."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.ThresholdDetector(threshold=10.0)

    detections = detector.detect(np.array([0.0, 1.0, 2.0]))

    assert detections.shape == (0, 2)


def test_peak_detector_respects_distance(monkeypatch) -> None:
    """Peaks separated by at least the configured distance should all survive."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.PeakDetector(distance=3)

    detections = detector.detect(np.array([0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0]))

    np.testing.assert_array_equal(
        detections,
        np.array([[1.0, 3.0], [4.0, 2.0], [7.0, 1.0]]),
    )


def test_peak_detector_returns_empty_array_when_no_local_maxima(monkeypatch) -> None:
    """Monotonic data should produce no peaks."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.PeakDetector(distance=1)

    detections = detector.detect(np.array([0.0, 1.0, 2.0, 3.0]))

    assert detections.shape == (0, 2)


def test_peak_detector_suppresses_smaller_nearby_peak(monkeypatch) -> None:
    """When peaks are too close together, only the taller one should remain."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.PeakDetector(distance=3)

    detections = detector.detect(np.array([0.0, 5.0, 0.0, 4.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[1.0, 5.0]]))


def test_cacfar_detector_detects_isolated_peak(monkeypatch) -> None:
    """CA-CFAR should detect a strong isolated cell above a flat background."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        threshold_factor=2.0,
    )

    detections = detector.detect(np.array([0.0, 0.0, 20.0, 0.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[2.0, 20.0]]))


def test_cacfar_default_mode_is_wrap(monkeypatch) -> None:
    """The default CA-CFAR mode should use circular edge handling."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        threshold_factor=2.0,
    )

    detections = detector.detect(np.array([20.0, 0.0, 0.0, 0.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[0.0, 20.0]]))


def test_cacfar_detector_returns_no_detections_on_flat_signal(monkeypatch) -> None:
    """A uniform background above no target should not create detections."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0,
        num_training_cells=2,
        threshold_factor=1.1,
    )

    detections = detector.detect(np.zeros(7))

    assert detections.shape == (0, 2)


def test_cacfar_wrap_mode_handles_edge_peak(monkeypatch) -> None:
    """Wrap mode should use circular training cells and detect peaks at the array edge."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        threshold_factor=2.0,
        mode="wrap",
    )

    detections = detector.detect(np.array([20.0, 0.0, 0.0, 0.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[0.0, 20.0]]))


def test_cacfar_same_mode_handles_edge_peak(monkeypatch) -> None:
    """The documented ``same`` mode should also detect peaks at the array edge."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        threshold_factor=2.0,
        mode="same",
    )

    detections = detector.detect(np.array([20.0, 0.0, 0.0, 0.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[0.0, 20.0]]))


def test_cacfar_handles_training_window_larger_than_input(monkeypatch) -> None:
    """Short inputs should not crash even when the training window exceeds the data length."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0,
        num_training_cells=4,
        threshold_factor=2.0,
    )

    detections = detector.detect(np.array([0.0, 20.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[1.0, 20.0]]))


def test_oscfar_detector_validates_rank(monkeypatch) -> None:
    """Rank must stay within the available training-cell count."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="Rank"):
        algorithms.OSCFARDetector(
            num_guard_cells=0,
            num_training_cells=2,
            rank=5,
            threshold_factor=1.0,
        )


def test_oscfar_detector_detects_isolated_peak(monkeypatch) -> None:
    """OS-CFAR should detect an isolated strong cell on a flat background."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        rank=1,
        threshold_factor=2.0,
    )

    detections = detector.detect(np.array([0.0, 0.0, 20.0, 0.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[2.0, 20.0]]))


def test_oscfar_detector_detects_edge_peak_with_wrap_processing(monkeypatch) -> None:
    """OS-CFAR should detect strong peaks at the array boundary."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        rank=1,
        threshold_factor=2.0,
    )

    detections = detector.detect(np.array([20.0, 0.0, 0.0, 0.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[0.0, 20.0]]))


def test_oscfar_detector_casts_float_rank_from_parameter_sweeps(monkeypatch) -> None:
    """Float rank values should be cast safely during detection."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=0,
        num_training_cells=2,
        rank=1.9,
        threshold_factor=2.0,
    )

    detections = detector.detect(np.array([0.0, 0.0, 20.0, 0.0, 0.0]))

    np.testing.assert_array_equal(detections, np.array([[2.0, 20.0]]))
