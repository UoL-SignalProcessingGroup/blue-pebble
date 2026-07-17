"""Tests for detection-metrics helper logic and synthetic sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from .support import (
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
)


def _load_detector_modules(monkeypatch):
    """Load detector algorithms and metrics with minimal package scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    algorithms = load_package_module_from_repo(
        "bluepebble/detector/algorithms.py",
        "bluepebble.detector.algorithms",
    )
    metrics = load_package_module_from_repo(
        "bluepebble/detector/metrics.py",
        "bluepebble.detector.metrics",
    )
    return algorithms, metrics


def test_compute_timestep_metrics_handles_empty_cases(monkeypatch) -> None:
    """No-target and no-detection cases should map to the expected counts."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    no_targets = metrics._compute_timestep_metrics(
        detected_bearings_rad=np.array([0.1, 0.2]),
        ground_truth_bearings_rad=np.array([]),
        association_threshold_rad=0.05,
        num_beam_cells=8,
    )
    assert no_targets == metrics._TimestepMetrics(tp=0, fp=2, fn=0, tn=8)

    no_detections = metrics._compute_timestep_metrics(
        detected_bearings_rad=np.array([]),
        ground_truth_bearings_rad=np.array([0.1, 0.2]),
        association_threshold_rad=0.05,
        num_beam_cells=8,
    )
    assert no_detections == metrics._TimestepMetrics(tp=0, fp=0, fn=2, tn=6)


def test_compute_timestep_metrics_respects_circular_wraparound(monkeypatch) -> None:
    """Bearings near ±pi should match across the wrap boundary."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    wrapped = metrics._compute_timestep_metrics(
        detected_bearings_rad=np.array([-np.pi + 0.01]),
        ground_truth_bearings_rad=np.array([np.pi - 0.01]),
        association_threshold_rad=0.05,
        num_beam_cells=16,
    )

    assert wrapped == metrics._TimestepMetrics(tp=1, fp=0, fn=0, tn=15)


def test_compute_timestep_metrics_counts_extra_detections_as_false_positives(monkeypatch) -> None:
    """Once a target is matched, additional detections should remain false alarms."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    result = metrics._compute_timestep_metrics(
        detected_bearings_rad=np.array([0.0, 0.02]),
        ground_truth_bearings_rad=np.array([0.01]),
        association_threshold_rad=0.05,
        num_beam_cells=10,
    )

    assert result == metrics._TimestepMetrics(tp=1, fp=1, fn=0, tn=8)


def test_run_detection_chain_passes_sparse_intermediate_output(monkeypatch) -> None:
    """Intermediate stages should see only prior detections, with all other cells at -inf."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    @dataclass
    class ThresholdStage:
        threshold: float

        def detect(self, data: np.ndarray) -> np.ndarray:
            indices = np.where(data > self.threshold)[0]
            if indices.size == 0:
                return np.empty((0, 2))
            return np.column_stack((indices, data[indices]))

    @dataclass
    class MaxStage:
        seen_input: np.ndarray | None = None

        def detect(self, data: np.ndarray) -> np.ndarray:
            self.seen_input = data.copy()
            finite_indices = np.where(np.isfinite(data))[0]
            if finite_indices.size == 0:
                return np.empty((0, 2))
            best = finite_indices[np.argmax(data[finite_indices])]
            return np.array([[best, data[best]]], dtype=float)

    final_stage = MaxStage()
    result = metrics.run_detection_chain(
        detection_chain=[ThresholdStage(threshold=1.0), final_stage],
        snr_vector=np.array([0.0, 10.0, 5.0]),
    )

    np.testing.assert_array_equal(final_stage.seen_input, np.array([-np.inf, 10.0, 5.0]))
    np.testing.assert_array_equal(result, np.array([[1.0, 10.0]]))


def test_sweep_result_properties_and_best_param_are_computed_correctly(monkeypatch) -> None:
    """Derived scalar metrics should match hand-computed values."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    result = metrics.SweepResult(
        param_values=np.array([0.1, 0.2, 0.3]),
        tp=np.array([4, 3, 1]),
        fp=np.array([4, 2, 1]),
        fn=np.array([0, 1, 3]),
        tn=np.array([0, 2, 3]),
        label="synthetic",
    )

    np.testing.assert_allclose(result.precision, np.array([0.5, 0.6, 0.5]))
    np.testing.assert_allclose(result.recall, np.array([1.0, 0.75, 0.25]))
    np.testing.assert_allclose(result.fpr, np.array([1.0, 0.5, 0.25]))
    np.testing.assert_allclose(result.f1, np.array([2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0]))
    assert result.best_param == pytest.approx(0.2)
    assert result.auc_roc == pytest.approx(0.5625)
    assert result.auc_pr == pytest.approx(0.4125)


def test_param_at_fpr_breaks_ties_using_higher_tpr(monkeypatch) -> None:
    """Equal FPR distance should prefer the stronger true-positive rate."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    result = metrics.SweepResult(
        param_values=np.array([1.0, 2.0]),
        tp=np.array([5, 6]),
        fp=np.array([1, 3]),
        fn=np.array([5, 4]),
        tn=np.array([9, 7]),
        label="tie-break",
    )

    assert result.param_at_fpr(0.2) == pytest.approx(2.0)


def test_sweep_detection_parameter_accumulates_counts_for_synthetic_chain(monkeypatch) -> None:
    """A tiny synthetic sweep should produce predictable aggregated confusion counts."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    @dataclass
    class ThresholdStage:
        threshold: float

        def detect(self, data: np.ndarray) -> np.ndarray:
            indices = np.where(data > self.threshold)[0]
            if indices.size == 0:
                return np.empty((0, 2))
            return np.column_stack((indices, data[indices]))

    @dataclass
    class FakeState:
        state_vector: np.ndarray

    @dataclass
    class FakePath:
        states: list[FakeState]

    snr_map = np.array(
        [
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    steering = np.array([0.0, 1.0, 2.0])
    ground_truth_paths = [
        FakePath(states=[FakeState(np.array([1.0])), FakeState(np.array([2.0]))]),
    ]

    results = metrics.sweep_detection_parameter(
        snr_map=snr_map,
        sweep_specs=[
            metrics.SweepSpec(
                detection_chain=[ThresholdStage(threshold=0.0)],
                algorithm_index=0,
                param_name="threshold",
                param_values=np.array([1.0, 10.0]),
                label="threshold sweep",
            )
        ],
        ground_truth_paths=ground_truth_paths,
        steering_azimuths_rad=steering,
        association_threshold_rad=0.01,
    )

    assert len(results) == 1
    result = results[0]
    np.testing.assert_array_equal(result.param_values, np.array([1.0, 10.0]))
    np.testing.assert_array_equal(result.tp, np.array([2, 0]))
    np.testing.assert_array_equal(result.fp, np.array([0, 0]))
    np.testing.assert_array_equal(result.fn, np.array([0, 2]))
    np.testing.assert_array_equal(result.tn, np.array([4, 4]))
    np.testing.assert_allclose(result.recall, np.array([1.0, 0.0]))


@dataclass
class _ThresholdStage:
    """Minimal detection stage selecting cells strictly above a threshold."""

    threshold: float

    def detect(self, data: np.ndarray) -> np.ndarray:
        """Return ``[index, value]`` rows for cells above the threshold."""
        indices = np.where(data > self.threshold)[0]
        if indices.size == 0:
            return np.empty((0, 2))
        return np.column_stack((indices, data[indices]))


@dataclass
class _FakeState:
    state_vector: np.ndarray


@dataclass
class _FakePath:
    states: list


def _multiband_spec(metrics, threshold=0.0, param_values=(1.0, 10.0)):
    """Build a sweep spec over a threshold stage."""
    return metrics.SweepSpec(
        detection_chain=[_ThresholdStage(threshold=threshold)],
        algorithm_index=0,
        param_name="threshold",
        param_values=np.array(param_values),
        label="threshold sweep",
    )


def test_multiband_sweep_with_one_band_matches_the_single_band_sweep(monkeypatch) -> None:
    """A one-band multiband sweep must reproduce the existing single-band sweep exactly."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    snr_map = np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    steering = np.array([0.0, 1.0, 2.0])
    paths = [_FakePath(states=[_FakeState(np.array([1.0])), _FakeState(np.array([2.0]))])]

    single = metrics.sweep_detection_parameter(
        snr_map=snr_map,
        sweep_specs=[_multiband_spec(metrics)],
        ground_truth_paths=paths,
        steering_azimuths_rad=steering,
        association_threshold_rad=0.01,
    )[0]
    multi = metrics.sweep_detection_parameter_multiband(
        snr_maps={"only": snr_map},
        sweep_specs={"only": _multiband_spec(metrics)},
        ground_truth_paths=paths,
        steering_azimuths_rad=steering,
        association_threshold_rad=0.01,
    )

    for field in ("tp", "fp", "fn", "tn"):
        np.testing.assert_array_equal(getattr(multi, field), getattr(single, field))
    np.testing.assert_array_equal(multi.param_values, single.param_values)


def test_multiband_sweep_unions_bands_and_deduplicates_shared_beams(monkeypatch) -> None:
    """A beam found in two bands must count once, not as a detection plus a false alarm."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    # Both bands see the target in beam 1; only band 'b' also raises beam 2 (a false alarm).
    band_a = np.array([[0.0, 5.0, 0.0]])
    band_b = np.array([[0.0, 5.0, 5.0]])
    steering = np.array([0.0, 1.0, 2.0])
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    specs = {label: _multiband_spec(metrics, param_values=(1.0,)) for label in ("a", "b")}
    result = metrics.sweep_detection_parameter_multiband(
        snr_maps={"a": band_a, "b": band_b},
        sweep_specs=specs,
        ground_truth_paths=paths,
        steering_azimuths_rad=steering,
        association_threshold_rad=0.01,
    )

    # Beam 1 is detected by both bands but scores a single TP; beam 2 is one FP.
    assert result.tp[0] == 1
    assert result.fp[0] == 1
    assert result.fn[0] == 0
    # The FP denominator stays in beam space: (num_beams - num_targets) - fp.
    assert result.tn[0] == 1


def test_multiband_sweep_uses_each_bands_own_chain(monkeypatch) -> None:
    """Per-band geometry lives in each band's chain, so bands can differ in sensitivity."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    band_a = np.array([[0.0, 5.0, 0.0]])
    band_b = np.array([[0.0, 0.0, 3.0]])
    steering = np.array([0.0, 1.0, 2.0])
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    # Band 'b' has a base stage that rejects its own peak once the swept value exceeds it.
    result = metrics.sweep_detection_parameter_multiband(
        snr_maps={"a": band_a, "b": band_b},
        sweep_specs={
            "a": _multiband_spec(metrics, param_values=(1.0, 4.0)),
            "b": _multiband_spec(metrics, param_values=(1.0, 4.0)),
        },
        ground_truth_paths=paths,
        steering_azimuths_rad=steering,
        association_threshold_rad=0.01,
    )

    # At threshold 1.0 both bands fire: TP on beam 1, FP on beam 2.
    assert (result.tp[0], result.fp[0]) == (1, 1)
    # At 4.0 only band 'a' still fires, so the false alarm from 'b' disappears.
    assert (result.tp[1], result.fp[1]) == (1, 0)


def test_multiband_sweep_rejects_mismatched_bands(monkeypatch) -> None:
    """Maps and specs must describe the same bands."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    snr_map = np.array([[0.0, 5.0, 0.0]])
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    with pytest.raises(ValueError, match="must cover the same bands"):
        metrics.sweep_detection_parameter_multiband(
            snr_maps={"a": snr_map, "b": snr_map},
            sweep_specs={"a": _multiband_spec(metrics)},
            ground_truth_paths=paths,
            steering_azimuths_rad=np.array([0.0, 1.0, 2.0]),
            association_threshold_rad=0.01,
        )


def test_multiband_sweep_rejects_mismatched_param_values(monkeypatch) -> None:
    """Combining bands swept at different operating points would be meaningless."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    snr_map = np.array([[0.0, 5.0, 0.0]])
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    with pytest.raises(ValueError, match="identical param_values"):
        metrics.sweep_detection_parameter_multiband(
            snr_maps={"a": snr_map, "b": snr_map},
            sweep_specs={
                "a": _multiband_spec(metrics, param_values=(1.0, 2.0)),
                "b": _multiband_spec(metrics, param_values=(1.0, 3.0)),
            },
            ground_truth_paths=paths,
            steering_azimuths_rad=np.array([0.0, 1.0, 2.0]),
            association_threshold_rad=0.01,
        )


def test_multiband_sweep_rejects_mismatched_map_shapes(monkeypatch) -> None:
    """Bands must share a timestep and beam grid."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    with pytest.raises(ValueError, match="share one SNR map shape"):
        metrics.sweep_detection_parameter_multiband(
            snr_maps={"a": np.zeros((1, 3)), "b": np.zeros((1, 4))},
            sweep_specs={label: _multiband_spec(metrics) for label in ("a", "b")},
            ground_truth_paths=paths,
            steering_azimuths_rad=np.array([0.0, 1.0, 2.0]),
            association_threshold_rad=0.01,
        )


def test_multiband_sweep_rejects_empty_snr_maps(monkeypatch) -> None:
    """An empty band set is a configuration error."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    with pytest.raises(ValueError, match="at least one band"):
        metrics.sweep_detection_parameter_multiband(
            snr_maps={},
            sweep_specs={},
            ground_truth_paths=[],
            steering_azimuths_rad=np.array([0.0]),
            association_threshold_rad=0.01,
        )
