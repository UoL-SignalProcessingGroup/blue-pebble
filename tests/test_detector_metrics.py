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


@dataclass
class _ThresholdDetector:
    """Minimal detector: raw mean power per beam, thresholded directly."""

    threshold: float

    def detect(self, beamformed_data: np.ndarray) -> np.ndarray:
        """Return ``[index, power]`` rows for beams whose power exceeds the threshold."""
        power = np.mean(np.abs(np.asarray(beamformed_data)) ** 2, axis=1)
        indices = np.nonzero(power > self.threshold)[0]
        if indices.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack((indices, power[indices])).astype(np.float64)


@dataclass
class _FakeState:
    state_vector: np.ndarray


@dataclass
class _FakePath:
    states: list


def _spec(metrics, threshold=0.0, param_values=(1.0, 100.0)):
    """Build a sweep spec over a power threshold detector."""
    return metrics.SweepSpec(
        detector=_ThresholdDetector(threshold=threshold),
        param_name="threshold",
        param_values=np.array(param_values),
        label="threshold sweep",
    )


def test_sweep_detection_parameter_accumulates_counts_for_synthetic_data(monkeypatch) -> None:
    """A tiny synthetic sweep should produce predictable aggregated confusion counts."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    # Two timesteps, raw beamformed data (num_beams, num_frames=1); beam 1 has power 25 at
    # t=0, beam 2 has power 25 at t=1 -- both above the loose threshold, neither above the
    # strict one.
    beamformed_data = [
        np.array([[0.0], [5.0], [0.0]]),
        np.array([[0.0], [0.0], [5.0]]),
    ]
    steering = np.array([0.0, 1.0, 2.0])
    ground_truth_paths = [
        _FakePath(states=[_FakeState(np.array([1.0])), _FakeState(np.array([2.0]))]),
    ]

    results = metrics.sweep_detection_parameter(
        beamformed_data=beamformed_data,
        sweep_specs=[_spec(metrics)],
        ground_truth_paths=ground_truth_paths,
        steering_azimuths_rad=steering,
        association_threshold_rad=0.01,
    )

    assert len(results) == 1
    result = results[0]
    np.testing.assert_array_equal(result.param_values, np.array([1.0, 100.0]))
    np.testing.assert_array_equal(result.tp, np.array([2, 0]))
    np.testing.assert_array_equal(result.fp, np.array([0, 0]))
    np.testing.assert_array_equal(result.fn, np.array([0, 2]))
    np.testing.assert_array_equal(result.tn, np.array([4, 4]))
    np.testing.assert_allclose(result.recall, np.array([1.0, 0.0]))


def test_multiband_sweep_with_one_band_matches_the_single_band_sweep(monkeypatch) -> None:
    """A one-band multiband sweep must reproduce the existing single-band sweep exactly."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    beamformed_data = [
        np.array([[0.0], [5.0], [0.0]]),
        np.array([[0.0], [0.0], [5.0]]),
    ]
    steering = np.array([0.0, 1.0, 2.0])
    paths = [_FakePath(states=[_FakeState(np.array([1.0])), _FakeState(np.array([2.0]))])]

    single = metrics.sweep_detection_parameter(
        beamformed_data=beamformed_data,
        sweep_specs=[_spec(metrics)],
        ground_truth_paths=paths,
        steering_azimuths_rad=steering,
        association_threshold_rad=0.01,
    )[0]
    multi = metrics.sweep_detection_parameter_multiband(
        beamformed_data={"only": beamformed_data},
        sweep_specs={"only": _spec(metrics)},
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
    band_a = [np.array([[0.0], [5.0], [0.0]])]
    band_b = [np.array([[0.0], [5.0], [5.0]])]
    steering = np.array([0.0, 1.0, 2.0])
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    specs = {label: _spec(metrics, param_values=(1.0,)) for label in ("a", "b")}
    result = metrics.sweep_detection_parameter_multiband(
        beamformed_data={"a": band_a, "b": band_b},
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


def test_multiband_sweep_uses_each_bands_own_detector(monkeypatch) -> None:
    """Per-band geometry lives in each band's detector, so bands can differ in sensitivity."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    band_a = [np.array([[0.0], [5.0], [0.0]])]
    band_b = [np.array([[0.0], [0.0], [3.0]])]
    steering = np.array([0.0, 1.0, 2.0])
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    # Band 'b's peak (power 9) is rejected once the swept threshold exceeds it.
    result = metrics.sweep_detection_parameter_multiband(
        beamformed_data={"a": band_a, "b": band_b},
        sweep_specs={
            "a": _spec(metrics, param_values=(1.0, 20.0)),
            "b": _spec(metrics, param_values=(1.0, 20.0)),
        },
        ground_truth_paths=paths,
        steering_azimuths_rad=steering,
        association_threshold_rad=0.01,
    )

    # At threshold 1.0 both bands fire: TP on beam 1, FP on beam 2.
    assert (result.tp[0], result.fp[0]) == (1, 1)
    # At 20.0 only band 'a' still fires (power 25 > 20), so 'b's false alarm disappears.
    assert (result.tp[1], result.fp[1]) == (1, 0)


def test_multiband_sweep_rejects_mismatched_bands(monkeypatch) -> None:
    """Maps and specs must describe the same bands."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    beamformed_data = [np.array([[0.0], [5.0], [0.0]])]
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    with pytest.raises(ValueError, match="must cover the same bands"):
        metrics.sweep_detection_parameter_multiband(
            beamformed_data={"a": beamformed_data, "b": beamformed_data},
            sweep_specs={"a": _spec(metrics)},
            ground_truth_paths=paths,
            steering_azimuths_rad=np.array([0.0, 1.0, 2.0]),
            association_threshold_rad=0.01,
        )


def test_multiband_sweep_rejects_mismatched_param_values(monkeypatch) -> None:
    """Combining bands swept at different operating points would be meaningless."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    beamformed_data = [np.array([[0.0], [5.0], [0.0]])]
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    with pytest.raises(ValueError, match="identical param_values"):
        metrics.sweep_detection_parameter_multiband(
            beamformed_data={"a": beamformed_data, "b": beamformed_data},
            sweep_specs={
                "a": _spec(metrics, param_values=(1.0, 2.0)),
                "b": _spec(metrics, param_values=(1.0, 3.0)),
            },
            ground_truth_paths=paths,
            steering_azimuths_rad=np.array([0.0, 1.0, 2.0]),
            association_threshold_rad=0.01,
        )


def test_multiband_sweep_rejects_mismatched_timestep_counts(monkeypatch) -> None:
    """Bands must cover the same number of timesteps."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    paths = [_FakePath(states=[_FakeState(np.array([1.0]))])]

    with pytest.raises(ValueError, match="must share the same number of timesteps"):
        metrics.sweep_detection_parameter_multiband(
            beamformed_data={
                "a": [np.array([[0.0], [5.0], [0.0]])],
                "b": [np.array([[0.0], [5.0], [0.0]])] * 2,
            },
            sweep_specs={label: _spec(metrics) for label in ("a", "b")},
            ground_truth_paths=paths,
            steering_azimuths_rad=np.array([0.0, 1.0, 2.0]),
            association_threshold_rad=0.01,
        )


def test_multiband_sweep_rejects_empty_beamformed_data(monkeypatch) -> None:
    """An empty band set is a configuration error."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    with pytest.raises(ValueError, match="at least one band"):
        metrics.sweep_detection_parameter_multiband(
            beamformed_data={},
            sweep_specs={},
            ground_truth_paths=[],
            steering_azimuths_rad=np.array([0.0]),
            association_threshold_rad=0.01,
        )


def test_clone_detector_with_param_recalibrates_instead_of_reusing_stale_alpha(
    monkeypatch,
) -> None:
    """A spec.detector already used to detect() must not hand its stale alpha to clones.

    CFAR detectors memoize alpha per num_frames in _alpha_cache. Deep-copying a detector
    that was already run once would otherwise carry that cache along, so every swept
    parameter value would silently reuse the alpha calibrated for the OLD parameter
    instead of recalibrating for the new one.
    """
    algorithms, metrics = _load_detector_modules(monkeypatch)

    base = algorithms.CACFARDetector(num_guard_cells=2, num_training_cells=10, target_pfa=1e-2)
    base.detect(np.zeros((50, 1)))  # warms _alpha_cache[1] at target_pfa=1e-2
    stale_alpha = base._alpha_cache[1]

    spec = metrics.SweepSpec(
        detector=base, param_name="target_pfa", param_values=np.array([1e-4])
    )
    clone = metrics._clone_detector_with_param(spec, 1e-4)

    assert clone._alpha_cache == {}
    clone.detect(np.zeros((50, 1)))
    recalibrated_alpha = clone._alpha_cache[1]
    expected_alpha = algorithms.solve_ca_cfar_alpha(1e-4, clone.num_training_total, num_frames=1)

    assert recalibrated_alpha == pytest.approx(expected_alpha)
    assert recalibrated_alpha != stale_alpha


def _complex_gaussian_amplitude(rng, shape, power):
    """Draw complex Gaussian samples with the given mean power (|x|^2 ~ Exponential(power))."""
    scale = np.sqrt(power / 2)
    return rng.normal(size=shape, scale=scale) + 1j * rng.normal(size=shape, scale=scale)


def _gamma_scans(rng, k, n_scans=150, n_beams=64, n_frames=6, level=None):
    """Build scans whose per-frame looks are Gamma(k, 1/k), optionally scaled per beam."""
    scans = []
    for _ in range(n_scans):
        scan = rng.gamma(k, 1.0 / k, size=(n_beams, n_frames))
        if level is not None:
            scan = scan * level[:, None]
        scans.append(scan)
    return scans
