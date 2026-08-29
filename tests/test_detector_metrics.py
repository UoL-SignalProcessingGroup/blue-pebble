"""Tests for detection-metrics helper logic and synthetic sweeps."""

from __future__ import annotations

import sys
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
    load_package_module_from_repo(
        "bluepebble/detector/fluctuation_models.py",
        "bluepebble.detector.fluctuation_models",
    )
    metrics = load_package_module_from_repo(
        "bluepebble/detector/metrics.py",
        "bluepebble.detector.metrics",
    )
    return algorithms, metrics


def _load_detector_modules_with_fluctuation_models(monkeypatch):
    """Load algorithms, fluctuation_models, and metrics with minimal scaffolding.

    fluctuation_models.py is already loaded as a side effect of _load_detector_modules
    (metrics.py imports from it), so it's just retrieved from sys.modules here rather
    than loaded a second time.
    """
    algorithms, metrics = _load_detector_modules(monkeypatch)
    fluctuation_models = sys.modules["bluepebble.detector.fluctuation_models"]
    return algorithms, fluctuation_models, metrics


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
    # auc_roc integrates over the full [0, 1] FPR range. This sweep's fpr already reaches 1.0,
    # so only the (0, 0) corner is added: 0.5625 over the measured span, plus the
    # (0, 0) -> (0.25, 0.25) triangle.
    assert result.auc_roc == pytest.approx(0.5625 + 0.03125)
    # auc_pr is deliberately NOT extended, so it stays the area over the measured recall span.
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


def test_clone_detector_with_param_tolerates_detectors_without_alpha_cache(
    monkeypatch,
) -> None:
    """Non-CFAR detectors (no _alpha_cache attribute) must not crash the clone helper."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    spec = _spec(metrics, threshold=0.0, param_values=(1.0, 2.0))
    clone = metrics._clone_detector_with_param(spec, 2.0)

    assert clone.threshold == 2.0
    assert not hasattr(clone, "_alpha_cache")


# ---------------------------------------------------------------------------
# Theoretical Pd-vs-Pfa curves (ca_cfar_roc / os_cfar_roc)
# ---------------------------------------------------------------------------


def test_ca_cfar_roc_matches_pointwise_alpha_pd_calls(monkeypatch) -> None:
    """The vectorised sweep must reproduce hand-called solve_ca_cfar_alpha + ca_cfar_pd."""
    algorithms, fluctuation_models, metrics = _load_detector_modules_with_fluctuation_models(
        monkeypatch
    )
    pfa_values = [1e-1, 1e-2, 1e-3, 1e-4]
    N, M, snr_linear = 20, 5, 2.0
    model = fluctuation_models.RayleighFluctuation()

    got = metrics.ca_cfar_roc(pfa_values, N, M, snr_linear)

    expected = [
        model.ca_cfar_pd(algorithms.solve_ca_cfar_alpha(pfa, N, M), N, M, snr_linear)
        for pfa in pfa_values
    ]
    np.testing.assert_allclose(got, expected)


def test_ca_cfar_roc_is_monotonically_non_decreasing_in_pfa(monkeypatch) -> None:
    """A more permissive Pfa (lower alpha) should never lower Pd."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    pfa_values = np.geomspace(1e-4, 1e-1, 20)

    pd = metrics.ca_cfar_roc(pfa_values, num_training_total=20, num_frames=5, snr_linear=2.0)

    assert np.all(np.diff(pd) >= 0)


def test_os_cfar_roc_single_look_matches_pointwise_alpha_pd_calls(monkeypatch) -> None:
    """At num_frames=1 the vectorised sweep must reproduce the hand-called closed forms."""
    algorithms, fluctuation_models, metrics = _load_detector_modules_with_fluctuation_models(
        monkeypatch
    )
    pfa_values = [1e-1, 1e-2, 1e-3, 1e-4]
    N, rank, snr_linear = 20, 15, 2.0
    model = fluctuation_models.RayleighFluctuation()

    got = metrics.os_cfar_roc(pfa_values, N, rank, num_frames=1, snr_linear=snr_linear)

    expected = [
        model.os_cfar_pd(
            algorithms.solve_os_cfar_alpha_single_look(pfa, N, rank),
            N,
            rank,
            num_frames=1,
            snr_linear=snr_linear,
        )
        for pfa in pfa_values
    ]
    np.testing.assert_allclose(got, expected)


def test_os_cfar_roc_single_look_is_monotonically_non_decreasing_in_pfa(monkeypatch) -> None:
    """A more permissive Pfa should never lower Pd for single-look OS-CFAR either."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    pfa_values = np.geomspace(1e-4, 1e-1, 20)

    pd = metrics.os_cfar_roc(
        pfa_values, num_training_total=20, rank=15, num_frames=1, snr_linear=2.0
    )

    assert np.all(np.diff(pd) >= 0)


def test_os_cfar_roc_multilook_agrees_with_independent_per_point_simulation(monkeypatch) -> None:
    """The shared-simulation sweep must agree with fully independent per-point Monte Carlo.

    Cross-checks the "common random numbers" reuse of noise_estimate across every Pfa
    point against calibrate_os_cfar_alpha_mc + RayleighFluctuation.os_cfar_pd run
    independently per point -- the same pattern already validated for those two functions
    individually.
    """
    algorithms, fluctuation_models, metrics = _load_detector_modules_with_fluctuation_models(
        monkeypatch
    )
    N, rank, M, snr_linear = 20, 15, 5, 1.0
    pfa_values = [1e-1, 1e-2, 1e-3]
    num_trials = 1_000_000
    model = fluctuation_models.RayleighFluctuation()

    shared = metrics.os_cfar_roc(
        pfa_values, N, rank, M, snr_linear, num_trials=num_trials, rng=np.random.default_rng(456)
    )

    rng_ref = np.random.default_rng(123)
    independent = []
    for pfa in pfa_values:
        alpha = algorithms.calibrate_os_cfar_alpha_mc(
            pfa, N, rank, num_frames=M, num_trials=num_trials, rng=rng_ref
        )
        independent.append(
            model.os_cfar_pd(
                alpha,
                N,
                rank,
                num_frames=M,
                snr_linear=snr_linear,
                num_trials=num_trials,
                rng=rng_ref,
            )
        )

    np.testing.assert_allclose(shared, independent, atol=0.01)


def test_os_cfar_roc_multilook_is_deterministic_given_fixed_seed(monkeypatch) -> None:
    """Two runs with the same seeded Generator should agree exactly."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    pfa_values = [1e-1, 1e-2]

    r1 = metrics.os_cfar_roc(
        pfa_values, 20, 15, num_frames=5, snr_linear=1.0, num_trials=10_000,
        rng=np.random.default_rng(0),
    )
    r2 = metrics.os_cfar_roc(
        pfa_values, 20, 15, num_frames=5, snr_linear=1.0, num_trials=10_000,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(r1, r2)


def test_os_cfar_roc_multilook_is_monotonically_non_decreasing_in_pfa(monkeypatch) -> None:
    """A more permissive Pfa should never lower Pd, even under Monte Carlo noise at scale."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    pfa_values = np.geomspace(1e-3, 1e-1, 8)

    pd = metrics.os_cfar_roc(
        pfa_values,
        num_training_total=20,
        rank=15,
        num_frames=5,
        snr_linear=2.0,
        num_trials=500_000,
        rng=np.random.default_rng(1),
    )

    assert np.all(np.diff(pd) >= 0)


# ---------------------------------------------------------------------------
# SweepResult.from_theoretical_roc
# ---------------------------------------------------------------------------


def test_sweep_result_requires_exactly_one_construction_mode(monkeypatch) -> None:
    """Neither both counts and a theoretical curve, nor neither, is a valid SweepResult."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    with pytest.raises(ValueError, match="exactly one"):
        metrics.SweepResult(param_values=np.array([0.1]))  # neither

    with pytest.raises(ValueError, match="exactly one"):
        metrics.SweepResult(
            param_values=np.array([0.1]),
            tp=np.array([1]),
            fp=np.array([0]),
            fn=np.array([0]),
            tn=np.array([1]),
            _fpr=np.array([0.1]),
            _tpr=np.array([0.9]),
        )  # both


def test_sweep_result_from_theoretical_roc_exposes_fpr_tpr_directly(monkeypatch) -> None:
    """fpr/tpr on a theoretical result should be exactly the pfa/pd arrays given."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    pfa = np.array([0.1, 0.01, 0.001])
    pd = np.array([0.9, 0.6, 0.3])

    result = metrics.SweepResult.from_theoretical_roc(pfa, pd, label="theoretical")

    np.testing.assert_array_equal(result.fpr, pfa)
    np.testing.assert_array_equal(result.tpr, pd)
    np.testing.assert_array_equal(result.recall, pd)
    np.testing.assert_array_equal(result.param_values, pfa)
    assert result.label == "theoretical"


def test_sweep_result_from_theoretical_roc_supports_auc_roc_and_best_param(monkeypatch) -> None:
    """Properties that only need fpr/tpr should work normally on a theoretical result."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    pfa = np.array([0.001, 0.01, 0.1])
    pd = np.array([0.3, 0.6, 0.9])

    result = metrics.SweepResult.from_theoretical_roc(pfa, pd)

    # This curve spans only 10% of the FPR axis, so auc_roc must close it off at both ends
    # rather than reporting the ~0.07 area over the measured span alone.
    expected = np.trapezoid(
        np.concatenate(([0.0], pd, [1.0])), np.concatenate(([0.0], pfa, [1.0]))
    )
    assert result.auc_roc == pytest.approx(expected)
    assert result.auc_roc > np.trapezoid(pd, pfa)
    assert result.best_param in pfa
    assert result.param_at_fpr(0.01) == pytest.approx(0.01)


def test_auc_roc_is_not_capped_by_a_truncated_fpr_range(monkeypatch) -> None:
    """A saturating detector must not be penalised for never reaching FPR=1.

    Regression test for AUC being computed over only the sweep's own FPR span: an excellent
    detector whose achieved Pfa plateaus early (peak consolidation does exactly this on real
    beamformed data) previously scored below the 0.5 random-classifier line.
    """
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    # Pd already saturated at 1.0 by the time Pfa reaches 0.13, then the sweep stops.
    result = metrics.SweepResult.from_theoretical_roc(
        pfa=np.array([0.0, 0.002, 0.02, 0.13]), pd=np.array([0.82, 0.95, 1.0, 1.0])
    )

    assert result.auc_roc > 0.99
    assert result.auc_roc <= 1.0


def test_auc_roc_of_a_random_classifier_is_one_half(monkeypatch) -> None:
    """The diagonal must score 0.5 whether or not the sweep reaches the corners."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    full = metrics.SweepResult.from_theoretical_roc(
        pfa=np.array([0.0, 0.5, 1.0]), pd=np.array([0.0, 0.5, 1.0])
    )
    partial = metrics.SweepResult.from_theoretical_roc(
        pfa=np.array([0.3, 0.5]), pd=np.array([0.3, 0.5])
    )

    assert full.auc_roc == pytest.approx(0.5)
    assert partial.auc_roc == pytest.approx(0.5)


@pytest.mark.parametrize("prop_name", ["precision", "f1", "auc_pr"])
def test_sweep_result_from_theoretical_roc_rejects_pr_properties(monkeypatch, prop_name) -> None:
    """Precision/F1/AUC-PR need confusion-matrix counts a theoretical curve doesn't have."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    result = metrics.SweepResult.from_theoretical_roc(
        pfa=np.array([0.1, 0.01]), pd=np.array([0.9, 0.6])
    )

    with pytest.raises(ValueError, match="confusion-matrix counts"):
        getattr(result, prop_name)


def test_sweep_result_empirical_construction_is_unaffected(monkeypatch) -> None:
    """Constructing from counts (the existing, well-tested path) must still work unchanged."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    result = metrics.SweepResult(
        param_values=np.array([0.1, 0.2]),
        tp=np.array([4, 3]),
        fp=np.array([1, 0]),
        fn=np.array([0, 1]),
        tn=np.array([5, 6]),
        label="empirical",
    )

    assert result.precision[0] == pytest.approx(0.8)
    assert result.f1[0] == pytest.approx(2 * 0.8 * 1.0 / (0.8 + 1.0))
    assert result.auc_pr >= 0.0


# ---------------------------------------------------------------------------
# snr_linear_from_ground_truth_bearing
# ---------------------------------------------------------------------------


def _complex_gaussian_amplitude(rng, shape, power):
    """Draw complex Gaussian samples with the given mean power (|x|^2 ~ Exponential(power))."""
    scale = np.sqrt(power / 2)
    return rng.normal(size=shape, scale=scale) + 1j * rng.normal(size=shape, scale=scale)


def test_snr_linear_from_ground_truth_bearing_finds_nearest_beam(monkeypatch) -> None:
    """cut_index should be the beam closest to the true bearing."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    steering = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    rng = np.random.default_rng(0)
    data = _complex_gaussian_amplitude(rng, (180, 200), power=1.0)

    _snr_linear_hat, cut_index = metrics.snr_linear_from_ground_truth_bearing(
        data, true_bearing_rad=steering[47] + 1e-3, steering_azimuths_rad=steering,
        validation_guard_bins=5,
    )

    assert cut_index == 47


def test_snr_linear_from_ground_truth_bearing_is_wrap_aware(monkeypatch) -> None:
    """A bearing just below +pi should resolve to whichever beam is truly nearest,

    including across the -pi/+pi wrap seam (-pi and +pi are the same point).
    """
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    steering = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    rng = np.random.default_rng(0)
    data = _complex_gaussian_amplitude(rng, (180, 200), power=1.0)
    true_bearing = np.pi - 1e-3  # infinitesimally less than +pi -- effectively at the seam

    _snr_linear_hat, cut_index = metrics.snr_linear_from_ground_truth_bearing(
        data, true_bearing_rad=true_bearing, steering_azimuths_rad=steering,
        validation_guard_bins=5,
    )

    # -pi and +pi are the same point on the circle, so beam 0 (steering[0] == -pi) is
    # genuinely closer to this bearing than beam 179 (steering[179] ~= +pi - one bin width).
    assert cut_index == 0


def test_snr_linear_from_ground_truth_bearing_recovers_known_snr_linear(monkeypatch) -> None:
    """A target planted at a known snr_linear should be recovered within Monte Carlo noise."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    steering = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    true_snr_linear = 8.0
    target_idx = 47
    rng = np.random.default_rng(1)

    data = _complex_gaussian_amplitude(rng, (180, 20_000), power=1.0)
    data[target_idx] = _complex_gaussian_amplitude(rng, (20_000,), power=1.0 + true_snr_linear)

    snr_linear_hat, cut_index = metrics.snr_linear_from_ground_truth_bearing(
        data, true_bearing_rad=steering[target_idx], steering_azimuths_rad=steering,
        validation_guard_bins=5,
    )

    assert cut_index == target_idx
    assert snr_linear_hat == pytest.approx(true_snr_linear, rel=0.1)


def test_snr_linear_from_ground_truth_bearing_rejects_guard_bins_covering_whole_array(
    monkeypatch,
) -> None:
    """An exclusion zone spanning every beam leaves no cells to estimate noise from."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    steering = np.linspace(-np.pi, np.pi, 180, endpoint=False)
    rng = np.random.default_rng(0)
    data = _complex_gaussian_amplitude(rng, (180, 50), power=1.0)

    with pytest.raises(ValueError, match="excludes the entire array"):
        metrics.snr_linear_from_ground_truth_bearing(
            data, true_bearing_rad=steering[0], steering_azimuths_rad=steering,
            validation_guard_bins=200,
        )


def test_snr_linear_from_ground_truth_bearing_treats_real_power_and_complex_consistently(
    monkeypatch,
) -> None:
    """Real (already-power) input must not be double-squared relative to the equivalent complex
    amplitude it was computed from.

    Regression test for the same _directional_power bug covered in test_detector_algorithms.py:
    BeamformedData is deliberately either complex amplitude or already-real power (see
    bluepebble.types.sensordata.BeamformedData), and this function must treat both consistently.
    """
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    steering = np.array([0.0, 1.0, 2.0])
    amplitude = np.array([[1.0 + 0.0j], [1.0 + 0.0j], [3.0 + 0.0j]])
    real_power = np.abs(amplitude) ** 2

    from_complex = metrics.snr_linear_from_ground_truth_bearing(
        amplitude, true_bearing_rad=2.0, steering_azimuths_rad=steering, validation_guard_bins=0
    )
    from_real_power = metrics.snr_linear_from_ground_truth_bearing(
        real_power, true_bearing_rad=2.0, steering_azimuths_rad=steering, validation_guard_bins=0
    )

    assert from_complex == from_real_power


# ---------------------------------------------------------------------------
# estimate_effective_looks_per_frame
# ---------------------------------------------------------------------------


def _gamma_scans(rng, k, n_scans=150, n_beams=64, n_frames=6, level=None):
    """Build scans whose per-frame looks are Gamma(k, 1/k), optionally scaled per beam."""
    scans = []
    for _ in range(n_scans):
        scan = rng.gamma(k, 1.0 / k, size=(n_beams, n_frames))
        if level is not None:
            scan = scan * level[:, None]
        scans.append(scan)
    return scans


@pytest.mark.parametrize("k_true", [1.0, 5.0, 40.0])
def test_estimate_effective_looks_recovers_a_known_k(monkeypatch, k_true) -> None:
    """The estimator must invert CV = 1/sqrt(K*M) across two orders of magnitude in K."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    scans = _gamma_scans(np.random.default_rng(0), k_true)

    estimate = metrics.estimate_effective_looks_per_frame(scans)

    assert estimate == pytest.approx(k_true, rel=0.15)


def test_estimate_effective_looks_ignores_beam_to_beam_level_structure(monkeypatch) -> None:
    """Array geometry makes mean power vary between beams; that is not per-look fluctuation.

    Measuring each beam's spread across scans (rather than pooling across beams within a scan)
    is what makes the estimate blind to this. A spatially-pooled estimator would read the level
    structure as extra variance and report a far lower K.
    """
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    rng = np.random.default_rng(1)
    k_true = 25.0
    level = np.exp(rng.normal(0.0, 0.8, size=64))

    flat = metrics.estimate_effective_looks_per_frame(_gamma_scans(rng, k_true))
    shaped = metrics.estimate_effective_looks_per_frame(_gamma_scans(rng, k_true, level=level))

    assert shaped == pytest.approx(k_true, rel=0.15)
    assert shaped == pytest.approx(flat, rel=0.25)


def test_estimate_effective_looks_needs_the_target_masked_out(monkeypatch) -> None:
    """Target energy in the sample destroys the estimate, so the mask is not optional.

    A moving contact inflates each contaminated beam's across-scan spread enormously. Masking
    those cells recovers the true K; leaving them in collapses the estimate by orders of
    magnitude, which would in turn produce a wildly over-permissive threshold.
    """
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    rng = np.random.default_rng(2)
    k_true, n_scans, n_beams = 25.0, 150, 64

    scans, mask = [], np.ones((n_scans, n_beams), dtype=bool)
    for t in range(n_scans):
        scan = rng.gamma(k_true, 1.0 / k_true, size=(n_beams, 6))
        target_bin = (t * 2) % n_beams
        lo, hi = max(0, target_bin - 3), min(n_beams, target_bin + 4)
        scan[lo:hi] *= 40.0
        mask[t, lo:hi] = False
        scans.append(scan)

    masked = metrics.estimate_effective_looks_per_frame(scans, mask)
    unmasked = metrics.estimate_effective_looks_per_frame(scans)

    assert masked == pytest.approx(k_true, rel=0.15)
    assert unmasked < k_true / 10


def test_estimate_effective_looks_treats_complex_and_real_power_consistently(monkeypatch) -> None:
    """Complex amplitude must be squared to power first, via the shared _directional_power."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    rng = np.random.default_rng(3)
    amplitude = [
        rng.normal(size=(48, 5), scale=np.sqrt(0.5))
        + 1j * rng.normal(size=(48, 5), scale=np.sqrt(0.5))
        for _ in range(120)
    ]
    real_power = [np.abs(scan) ** 2 for scan in amplitude]

    assert metrics.estimate_effective_looks_per_frame(amplitude) == pytest.approx(
        metrics.estimate_effective_looks_per_frame(real_power)
    )


def test_estimate_effective_looks_of_single_bin_data_is_about_one(monkeypatch) -> None:
    """Unit-power complex Gaussian data is the K=1 case the idealised model already assumes."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    rng = np.random.default_rng(4)
    scans = [
        rng.normal(size=(64, 6), scale=np.sqrt(0.5))
        + 1j * rng.normal(size=(64, 6), scale=np.sqrt(0.5))
        for _ in range(200)
    ]

    assert metrics.estimate_effective_looks_per_frame(scans) == pytest.approx(1.0, rel=0.15)


def test_estimate_effective_looks_rejects_empty_input(monkeypatch) -> None:
    """No scans means nothing to estimate from."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)

    with pytest.raises(ValueError, match="empty"):
        metrics.estimate_effective_looks_per_frame([])


def test_estimate_effective_looks_rejects_ragged_scans(monkeypatch) -> None:
    """Scans with differing beam counts cannot be stacked into one array."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    scans = [np.ones((10, 4)), np.ones((11, 4))]

    with pytest.raises(ValueError, match="share a shape"):
        metrics.estimate_effective_looks_per_frame(scans)


def test_estimate_effective_looks_rejects_a_mismatched_mask(monkeypatch) -> None:
    """A mask that doesn't line up with the data would silently mis-select cells."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    scans = _gamma_scans(np.random.default_rng(5), 10.0, n_scans=20, n_beams=16)

    with pytest.raises(ValueError, match="does not match"):
        metrics.estimate_effective_looks_per_frame(scans, np.ones((20, 8), dtype=bool))


def test_estimate_effective_looks_rejects_too_few_usable_scans(monkeypatch) -> None:
    """A CV from a handful of samples is too noisy to be worth returning."""
    _algorithms, metrics = _load_detector_modules(monkeypatch)
    scans = _gamma_scans(np.random.default_rng(6), 10.0, n_scans=20, n_beams=16)

    with pytest.raises(ValueError, match="min_scans_per_beam"):
        metrics.estimate_effective_looks_per_frame(scans, min_scans_per_beam=50)


def test_estimated_k_calibrates_a_threshold_that_hits_target_pfa(monkeypatch) -> None:
    """End-to-end: measure K off the data, calibrate with it, achieve the target Pfa.

    This is the whole point of the helper, so it is checked as one chain rather than as two
    independently-correct halves.
    """
    algorithms, metrics = _load_detector_modules(monkeypatch)
    k_true, num_frames, num_training_total, rank, target_pfa = 20.0, 6, 20, 15, 0.05

    scans = _gamma_scans(
        np.random.default_rng(7), k_true, n_scans=400, n_beams=64, n_frames=num_frames
    )
    k_hat = metrics.estimate_effective_looks_per_frame(scans)

    alpha = algorithms.calibrate_os_cfar_alpha_mc(
        target_pfa, num_training_total, rank, num_frames,
        num_trials=200_000, rng=np.random.default_rng(8),
        effective_looks_per_frame=k_hat,
    )

    rng = np.random.default_rng(9)
    trials = 200_000
    ref = rng.gamma(k_true, 1.0 / k_true, size=(trials, num_training_total, num_frames)).mean(
        axis=2
    )
    ref.sort(axis=1)
    cut = rng.gamma(k_true, 1.0 / k_true, size=(trials, num_frames)).mean(axis=1)
    achieved = float(np.mean(cut > alpha * ref[:, rank - 1]))

    assert achieved == pytest.approx(target_pfa, rel=0.15)
