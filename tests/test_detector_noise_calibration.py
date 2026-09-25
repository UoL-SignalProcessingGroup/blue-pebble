"""Tests for calibrating CFAR detectors against noise-only data (calibration.py).

Accuracy is checked three ways: against the exact Pfa of i.i.d. Gamma data (quadrature), against
a heavy-tailed ratio distribution with a known quantile function, and end to end on held-out
scans of spatially correlated noise where the nominal Gamma model is demonstrably wrong.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from .support import (
    ca_pfa_by_quadrature,
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
    os_pfa_by_quadrature,
)


def _load(monkeypatch):
    """Load algorithms (which imports calibration) and calibration with minimal scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    algorithms = load_package_module_from_repo(
        "bluepebble/detector/algorithms.py", "bluepebble.detector.algorithms"
    )
    calibration = load_package_module_from_repo(
        "bluepebble/detector/calibration.py", "bluepebble.detector.calibration"
    )
    return algorithms, calibration


def _make_detector(algorithms, name, **overrides):
    common = dict(num_guard_cells=2, num_training_cells=10, target_pfa=1e-2)
    common.update(overrides)
    if name == "ca":
        return algorithms.CACFARDetector(**common)
    return algorithms.OSCFARDetector(**{"rank": 15, **common})


def _gamma_scans(rng, num_scans, num_beams, num_frames):
    """i.i.d. Exponential(1) looks: the model the closed-form calibration assumes."""
    return [rng.exponential(size=(num_beams, num_frames)) for _ in range(num_scans)]


def _correlated_scans(rng, num_scans, num_beams, num_frames, width=7):
    """Average looks over ``width`` neighbouring beams (circularly), like finite beamwidth.

    Beam-to-beam correlation breaks the i.i.d. model: the CUT and its reference cells share
    noise, so the nominal alpha no longer achieves target_pfa.
    """
    looks = rng.exponential(size=(num_scans, num_beams, num_frames))
    kernel = np.zeros(num_beams)
    kernel[:width] = 1.0 / width
    kernel = np.roll(kernel, -(width // 2))
    smoothed = np.fft.ifft(np.fft.fft(looks, axis=1) * np.fft.fft(kernel)[None, :, None], axis=1)
    return list(np.real(smoothed))


def _achieved_pfa(detector, scans):
    """Fraction of cells flagged by unconsolidated detect() across the scans."""
    crossings = sum(len(detector.detect(scan)) for scan in scans)
    return crossings / sum(scan.shape[0] for scan in scans)


# ---------------------------------------------------------------------------
# 1. Accuracy against exact references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_calibration_on_iid_gamma_data_recovers_the_exact_pfa(monkeypatch, detector_name) -> None:
    """On data that satisfies the model, calibrated alpha has the requested exact Pfa.

    Empirical region (1e-2) and generalised Pareto tail region (1e-4) are both checked against
    quadrature of the true Pfa at the calibrated alpha.
    """
    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, detector_name)
    num_frames = 4
    scans = _gamma_scans(np.random.default_rng(0), 500, 256, num_frames)

    calibration = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(scans)

    def exact_pfa(alpha):
        if detector_name == "ca":
            return ca_pfa_by_quadrature(alpha, 20, num_frames)
        return os_pfa_by_quadrature(alpha, 20, 15, num_frames)

    assert exact_pfa(calibration.alpha(1e-2)) == pytest.approx(1e-2, rel=0.1)
    achieved = exact_pfa(calibration.alpha(1e-4))
    assert 1e-4 / 1.3 < achieved < 1e-4 * 1.3


def test_generalised_pareto_tail_extrapolates_a_known_heavy_tail(monkeypatch) -> None:
    """Below the observable rate, alpha follows a heavy tail an exponential tail would miss.

    Ratios are drawn from a generalised Pareto distribution above a body, so the true quantile
    at any Pfa is known in closed form. The fitted tail must reach Pfa 1e-5, where only a couple
    of the 2e5 calibration cells lie beyond the threshold; forcing an exponential tail (shape 0)
    must be clearly worse.
    """
    _, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(1)
    num_cells, tail_rate, shape, scale, threshold = 200_000, 0.05, 0.25, 0.2, 1.5
    in_tail = rng.random(num_cells) < tail_rate
    ratios = np.where(
        in_tail,
        threshold + scale / shape * (rng.random(num_cells) ** -shape - 1.0),
        rng.uniform(0.5, threshold, num_cells),
    )

    def true_quantile(pfa):
        return threshold + scale / shape * ((pfa / tail_rate) ** -shape - 1.0)

    def true_pfa(alpha):
        return tail_rate * (1.0 + shape * (alpha - threshold) / scale) ** (-1.0 / shape)

    detector = SimpleNamespace(
        num_guard_cells=0,
        num_training_cells=1,
        circular=True,
        _power_and_noise=lambda data: (np.asarray(data)[:, 0], np.ones(len(data))),
    )
    calibration = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(
        [ratios[:, None]], tail_pfa=1e-2
    )

    alpha = calibration.alpha(1e-5)  # below tail_pfa, within what 2e5 cells resolve
    assert alpha == pytest.approx(true_quantile(1e-5), rel=0.1)
    assert true_pfa(alpha) == pytest.approx(1e-5, rel=0.3)

    exponential_tail = dataclasses.replace(calibration, tail_shape=0.0)
    assert true_pfa(exponential_tail.alpha(1e-5)) > 3e-5


# ---------------------------------------------------------------------------
# 2. End to end on correlated noise, held out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_calibrated_detector_achieves_target_pfa_on_held_out_correlated_noise(
    monkeypatch, detector_name, circular
) -> None:
    """Calibrate on scans A, detect on independent scans B: target_pfa is met.

    Beam correlation leaves far fewer independent cells than cells, so the scan counts are
    sized from a seed study: at 3000 scans per set, achieved/target stayed within 0.88-1.07 at
    both Pfa values across eight seeds.
    """
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(2)
    calibration_scans = _correlated_scans(rng, 3000, 128, 3)
    held_out_scans = _correlated_scans(rng, 3000, 128, 3)

    kwargs = dict(consolidate_peaks=False, circular=circular)
    reference = _make_detector(algorithms, detector_name, **kwargs)
    calibration = calibration_module.NoiseCalibrator(reference).calibrate_from_noise(
        calibration_scans
    )
    calibrated = _make_detector(algorithms, detector_name, **kwargs, noise_calibration=calibration)

    for target_pfa, tolerance in ((1e-2, 1.15), (1e-3, 1.3)):
        calibrated.target_pfa = target_pfa
        achieved = _achieved_pfa(calibrated, held_out_scans)
        assert target_pfa / tolerance < achieved < target_pfa * tolerance, (target_pfa, achieved)


# ---------------------------------------------------------------------------
# 3. Detector integration
# ---------------------------------------------------------------------------


def _small_calibration(algorithms, calibration_module, detector, num_frames=2):
    scans = _gamma_scans(np.random.default_rng(3), 120, 200, num_frames)
    return calibration_module.NoiseCalibrator(detector).calibrate_from_noise(
        scans, min_tail_exceedances=100
    )


def test_calibrated_alpha_replaces_the_model_alpha(monkeypatch) -> None:
    """With a calibration set, _alpha_for returns calibration.alpha(target_pfa)."""
    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, "ca")
    calibration = _small_calibration(algorithms, calibration_module, detector)

    detector.noise_calibration = calibration

    assert detector._alpha_for(2) == calibration.alpha(1e-2)


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(num_guard_cells=3), "num_guard_cells"),
        (dict(num_training_cells=12), "num_training_cells"),
        (dict(circular=False), "circular"),
    ],
)
def test_calibration_mismatch_raises_naming_the_setting(monkeypatch, change, match) -> None:
    """A calibration measured with different window or edge settings is rejected."""
    algorithms, calibration_module = _load(monkeypatch)
    calibration = _small_calibration(
        algorithms, calibration_module, _make_detector(algorithms, "ca")
    )
    detector = _make_detector(algorithms, "ca", **change, noise_calibration=calibration)

    with pytest.raises(ValueError, match=match):
        detector.detect(np.ones((64, 2)))


def test_calibration_mismatch_on_frames_rank_or_detector_type_raises(monkeypatch) -> None:
    """Frame count, OS rank and detector type are all part of what a calibration applies to."""
    algorithms, calibration_module = _load(monkeypatch)
    os_detector = _make_detector(algorithms, "os")
    calibration = _small_calibration(algorithms, calibration_module, os_detector)

    with pytest.raises(ValueError, match="frames"):
        _make_detector(algorithms, "os", noise_calibration=calibration).detect(np.ones((64, 5)))
    with pytest.raises(ValueError, match="rank"):
        _make_detector(algorithms, "os", rank=12, noise_calibration=calibration)._alpha_for(2)
    with pytest.raises(ValueError, match="detector type"):
        _make_detector(algorithms, "ca", noise_calibration=calibration)._alpha_for(2)


def test_setting_or_replacing_calibration_and_changing_pfa_invalidate_alpha(monkeypatch) -> None:
    """Cached alpha follows the calibration and target_pfa after the detector has been used."""
    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, "ca")

    first = _small_calibration(algorithms, calibration_module, detector)
    detector.noise_calibration = first
    assert detector._alpha_for(2) == first.alpha(1e-2)

    second = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(
        _correlated_scans(np.random.default_rng(4), 120, 200, 2),
        min_tail_exceedances=100,
    )
    detector.noise_calibration = second
    assert detector._alpha_for(2) == second.alpha(1e-2) != first.alpha(1e-2)

    detector.target_pfa = 5e-2
    assert detector._alpha_for(2) == second.alpha(5e-2)

    # No model-based fallback remains: clearing the calibration must raise, not silently
    # thresholding on the wrong distribution.
    detector.noise_calibration = None
    with pytest.raises(ValueError, match="target_pfa set but no noise_calibration"):
        detector._alpha_for(2)


def test_sweeping_target_pfa_on_clones_uses_the_calibration(monkeypatch) -> None:
    """ROC sweeps deep-copy the detector and set target_pfa; the calibration must carry over."""
    algorithms, calibration_module = _load(monkeypatch)
    metrics = load_package_module_from_repo(
        "bluepebble/detector/metrics.py", "bluepebble.detector.metrics"
    )
    detector = _make_detector(algorithms, "ca")
    detector.noise_calibration = _small_calibration(algorithms, calibration_module, detector)
    detector.detect(np.ones((64, 2)))
    spec = metrics.SweepSpec(detector=detector, param_name="target_pfa", param_values=[1e-3])

    clone = metrics._clone_detector_with_param(spec, 3e-2)

    assert clone._alpha_for(2) == pytest.approx(detector.noise_calibration.alpha(3e-2))
    assert copy.deepcopy(detector)._alpha_for(2) == pytest.approx(
        detector.noise_calibration.alpha(1e-2)
    )


# ---------------------------------------------------------------------------
# 4. Input validation and guards
# ---------------------------------------------------------------------------


def test_too_few_tail_exceedances_raises_with_guidance(monkeypatch) -> None:
    """Calibration refuses to fit a tail from too few exceedances, saying how many cells."""
    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, "ca")
    scans = _gamma_scans(np.random.default_rng(5), 10, 100, 2)

    with pytest.raises(ValueError, match="cells"):
        calibration_module.NoiseCalibrator(detector).calibrate_from_noise(scans)


def test_calibrating_a_fixed_mode_detector_is_refused(monkeypatch) -> None:
    """Attaching a calibration would put a threshold_factor detector in both modes.

    cell_noise_ratios attaches nothing, so it stays available for inspecting such a detector.
    """
    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, "ca", target_pfa=None, threshold_factor=2.0)
    calibrator = calibration_module.NoiseCalibrator(detector)
    scans = _gamma_scans(np.random.default_rng(6), 120, 200, 2)

    with pytest.raises(ValueError, match="threshold_factor set"):
        calibrator.calibrate_from_noise(scans, min_tail_exceedances=100)
    assert detector.noise_calibration is None
    assert calibrator.cell_noise_ratios(scans).shape == (120 * 200,)


def test_mixed_frame_counts_empty_input_and_zero_noise_raise(monkeypatch) -> None:
    """Inconsistent, empty or degenerate noise scans and invalid tail_pfa are rejected."""
    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, "ca")
    calibrator = calibration_module.NoiseCalibrator(detector)

    with pytest.raises(ValueError, match="frame count"):
        calibrator.cell_noise_ratios([np.ones((50, 2)), np.ones((50, 3))])
    with pytest.raises(ValueError, match="empty"):
        calibrator.cell_noise_ratios([])
    with pytest.raises(ValueError, match="non-finite"):
        calibrator.cell_noise_ratios([np.zeros((50, 2))])
    with pytest.raises(ValueError, match="tail_pfa"):
        calibrator.calibrate_from_noise([np.ones((50, 2))], tail_pfa=1.0)


def test_extrapolation_warning_is_issued_once_per_pfa(monkeypatch) -> None:
    """Far extrapolation warns once per Pfa; the empirical region never warns."""
    algorithms, calibration_module = _load(monkeypatch)
    calibration = _small_calibration(
        algorithms, calibration_module, _make_detector(algorithms, "ca")
    )
    tiny_pfa = 1e-3 / calibration.num_cells  # 1000x beyond what the cells resolve

    with pytest.warns(UserWarning, match="extrapolated tail"):
        calibration.alpha(tiny_pfa)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calibration.alpha(tiny_pfa)
        calibration.alpha(1e-2)  # empirical region never warns


def test_alpha_is_monotone_and_continuous_across_the_tail_threshold(monkeypatch) -> None:
    """Stricter Pfa never lowers alpha, and the empirical/GPD hand-over has no jump."""
    algorithms, calibration_module = _load(monkeypatch)
    calibration = _small_calibration(
        algorithms, calibration_module, _make_detector(algorithms, "ca")
    )

    pfas = np.logspace(-1, -5, 60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alphas = np.array([calibration.alpha(p) for p in pfas])
    assert np.all(np.diff(alphas) >= 0)

    just_above = calibration.alpha(calibration.tail_pfa * 1.0001)
    just_below = calibration.alpha(calibration.tail_pfa * 0.9999)
    assert just_below == pytest.approx(just_above, rel=1e-3)


def test_beamformed_scans_from_sensor_data_skips_missing_and_empty(monkeypatch) -> None:
    """Only non-empty beamformed arrays are yielded from simulator output."""
    _, calibration_module = _load(monkeypatch)

    class SensorData:  # hashable stand-in for PassiveSonarSensorData
        def __init__(self, beamformed_data):
            self.beamformed_data = beamformed_data

    full = np.ones((4, 2))
    sensor_data_gen = [
        (0, {SensorData(full)}),
        (1, {SensorData(None)}),
        (2, {SensorData(np.empty((0, 0)))}),
    ]

    scans = list(calibration_module.beamformed_scans_from_sensor_data(sensor_data_gen))

    assert len(scans) == 1 and scans[0] is full


def test_beamformed_scans_progress_bar_routes_through_lazy_progress_bar(monkeypatch) -> None:
    """progress_bar=True should defer to passive._lazy_progress_bar, not wrap eagerly.

    The import is local (breaking a module cycle: .passive imports from .algorithms, which
    imports NoiseCalibration from this module), so this stubs bluepebble.detector.passive in
    sys.modules rather than pulling in the real Stone Soup dependencies passive.py needs.
    """
    _, calibration_module = _load(monkeypatch)

    calls = []

    def fake_lazy_progress_bar(iterable, desc, total):
        calls.append((desc, total))
        yield from iterable

    monkeypatch.setitem(
        sys.modules,
        "bluepebble.detector.passive",
        SimpleNamespace(_lazy_progress_bar=fake_lazy_progress_bar),
    )

    class SensorData:  # hashable stand-in for PassiveSonarSensorData
        def __init__(self, beamformed_data):
            self.beamformed_data = beamformed_data

    full = np.ones((4, 2))
    sensor_data_gen = [(0, {SensorData(full)})]

    scans = list(
        calibration_module.beamformed_scans_from_sensor_data(
            sensor_data_gen, progress_bar=True, total=1
        )
    )

    assert len(scans) == 1 and scans[0] is full
    assert calls == [("Calibrating noise", 1)]


# ---------------------------------------------------------------------------
# 5. Further correctness properties
# ---------------------------------------------------------------------------


def _ratios_of(detector, scans):
    return np.concatenate([np.divide(*detector._power_and_noise(scan)) for scan in scans])


@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_calibration_is_unbiased_across_independent_seeds(monkeypatch, detector_name) -> None:
    """Averaged over many independent calibration/held-out pairs, achieved Pfa equals target.

    A single seed can pass by luck; the ensemble mean exposes systematic bias (for example an
    off-by-one quantile or a tail threshold rate that is not the measured one).
    """
    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, detector_name)
    ratios_of_target = {1e-2: [], 1e-3: []}
    for seed in range(24):
        rng = np.random.default_rng(100 + seed)
        calibration = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(
            _correlated_scans(rng, 800, 128, 3)
        )
        held_out = _ratios_of(detector, _correlated_scans(rng, 800, 128, 3))
        for pfa, collected in ratios_of_target.items():
            collected.append(np.mean(held_out > calibration.alpha(pfa)) / pfa)

    assert np.mean(ratios_of_target[1e-2]) == pytest.approx(1.0, abs=0.05)
    assert np.mean(ratios_of_target[1e-3]) == pytest.approx(1.0, abs=0.12)


def test_calibration_transfers_exactly_across_ambient_level(monkeypatch) -> None:
    """CFAR ratios do not depend on absolute noise power, so neither does a calibration.

    Calibrating at one ambient level and detecting at a level 60 dB higher flags exactly the
    same cells as detecting at the calibration level.
    """
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(7)
    scans = _correlated_scans(rng, 300, 128, 3)
    held_out = _correlated_scans(rng, 50, 128, 3)
    reference = _make_detector(algorithms, "ca", consolidate_peaks=False)
    calibration = calibration_module.NoiseCalibrator(reference).calibrate_from_noise(scans)
    detector = _make_detector(
        algorithms, "ca", consolidate_peaks=False, target_pfa=1e-2, noise_calibration=calibration
    )

    for scan in held_out:
        np.testing.assert_array_equal(
            detector.detect(scan)[:, 0], detector.detect(scan * 2.0**20)[:, 0]
        )


def test_consolidated_false_alarms_never_exceed_the_calibrated_cell_rate(monkeypatch) -> None:
    """target_pfa calibrates per-cell crossings; consolidation can only remove detections."""
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(8)
    calibration = calibration_module.NoiseCalibrator(
        _make_detector(algorithms, "ca")
    ).calibrate_from_noise(_correlated_scans(rng, 1500, 128, 3))
    held_out = _correlated_scans(rng, 1500, 128, 3)
    common = dict(target_pfa=2e-2, noise_calibration=calibration)

    cells = _achieved_pfa(
        _make_detector(algorithms, "ca", consolidate_peaks=False, **common), held_out
    )
    consolidated = _achieved_pfa(
        _make_detector(algorithms, "ca", peak_distance=3, **common), held_out
    )

    assert cells == pytest.approx(2e-2, rel=0.15)
    assert consolidated <= cells


def test_bounded_tail_is_extrapolated_without_passing_its_endpoint(monkeypatch) -> None:
    """A negative-shape (bounded) tail gives alpha that approaches, never exceeds, the endpoint."""
    _, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(9)
    ratios = rng.beta(2.0, 5.0, 300_000) + 0.5  # bounded above at 1.5
    detector = SimpleNamespace(
        num_guard_cells=0,
        num_training_cells=1,
        circular=True,
        _power_and_noise=lambda data: (np.asarray(data)[:, 0], np.ones(len(data))),
    )
    calibration = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(
        [ratios[:, None]]
    )

    assert calibration.tail_shape < 0
    endpoint = calibration.tail_threshold - calibration.tail_scale / calibration.tail_shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alphas = [calibration.alpha(p) for p in (1e-4, 1e-6, 1e-9, 1e-12)]
    assert np.all(np.diff(alphas) >= 0)
    assert alphas[-1] <= endpoint + 1e-12
    from scipy.stats import beta as beta_dist

    true_alpha = 0.5 + beta_dist.ppf(1 - 1e-4, 2.0, 5.0)
    assert alphas[0] == pytest.approx(true_alpha, rel=0.02)


def test_duplicated_mirror_beams_do_not_bias_the_calibration(monkeypatch) -> None:
    """Mirrored half-plane steering duplicates every beam; ties must not shift the quantiles."""
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(10)
    half = _correlated_scans(rng, 1000, 64, 3)
    mirrored = [np.concatenate([scan, scan[::-1]]) for scan in half]
    held_out = [np.concatenate([scan, scan[::-1]]) for scan in _correlated_scans(rng, 1000, 64, 3)]
    detector = _make_detector(algorithms, "ca")

    calibration = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(mirrored)
    achieved = np.mean(_ratios_of(detector, held_out) > calibration.alpha(1e-2))

    assert achieved == pytest.approx(1e-2, rel=0.2)


def test_alpha_is_finite_and_monotone_at_extreme_pfa(monkeypatch) -> None:
    """No overflow, underflow or non-monotonic step for Pfa from 0.5 down to 1e-15."""
    algorithms, calibration_module = _load(monkeypatch)
    calibration = _small_calibration(
        algorithms, calibration_module, _make_detector(algorithms, "ca")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alphas = np.array([calibration.alpha(p) for p in np.logspace(np.log10(0.5), -15, 80)])

    assert np.all(np.isfinite(alphas))
    assert np.all(np.diff(alphas) >= 0)
    with pytest.raises(ValueError):
        calibration.alpha(0.0)
    with pytest.raises(ValueError):
        calibration.alpha(1.0)


def test_complex_amplitude_and_its_power_give_the_same_calibration(monkeypatch) -> None:
    """DAS time/frequency output is complex amplitude, MVDR is real power: same statistics."""
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(11)
    amplitudes = [
        (rng.normal(size=(128, 4)) + 1j * rng.normal(size=(128, 4))) / np.sqrt(2)
        for _ in range(200)
    ]
    detector = _make_detector(algorithms, "ca")

    from_amplitude = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(amplitudes)
    from_power = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(
        [np.abs(a) ** 2 for a in amplitudes]
    )

    np.testing.assert_allclose(from_amplitude.sorted_ratios, from_power.sorted_ratios)
    assert from_amplitude.alpha(1e-4) == pytest.approx(from_power.alpha(1e-4))


def test_calibrated_detector_survives_pickling_and_deep_copy(monkeypatch) -> None:
    """Calibrations are cached to disk and detectors are deep-copied by sweeps."""
    import pickle

    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, "ca")
    detector.noise_calibration = _small_calibration(algorithms, calibration_module, detector)
    expected = detector._alpha_for(2)

    restored = pickle.loads(pickle.dumps(detector.noise_calibration))
    assert restored.alpha(1e-2) == detector.noise_calibration.alpha(1e-2)
    assert copy.deepcopy(detector)._alpha_for(2) == expected


@pytest.mark.filterwarnings("ignore:.*threshold is only")
def test_frame_counts_within_tolerance_are_accepted_and_beyond_it_rejected(monkeypatch) -> None:
    """A continuous simulator's final scan runs long; a very different integration time is not."""
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(12)
    scans = [rng.exponential(size=(128, 1998)) for _ in range(59)]
    scans.append(rng.exponential(size=(128, 2398)))  # final scan, 20% longer
    detector = _make_detector(algorithms, "ca", consolidate_peaks=False)

    calibrator = calibration_module.NoiseCalibrator(detector)
    calibration = calibrator.calibrate_from_noise(scans, min_tail_exceedances=50)
    assert (calibration.min_num_frames, calibration.max_num_frames) == (1998, 2398)
    assert calibration.num_frames == 1998

    detector.detect(rng.exponential(size=(128, 2398)))
    detector.detect(rng.exponential(size=(128, 1700)))
    with pytest.raises(ValueError, match="frames"):
        detector.detect(rng.exponential(size=(128, 3100)))
    with pytest.raises(ValueError, match="frame count"):
        calibrator.calibrate_from_noise([np.ones((128, 9)), np.ones((128, 12))])


@pytest.mark.parametrize("detect_frames", [300, 500])
def test_alpha_is_referred_to_the_frame_count_being_detected_on(
    monkeypatch, detect_frames
) -> None:
    """A calibration at 400 frames stays accurate at 300 or 500 frames once referred.

    Without referral, integrating 25% more (or fewer) looks moves the achieved rate well away
    from target; referring the threshold's excess over 1 by sqrt(frames) brings it back.
    """
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(13)
    detector = _make_detector(algorithms, "ca")
    calibration = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(
        [rng.exponential(size=(128, 400)) for _ in range(1200)]
    )
    held_out = _ratios_of(
        detector, [rng.exponential(size=(128, detect_frames)) for _ in range(1200)]
    )

    for pfa, tolerance in ((1e-2, 1.12), (1e-3, 1.25)):
        referred = np.mean(held_out > calibration.alpha(pfa, detect_frames)) / pfa
        unreferred = np.mean(held_out > calibration.alpha(pfa)) / pfa
        assert 1 / tolerance < referred < tolerance, (pfa, referred)
        assert abs(np.log(unreferred)) > 2 * abs(np.log(referred)), (pfa, unreferred, referred)


def test_calibration_pooled_over_mixed_frame_counts_is_unbiased(monkeypatch) -> None:
    """Scans of differing length are referred to the median before pooling.

    Pooling 320- and 400-frame scans unreferred leaves achieved Pfa at about 0.5-0.65x target
    at the reference frame count; referred, it stays within a few percent.
    """
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(14)
    detector = _make_detector(algorithms, "ca")
    frame_counts = rng.choice([320, 400], size=1500)
    calibration = calibration_module.NoiseCalibrator(detector).calibrate_from_noise(
        [rng.exponential(size=(128, m)) for m in frame_counts]
    )
    reference = calibration.num_frames
    held_out = _ratios_of(detector, [rng.exponential(size=(128, reference)) for _ in range(1500)])

    assert np.mean(held_out > calibration.alpha(1e-2)) / 1e-2 == pytest.approx(1.0, abs=0.1)
    assert np.mean(held_out > calibration.alpha(1e-3)) / 1e-3 == pytest.approx(1.0, abs=0.15)


@pytest.mark.filterwarnings("ignore:.*threshold is only")
def test_calibrated_detect_refers_alpha_to_each_scans_frame_count(monkeypatch) -> None:
    """End to end: calibrated at 400 frames, detect() on 500-frame scans still meets target."""
    algorithms, calibration_module = _load(monkeypatch)
    rng = np.random.default_rng(15)
    calibration = calibration_module.NoiseCalibrator(
        _make_detector(algorithms, "ca")
    ).calibrate_from_noise([rng.exponential(size=(128, 400)) for _ in range(1200)])
    detector = _make_detector(
        algorithms, "ca", consolidate_peaks=False, noise_calibration=calibration
    )

    achieved = _achieved_pfa(detector, [rng.exponential(size=(128, 500)) for _ in range(1200)])

    assert achieved == pytest.approx(1e-2, rel=0.12)


def test_band_label_selects_one_band_of_multiband_sensor_data(monkeypatch) -> None:
    """Multiband output is calibrated per band; unlabelled multiband use fails clearly."""
    _, calibration_module = _load(monkeypatch)

    class SensorData:
        def __init__(self, beamformed_data, band_labels=None):
            self.beamformed_data = beamformed_data
            self.band_labels = band_labels

    multiband = np.stack([np.full((4, 2), 1.0), np.full((4, 2), 2.0)])
    generator = [(0, {SensorData(multiband, ["low", "high"])})]

    scans = list(
        calibration_module.beamformed_scans_from_sensor_data(generator, band_label="high")
    )
    np.testing.assert_array_equal(scans[0], multiband[1])
    with pytest.raises(ValueError, match="pass band_label"):
        list(calibration_module.beamformed_scans_from_sensor_data(generator))
    with pytest.raises(ValueError, match="not one of"):
        list(calibration_module.beamformed_scans_from_sensor_data(generator, band_label="mid"))
    with pytest.raises(ValueError, match="single-band"):
        list(
            calibration_module.beamformed_scans_from_sensor_data(
                [(0, {SensorData(np.ones((4, 2)))})], band_label="low"
            )
        )


@pytest.mark.parametrize(
    "param_name, value", [("num_guard_cells", 3), ("num_training_cells", 12), ("circular", 0)]
)
def test_sweep_clone_rejects_parameters_that_invalidate_the_calibration(
    monkeypatch, param_name, value
) -> None:
    """Sweeping a calibrated detector's window or edge handling fails at clone time."""
    algorithms, calibration_module = _load(monkeypatch)
    metrics = load_package_module_from_repo(
        "bluepebble/detector/metrics.py", "bluepebble.detector.metrics"
    )
    detector = _make_detector(algorithms, "ca")
    detector.noise_calibration = _small_calibration(algorithms, calibration_module, detector)
    spec = metrics.SweepSpec(detector=detector, param_name=param_name, param_values=[value])

    with pytest.raises(ValueError, match="invalidates the detector's noise_calibration"):
        metrics._clone_detector_with_param(spec, value)


def test_sweep_clone_rejects_rank_and_allows_calibration_independent_parameters(
    monkeypatch,
) -> None:
    """OS rank is part of the calibration; target_pfa and peak_distance are not."""
    algorithms, calibration_module = _load(monkeypatch)
    metrics = load_package_module_from_repo(
        "bluepebble/detector/metrics.py", "bluepebble.detector.metrics"
    )
    detector = _make_detector(algorithms, "os")
    detector.noise_calibration = _small_calibration(algorithms, calibration_module, detector)

    with pytest.raises(ValueError, match="invalidates"):
        metrics._clone_detector_with_param(
            metrics.SweepSpec(detector=detector, param_name="rank", param_values=[12]), 12
        )
    for name, value in (("target_pfa", 1e-3), ("peak_distance", 4)):
        clone = metrics._clone_detector_with_param(
            metrics.SweepSpec(detector=detector, param_name=name, param_values=[value]), value
        )
        assert clone.noise_calibration is not None
        clone.detect(np.random.default_rng(0).exponential(size=(200, 2)))


def test_replaced_calibration_is_kept_alive_until_the_cache_is_revalidated(monkeypatch) -> None:
    """The cache signature holds the calibration itself, so its id cannot be reused.

    Were the signature an id(), a replaced calibration could be freed and a new one allocated
    at the same address before the next detection, and the stale alpha would be reused.
    """
    import gc
    import weakref

    algorithms, calibration_module = _load(monkeypatch)
    detector = _make_detector(algorithms, "ca")
    calibration = _small_calibration(algorithms, calibration_module, detector)
    detector.noise_calibration = calibration
    detector._alpha_for(2)
    reference = weakref.ref(calibration)

    detector.noise_calibration = None
    del calibration
    gc.collect()
    assert reference() is not None

    # Revalidating the cache against a replacement calibration is what drops the last
    # reference to the old one. A detection with no calibration raises before revalidating, so
    # the old one outlives it; that is harmless, since it is released on the next valid call.
    replacement = _small_calibration(algorithms, calibration_module, detector)
    assert detector._alpha_for(2) == replacement.alpha(detector.target_pfa, 2)
    gc.collect()
    assert reference() is None
