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


def test_os_cfar_log_pfa_rank_one_matches_simple_formula(monkeypatch) -> None:
    """Rank 1 is a single spacing term: Pfa(alpha) = N / (alpha + N)."""
    algorithms = _load_detector_algorithms(monkeypatch)

    log_pfa = algorithms._os_cfar_log_pfa(alpha=2.0, num_training_total=5, rank=1)

    assert log_pfa == pytest.approx(np.log(5 / (2.0 + 5)))


def test_solve_os_cfar_alpha_single_look_inverts_log_pfa(monkeypatch) -> None:
    """The solved alpha should drive _os_cfar_log_pfa back to log(target_pfa)."""
    algorithms = _load_detector_algorithms(monkeypatch)
    target_pfa = 0.05

    alpha = algorithms.solve_os_cfar_alpha_single_look(target_pfa, num_training_total=5, rank=2)

    assert algorithms._os_cfar_log_pfa(alpha, 5, 2) == pytest.approx(np.log(target_pfa))


@pytest.mark.parametrize("target_pfa", [0.0, 1.0, -0.1, 1.5])
def test_solve_os_cfar_alpha_single_look_rejects_invalid_pfa(monkeypatch, target_pfa) -> None:
    """Pfa outside (0, 1) is not a valid probability and should be rejected."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="target_pfa"):
        algorithms.solve_os_cfar_alpha_single_look(target_pfa, num_training_total=5, rank=1)


def test_solve_ca_cfar_alpha_single_look_matches_closed_form(monkeypatch) -> None:
    """num_frames=1 should reduce to the classic N*(Pfa**(-1/N) - 1) CA-CFAR formula."""
    algorithms = _load_detector_algorithms(monkeypatch)
    target_pfa = 0.01
    num_training_total = 8

    alpha = algorithms.solve_ca_cfar_alpha(target_pfa, num_training_total, num_frames=1)

    closed_form = num_training_total * (target_pfa ** (-1 / num_training_total) - 1)
    assert alpha == pytest.approx(closed_form)


@pytest.mark.parametrize("target_pfa", [0.0, 1.0, -0.1, 1.5])
def test_solve_ca_cfar_alpha_rejects_invalid_pfa(monkeypatch, target_pfa) -> None:
    """Pfa outside (0, 1) is not a valid probability and should be rejected."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="target_pfa"):
        algorithms.solve_ca_cfar_alpha(target_pfa, num_training_total=8, num_frames=1)


def test_calibrate_os_cfar_alpha_mc_matches_closed_form_at_single_look(monkeypatch) -> None:
    """At num_frames=1 the Monte Carlo calibration should agree with the exact closed form."""
    algorithms = _load_detector_algorithms(monkeypatch)
    target_pfa = 0.05
    num_training_total = 5
    rank = 2
    exact_alpha = algorithms.solve_os_cfar_alpha_single_look(target_pfa, num_training_total, rank)

    mc_alpha = algorithms.calibrate_os_cfar_alpha_mc(
        target_pfa,
        num_training_total,
        rank,
        num_frames=1,
        num_trials=20_000,
        rng=np.random.default_rng(42),
    )

    assert mc_alpha == pytest.approx(exact_alpha, rel=0.15)


def test_cacfar_detector_validates_num_guard_cells(monkeypatch) -> None:
    """Negative guard-cell counts are not physically meaningful."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="num_guard_cells"):
        algorithms.CACFARDetector(num_guard_cells=-1, num_training_cells=1, target_pfa=0.1)


def test_cacfar_detector_validates_num_training_cells(monkeypatch) -> None:
    """At least one training cell per side is required to estimate the noise floor."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="num_training_cells"):
        algorithms.CACFARDetector(num_guard_cells=0, num_training_cells=0, target_pfa=0.1)


def test_cacfar_detector_validates_target_pfa(monkeypatch) -> None:
    """target_pfa must be a valid probability."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="target_pfa"):
        algorithms.CACFARDetector(num_guard_cells=0, num_training_cells=1, target_pfa=1.5)


@pytest.mark.parametrize("kwarg", ["peak_distance", "wrap_pad_width"])
def test_cacfar_detector_validates_positive_padding_params(monkeypatch, kwarg) -> None:
    """peak_distance and wrap_pad_width must both be at least 1 bearing bin."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match=kwarg):
        algorithms.CACFARDetector(
            num_guard_cells=0, num_training_cells=1, target_pfa=0.5, **{kwarg: 0}
        )


def test_oscfar_detector_validates_rank(monkeypatch) -> None:
    """Rank must stay within the available training-cell count."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="Rank"):
        algorithms.OSCFARDetector(
            num_guard_cells=0, num_training_cells=2, rank=5, target_pfa=0.1
        )


def test_oscfar_detector_warns_on_low_rank(monkeypatch) -> None:
    """A rank below half the training-cell count risks underestimating the noise floor."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.warns(UserWarning, match="likely underestimates the noise floor"):
        algorithms.OSCFARDetector(
            num_guard_cells=0, num_training_cells=2, rank=1, target_pfa=0.1
        )


# --------------------------------------------------------------------------
# CA-CFAR detection on beamformed data (shape: num_beams x num_frames)
# --------------------------------------------------------------------------


def test_cacfar_detector_detects_isolated_peak(monkeypatch) -> None:
    """CA-CFAR should detect a strong isolated beam above a zero-power background."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0, num_training_cells=1, target_pfa=0.5
    )

    detections = detector.detect(np.array([[0.0], [0.0], [20.0], [0.0], [0.0]]))

    assert detections.shape == (1, 2)
    assert detections[0, 0] == 2.0


def test_cacfar_detector_returns_no_detections_on_flat_signal(monkeypatch) -> None:
    """A uniform (all-zero) background should produce no detections."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0, num_training_cells=2, target_pfa=0.3
    )

    detections = detector.detect(np.zeros((7, 1)))

    assert detections.shape == (0, 2)


def test_cacfar_default_is_circular_and_detects_edge_peak(monkeypatch) -> None:
    """The default (circular=True) mode should wrap training cells across the array edge."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0, num_training_cells=1, target_pfa=0.5
    )

    detections = detector.detect(np.array([[20.0], [0.0], [0.0], [0.0], [0.0]]))

    assert detections.shape == (1, 2)
    assert detections[0, 0] == 0.0


def test_cacfar_non_circular_mode_folds_edge_peak_into_its_own_noise_estimate(
    monkeypatch,
) -> None:
    """With circular=False, edge padding replicates the boundary cell (the peak itself),

    inflating the noise estimate right at the edge and suppressing the detection --
    unlike circular=True, which wraps in independent (zero-power) cells instead.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0, num_training_cells=1, target_pfa=0.5, circular=False
    )

    detections = detector.detect(np.array([[20.0], [0.0], [0.0], [0.0], [0.0]]))

    assert detections.shape == (0, 2)


def test_cacfar_handles_training_window_larger_than_input(monkeypatch) -> None:
    """Short inputs should not crash even when the training window exceeds the data length."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0, num_training_cells=4, target_pfa=0.5
    )

    detections = detector.detect(np.array([[0.0], [20.0], [0.0]]))

    assert detections.shape == (1, 2)
    assert detections[0, 0] == 1.0


def test_cacfar_alpha_is_memoized_per_num_frames(monkeypatch) -> None:
    """Alpha calibration should be cached per num_frames rather than recomputed every call."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0, num_training_cells=1, target_pfa=0.5
    )
    data = np.array([[0.0], [0.0], [20.0], [0.0], [0.0]])

    assert detector._alpha_cache == {}
    detector.detect(data)
    assert list(detector._alpha_cache.keys()) == [1]
    cached_alpha = detector._alpha_cache[1]
    detector.detect(data)
    assert detector._alpha_cache == {1: cached_alpha}


def test_snr_map_matches_detect_output_values(monkeypatch) -> None:
    """snr_map should report the same per-beam SNR values that detect() surfaces."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=0, num_training_cells=1, target_pfa=0.5
    )
    data = np.array([[0.0], [0.0], [20.0], [0.0], [0.0]])

    snr_db = detector.snr_map(data)
    detections = detector.detect(data)

    assert detections[0, 1] == pytest.approx(snr_db[2])


# --------------------------------------------------------------------------
# OS-CFAR detection on beamformed data
# --------------------------------------------------------------------------


def test_oscfar_detector_detects_isolated_peak(monkeypatch) -> None:
    """OS-CFAR should detect an isolated strong beam on a zero-power background."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=0, num_training_cells=1, rank=1, target_pfa=0.5
    )

    detections = detector.detect(np.array([[0.0], [0.0], [20.0], [0.0], [0.0]]))

    assert detections.shape == (1, 2)
    assert detections[0, 0] == 2.0


def test_oscfar_detector_detects_edge_peak_with_wrap_processing(monkeypatch) -> None:
    """OS-CFAR should detect strong peaks at the array boundary under the default wrap mode."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=0, num_training_cells=1, rank=1, target_pfa=0.5
    )

    detections = detector.detect(np.array([[20.0], [0.0], [0.0], [0.0], [0.0]]))

    assert detections.shape == (1, 2)
    assert detections[0, 0] == 0.0


def test_oscfar_consolidate_peaks_suppresses_closer_smaller_peak(monkeypatch) -> None:
    """Two candidate peaks closer than peak_distance should collapse to the taller one."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        rank=1,
        target_pfa=0.9,
        peak_distance=3,
        circular=False,
    )

    detections = detector.detect(np.array([[0.0], [5.0], [0.0], [4.0], [0.0]]))

    assert detections.shape == (1, 2)
    assert detections[0, 0] == 1.0


def test_oscfar_consolidate_peaks_keeps_both_when_far_enough_apart(monkeypatch) -> None:
    """Candidate peaks at or beyond peak_distance should both survive consolidation."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        rank=1,
        target_pfa=0.9,
        peak_distance=3,
        circular=False,
    )

    detections = detector.detect(np.array([[0.0], [5.0], [0.0], [0.0], [0.0], [4.0], [0.0]]))

    np.testing.assert_array_equal(detections[:, 0], [1.0, 5.0])


def test_oscfar_peak_prominence_suppresses_low_prominence_shoulder(monkeypatch) -> None:
    """peak_prominence should drop a low-contrast shoulder that peak_distance alone would keep."""
    algorithms = _load_detector_algorithms(monkeypatch)
    snr_db = np.array([0.0, 10.0, 4.0, 6.0, 0.0])
    candidate_mask = np.array([False, True, True, True, False])

    permissive = algorithms.OSCFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        rank=1,
        target_pfa=0.9,
        peak_distance=1,
        circular=False,
    )
    strict = algorithms.OSCFARDetector(
        num_guard_cells=0,
        num_training_cells=1,
        rank=1,
        target_pfa=0.9,
        peak_distance=1,
        peak_prominence=3.0,
        circular=False,
    )

    np.testing.assert_array_equal(
        permissive._consolidate_peaks(snr_db, candidate_mask, 5), [1, 3]
    )
    np.testing.assert_array_equal(strict._consolidate_peaks(snr_db, candidate_mask, 5), [1])


# --------------------------------------------------------------------------
# _directional_power (real vs complex beamformed_data)
# --------------------------------------------------------------------------


def test_directional_power_squares_complex_amplitude(monkeypatch) -> None:
    """Complex input is raw amplitude; power is the squared magnitude."""
    algorithms = _load_detector_algorithms(monkeypatch)
    data = np.array([[3.0 + 4.0j]])

    result = algorithms._directional_power(data)

    np.testing.assert_allclose(result, [[25.0]])


def test_directional_power_passes_real_input_through_unsquared(monkeypatch) -> None:
    """Real input is already power (e.g. an MVDR beamformer's output) -- must not be re-squared."""
    algorithms = _load_detector_algorithms(monkeypatch)
    data = np.array([[7.0]])

    result = algorithms._directional_power(data)

    np.testing.assert_allclose(result, [[7.0]])


def test_detect_gives_identical_results_for_real_power_and_complex_amplitude_input(
    monkeypatch,
) -> None:
    """detect() must treat already-real power the same as the amplitude it was computed from.

    Regression test: BeamformedData is deliberately either complex amplitude
    (DelayAndSumBeamformer in 'time'/'frequency' domain) or already-real power
    (MinimumVarianceDistortionlessResponseBeamformer, and DelayAndSumBeamformer in
    'broadband_power' domain -- see bluepebble.types.sensordata.BeamformedData). Before
    _directional_power existed, detect() applied |x|**2 unconditionally, so real power
    (already |amplitude|**2) was squared a second time, silently breaking the calibrated
    Pfa for those beamformers.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=2, num_training_cells=8, target_pfa=0.1, peak_distance=1, circular=True
    )
    rng = np.random.default_rng(7)
    amplitude = rng.normal(size=(100, 1), scale=np.sqrt(0.5)) + 1j * rng.normal(
        size=(100, 1), scale=np.sqrt(0.5)
    )
    real_power = np.abs(amplitude) ** 2

    from_complex = detector.detect(amplitude)
    from_real_power = detector.detect(real_power)

    np.testing.assert_array_equal(from_complex, from_real_power)


# --------------------------------------------------------------------------
# effective_looks_per_frame (band-integrated per-frame statistics)
# --------------------------------------------------------------------------


def test_effective_looks_per_frame_defaults_to_the_exponential_model(monkeypatch) -> None:
    """K=1 is exactly the classic Exponential(1)-per-look model, bit for bit.

    Gamma(1, 1) IS Exponential(1), so passing the default explicitly must not perturb the
    calibration at all -- only the generator call changes, and it is fed the same seed.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    kwargs = dict(target_pfa=0.05, num_training_total=20, rank=15, num_frames=4)

    default = algorithms.calibrate_os_cfar_alpha_mc(
        **kwargs, num_trials=20_000, rng=np.random.default_rng(3)
    )
    explicit = algorithms.calibrate_os_cfar_alpha_mc(
        **kwargs, num_trials=20_000, rng=np.random.default_rng(3), effective_looks_per_frame=1.0
    )

    assert default == pytest.approx(explicit, rel=1e-12)


def test_alpha_decreases_as_effective_looks_per_frame_rises(monkeypatch) -> None:
    """More looks integrated per frame means a tighter null, so a lower threshold suffices."""
    algorithms = _load_detector_algorithms(monkeypatch)
    alphas = [
        algorithms.calibrate_os_cfar_alpha_mc(
            0.01, 20, 15, num_frames=4, num_trials=40_000,
            rng=np.random.default_rng(11), effective_looks_per_frame=k,
        )
        for k in (1.0, 5.0, 25.0, 100.0)
    ]

    assert alphas == sorted(alphas, reverse=True)
    # The K=1 threshold is far above the K=100 one; this gap is the whole calibration bug.
    assert alphas[0] > 1.5 * alphas[-1]


@pytest.mark.parametrize("effective_looks", [1.0, 8.0, 30.0])
def test_calibrated_alpha_achieves_target_pfa_on_matching_data(monkeypatch, effective_looks):
    """The core claim: alpha calibrated at K delivers target_pfa on Gamma(K, 1/K) data."""
    algorithms = _load_detector_algorithms(monkeypatch)
    target_pfa, num_training_total, rank, num_frames = 0.05, 20, 15, 4

    alpha = algorithms.calibrate_os_cfar_alpha_mc(
        target_pfa, num_training_total, rank, num_frames,
        num_trials=200_000, rng=np.random.default_rng(5),
        effective_looks_per_frame=effective_looks,
    )

    # Independent draws from the same model, with a different seed than the calibration used.
    rng = np.random.default_rng(99)
    trials = 200_000
    shape, scale = effective_looks, 1.0 / effective_looks
    ref = rng.gamma(shape, scale, size=(trials, num_training_total, num_frames)).mean(axis=2)
    ref.sort(axis=1)
    cut = rng.gamma(shape, scale, size=(trials, num_frames)).mean(axis=1)

    achieved = float(np.mean(cut > alpha * ref[:, rank - 1]))

    assert achieved == pytest.approx(target_pfa, rel=0.06)


def test_idealised_alpha_badly_overshoots_on_band_integrated_data(monkeypatch) -> None:
    """Regression test for the calibration bug this parameter exists to fix.

    Calibrating at K=1 but running on K=30 data does not merely bias Pfa slightly: the null
    distribution is so much tighter than assumed that the threshold admits essentially nothing.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    target_pfa, num_training_total, rank, num_frames, k_true = 0.05, 20, 15, 4, 30.0

    alpha_idealised = algorithms.calibrate_os_cfar_alpha_mc(
        target_pfa, num_training_total, rank, num_frames,
        num_trials=100_000, rng=np.random.default_rng(5),
    )

    rng = np.random.default_rng(99)
    trials = 100_000
    ref = rng.gamma(k_true, 1.0 / k_true, size=(trials, num_training_total, num_frames)).mean(
        axis=2
    )
    ref.sort(axis=1)
    cut = rng.gamma(k_true, 1.0 / k_true, size=(trials, num_frames)).mean(axis=1)

    achieved = float(np.mean(cut > alpha_idealised * ref[:, rank - 1]))

    assert achieved < target_pfa / 100


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_oscfar_detector_rejects_non_positive_effective_looks(monkeypatch, bad_value) -> None:
    """A Gamma shape parameter must be positive."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="effective_looks_per_frame"):
        algorithms.OSCFARDetector(
            num_guard_cells=0, num_training_cells=10, rank=8, target_pfa=0.1,
            effective_looks_per_frame=bad_value,
        )


def test_oscfar_detector_passes_effective_looks_through_to_calibration(monkeypatch) -> None:
    """The detector property must reach calibrate_os_cfar_alpha_mc, not be silently dropped."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=0, num_training_cells=10, rank=15, target_pfa=0.05,
        rng=np.random.default_rng(17), effective_looks_per_frame=12.0,
    )

    expected = algorithms.calibrate_os_cfar_alpha_mc(
        0.05, 20, 15, num_frames=4, num_trials=detector.mc_trials,
        rng=np.random.default_rng(17), effective_looks_per_frame=12.0,
    )

    assert detector._alpha_for(4) == pytest.approx(expected)


def test_single_frame_closed_form_is_skipped_when_looks_are_integrated(monkeypatch) -> None:
    """The Rohling closed form assumes Exponential(1) cells, so K != 1 must bypass it.

    At num_frames == 1 the detector normally takes an exact closed-form shortcut. That form is
    only valid for single-look exponential statistics, so band-integrated data has to fall
    through to Monte Carlo even though there is just one frame.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    closed_form = algorithms.solve_os_cfar_alpha_single_look(0.05, 20, 15)

    plain = algorithms.OSCFARDetector(
        num_guard_cells=0, num_training_cells=10, rank=15, target_pfa=0.05,
        rng=np.random.default_rng(17),
    )
    integrated = algorithms.OSCFARDetector(
        num_guard_cells=0, num_training_cells=10, rank=15, target_pfa=0.05,
        rng=np.random.default_rng(17), effective_looks_per_frame=20.0,
    )

    assert plain._alpha_for(1) == pytest.approx(closed_form)
    assert integrated._alpha_for(1) != pytest.approx(closed_form)
    assert integrated._alpha_for(1) < plain._alpha_for(1)


def test_ca_cfar_alpha_generalises_to_integrated_looks(monkeypatch) -> None:
    """The Beta closed form extends exactly to K looks per frame via Beta(K*M, N*K*M)."""
    algorithms = _load_detector_algorithms(monkeypatch)

    # K and M enter the Beta parameters only as their product, so these must coincide.
    assert algorithms.solve_ca_cfar_alpha(0.05, 16, 6, 1.0) == pytest.approx(
        algorithms.solve_ca_cfar_alpha(0.05, 16, 2, 3.0)
    )
    # Default stays the classic single-look result.
    assert algorithms.solve_ca_cfar_alpha(0.05, 16, 4) == pytest.approx(
        algorithms.solve_ca_cfar_alpha(0.05, 16, 4, 1.0)
    )
    # More looks integrated -> tighter null -> lower threshold.
    alphas = [algorithms.solve_ca_cfar_alpha(0.01, 16, 4, k) for k in (1.0, 5.0, 25.0)]
    assert alphas == sorted(alphas, reverse=True)


def test_ca_cfar_alpha_rejects_non_positive_effective_looks(monkeypatch) -> None:
    """A Gamma/Beta shape parameter must be positive."""
    algorithms = _load_detector_algorithms(monkeypatch)

    with pytest.raises(ValueError, match="effective_looks_per_frame"):
        algorithms.solve_ca_cfar_alpha(0.05, 16, 4, 0.0)


# --------------------------------------------------------------------------
# signal_looks_per_frame (target bandwidth vs processing bandwidth)
# --------------------------------------------------------------------------
