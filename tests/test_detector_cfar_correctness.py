"""Independent correctness checks for the CFAR detectors and their Pfa calibration.

Complements ``test_detector_algorithms_validation.py``, which pins single-look literature values
and single-frame end-to-end Pfa. The checks here target what that file leaves open:

1. Noise-floor estimators against a brute-force, loop-based reference (guard/training layout,
   circular wrap, edge padding, windows longer than the data).
2. CA-CFAR alpha against numerical quadrature and direct simulation for multi-frame and
   non-integer effective looks, including fewer than one look per frame.
3. OS-CFAR alpha against an independent quadrature of the multi-look order statistic, and
   against the single-look closed form.
4. End-to-end ``detect()`` Pfa for multi-frame, band-integrated (Gamma) data, circular and
   non-circular, for both detectors.
5. The CFAR property as an exact invariant, and calibration staying in step with parameters
   changed after a detector has already been used.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from .support import (
    ca_pfa_by_quadrature,
    install_fake_stonesoup,
    install_repo_package,
    load_package_module_from_repo,
    os_pfa_by_quadrature,
)


def _load_detector_algorithms(monkeypatch):
    """Load detector algorithms with minimal package scaffolding."""
    install_fake_stonesoup(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble", "bluepebble")
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    return load_package_module_from_repo(
        "bluepebble/detector/algorithms.py",
        "bluepebble.detector.algorithms",
    )


def _load_detector_algorithms_and_calibration(monkeypatch):
    """Load algorithms.py and calibration.py, with minimal scaffolding."""
    algorithms = _load_detector_algorithms(monkeypatch)
    calibration = load_package_module_from_repo(
        "bluepebble/detector/calibration.py",
        "bluepebble.detector.calibration",
    )
    return algorithms, calibration


def _load_fluctuation_models(monkeypatch):
    """Load _theory.py and the algorithms.py it needs, with minimal scaffolding."""
    algorithms = _load_detector_algorithms(monkeypatch)
    fluctuation_models = load_package_module_from_repo(
        "bluepebble/detector/_theory.py",
        "bluepebble.detector._theory",
    )
    return algorithms, fluctuation_models


def _training_indices(cut, num_beams, num_guard_cells, num_training_cells, circular):
    """Reference-cell indices for one CUT, written out explicitly.

    Circular data wraps modulo ``num_beams``. Non-circular data replicates the boundary cell,
    which is what ``np.pad(mode="edge")`` does, so out-of-range indices clamp to the edge.
    """
    offsets = np.arange(num_guard_cells + 1, num_guard_cells + num_training_cells + 1)
    raw = np.concatenate([cut - offsets[::-1], cut + offsets])
    if circular:
        return raw % num_beams
    return np.clip(raw, 0, num_beams - 1)


def _gamma_frames(rng, shape, looks_per_frame):
    """Unit-mean per-frame power with ``looks_per_frame`` effective Exponential(1) looks."""
    return rng.gamma(shape=looks_per_frame, scale=1.0 / looks_per_frame, size=shape)


def _calibrate_with_matching_noise(
    detector,
    calibration_module,
    num_beams,
    num_frames=1,
    looks_per_frame=1.0,
    num_scans=None,
    seed=0,
    **kwargs,
):
    """Calibrate ``detector`` on synthetic noise-only scans with the given look structure.

    Defaults to enough scans for the default tail fit (``min_tail_exceedances / tail_pfa``
    pooled cells, about 20,000), plus margin, regardless of ``num_beams``.
    """
    if num_scans is None:
        num_scans = max(200, -(-22_000 // num_beams))  # ceil division
    rng = np.random.default_rng(seed)
    scans = [
        _gamma_frames(rng, (num_beams, num_frames), looks_per_frame) for _ in range(num_scans)
    ]
    return calibration_module.NoiseCalibrator(detector).calibrate_from_noise(scans, **kwargs)


# ---------------------------------------------------------------------------
# 1. Noise-floor estimators against a brute-force reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize(
    "num_beams, num_guard_cells, num_training_cells",
    [(40, 0, 1), (40, 2, 5), (40, 3, 10), (9, 2, 6), (5, 0, 8)],
)
def test_ca_noise_floor_matches_brute_force_mean(
    monkeypatch, circular, num_beams, num_guard_cells, num_training_cells
) -> None:
    """Every cell's CA noise estimate is the mean of exactly its reference cells."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=num_guard_cells,
        num_training_cells=num_training_cells,
        target_pfa=1e-3,
        circular=circular,
    )
    power = np.random.default_rng(num_beams + num_training_cells).exponential(size=num_beams)

    expected = np.array(
        [
            power[
                _training_indices(cut, num_beams, num_guard_cells, num_training_cells, circular)
            ].mean()
            for cut in range(num_beams)
        ]
    )

    np.testing.assert_allclose(detector._local_noise_floor(power), expected, rtol=1e-12)


@pytest.mark.parametrize("circular", [True, False])
@pytest.mark.parametrize(
    "num_beams, num_guard_cells, num_training_cells, rank",
    [(40, 0, 1, 1), (40, 2, 5, 8), (40, 3, 10, 15), (40, 3, 10, 20), (9, 2, 6, 9)],
)
def test_os_noise_floor_matches_brute_force_order_statistic(
    monkeypatch, circular, num_beams, num_guard_cells, num_training_cells, rank
) -> None:
    """Every cell's OS noise estimate is the rank-th smallest of exactly its reference cells."""
    algorithms = _load_detector_algorithms(monkeypatch)
    with pytest.warns(UserWarning) if rank < num_training_cells else _no_warning():
        detector = algorithms.OSCFARDetector(
            num_guard_cells=num_guard_cells,
            num_training_cells=num_training_cells,
            rank=rank,
            target_pfa=1e-3,
            circular=circular,
        )
    power = np.random.default_rng(rank + num_beams).exponential(size=num_beams)

    expected = np.array(
        [
            np.sort(
                power[
                    _training_indices(
                        cut, num_beams, num_guard_cells, num_training_cells, circular
                    )
                ]
            )[rank - 1]
            for cut in range(num_beams)
        ]
    )

    np.testing.assert_allclose(detector._local_noise_floor(power), expected, rtol=1e-12)


class _no_warning:  # noqa: N801 - used like pytest.warns in a conditional context manager
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_detect_thresholds_frame_averaged_power_against_alpha_times_noise_floor(
    monkeypatch,
) -> None:
    """detect() flags exactly the cells whose frame-mean power exceeds alpha * noise floor."""
    algorithms, calibration = _load_detector_algorithms_and_calibration(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=1,
        num_training_cells=4,
        target_pfa=0.05,
        consolidate_peaks=False,
    )
    _calibrate_with_matching_noise(detector, calibration, num_beams=64, num_frames=6, seed=9)
    data = np.random.default_rng(5).exponential(size=(64, 6))

    power = data.mean(axis=1)
    threshold = detector._alpha_for(6) * detector._local_noise_floor(power)
    expected = np.flatnonzero(power > threshold)

    np.testing.assert_array_equal(detector.detect(data)[:, 0].astype(int), expected)


# ---------------------------------------------------------------------------
# 2. CA-CFAR alpha: independent quadrature and simulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_training_total, num_frames, looks_per_frame, target_pfa",
    [
        (8, 1, 1.0, 1e-3),
        (20, 5, 1.0, 1e-4),
        (32, 10, 2.5, 1e-6),
        (16, 3, 3.7, 1e-2),
        (24, 40, 0.1, 1e-3),  # correlated frames: 4 effective looks from 40 frames
    ],
)
def test_ca_cfar_alpha_achieves_target_pfa_by_quadrature(
    monkeypatch, num_training_total, num_frames, looks_per_frame, target_pfa
) -> None:
    """The Beta closed form inverts an independently integrated Pfa, to quadrature accuracy."""
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    alpha = fluctuation_models.solve_ca_cfar_alpha(
        target_pfa, num_training_total, num_frames, effective_looks_per_frame=looks_per_frame
    )

    pfa = ca_pfa_by_quadrature(alpha, num_training_total, num_frames * looks_per_frame)

    assert pfa == pytest.approx(target_pfa, rel=1e-5)


@pytest.mark.parametrize("looks_per_frame", [0.5, 1.0, 4.0])
def test_ca_cfar_alpha_achieves_target_pfa_by_direct_simulation(
    monkeypatch, looks_per_frame
) -> None:
    """Simulating actual frames (not the closed-form cell distribution) reproduces target_pfa.

    Per-frame Gamma(K) samples are averaged over M frames, so this also checks that K and M
    enter the calibration as the product K * M.
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    num_training_total, num_frames, target_pfa = 12, 4, 1e-2
    alpha = fluctuation_models.solve_ca_cfar_alpha(
        target_pfa, num_training_total, num_frames, effective_looks_per_frame=looks_per_frame
    )

    rng = np.random.default_rng(21)
    trials = 200_000
    refs = _gamma_frames(rng, (trials, num_training_total, num_frames), looks_per_frame)
    cut = _gamma_frames(rng, (trials, num_frames), looks_per_frame).mean(axis=1)
    empirical = np.mean(cut > alpha * refs.mean(axis=2).mean(axis=1))

    # ~2000 expected exceedances: 5 sigma is ~11%.
    assert empirical == pytest.approx(target_pfa, rel=0.12)


def test_ca_cfar_alpha_is_monotone_in_pfa_and_looks(monkeypatch) -> None:
    """Stricter Pfa raises alpha; more looks lower it towards 1 but never below."""
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    pfas = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8]
    alphas = [fluctuation_models.solve_ca_cfar_alpha(p, 16, 1) for p in pfas]
    assert np.all(np.diff(alphas) > 0)

    looks = [1, 4, 16, 64, 256, 4096]
    alphas = [fluctuation_models.solve_ca_cfar_alpha(1e-3, 16, m) for m in looks]
    assert np.all(np.diff(alphas) < 0)
    assert alphas[-1] > 1.0


# ---------------------------------------------------------------------------
# 3. OS-CFAR alpha against quadrature of the multi-look order statistic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_training_total, rank, num_frames, looks_per_frame",
    [(16, 12, 4, 1.0), (20, 15, 1, 3.5), (24, 18, 10, 0.5), (32, 24, 8, 12.0)],
)
@pytest.mark.parametrize("target_pfa", [1e-2, 1e-4, 1e-6])
def test_os_cfar_alpha_achieves_target_pfa_by_quadrature(
    monkeypatch, num_training_total, rank, num_frames, looks_per_frame, target_pfa
) -> None:
    """solve_os_cfar_alpha lands on an alpha whose Pfa, integrated independently, is the target.

    The reference integrates in the noise estimate itself (tests/support.py); the solver
    integrates in its logarithm on a fixed grid, so they share no numerical machinery.
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    alpha = fluctuation_models.solve_os_cfar_alpha(
        target_pfa,
        num_training_total,
        rank,
        num_frames,
        effective_looks_per_frame=looks_per_frame,
    )

    pfa = os_pfa_by_quadrature(alpha, num_training_total, rank, num_frames * looks_per_frame)

    assert pfa == pytest.approx(target_pfa, rel=1e-5)


@pytest.mark.parametrize("num_training_total, rank", [(2, 1), (5, 1), (5, 5), (40, 30), (128, 96)])
@pytest.mark.parametrize("target_pfa", [0.9, 1e-3, 1e-9])
def test_os_cfar_quadrature_reproduces_the_single_look_closed_form_when_forced(
    monkeypatch, num_training_total, rank, target_pfa
) -> None:
    """With one look the integral must equal Rohling's closed form, including extreme alphas.

    Looks of exactly 1 take the closed-form shortcut, so the integrator is called directly.
    Rank 1 at Pfa 1e-9 needs alpha ~ N / Pfa, far outside a fixed root-finding bracket.
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    closed_form = fluctuation_models.solve_os_cfar_alpha_single_look(
        target_pfa, num_training_total, rank
    )
    log_pfa = fluctuation_models._os_cfar_log_pfa_integrator(
        num_training_total, rank, 1.0, target_pfa
    )

    assert log_pfa(closed_form) == pytest.approx(np.log(target_pfa), abs=1e-8)


def test_os_cfar_alpha_is_accurate_for_very_many_looks(monkeypatch) -> None:
    """Broadband power over thousands of frames concentrates the cells tightly around 1.

    Checked by simulation, drawing the order statistic exactly as F^-1(Beta(k, N - k + 1)).
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    n, k, frames, looks_per_frame, target_pfa = 128, 96, 2497, 50.0, 1e-3
    looks = frames * looks_per_frame
    alpha = fluctuation_models.solve_os_cfar_alpha(
        target_pfa, n, k, num_frames=frames, effective_looks_per_frame=looks_per_frame
    )

    rng = np.random.default_rng(0)
    trials = 2_000_000
    noise_estimate = gamma_dist(a=looks, scale=1.0 / looks).ppf(rng.beta(k, n - k + 1, trials))
    cut = rng.gamma(looks, 1.0 / looks, size=trials)
    achieved = np.mean(cut > alpha * noise_estimate)

    # 2000 expected exceedances: binomial 3 sigma is about 7%.
    assert achieved == pytest.approx(target_pfa, rel=0.1)
    assert 1.0 < alpha < 1.01


def test_os_cfar_alpha_at_pfa_of_one_passes_every_cell(monkeypatch) -> None:
    """Sweeps end ROC curves at Pfa 1; alpha 0 is the threshold that achieves it."""
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)

    assert fluctuation_models.solve_os_cfar_alpha(1.0, 20, 15, num_frames=4) == 0.0
    with pytest.raises(ValueError, match="target_pfa"):
        fluctuation_models.solve_os_cfar_alpha(0.0, 20, 15, num_frames=4)
    with pytest.raises(ValueError, match="rank"):
        fluctuation_models.solve_os_cfar_alpha(0.1, 20, 21, num_frames=4)


def test_os_cfar_quadrature_reproduces_the_single_look_closed_form(monkeypatch) -> None:
    """Sanity check on the reference itself: at one look it matches Rohling's closed form."""
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    for n, k, pfa in [(24, 17, 1e-6), (16, 8, 1e-4), (20, 15, 1e-2)]:
        alpha = fluctuation_models.solve_os_cfar_alpha_single_look(pfa, n, k)
        assert os_pfa_by_quadrature(alpha, n, k, 1.0) == pytest.approx(pfa, rel=1e-5)


# ---------------------------------------------------------------------------
# 4. End-to-end detect() Pfa on multi-frame, band-integrated data
# ---------------------------------------------------------------------------


def _empirical_pfa(detector, rng, num_scans, num_beams, num_frames, looks_per_frame, cells):
    """Fraction of the given cells flagged by unconsolidated detect() on pure noise."""
    crossings = 0
    for _ in range(num_scans):
        data = _gamma_frames(rng, (num_beams, num_frames), looks_per_frame)
        flagged = detector.detect(data)[:, 0].astype(int)
        crossings += np.isin(flagged, cells).sum()
    return crossings / (num_scans * len(cells))


@pytest.mark.parametrize("detector_name", ["ca", "os"])
@pytest.mark.parametrize("num_frames, looks_per_frame", [(1, 1.0), (4, 1.0), (3, 2.5), (20, 0.5)])
def test_detect_achieves_target_pfa_with_integrated_looks(
    monkeypatch, detector_name, num_frames, looks_per_frame
) -> None:
    """Unconsolidated detect() on circular noise reproduces target_pfa for any look structure."""
    algorithms, calibration = _load_detector_algorithms_and_calibration(monkeypatch)
    common = dict(
        num_guard_cells=2,
        num_training_cells=8,
        target_pfa=1e-2,
        consolidate_peaks=False,
    )
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
    else:
        detector = algorithms.OSCFARDetector(**common, rank=12)
    num_beams = 128

    _calibrate_with_matching_noise(
        detector, calibration, num_beams, num_frames, looks_per_frame, seed=1000 + num_frames
    )

    empirical = _empirical_pfa(
        detector,
        np.random.default_rng(num_frames),
        num_scans=800,
        num_beams=num_beams,
        num_frames=num_frames,
        looks_per_frame=looks_per_frame,
        cells=np.arange(num_beams),
    )

    # ~1000 expected crossings; overlapping reference windows correlate cells within a scan,
    # so the tolerance is wider than the binomial 3 sigma of ~10%.
    assert empirical == pytest.approx(1e-2, rel=0.15)


@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_non_circular_detect_achieves_target_pfa_away_from_the_edges(
    monkeypatch, detector_name
) -> None:
    """Interior cells, whose windows never touch the padding, are calibrated without wrap."""
    algorithms, calibration = _load_detector_algorithms_and_calibration(monkeypatch)
    num_guard_cells, num_training_cells, num_beams = 2, 8, 96
    common = dict(
        num_guard_cells=num_guard_cells,
        num_training_cells=num_training_cells,
        target_pfa=1e-2,
        consolidate_peaks=False,
        circular=False,
    )
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
    else:
        detector = algorithms.OSCFARDetector(**common, rank=12)
    margin = num_guard_cells + num_training_cells
    interior = np.arange(margin, num_beams - margin)

    # Calibrate on a much wider scan than the one _empirical_pfa measures: edge-padded cells
    # have a biased ratio distribution (non-circular detect() replicates the boundary cell
    # into the window), and pooling them at this array's width would skew the calibration away
    # from what the *interior* cells being measured actually see. A 10x wider array dilutes
    # that edge fraction from ~20% of pooled cells to ~2%.
    _calibrate_with_matching_noise(detector, calibration, num_beams * 10, seed=1011)

    empirical = _empirical_pfa(
        detector,
        np.random.default_rng(11),
        num_scans=1500,
        num_beams=num_beams,
        num_frames=1,
        looks_per_frame=1.0,
        cells=interior,
    )

    assert empirical == pytest.approx(1e-2, rel=0.15)


# ---------------------------------------------------------------------------
# 5. CFAR invariance and calibration consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_detections_are_exactly_invariant_to_noise_power_scaling(
    monkeypatch, detector_name
) -> None:
    """Scaling every cell by a constant changes neither which cells are flagged nor their SNR."""
    algorithms, calibration = _load_detector_algorithms_and_calibration(monkeypatch)
    common = dict(num_guard_cells=1, num_training_cells=6, target_pfa=0.05, peak_distance=2)
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
    else:
        detector = algorithms.OSCFARDetector(**common, rank=9)
    data = np.random.default_rng(2).exponential(size=(90, 3))

    _calibrate_with_matching_noise(detector, calibration, num_beams=90, num_frames=3, seed=3)
    reference = detector.detect(data)
    for scale in (2.0**-20, 2.0**-3, 2.0**7, 2.0**30):  # powers of two scale exactly
        scaled = detector.detect(data * scale)
        np.testing.assert_array_equal(scaled[:, 0], reference[:, 0])
        np.testing.assert_allclose(scaled[:, 1], reference[:, 1], atol=1e-9)
    assert len(reference) > 0


@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_alpha_follows_calibration_parameters_changed_after_first_use(
    monkeypatch, detector_name
) -> None:
    """A detector whose target_pfa changes after detect() must not reuse a stale alpha.

    Stone Soup properties are ordinary mutable attributes, and sweeps set them on copies of
    detectors that have already been used. target_pfa doesn't affect noise_calibration itself
    (see calibration.detector_signature), so the same calibration applies to both detectors;
    only the memoised alpha must track the change.
    """
    algorithms, calibration = _load_detector_algorithms_and_calibration(monkeypatch)
    common = dict(num_guard_cells=1, num_training_cells=8, target_pfa=1e-2)
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
        fresh_kwargs = common
    else:
        common = dict(common, rank=12)
        detector = algorithms.OSCFARDetector(**common)
        fresh_kwargs = common
    _calibrate_with_matching_noise(detector, calibration, num_beams=64, num_frames=4, seed=0)
    data = np.random.default_rng(0).exponential(size=(64, 4))
    detector.detect(data)

    detector.target_pfa = 1e-3
    fresh_kwargs = dict(fresh_kwargs, target_pfa=1e-3)
    fresh = type(detector)(**fresh_kwargs)
    fresh.noise_calibration = detector.noise_calibration

    assert detector._alpha_for(4) == pytest.approx(fresh._alpha_for(4), rel=1e-12)


@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_training_window_changed_after_construction_is_used_consistently(
    monkeypatch, detector_name
) -> None:
    """Changing num_training_cells must update both the window and the calibrated N together."""
    algorithms, calibration = _load_detector_algorithms_and_calibration(monkeypatch)
    common = dict(num_guard_cells=1, num_training_cells=4, target_pfa=1e-3)
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
        fresh = algorithms.CACFARDetector(**dict(common, num_training_cells=12))
    else:
        common = dict(common, rank=6)
        detector = algorithms.OSCFARDetector(**common)
        fresh = algorithms.OSCFARDetector(**dict(common, num_training_cells=12))
    _calibrate_with_matching_noise(detector, calibration, num_beams=40, seed=6)
    detector.detect(np.ones((40, 1)))

    detector.num_training_cells = 12
    # The window just changed, so the old calibration (built for num_training_cells=4) no
    # longer matches; recalibrate for the new window and hand `fresh` the identical result, so
    # any remaining difference in _alpha_for reflects only whether the geometry stayed in sync.
    _calibrate_with_matching_noise(fresh, calibration, num_beams=40, seed=7)
    detector.noise_calibration = fresh.noise_calibration

    assert detector.num_training_total == 24
    assert detector._alpha_for(1) == pytest.approx(fresh._alpha_for(1))
    power = np.random.default_rng(6).exponential(size=40)
    np.testing.assert_allclose(
        detector._local_noise_floor(power), fresh._local_noise_floor(power), rtol=1e-12
    )


def test_deep_copied_detector_with_changed_parameter_recalibrates(monkeypatch) -> None:
    """The metrics sweep path: deep-copy a used detector, change a parameter, detect again."""
    algorithms, calibration = _load_detector_algorithms_and_calibration(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    detector = algorithms.CACFARDetector(num_guard_cells=1, num_training_cells=8, target_pfa=1e-2)
    _calibrate_with_matching_noise(detector, calibration, num_beams=32, seed=4)
    detector.detect(np.ones((32, 1)))

    clone = copy.deepcopy(detector)
    clone.target_pfa = 1e-5

    # target_pfa doesn't affect noise_calibration (see calibration.detector_signature), so the
    # deep-copied calibration still applies; only the memoised alpha must track the new target.
    expected = clone.noise_calibration.alpha(1e-5, 1)
    assert clone._alpha_for(1) == pytest.approx(expected)
    assert clone._alpha_for(1) != pytest.approx(detector._alpha_for(1))


@pytest.mark.parametrize(
    "param_name, value", [("num_training_cells", 12), ("num_guard_cells", 3), ("rank", 10)]
)
def test_sweep_clone_keeps_integer_parameters_integral_and_detects(
    monkeypatch, param_name, value
) -> None:
    """Sweeping an integer detector parameter must not turn it into a float and crash.

    The detector is left uncalibrated before the sweep: num_training_cells, num_guard_cells and
    rank are all structural, so sweeping any of them would invalidate an existing
    noise_calibration (_clone_detector_with_param raises rather than silently keeping a stale
    one -- see test_detector_metrics.py for that check). The clone is calibrated separately,
    after the swept value is in place.
    """
    algorithms, calibration = _load_detector_algorithms_and_calibration(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    metrics = load_package_module_from_repo(
        "bluepebble/detector/metrics.py", "bluepebble.detector.metrics"
    )
    detector = algorithms.OSCFARDetector(
        num_guard_cells=1, num_training_cells=8, rank=12, target_pfa=1e-2
    )
    spec = metrics.SweepSpec(detector=detector, param_name=param_name, param_values=[value])

    clone = metrics._clone_detector_with_param(spec, float(value))

    assert getattr(clone, param_name) == value
    assert isinstance(getattr(clone, param_name), int)
    _calibrate_with_matching_noise(clone, calibration, num_beams=48, seed=5)
    clone.detect(np.random.default_rng(0).exponential(size=(48, 1)))
    with pytest.raises(ValueError, match="integer parameter"):
        metrics._clone_detector_with_param(spec, 10.5)
