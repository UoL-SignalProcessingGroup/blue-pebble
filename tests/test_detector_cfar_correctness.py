"""Independent correctness checks for the CFAR detectors and their Pfa calibration.

Complements ``test_detector_algorithms_validation.py``, which pins single-look literature values
and single-frame end-to-end Pfa. The checks here target what that file leaves open:

1. Noise-floor estimators against a brute-force, loop-based reference (guard/training layout,
   circular wrap, edge padding, windows longer than the data).
2. CA-CFAR alpha against numerical quadrature and direct simulation for multi-frame and
   non-integer effective looks, including fewer than one look per frame.
3. OS-CFAR Monte Carlo alpha against numerical quadrature of the multi-look order statistic.
4. End-to-end ``detect()`` Pfa for multi-frame, band-integrated (Gamma) data, circular and
   non-circular, for both detectors.
5. The CFAR property as an exact invariant, and calibration staying in step with parameters
   changed after a detector has already been used.
"""

from __future__ import annotations

import copy
from math import comb

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import gamma as gamma_dist

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
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.CACFARDetector(
        num_guard_cells=1,
        num_training_cells=4,
        target_pfa=0.05,
        consolidate_peaks=False,
    )
    data = np.random.default_rng(5).exponential(size=(64, 6))

    power = data.mean(axis=1)
    threshold = detector._alpha_for(6) * detector._local_noise_floor(power)
    expected = np.flatnonzero(power > threshold)

    np.testing.assert_array_equal(detector.detect(data)[:, 0].astype(int), expected)


# ---------------------------------------------------------------------------
# 2. CA-CFAR alpha: independent quadrature and simulation
# ---------------------------------------------------------------------------


def _ca_pfa_by_quadrature(alpha, num_training_total, total_looks):
    """P(CUT > alpha * mean(refs)) with CUT, refs unit-mean Gamma(total_looks) cells.

    Integrates over the sum of reference cells, S ~ Gamma(N * L, scale 1/L), of the CUT
    survival function, CUT ~ Gamma(L, scale 1/L). Written without the Beta-distribution identity
    used by solve_ca_cfar_alpha.
    """
    looks = total_looks
    ref_sum = gamma_dist(a=num_training_total * looks, scale=1.0 / looks)
    cut = gamma_dist(a=looks, scale=1.0 / looks)
    upper = ref_sum.ppf(1 - 1e-14)

    def integrand(s):
        return ref_sum.pdf(s) * cut.sf(alpha * s / num_training_total)

    value, _ = integrate.quad(integrand, 0.0, upper, limit=500, epsabs=1e-14, epsrel=1e-10)
    return value


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
    algorithms = _load_detector_algorithms(monkeypatch)
    alpha = algorithms.solve_ca_cfar_alpha(
        target_pfa, num_training_total, num_frames, effective_looks_per_frame=looks_per_frame
    )

    pfa = _ca_pfa_by_quadrature(alpha, num_training_total, num_frames * looks_per_frame)

    assert pfa == pytest.approx(target_pfa, rel=1e-5)


@pytest.mark.parametrize("looks_per_frame", [0.5, 1.0, 4.0])
def test_ca_cfar_alpha_achieves_target_pfa_by_direct_simulation(
    monkeypatch, looks_per_frame
) -> None:
    """Simulating actual frames (not the closed-form cell distribution) reproduces target_pfa.

    Per-frame Gamma(K) samples are averaged over M frames, so this also checks that K and M
    enter the calibration as the product K * M.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    num_training_total, num_frames, target_pfa = 12, 4, 1e-2
    alpha = algorithms.solve_ca_cfar_alpha(
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
    algorithms = _load_detector_algorithms(monkeypatch)
    pfas = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8]
    alphas = [algorithms.solve_ca_cfar_alpha(p, 16, 1) for p in pfas]
    assert np.all(np.diff(alphas) > 0)

    looks = [1, 4, 16, 64, 256, 4096]
    alphas = [algorithms.solve_ca_cfar_alpha(1e-3, 16, m) for m in looks]
    assert np.all(np.diff(alphas) < 0)
    assert alphas[-1] > 1.0


# ---------------------------------------------------------------------------
# 3. OS-CFAR Monte Carlo alpha against quadrature of the multi-look order statistic
# ---------------------------------------------------------------------------


def _os_pfa_by_quadrature(alpha, num_training_total, rank, total_looks):
    """P(CUT > alpha * X_(k)) for unit-mean Gamma(total_looks) cells, by quadrature.

    X_(k), the k-th smallest of N iid cells with CDF F and PDF f, has density
    ``k * C(N, k) * F^(k-1) * (1 - F)^(N-k) * f``.
    """
    n, k = num_training_total, rank
    cell = gamma_dist(a=total_looks, scale=1.0 / total_looks)

    def integrand(x):
        f_cdf = cell.cdf(x)
        density = k * comb(n, k) * f_cdf ** (k - 1) * (1 - f_cdf) ** (n - k) * cell.pdf(x)
        return density * cell.sf(alpha * x)

    upper = cell.ppf(1 - 1e-12)
    value, _ = integrate.quad(integrand, 0.0, upper, limit=500, epsabs=1e-13, epsrel=1e-9)
    return value


@pytest.mark.parametrize(
    "num_training_total, rank, num_frames, looks_per_frame",
    [(16, 12, 4, 1.0), (20, 15, 1, 3.5), (24, 18, 10, 0.5), (32, 24, 8, 12.0)],
)
def test_os_cfar_mc_alpha_achieves_target_pfa_by_quadrature(
    monkeypatch, num_training_total, rank, num_frames, looks_per_frame
) -> None:
    """Monte Carlo calibration lands on an alpha whose exact Pfa is the target."""
    algorithms = _load_detector_algorithms(monkeypatch)
    target_pfa = 1e-2
    alpha = algorithms.calibrate_os_cfar_alpha_mc(
        target_pfa,
        num_training_total,
        rank,
        num_frames,
        num_trials=400_000,
        rng=np.random.default_rng(3),
        effective_looks_per_frame=looks_per_frame,
    )

    pfa = _os_pfa_by_quadrature(alpha, num_training_total, rank, num_frames * looks_per_frame)

    # 4000 exceedances: the quantile's own sampling error is a few percent.
    assert pfa == pytest.approx(target_pfa, rel=0.08)


def test_os_cfar_quadrature_reproduces_the_single_look_closed_form(monkeypatch) -> None:
    """Sanity check on the reference itself: at one look it matches Rohling's closed form."""
    algorithms = _load_detector_algorithms(monkeypatch)
    for n, k, pfa in [(24, 17, 1e-6), (16, 8, 1e-4), (20, 15, 1e-2)]:
        alpha = algorithms.solve_os_cfar_alpha_single_look(pfa, n, k)
        assert _os_pfa_by_quadrature(alpha, n, k, 1.0) == pytest.approx(pfa, rel=1e-5)


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
    algorithms = _load_detector_algorithms(monkeypatch)
    common = dict(
        num_guard_cells=2,
        num_training_cells=8,
        target_pfa=1e-2,
        consolidate_peaks=False,
        effective_looks_per_frame=looks_per_frame,
    )
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
    else:
        detector = algorithms.OSCFARDetector(
            **common, rank=12, mc_trials=400_000, rng=np.random.default_rng(1)
        )
    num_beams = 128

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
    algorithms = _load_detector_algorithms(monkeypatch)
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
    algorithms = _load_detector_algorithms(monkeypatch)
    common = dict(num_guard_cells=1, num_training_cells=6, target_pfa=0.05, peak_distance=2)
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
    else:
        detector = algorithms.OSCFARDetector(**common, rank=9)
    data = np.random.default_rng(2).exponential(size=(90, 3))

    reference = detector.detect(data)
    for scale in (2.0**-20, 2.0**-3, 2.0**7, 2.0**30):  # powers of two scale exactly
        scaled = detector.detect(data * scale)
        np.testing.assert_array_equal(scaled[:, 0], reference[:, 0])
        np.testing.assert_allclose(scaled[:, 1], reference[:, 1], atol=1e-9)
    assert len(reference) > 0


@pytest.mark.parametrize(
    "param_name, value",
    [("target_pfa", 1e-3), ("effective_looks_per_frame", 0.25)],
)
@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_alpha_follows_calibration_parameters_changed_after_first_use(
    monkeypatch, detector_name, param_name, value
) -> None:
    """A detector whose calibration inputs change after detect() must not reuse a stale alpha.

    Stone Soup properties are ordinary mutable attributes, and sweeps set them on copies of
    detectors that have already been used.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    common = dict(num_guard_cells=1, num_training_cells=8, target_pfa=1e-2)
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
        fresh_kwargs = common
    else:
        common = dict(common, rank=12, mc_trials=400_000, rng=np.random.default_rng(0))
        detector = algorithms.OSCFARDetector(**common)
        fresh_kwargs = dict(common, rng=np.random.default_rng(0))
    data = np.random.default_rng(0).exponential(size=(64, 4))
    detector.detect(data)

    setattr(detector, param_name, value)
    fresh_kwargs = dict(fresh_kwargs, **{param_name: value})
    fresh = type(detector)(**fresh_kwargs)

    # OS-CFAR calibrates by Monte Carlo from different generator states, hence the tolerance;
    # the stale alpha (target_pfa 1e-2 or 4x the looks) differs by far more than this.
    assert detector._alpha_for(4) == pytest.approx(fresh._alpha_for(4), rel=0.05)


@pytest.mark.parametrize("detector_name", ["ca", "os"])
def test_training_window_changed_after_construction_is_used_consistently(
    monkeypatch, detector_name
) -> None:
    """Changing num_training_cells must update both the window and the calibrated N together."""
    algorithms = _load_detector_algorithms(monkeypatch)
    common = dict(num_guard_cells=1, num_training_cells=4, target_pfa=1e-3)
    if detector_name == "ca":
        detector = algorithms.CACFARDetector(**common)
        fresh = algorithms.CACFARDetector(**dict(common, num_training_cells=12))
    else:
        common = dict(common, rank=6)
        detector = algorithms.OSCFARDetector(**common)
        fresh = algorithms.OSCFARDetector(**dict(common, num_training_cells=12))
    detector.detect(np.ones((40, 1)))

    detector.num_training_cells = 12

    assert detector.num_training_total == 24
    assert detector._alpha_for(1) == pytest.approx(fresh._alpha_for(1))
    power = np.random.default_rng(6).exponential(size=40)
    np.testing.assert_allclose(
        detector._local_noise_floor(power), fresh._local_noise_floor(power), rtol=1e-12
    )


def test_deep_copied_detector_with_changed_parameter_recalibrates(monkeypatch) -> None:
    """The metrics sweep path: deep-copy a used detector, change a parameter, detect again."""
    algorithms = _load_detector_algorithms(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    detector = algorithms.CACFARDetector(num_guard_cells=1, num_training_cells=8, target_pfa=1e-2)
    detector.detect(np.ones((32, 1)))

    clone = copy.deepcopy(detector)
    clone.target_pfa = 1e-5

    expected = algorithms.solve_ca_cfar_alpha(1e-5, 16, 1)
    assert clone._alpha_for(1) == pytest.approx(expected)


@pytest.mark.parametrize(
    "param_name, value", [("num_training_cells", 12), ("num_guard_cells", 3), ("rank", 10)]
)
def test_sweep_clone_keeps_integer_parameters_integral_and_detects(
    monkeypatch, param_name, value
) -> None:
    """Sweeping an integer detector parameter must not turn it into a float and crash."""
    algorithms = _load_detector_algorithms(monkeypatch)
    install_repo_package(monkeypatch, "bluepebble.detector", "bluepebble/detector")
    metrics = load_package_module_from_repo(
        "bluepebble/detector/metrics.py", "bluepebble.detector.metrics"
    )
    detector = algorithms.OSCFARDetector(
        num_guard_cells=1, num_training_cells=8, rank=12, target_pfa=1e-2
    )
    detector.detect(np.ones((48, 1)))
    spec = metrics.SweepSpec(detector=detector, param_name=param_name, param_values=[value])

    clone = metrics._clone_detector_with_param(spec, float(value))

    assert getattr(clone, param_name) == value
    assert isinstance(getattr(clone, param_name), int)
    clone.detect(np.random.default_rng(0).exponential(size=(48, 1)))
    with pytest.raises(ValueError, match="integer parameter"):
        metrics._clone_detector_with_param(spec, 10.5)
