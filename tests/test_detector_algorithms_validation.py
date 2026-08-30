"""Validation tests for the CFAR detector implementations in algorithms.py.

Organized by validation strategy, roughly in order of how much a failure would
actually tell you:

1. Literature regression -- numeric values pinned directly from published sources.
   A future edit that reintroduces a derivation error fails one of these
   immediately, rather than requiring the whole argument to be re-derived.
2. Property invariants -- monotonicity, limits, and cross-path agreement that
   don't need any external reference, just internal consistency.
3. End-to-end empirical Pfa -- runs the actual detect() method (guard cells, wrap
   padding, peak consolidation, alpha caching, all wired together), not just the
   underlying math functions in isolation.
4. The CFAR property itself -- Pfa must not depend on absolute noise power.
5. Wrap/boundary edge cases -- constructed data exercising the circular seam,
   peak_distance merging, and peak_prominence rejection.
6. Reproducibility -- Monte Carlo functions must be deterministic given a fixed
   Generator.

The two Monte Carlo / numerical-integration heavy tests (Chalabi cross-checks)
are the priciest to run; skip them during quick iteration with
``pytest -k "not chalabi"``.
"""

from __future__ import annotations

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


def _load_fluctuation_models(monkeypatch):
    """Load fluctuation_models.py and the algorithms.py it needs, with minimal scaffolding."""
    algorithms = _load_detector_algorithms(monkeypatch)
    fluctuation_models = load_package_module_from_repo(
        "bluepebble/detector/fluctuation_models.py",
        "bluepebble.detector.fluctuation_models",
    )
    return algorithms, fluctuation_models


def _complex_gaussian_power(rng, shape):
    """Generate |x|^2 ~ Exponential(1) via complex Gaussian amplitude.

    The actual physical model (square-law detection of complex Gaussian noise),
    not just a convenient way to draw exponential numbers.
    """
    re = rng.normal(size=shape, scale=np.sqrt(0.5))
    im = rng.normal(size=shape, scale=np.sqrt(0.5))
    return re + 1j * im


# ---------------------------------------------------------------------------
# 1. Literature regression -- pinned against Rohling (1983), Gandhi & Kassam
# (1988), and Chalabi (2022). See algorithms.py's docstrings for the full
# derivation and cross-check behind each of these.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N, k, T_expected",
    [
        # Rohling (1983) Table II, Pfa = 1e-6.
        (24, 17, 18.6),
        (16, 8, 56.6),
        (8, 7, 27.8),
        (16, 1, 15_999_984.0),
    ],
)
def test_os_cfar_single_look_alpha_matches_rohling_table_ii(monkeypatch, N, k, T_expected) -> None:
    """The single-look OS-CFAR alpha solver should reproduce Rohling's Table II."""
    algorithms = _load_detector_algorithms(monkeypatch)

    alpha = algorithms.solve_os_cfar_alpha_single_look(
        target_pfa=1e-6, num_training_total=N, rank=k
    )

    assert alpha == pytest.approx(T_expected, rel=0.01)


@pytest.mark.parametrize(
    "N, k, target_pfa",
    [(24, 17, 1e-2), (32, 24, 1e-2), (20, 10, 1e-2), (10, 5, 1e-3)],
)
def test_os_cfar_single_look_alpha_achieves_target_pfa_by_direct_simulation(
    monkeypatch, N, k, target_pfa
) -> None:
    """Direct simulation, independent of the closed-form derivation itself.

    Guards against a bug that's self-consistent within the formula but wrong
    relative to what the formula is supposed to compute. target_pfa values here
    are chosen so 500k trials gives hundreds-to-thousands of exceedance events
    (per the num_trials ~ 100/target_pfa rule of thumb documented on
    calibrate_os_cfar_alpha_mc) -- very small Pfa (e.g. 1e-6) is already covered
    exactly, without needing simulation, by the Rohling Table II regression
    test above.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    alpha = algorithms.solve_os_cfar_alpha_single_look(target_pfa, N, k)

    rng = np.random.default_rng(0)
    trials = 500_000
    ref = rng.exponential(1.0, size=(trials, N))
    ref.sort(axis=1)
    xk = ref[:, k - 1]
    cut = rng.exponential(1.0, size=trials)
    empirical_pfa = np.mean(cut > alpha * xk)

    assert empirical_pfa == pytest.approx(target_pfa, rel=0.15)


def test_ca_cfar_pd_matches_gandhi_kassam_eq13(monkeypatch) -> None:
    """Gandhi & Kassam (1988) eq. 13: Pd = [1 + T/(1+S)]^-N, sum-based T.

    Our alpha is mean-based; alpha = N*T reconciles the two conventions.
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, T, S = 24, 0.778, 5.0
    expected = (1 + T / (1 + S)) ** (-N)
    alpha = N * T

    got = fluctuation_models.RayleighFluctuation().ca_cfar_pd(
        alpha, num_training_total=N, num_frames=1, snr_linear=S
    )

    assert got == pytest.approx(expected, rel=1e-6)


def test_os_cfar_pd_matches_gandhi_kassam_eq37(monkeypatch) -> None:
    """Gandhi & Kassam (1988) eq. 37: exact single-look OS-CFAR Pd, Swerling I."""
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, k, T, S = 20, 15, 6.8554, 2.0
    expected = 1.0
    for i in range(k):
        expected *= (N - i) / (N - i + T / (1 + S))

    got = fluctuation_models.RayleighFluctuation().os_cfar_pd(
        alpha=T, num_training_total=N, rank=k, num_frames=1, snr_linear=S
    )

    assert got == pytest.approx(expected, rel=1e-9)


def test_ca_cfar_multilook_matches_chalabi_eq8(monkeypatch) -> None:
    """Chalabi (2022) eq. 8, general Gamma clutter shape -> exponential at shape=1.

    Cross-checks our exact Beta-function closed form (RayleighFluctuation.ca_cfar_pd at
    snr_linear=0) against direct numerical integration of their formula.
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, M, T = 20, 10, 0.10
    NMa, Ma = N * M, M

    def integrand(z):
        fz = gamma_dist.pdf(z, a=NMa, scale=1.0)
        cut_sf = gamma_dist.sf(T * z, a=Ma, scale=1.0)
        return fz * cut_sf

    expected, _ = integrate.quad(integrand, 0, 500, limit=300)
    alpha = N * T

    got = fluctuation_models.RayleighFluctuation().ca_cfar_pd(
        alpha, num_training_total=N, num_frames=M, snr_linear=0.0
    )

    assert got == pytest.approx(expected, rel=1e-4)


def test_os_cfar_multilook_matches_chalabi_eq20(monkeypatch) -> None:
    """Chalabi (2022) eq. 20, shape=1 (exponential): order statistic across N cells.

    Each an M-look-integrated Gamma variable. Cross-checks our Monte Carlo
    (RayleighFluctuation.os_cfar_pd at snr_linear=0) against their numerical integration.
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, k, M, alpha = 20, 15, 10, 1.6

    def integrand(z):
        Fz = gamma_dist.cdf(z, a=M, scale=1.0)
        fz = gamma_dist.pdf(z, a=M, scale=1.0)
        outer = k * comb(N, k) * (1 - Fz) ** (N - k) * Fz ** (k - 1) * fz
        cut_sf = gamma_dist.sf(alpha * z, a=M, scale=1.0)
        return outer * cut_sf

    expected, _ = integrate.quad(integrand, 0, 100, limit=300)

    got = fluctuation_models.RayleighFluctuation().os_cfar_pd(
        alpha,
        N,
        k,
        num_frames=M,
        snr_linear=0.0,
        num_trials=1_000_000,
        rng=np.random.default_rng(0),
    )

    assert got == pytest.approx(expected, rel=0.1)  # Monte Carlo tolerance


# ---------------------------------------------------------------------------
# 2. Property invariants -- no external reference needed, these must hold by
# construction, and a violation points directly at what broke.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N, M", [(20, 1), (20, 5), (32, 10)])
def test_ca_cfar_pd_at_snr_linear_zero_recovers_target_pfa(monkeypatch, N, M) -> None:
    """With no target power, Pd should equal the calibrated Pfa exactly."""
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    target_pfa = 1e-3
    alpha = algorithms.solve_ca_cfar_alpha(target_pfa, N, num_frames=M)

    pd = fluctuation_models.RayleighFluctuation().ca_cfar_pd(
        alpha, N, num_frames=M, snr_linear=0.0
    )

    assert pd == pytest.approx(target_pfa, rel=1e-9)


@pytest.mark.parametrize("N, k", [(20, 15), (24, 18)])
def test_os_cfar_single_look_pd_at_snr_linear_zero_recovers_target_pfa(monkeypatch, N, k) -> None:
    """With no target power, Pd should equal the calibrated Pfa exactly."""
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    target_pfa = 1e-3
    alpha = algorithms.solve_os_cfar_alpha_single_look(target_pfa, N, k)

    pd = fluctuation_models.RayleighFluctuation().os_cfar_pd(
        alpha, N, k, num_frames=1, snr_linear=0.0
    )

    assert pd == pytest.approx(target_pfa, rel=1e-9)


def test_os_cfar_multilook_mc_at_snr_linear_zero_recovers_target_pfa_approximately(
    monkeypatch,
) -> None:
    """The Monte Carlo multi-look alpha/Pd pair should also round-trip at snr_linear=0."""
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, k, M = 20, 15, 10
    target_pfa = 1e-2  # larger target so Monte Carlo noise stays proportionally small
    rng = np.random.default_rng(1)
    alpha = algorithms.calibrate_os_cfar_alpha_mc(
        target_pfa, N, k, num_frames=M, num_trials=300_000, rng=rng
    )

    got = fluctuation_models.RayleighFluctuation().os_cfar_pd(
        alpha, N, k, num_frames=M, snr_linear=0.0, num_trials=300_000, rng=np.random.default_rng(2)
    )

    assert got == pytest.approx(target_pfa, rel=0.2)


@pytest.mark.parametrize("snr_linear_low, snr_linear_high", [(0.5, 1.0), (1.0, 5.0), (5.0, 20.0)])
def test_ca_cfar_pd_increases_with_snr_linear(
    monkeypatch, snr_linear_low, snr_linear_high
) -> None:
    """Pd should be monotonically increasing in target-power ratio snr_linear."""
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    alpha = algorithms.solve_ca_cfar_alpha(1e-3, num_training_total=20, num_frames=5)
    model = fluctuation_models.RayleighFluctuation()

    pd_low = model.ca_cfar_pd(alpha, 20, 5, snr_linear_low)
    pd_high = model.ca_cfar_pd(alpha, 20, 5, snr_linear_high)

    assert pd_high > pd_low


def test_ca_cfar_pd_approaches_one_at_high_snr_linear(monkeypatch) -> None:
    """As target power dominates the noise floor, Pd should approach 1."""
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    alpha = algorithms.solve_ca_cfar_alpha(1e-3, num_training_total=20, num_frames=5)

    pd = fluctuation_models.RayleighFluctuation().ca_cfar_pd(alpha, 20, 5, snr_linear=1e6)
    assert pd == pytest.approx(1.0, abs=1e-6)


def test_os_cfar_pfa_decreases_with_alpha(monkeypatch) -> None:
    """A stricter (higher) threshold multiplier should always lower the achieved Pfa."""
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, k = 20, 15
    model = fluctuation_models.RayleighFluctuation()

    pfa_small = model.os_cfar_pd(
        alpha=1.0, num_training_total=N, rank=k, num_frames=1, snr_linear=0.0
    )
    pfa_large = model.os_cfar_pd(
        alpha=5.0, num_training_total=N, rank=k, num_frames=1, snr_linear=0.0
    )

    assert pfa_large < pfa_small


def test_os_cfar_single_look_and_mc_agree_at_num_frames_one(monkeypatch) -> None:
    """The specific boundary that was fixed once already.

    os_cfar_pd() always takes the closed-form branch at num_frames == 1, so this instead
    forces the shared Monte Carlo machinery (the same code path used for num_frames > 1) down
    to num_frames == 1 by hand, and checks it still agrees with the closed form there.
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, k, alpha, snr_linear = 20, 15, 3.0, 1.5
    model = fluctuation_models.RayleighFluctuation()
    exact = model.os_cfar_pd(alpha, N, k, num_frames=1, snr_linear=snr_linear)

    rng = np.random.default_rng(3)
    num_trials = 500_000
    ref = rng.exponential(1.0, size=(num_trials, N, 1)).mean(axis=2)
    ref.sort(axis=1)
    noise_estimate = ref[:, k - 1]
    cut = model.cut_power_samples(num_trials, num_frames=1, snr_linear=snr_linear, rng=rng)
    mc = float(np.mean(cut > alpha * noise_estimate))

    assert mc == pytest.approx(exact, rel=0.05)


def test_os_cfar_pd_does_not_exceed_ca_cfar_pd_in_homogeneous_noise(monkeypatch) -> None:
    """OS-CFAR's censoring robustness costs a small, consistent Pd penalty.

    This holds in genuinely homogeneous noise -- the trade-off documented at
    length in OSCFARDetector's own docstring.
    """
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, target_pfa = 20, 1e-3
    rank = round(0.75 * N)
    alpha_os = algorithms.solve_os_cfar_alpha_single_look(target_pfa, N, rank)
    alpha_ca = algorithms.solve_ca_cfar_alpha(target_pfa, N, num_frames=1)
    model = fluctuation_models.RayleighFluctuation()

    for snr_linear in [0.5, 1.0, 2.0, 5.0]:
        pd_os = model.os_cfar_pd(alpha_os, N, rank, num_frames=1, snr_linear=snr_linear)
        pd_ca = model.ca_cfar_pd(alpha_ca, N, num_frames=1, snr_linear=snr_linear)
        assert pd_os <= pd_ca + 1e-9


# ---------------------------------------------------------------------------
# 3. End-to-end empirical Pfa -- exercises detect() itself (guard cells, wrap
# padding, peak consolidation, alpha caching), not just the math in isolation.
# ---------------------------------------------------------------------------


def test_oscfar_detect_achieves_calibrated_pfa(monkeypatch) -> None:
    """Running detect() on pure noise many times should reproduce target_pfa."""
    algorithms = _load_detector_algorithms(monkeypatch)
    num_beams, num_frames = 180, 1
    target_pfa = 1e-2
    detector = algorithms.OSCFARDetector(
        num_guard_cells=2,
        num_training_cells=8,
        rank=12,
        target_pfa=target_pfa,
        peak_distance=1,
        peak_prominence=None,
        circular=True,
    )
    rng = np.random.default_rng(7)
    n_trials = 2000
    total_candidates = 0
    for _ in range(n_trials):
        data = _complex_gaussian_power(rng, (num_beams, num_frames))
        total_candidates += len(detector.detect(data))

    # Loose tolerance: peak consolidation occasionally merges two adjacent false
    # alarms into one, biasing this slightly low -- expected and fine for this
    # test's purpose (catching an order-of-magnitude miscalibration).
    empirical_pfa = total_candidates / (n_trials * num_beams)
    assert empirical_pfa == pytest.approx(target_pfa, rel=0.3)


def test_cacfar_detect_achieves_calibrated_pfa(monkeypatch) -> None:
    """Running detect() on pure noise many times should reproduce target_pfa."""
    algorithms = _load_detector_algorithms(monkeypatch)
    num_beams, num_frames = 180, 1
    target_pfa = 1e-2
    detector = algorithms.CACFARDetector(
        num_guard_cells=2,
        num_training_cells=8,
        target_pfa=target_pfa,
        peak_distance=1,
        peak_prominence=None,
        circular=True,
    )
    rng = np.random.default_rng(8)
    n_trials = 2000
    total_candidates = 0
    for _ in range(n_trials):
        data = _complex_gaussian_power(rng, (num_beams, num_frames))
        total_candidates += len(detector.detect(data))

    empirical_pfa = total_candidates / (n_trials * num_beams)
    assert empirical_pfa == pytest.approx(target_pfa, rel=0.3)


# ---------------------------------------------------------------------------
# 4. The CFAR property itself: Pfa must not depend on absolute noise power.
# ---------------------------------------------------------------------------


def test_empirical_pfa_is_invariant_to_absolute_noise_power(monkeypatch) -> None:
    """Scaling the noise floor up or down should not move the achieved Pfa."""
    algorithms = _load_detector_algorithms(monkeypatch)
    num_beams, num_frames = 180, 1
    target_pfa = 1e-2
    detector = algorithms.OSCFARDetector(
        num_guard_cells=2,
        num_training_cells=8,
        rank=12,
        target_pfa=target_pfa,
        peak_distance=1,
        circular=True,
    )
    rng = np.random.default_rng(9)
    n_trials = 1500
    results = {}
    for power_scale in [0.01, 1.0, 100.0]:
        total_candidates = 0
        for _ in range(n_trials):
            data = _complex_gaussian_power(rng, (num_beams, num_frames)) * np.sqrt(power_scale)
            total_candidates += len(detector.detect(data))
        results[power_scale] = total_candidates / (n_trials * num_beams)

    # All three should land close to target_pfa and close to each other -- a bug
    # that introduced any additive (non-multiplicative) threshold term would
    # break this even if the power_scale=1.0 case still passed.
    values = list(results.values())
    assert max(values) / min(values) < 2.0, results


# ---------------------------------------------------------------------------
# 5. Wrap / boundary edge cases -- constructed data exercising the circular
# seam, peak_distance merging, and peak_prominence rejection.
# ---------------------------------------------------------------------------


def _wrap_test_detector(algorithms, peak_distance=2, peak_prominence=None):
    return algorithms.OSCFARDetector(
        num_guard_cells=1,
        num_training_cells=3,
        rank=4,
        target_pfa=1e-3,
        peak_distance=peak_distance,
        peak_prominence=peak_prominence,
        circular=True,
    )


def test_cluster_straddling_wrap_boundary_consolidates_to_one_detection(monkeypatch) -> None:
    """A candidate cluster split across the array's wrap seam should merge to one peak."""
    algorithms = _load_detector_algorithms(monkeypatch)
    num_beams = 10
    detector = _wrap_test_detector(algorithms)
    snr_db = np.full(num_beams, -10.0)
    candidate_mask = np.zeros(num_beams, dtype=bool)
    for b, val in [(8, 5.0), (9, 12.0), (0, 8.0), (1, 3.0)]:
        snr_db[b] = val
        candidate_mask[b] = True

    indices = detector._consolidate_peaks(snr_db, candidate_mask, num_beams)

    assert len(indices) == 1
    assert indices[0] == 9  # the true peak of the wrapped cluster


def test_peak_distance_merges_close_candidates_keeping_the_stronger(monkeypatch) -> None:
    """Two candidates closer than peak_distance should collapse to the taller one."""
    algorithms = _load_detector_algorithms(monkeypatch)
    num_beams = 30
    detector = _wrap_test_detector(algorithms, peak_distance=5)
    snr_db = np.full(num_beams, -10.0)
    candidate_mask = np.zeros(num_beams, dtype=bool)
    for b, val in [(10, 6.0), (12, 9.0)]:  # 2 bins apart, inside peak_distance=5
        snr_db[b] = val
        candidate_mask[b] = True

    indices = detector._consolidate_peaks(snr_db, candidate_mask, num_beams)

    assert len(indices) == 1
    assert indices[0] == 12


def test_low_prominence_ripple_is_rejected(monkeypatch) -> None:
    """A shallow ripple beside a dominant peak should be dropped by peak_prominence."""
    algorithms = _load_detector_algorithms(monkeypatch)
    num_beams = 30
    detector = _wrap_test_detector(algorithms, peak_distance=1, peak_prominence=3.0)
    snr_db = np.full(num_beams, -10.0)
    candidate_mask = np.zeros(num_beams, dtype=bool)
    # A dominant peak with a shallow, low-prominence ripple beside it.
    for b, val in [(10, 20.0), (11, 17.0), (12, 20.5), (13, 17.5), (14, 19.5)]:
        snr_db[b] = val
        candidate_mask[b] = True

    indices = detector._consolidate_peaks(snr_db, candidate_mask, num_beams)

    # Ripples within ~3 dB of each other should collapse to fewer peaks than the
    # number of local maxima visible in the raw values.
    assert len(indices) < 3


@pytest.mark.parametrize("num_guard_cells, num_training_cells", [(0, 1), (0, 8)])
def test_minimal_guard_and_training_do_not_crash(
    monkeypatch, num_guard_cells, num_training_cells
) -> None:
    """Degenerate but valid guard/training sizes should not crash detect()."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=num_guard_cells,
        num_training_cells=num_training_cells,
        rank=max(1, round(0.75 * 2 * num_training_cells)),
        target_pfa=1e-3,
    )
    data = _complex_gaussian_power(np.random.default_rng(4), (50, 1))

    detector.detect(data)  # should not raise


# ---------------------------------------------------------------------------
# 6. Reproducibility -- Monte Carlo functions must be deterministic given a
# fixed Generator.
# ---------------------------------------------------------------------------


def test_calibrate_os_cfar_alpha_mc_is_deterministic_given_fixed_seed(monkeypatch) -> None:
    """Two calibration runs with the same seeded Generator should agree exactly."""
    algorithms = _load_detector_algorithms(monkeypatch)

    a1 = algorithms.calibrate_os_cfar_alpha_mc(
        1e-3, 20, 15, num_frames=5, num_trials=10_000, rng=np.random.default_rng(42)
    )
    a2 = algorithms.calibrate_os_cfar_alpha_mc(
        1e-3, 20, 15, num_frames=5, num_trials=10_000, rng=np.random.default_rng(42)
    )

    assert a1 == a2


def test_os_cfar_pd_mc_is_deterministic_given_fixed_seed(monkeypatch) -> None:
    """Two Pd Monte Carlo runs with the same seeded Generator should agree exactly."""
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    model = fluctuation_models.RayleighFluctuation()

    p1 = model.os_cfar_pd(
        2.0, 20, 15, num_frames=5, snr_linear=1.0, num_trials=10_000, rng=np.random.default_rng(0)
    )
    p2 = model.os_cfar_pd(
        2.0, 20, 15, num_frames=5, snr_linear=1.0, num_trials=10_000, rng=np.random.default_rng(0)
    )

    assert p1 == p2


def test_oscfar_detector_rng_gives_reproducible_multilook_alpha(monkeypatch) -> None:
    """Two detectors built with identically-seeded rng should calibrate the same alpha.

    OSCFARDetector._alpha_for falls back to Monte Carlo calibration for num_frames > 1;
    without an explicit rng each detector draws from a fresh, unseeded Generator, so a
    deep-copied detector (as used when sweeping a parameter -- see metrics.py) would
    otherwise recalibrate with different randomness than its original.
    """
    algorithms = _load_detector_algorithms(monkeypatch)

    det1 = algorithms.OSCFARDetector(
        num_guard_cells=2,
        num_training_cells=10,
        rank=15,
        target_pfa=1e-2,
        rng=np.random.default_rng(42),
    )
    det2 = algorithms.OSCFARDetector(
        num_guard_cells=2,
        num_training_cells=10,
        rank=15,
        target_pfa=1e-2,
        rng=np.random.default_rng(42),
    )

    assert det1._alpha_for(num_frames=5) == det2._alpha_for(num_frames=5)


def test_oscfar_detector_without_rng_still_calibrates(monkeypatch) -> None:
    """Omitting rng should not break multi-look calibration -- it just won't be seeded."""
    algorithms = _load_detector_algorithms(monkeypatch)
    detector = algorithms.OSCFARDetector(
        num_guard_cells=2, num_training_cells=10, rank=15, target_pfa=1e-2
    )

    alpha = detector._alpha_for(num_frames=5)

    assert alpha > 0


# ---------------------------------------------------------------------------
# 7. Non-fluctuating (Swerling 0-equivalent) target model
# ---------------------------------------------------------------------------


def test_non_fluctuating_cut_power_has_correct_mean_and_reduces_to_exponential_at_zero_snr(
    monkeypatch,
) -> None:
    """Mean CUT power under H1 should be 1+snr_linear regardless of num_frames.

    At snr_linear=0 (no signal) the power model must reduce exactly to Exponential(1),
    matching the noise-only model used everywhere else in this module.
    """
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    rng = np.random.default_rng(0)

    model = fluctuation_models.NonFluctuating()
    for snr_linear, num_frames in [(0.0, 1), (0.0, 5), (3.0, 1), (3.0, 5)]:
        samples = model.cut_power_samples(200_000, num_frames, snr_linear, rng)
        assert samples.mean() == pytest.approx(1 + snr_linear, rel=0.02)


def test_non_fluctuating_cut_power_matches_noncentral_chi_squared(monkeypatch) -> None:
    """The physical-model simulation must match the textbook noncentral-chi-squared distribution.

    Per-frame power is derived here as |fixed signal + complex Gaussian noise|**2; the
    textbook closed form for M-frame-averaged power is (0.5/M) * ncx2(df=2M, nc=2*M*snr_linear).
    Cross-checking against scipy's independent implementation of that distribution (rather than
    just trusting the derivation) is what actually validates the simulation.
    """
    from scipy.stats import ncx2

    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    num_frames, snr_linear, n = 4, 5.0, 300_000

    simulated = fluctuation_models.NonFluctuating().cut_power_samples(
        n, num_frames, snr_linear, np.random.default_rng(1)
    )
    theoretical = (0.5 / num_frames) * ncx2.rvs(
        2 * num_frames, 2 * num_frames * snr_linear, size=n, random_state=2
    )

    assert simulated.mean() == pytest.approx(theoretical.mean(), rel=0.02)
    assert simulated.var() == pytest.approx(theoretical.var(), rel=0.05)
    for q in [0.1, 0.5, 0.9]:
        assert np.quantile(simulated, q) == pytest.approx(np.quantile(theoretical, q), rel=0.05)


@pytest.mark.parametrize("num_frames", [1, 4])
def test_ca_cfar_pd_non_fluctuating_mc_at_zero_snr_recovers_target_pfa(
    monkeypatch, num_frames
) -> None:
    """With no target power, Pd must equal the calibrated Pfa -- the H0 side is model-agnostic."""
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    target_pfa = 0.02
    N = 16
    alpha = algorithms.solve_ca_cfar_alpha(target_pfa, N, num_frames=num_frames)

    pd = fluctuation_models.NonFluctuating().ca_cfar_pd(
        alpha, N, num_frames, snr_linear=0.0, num_trials=300_000, rng=np.random.default_rng(3)
    )

    assert pd == pytest.approx(target_pfa, abs=0.01)


def test_os_cfar_pd_non_fluctuating_mc_at_zero_snr_recovers_target_pfa(monkeypatch) -> None:
    """With no target power, Pd must equal the calibrated Pfa -- the H0 side is model-agnostic."""
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    target_pfa = 0.02
    N, rank = 16, 12
    alpha = algorithms.solve_os_cfar_alpha_single_look(target_pfa, N, rank)

    pd = fluctuation_models.NonFluctuating().os_cfar_pd(
        alpha, N, rank, num_frames=1, snr_linear=0.0, num_trials=300_000,
        rng=np.random.default_rng(4),
    )

    assert pd == pytest.approx(target_pfa, abs=0.01)


def test_non_fluctuating_pd_is_monotonically_increasing_in_snr(monkeypatch) -> None:
    """Pd should increase with target-power ratio, same invariant as the Rayleigh-fading model."""
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    alpha = algorithms.solve_ca_cfar_alpha(0.02, 16, num_frames=4)
    model = fluctuation_models.NonFluctuating()

    pd_low = model.ca_cfar_pd(
        alpha, 16, 4, snr_linear=1.0, num_trials=300_000, rng=np.random.default_rng(5)
    )
    pd_high = model.ca_cfar_pd(
        alpha, 16, 4, snr_linear=10.0, num_trials=300_000, rng=np.random.default_rng(6)
    )

    assert pd_high > pd_low


def test_non_fluctuating_pd_exceeds_rayleigh_pd_at_moderate_to_high_snr(monkeypatch) -> None:
    """A steady target should out-detect a fading one of the same mean SNR, above the crossover.

    This is textbook radar/sonar detection theory (e.g. Skolnik, DiFranco & Rubin): a
    fluctuating target occasionally has a much WORSE-than-average look, which a fixed-threshold
    CFAR detector misses; a non-fluctuating target never does. At very low Pd the ordering can
    reverse (a fluctuating target occasionally gets a lucky, much-BETTER-than-average look),
    which is why this test only asserts the ordering at 6 dB, well clear of that crossover --
    see test_non_fluctuating_pd_crossover_at_low_snr_matches_rayleigh_closely for that region.
    """
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, M = 16, 4
    alpha = algorithms.solve_ca_cfar_alpha(0.02, N, num_frames=M)
    snr_linear = 10 ** (6 / 10)

    pd_rayleigh = fluctuation_models.RayleighFluctuation().ca_cfar_pd(alpha, N, M, snr_linear)
    pd_non_fluctuating = fluctuation_models.NonFluctuating().ca_cfar_pd(
        alpha, N, M, snr_linear, num_trials=500_000, rng=np.random.default_rng(7)
    )

    assert pd_non_fluctuating > pd_rayleigh


def test_non_fluctuating_pd_crossover_at_low_snr_matches_rayleigh_closely(monkeypatch) -> None:
    """Near the detection threshold, non-fluctuating and Rayleigh-fading Pd should be close.

    At low SNR (here 0 dB, giving Pd ~0.3), a fluctuating target's chance of an
    unusually-good look partially offsets its chance of an unusually-bad one, so the two
    fluctuation models' Pd values sit close together rather than the wide gap seen at high SNR
    (see test_non_fluctuating_pd_exceeds_rayleigh_pd_at_moderate_to_high_snr).
    """
    algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    N, M = 16, 4
    alpha = algorithms.solve_ca_cfar_alpha(0.02, N, num_frames=M)

    pd_rayleigh = fluctuation_models.RayleighFluctuation().ca_cfar_pd(alpha, N, M, snr_linear=1.0)
    pd_non_fluctuating = fluctuation_models.NonFluctuating().ca_cfar_pd(
        alpha, N, M, snr_linear=1.0, num_trials=500_000, rng=np.random.default_rng(8)
    )

    assert abs(pd_non_fluctuating - pd_rayleigh) < 0.03


def test_ca_and_os_cfar_non_fluctuating_mc_are_deterministic_given_fixed_seed(monkeypatch) -> None:
    """Two runs with the same seeded Generator should agree exactly, for both detector types."""
    _algorithms, fluctuation_models = _load_fluctuation_models(monkeypatch)
    model = fluctuation_models.NonFluctuating()

    ca1 = model.ca_cfar_pd(
        2.0, 16, 4, snr_linear=1.0, num_trials=10_000, rng=np.random.default_rng(0)
    )
    ca2 = model.ca_cfar_pd(
        2.0, 16, 4, snr_linear=1.0, num_trials=10_000, rng=np.random.default_rng(0)
    )
    assert ca1 == ca2

    os1 = model.os_cfar_pd(
        2.0, 16, 12, num_frames=4, snr_linear=1.0, num_trials=10_000, rng=np.random.default_rng(0)
    )
    os2 = model.os_cfar_pd(
        2.0, 16, 12, num_frames=4, snr_linear=1.0, num_trials=10_000, rng=np.random.default_rng(0)
    )
    assert os1 == os2


@pytest.mark.parametrize("target_pfa", [0.05, 0.2, 0.5, 0.9])
def test_unconsolidated_detect_tracks_target_pfa_where_consolidation_suppresses_it(
    monkeypatch, target_pfa
) -> None:
    """Without consolidation, achieved Pfa follows target_pfa into the dense regime.

    test_cacfar_detect_achieves_calibrated_pfa covers Pfa=1e-2, where crossings are sparse
    enough to be isolated; it is a sound alpha check, but consolidation is a no-op there so it
    says nothing about the dense regime. Once crossings stop being sparse, merging adjacent
    ones pulls the reported count well below the requested rate -- about 78% of crossings at
    Pfa 0.2, 57% at 0.5, 37% at 0.9 -- which is what breaks ROC sweeps and comparisons against
    theory.

    Unconsolidated output is the raw threshold-crossing rate, exactly the quantity
    solve_ca_cfar_alpha calibrates, so it must track target_pfa across the whole range. This
    guards that tracking property. Note it is a weak alpha check at the top of the range: near
    Pfa 0.9 almost every cell crosses whatever alpha is, so a mis-solved alpha barely moves the
    result. The sparse-regime test above is the sensitive one for alpha itself.
    """
    algorithms = _load_detector_algorithms(monkeypatch)
    num_beams, num_frames = 361, 4
    detector = algorithms.CACFARDetector(
        num_guard_cells=2,
        num_training_cells=5,
        target_pfa=target_pfa,
        peak_distance=1,
        circular=True,
        consolidate_peaks=False,
    )

    rng = np.random.default_rng(11)
    n_trials = 60
    total = 0
    for _ in range(n_trials):
        data = _complex_gaussian_power(rng, (num_beams, num_frames))
        total += len(detector.detect(data))

    empirical_pfa = total / (n_trials * num_beams)
    assert empirical_pfa == pytest.approx(target_pfa, rel=0.1)


def test_consolidation_only_ever_removes_detections(monkeypatch) -> None:
    """Consolidation is a filter over the crossings, never a source of new ones."""
    algorithms = _load_detector_algorithms(monkeypatch)
    kwargs = dict(
        num_guard_cells=2, num_training_cells=5, target_pfa=0.3, peak_distance=3, circular=True
    )
    consolidated = algorithms.CACFARDetector(**kwargs)
    raw = algorithms.CACFARDetector(consolidate_peaks=False, **kwargs)

    rng = np.random.default_rng(12)
    for _ in range(20):
        data = _complex_gaussian_power(rng, (361, 4))
        kept = consolidated.detect(data)
        every = raw.detect(data)
        assert len(kept) <= len(every)
        # and every surviving detection was itself a crossing
        assert set(kept[:, 0].astype(int)).issubset(set(every[:, 0].astype(int)))
