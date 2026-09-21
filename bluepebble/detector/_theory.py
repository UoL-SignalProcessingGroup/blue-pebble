"""Detection probability (Pd) under different target fluctuation models.

A :class:`FluctuationModel` determines the H1 (target-present) statistics of the CUT, and
therefore Pd, for a given detector, since Pd depends on how the target's amplitude fluctuates
from look to look, not merely on its mean SNR. Alpha/Pfa calibration (:func:`solve_ca_cfar_alpha`,
:func:`solve_os_cfar_alpha_single_look`, :func:`solve_os_cfar_alpha`) is a separate concern that
remains shared across every fluctuation model defined here: it depends only on H0 (noise-only)
statistics, which do not change with how the target fluctuates. Correspondingly,
``CACFARDetector``/``OSCFARDetector`` never call anything Pd-related in this module; only this
module's own :func:`ca_cfar_roc`/:func:`os_cfar_roc` theoretical ROC curve functions do, via a
``model: FluctuationModel`` parameter rather than by selecting a Pd function by name.

Fluctuation models
-------------------
- **Rayleigh-fading** (:class:`RayleighFluctuation`; Swerling II-equivalent): the target return
  is itself fully random, Rayleigh-amplitude / Exponential-power distributed, and decorrelated
  frame to frame, he same statistical family as noise, scaled only by (1 + snr_linear). A
  closed form is available for both CA-CFAR (:meth:`RayleighFluctuation.ca_cfar_pd`) and
  single-frame OS-CFAR (:meth:`RayleighFluctuation.os_cfar_pd` at ``num_frames == 1``); Monte
  Carlo estimation is used for multi-frame OS-CFAR (same method, ``num_frames > 1``), since no
  closed form exists in that case (see its docstring for the reasoning).
- **Non-fluctuating** (:class:`NonFluctuating`; Swerling 0-equivalent): the target return has a
  fixed amplitude at every look (only the ambient noise varies), giving Rice-amplitude /
  noncentral-chi-squared-power statistics under H1. No closed form is implemented for either
  detector; both :meth:`NonFluctuating.ca_cfar_pd` and :meth:`NonFluctuating.os_cfar_pd` rely on
  Monte Carlo estimation throughout.

See :meth:`RayleighFluctuation.ca_cfar_pd` for the full derivation of the Rayleigh-fading model,
and :meth:`NonFluctuating.cut_power_samples` for that of the non-fluctuating model.

Band integration and signal bandwidth
--------------------------------------
Every Pd path here accepts two look counts, because a broadband detector's cell under test
contains two populations that need not share a bandwidth:

- ``effective_looks_per_frame`` (K_n): independent Exponential(1) looks integrated into each
  per-frame sample. An STFT beamformer sums power over every frequency bin between ``fmin`` and
  ``fmax``, so K_n is the effective bin count (below the nominal one, since window leakage
  correlates neighbouring bins). This is the same quantity :func:`solve_os_cfar_alpha` calibrates
  alpha against, and :func:`~.metrics.estimate_effective_looks_per_frame` measures it from
  simulator output.
- ``signal_looks_per_frame`` (K_s): how many of those looks the TARGET actually occupies.
  Defaults to K_n, i.e. a target whose energy fills the processed band the way the noise does.

Noise always fills the band, so H0 depends on K_n alone, which is why alpha calibration needs
only that one number. H1 does not: a narrowband source in a wide processing band puts its energy
in a handful of bins while noise arrives in all of them, making the cell a MIXTURE of K_s
signal-bearing looks and K_n - K_s noise-only looks. Setting K_s = K_n when the target is
actually a tonal understates its look-to-look fluctuation badly, because a signal spread over
many looks self-averages (CV ~ 1/sqrt(K_s)) and a signal confined to two does not. The mean is
unaffected (``snr_linear`` is defined on the integrated cell either way), so this is purely a
statement about variance, and therefore about the shape of the Pd curve rather than its
midpoint.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist

FloatArray: TypeAlias = NDArray[np.float64]


def _os_cfar_log_pfa(alpha: float, num_training_total: int, rank: int) -> float:
    """Log Pfa for single-look OS-CFAR.

    Derivation: for N iid Exponential(1) reference cells, the k-th order statistic (ascending,
    1-indexed) admits the exponential order-statistic "spacings" representation::

        X_(k) = sum_{i=1}^{k} E_i / (N - i + 1),   E_i iid Exponential(1)

    i.e. a sum of k INDEPENDENT (not identical) exponential random variables. Since the detection
    threshold is alpha * X_(k) and the CUT ~ Exponential(1) independently of the reference cells,
    Pfa = P(CUT > alpha * X_(k)) = E[exp(-alpha * X_(k))], which factors into a product of Laplace
    transforms (one per spacing term)::

        Pfa(alpha) = prod_{l=0}^{k-1} (N - l) / (alpha + N - l)

    Parameters
    ----------
    alpha : float
        Threshold multiplier.
    num_training_total : int
        Total number of reference cells (N).
    rank : int
        Order-statistic rank used as the noise estimate (k, 1-indexed).

    Returns
    -------
    float
        log(Pfa(alpha)).

    References
    ----------
    .. [1] Rohling, H. "Radar CFAR Thresholding in Clutter and Multiple Target Situations."
           IEEE Transactions on Aerospace and Electronic Systems, AES-19(4), 608-621, 1983.
    .. [2] Renyi, A. "On the theory of order statistics." Acta Mathematica Academiae Scientiarum
           Hungaricae, 4(3-4), 191-231, 1953.

    This specific closed form was re-derived from [2] and confirmed algebraically identical to
    Rohling's own equation (14) in [1]: k*C(N,k)*(k-1)!*(T+N-k)!/(T+N)! reduces exactly to the
    product form above. Cross-checked numerically against four entries in [1]'s Table II
    (Pfa=1e-6): both eq. 14 and this formula reproduce the tabulated T values to within rounding.

    """
    log_pfa = 0.0
    for i in range(rank):
        log_pfa += np.log(num_training_total - i) - np.log(alpha + num_training_total - i)
    return log_pfa


def solve_ca_cfar_alpha(
    target_pfa: float,
    num_training_total: int,
    num_frames: int,
    effective_looks_per_frame: float = 1.0,
) -> float:
    """Exact alpha for target_pfa, for any num_frames (M-frame integration per look).

    Reference cells and the CUT are each Gamma(K*num_frames, 1/(K*num_frames)) (unit mean,
    M-frame averaged, K looks integrated per frame). Since threhsold = alpha * mean(N reference
    cells)), and mean(N cells) is itself Gamma(N*K*num_frames, ...) (same scale as the CUT),
    W = CUT / (CUT + sum_of_refs) is exactly Beta(K*num_frames, N*K*num_frames)-distributed.
    Reduces to the classic single-look closed form N*(Pfa**(-1/N) - 1) when num_frames == 1 and
    K == 1.

    Parameters
    ----------
    target_pfa : float
        The desired probability of false alarm.
    num_training_total : int
        The total number of training cells.
    num_frames : int
        The number of frames incoherently integrated into this one look (M).
    effective_looks_per_frame : float, optional
        Effective independent Exponential(1) looks already integrated into each per-frame
        power sample (K), by default 1.0 (classic single-bin square-law data). Broadband STFT
        beam power sums many frequency bins per frame, making K far larger; see
        :func:`~.metrics.estimate_effective_looks_per_frame` for how to estimate it. Need not be
        an integer.

    Returns
    -------
    float
        The calibrated alpha value.

    References
    ----------
    .. [1] Finn, H.M. and Johnson, R.S. "Adaptive detection mode with threshold control as a
           function of spatially sampled clutter level estimates." RCA Review, 29, 414-464, 1968.
           Origin of cell-averaging CFAR.
    .. [2] Gandhi, P.P. and Kassam, S.A. "Analysis of CFAR Processors in Nonhomogeneous
           Background." IEEE Transactions on Aerospace and Electronic Systems, 24(4), 427-445,
           1988. Equation (14) gives this exact single-look alpha/Pfa pair (sum-based threshold;
           reconciles with this module's mean-based alpha via alpha = N*T; see
           RayleighFluctuation.ca_cfar_pd).
    .. [3] Chalabi, I. "Application of CFAR detection to multiple pulses for gamma distributed
           clutter." Remote Sensing Letters, 13(10), 1011-1019, 2022. Independently derives the
           same multi-look CA-CFAR Pfa (their eq. 8) via direct numerical integration rather than
           this function's Beta-function closed form; see
           RayleighFluctuation.ca_cfar_pd for the numerical cross-check
           (6 decimal places across several threshold values).

    """
    if not 0 < target_pfa < 1:
        raise ValueError(f"target_pfa ({target_pfa}) must be in (0, 1)")
    if effective_looks_per_frame <= 0:
        raise ValueError(
            f"effective_looks_per_frame ({effective_looks_per_frame}) must be positive"
        )

    # w0 is the (1 - target_pfa) quantile of W = CUT / (CUT + sum_of_refs); alpha is then
    # recovered by inverting W = threshold / (threshold + N*mean_ref) for the equal-scale case.
    a = effective_looks_per_frame * num_frames
    b = num_training_total * effective_looks_per_frame * num_frames
    w0 = beta_dist.ppf(1 - target_pfa, a, b)
    return num_training_total * w0 / (1 - w0)  # pyright: ignore[reportReturnType]


def _solve_alpha_for_log_pfa(log_pfa: Callable[[float], float], target_pfa: float) -> float:
    """Solve ``log_pfa(alpha) = log(target_pfa)`` for alpha, given log Pfa decreasing in alpha.

    Works in log(alpha) with an expanding upper bracket, so alphas far outside any fixed range
    (e.g. ~N / Pfa for rank 1 at very small Pfa) are still found.
    """
    log_target = np.log(target_pfa)

    def f(log_alpha: float) -> float:
        return log_pfa(float(np.exp(log_alpha))) - log_target

    lower, upper = -30.0, 1.0
    while f(upper) > 0:
        upper += 4.0
    while f(lower) < 0:
        lower -= 30.0
    return float(np.exp(brentq(f, lower, upper, xtol=1e-12, rtol=1e-12)))


def solve_os_cfar_alpha_single_look(
    target_pfa: float, num_training_total: int, rank: int
) -> float:
    """Exact OS-CFAR alpha for target_pfa with single-look (exponential) cells.

    Solves Pfa(alpha) = target_pfa for alpha using the closed form from _os_cfar_log_pfa::

        target_pfa = prod_{l=0}^{k-1} (N - l) / (alpha + N - l)

    Pfa is strictly decreasing in alpha, so Brent's method converges reliably. For
    band-integrated or multi-frame cells use :func:`solve_os_cfar_alpha`, which reduces to this
    function when there is one look in total.

    Parameters
    ----------
    target_pfa : float
        Desired probability of false alarm, in (0, 1).
    num_training_total : int
        Total number of reference cells (N).
    rank : int
        Order-statistic rank used as the noise estimate (k, 1-indexed).

    Returns
    -------
    float
        The alpha satisfying Pfa(alpha) = target_pfa.

    References
    ----------
    .. [1] Rohling, H. "Radar CFAR Thresholding in Clutter and Multiple Target Situations."
            IEEE Transactions on Aerospace and Electronic Systems, AES-19(4), 608-621, 1983.
    .. [2] Renyi, A. "On the theory of order statistics." Acta Mathematica Academiae
            Scientiarum Hungaricae, 4(3-4), 191-231, 1953.

    """
    if not 0 < target_pfa < 1:
        raise ValueError(f"target_pfa ({target_pfa}) must be in (0, 1)")

    return _solve_alpha_for_log_pfa(
        lambda alpha: _os_cfar_log_pfa(alpha, num_training_total, rank), target_pfa
    )


# Trapezoid nodes over log(noise estimate) for the OS-CFAR Pfa integral. The integrand is smooth
# and decays at both ends, so the trapezoid rule converges quickly: raising this to 16001 moves
# alpha by less than 1e-10 relative across looks from 0.15 to 1e6 and Pfa from 0.9 to 1e-12.
_OS_PFA_NODES = 4001

# Integration range: the probability that the noise estimate falls outside it, relative to
# target_pfa, bounds the truncation error in Pfa.
_OS_PFA_RELATIVE_TAIL = 1e-10


def _os_cfar_log_pfa_integrator(
    num_training_total: int, rank: int, total_looks: float, target_pfa: float
) -> Callable[[float], float]:
    """Return ``alpha -> log Pfa`` for OS-CFAR with unit-mean Gamma(total_looks) cells.

    Pfa is the probability that the CUT exceeds alpha times the rank-th smallest of N reference
    cells. Conditioning on that order statistic X_(k), whose density is
    ``k * C(N, k) * F**(k-1) * (1 - F)**(N-k) * f`` for cell CDF F and PDF f, gives::

        Pfa(alpha) = integral f_X(k)(z) * S(alpha * z) dz

    with S the CUT survival function ([1]_, Eqs. (8.365) and (8.371), for any cell
    distribution). The integral is evaluated in t = log(z), where both factors are smooth, on a
    fixed trapezoid grid spanning the order statistic's quantiles at
    ``target_pfa * _OS_PFA_RELATIVE_TAIL`` and its complement. Everything except S is computed
    once, so each Pfa evaluation during root finding is a single vectorised survival-function
    call. Working in logs keeps very small Pfa from underflowing.

    References
    ----------
    .. [1] Abraham, D. A. "Underwater Acoustic Signal Processing: Modeling, Detection, and
           Estimation." Springer, Cham, 2019. doi:10.1007/978-3-319-92983-5

    """
    n, k = num_training_total, rank
    # Typed loosely: pyright models the frozen distribution as possibly discrete.
    cell: Any = gamma_dist(a=total_looks, scale=1.0 / total_looks)
    tail = target_pfa * _OS_PFA_RELATIVE_TAIL
    # X_(k) = F^-1(B) with B ~ Beta(k, N - k + 1), and 1 - B ~ Beta(N - k + 1, k).
    z_lower = cell.ppf(beta_dist.ppf(tail, k, n - k + 1))
    z_upper = cell.isf(beta_dist.ppf(tail, n - k + 1, k))
    log_z = np.linspace(np.log(z_lower), np.log(z_upper), _OS_PFA_NODES)
    z = np.exp(log_z)

    log_weights = (
        np.log(k)
        + gammaln(n + 1)
        - gammaln(k + 1)
        - gammaln(n - k + 1)
        + cell.logpdf(z)
        + log_z
        + np.log(log_z[1] - log_z[0])
    )
    # Guarded so a zero exponent never meets a -inf log CDF or survival value.
    if k > 1:
        log_weights += (k - 1) * cell.logcdf(z)
    if n > k:
        log_weights += (n - k) * cell.logsf(z)
    log_weights[[0, -1]] += np.log(0.5)

    def log_pfa(alpha: float) -> float:
        return float(logsumexp(log_weights + cell.logsf(alpha * z)))

    return log_pfa


def solve_os_cfar_alpha(
    target_pfa: float,
    num_training_total: int,
    rank: int,
    num_frames: int = 1,
    effective_looks_per_frame: float = 1.0,
) -> float:
    """Alpha achieving target_pfa for OS-CFAR on frame-averaged, band-integrated power.

    Reference cells and the CUT are each Gamma(K*M, 1/(K*M)) (unit mean, M frames averaged, K
    looks integrated per frame), the same model as :func:`solve_ca_cfar_alpha`. OS-CFAR has no
    closed form for more than one look, so Pfa(alpha) is integrated numerically over the
    distribution of the rank-th order statistic (see ``_os_cfar_log_pfa_integrator``) and
    inverted with Brent's method. With a single look in total it uses the exact closed form of
    :func:`solve_os_cfar_alpha_single_look`.

    The result is deterministic. Checked against the single-look closed form (alpha within
    1e-9 relative), adaptive quadrature (Pfa within 1e-6 relative down to 1e-12) and Monte Carlo,
    for looks from 0.15 to 1e6. It replaces the Monte Carlo quantile of
    :func:`calibrate_os_cfar_alpha_mc`, whose accuracy depended on the trial count.

    Parameters
    ----------
    target_pfa : float
        Desired probability of false alarm, in (0, 1]. A value of 1 returns alpha = 0, which
        passes every cell; parameter sweeps use it as the end point of an ROC curve.
    num_training_total : int
        Total number of reference cells (N).
    rank : int
        Order-statistic rank used as the noise estimate (k, 1-indexed).
    num_frames : int, optional
        Number of frames incoherently averaged into each cell (M), by default 1.
    effective_looks_per_frame : float, optional
        Effective independent Exponential(1) looks integrated into each per-frame sample (K),
        by default 1.0. See :func:`solve_ca_cfar_alpha`. Need not be an integer.

    Returns
    -------
    float
        The alpha satisfying Pfa(alpha) = target_pfa.

    References
    ----------
    .. [1] Rohling, H. "Radar CFAR Thresholding in Clutter and Multiple Target Situations."
           IEEE Transactions on Aerospace and Electronic Systems, AES-19(4), 608-621, 1983.
    .. [2] Abraham, D. A. "Underwater Acoustic Signal Processing: Modeling, Detection, and
           Estimation." Springer, Cham, 2019. doi:10.1007/978-3-319-92983-5. Sect. 8.6.3,
           Eqs. (8.365) and (8.371): Pfa of an order-statistic normaliser for any cell
           distribution.

    """
    if not 0 < target_pfa <= 1:
        raise ValueError(f"target_pfa ({target_pfa}) must be in (0, 1]")
    if effective_looks_per_frame <= 0:
        raise ValueError(
            f"effective_looks_per_frame ({effective_looks_per_frame}) must be positive"
        )
    if not 1 <= rank <= num_training_total:
        raise ValueError(f"rank ({rank}) must be between 1 and {num_training_total}")
    if target_pfa == 1:
        return 0.0

    total_looks = effective_looks_per_frame * num_frames
    if total_looks == 1:
        return solve_os_cfar_alpha_single_look(target_pfa, num_training_total, rank)
    log_pfa = _os_cfar_log_pfa_integrator(num_training_total, rank, total_looks, target_pfa)
    return _solve_alpha_for_log_pfa(log_pfa, target_pfa)


def calibrate_os_cfar_alpha_mc(
    target_pfa: float,
    num_training_total: int,
    rank: int,
    num_frames: int,
    num_trials: int = 200_000,
    rng: np.random.Generator | None = None,
    effective_looks_per_frame: float = 1.0,
) -> float:
    """Calibrate OS-CFAR alpha for a target Pfa using Monte Carlo simulation.

    .. deprecated::
        Use :func:`solve_os_cfar_alpha`, which computes the same alpha deterministically by
        numerical integration and is accurate at any target_pfa. This function is kept so
        existing code runs unchanged and will be removed in a future release.

    M-frame averaged power is Gamma(M, ...)-distributed, not exponential, so Rohling's closed form
    doesn't apply directly. Caveat: for target_pfa << 1, num_trials needs scaling up (rule of thumb
    ~100 / target_pfa) or the calibrated alpha will be noisy.

    Parameters
    ----------
    target_pfa : float
        The desired probability of false alarm.
    num_training_total : int
        The total number of training cells.
    rank : int
        The rank of the cell under test.
    num_frames : int
        The number of independent frames (averages).
    num_trials : int, optional
        The number of Monte Carlo trials to run, by default 200_000.
    rng : np.random.Generator | None, optional
        A random number generator for reproducibility, by default None.
    effective_looks_per_frame : float, optional
        Effective number of independent Exponential(1) looks already integrated into each
        per-frame power sample, by default 1.0 (the classic single-bin square-law model).
        Broadband STFT beamformers (``MinimumVarianceDistortionlessResponseBeamformer``,
        ``DelayAndSumBeamformer`` in ``'broadband_power'`` domain) sum power over every
        active frequency bin in [fmin, fmax] before the detector sees it, so each per-frame
        sample is Gamma(K, 1/K)-distributed (unit mean) with K well above 1. Window
        spectral leakage correlates adjacent bins, so K is smaller than the bin count;
        estimate it from noise-only per-cell power as ``1 / CV**2 / num_frames`` (CV of the
        frame-averaged cell power across independent scans). Need not be an integer.

    Returns
    -------
    float
        The calibrated alpha value.

    References
    ----------
    .. [1] Rohling, H. "Radar CFAR Thresholding in Clutter and Multiple Target Situations."
           IEEE Transactions on Aerospace and Electronic Systems, AES-19(4), 608-621, 1983.
    .. [2] Chalabi, I. "Application of CFAR detection to multiple pulses for gamma distributed
           clutter." Remote Sensing Letters, 13(10), 1011-1019, 2022. Independently derives the
           same multi-look OS-CFAR problem and states explicitly that no closed form exists for
           it; see RayleighFluctuation.os_cfar_pd for the numerical
           cross-check against their eq. 20 (shape=1). This function calibrates alpha for the
           same model RayleighFluctuation.os_cfar_pd evaluates Pd against.

    """
    warnings.warn(
        "calibrate_os_cfar_alpha_mc is deprecated; use solve_os_cfar_alpha, which gives the same "
        "alpha deterministically.",
        DeprecationWarning,
        stacklevel=2,
    )
    rng = rng or np.random.default_rng()

    # Each per-frame look is Gamma(K, 1/K): unit mean, K effective independent exponential
    # looks integrated per frame. K = 1 is exactly the classic Exponential(1) per-look model.
    k = effective_looks_per_frame

    # Simulate M-frame-averaged noise-only training cells: mean of `num_frames` unit-mean
    # per-frame looks per cell, for every training cell, over all trials.
    # Only the frame-average is ever used, and the mean of M iid Gamma(k, 1/k) draws is
    # exactly Gamma(M*k, 1/(M*k)), so the frame axis is drawn in closed form rather than
    # materialised. Sampling it would cost (num_trials, num_training_total, num_frames)
    # float64: over a gigabyte for a window sized to a wide mainlobe, and tens of gigabytes
    # for broadband STFT data with thousands of frames. This is the same distribution, not
    # an approximation.
    shape_m = k * num_frames
    ref_samples = rng.gamma(
        shape=shape_m, scale=1.0 / shape_m, size=(num_trials, num_training_total)
    )

    # OS-CFAR's noise estimate is the rank-th smallest training cell per trial.
    ref_sorted = np.sort(ref_samples, axis=1)
    noise_estimate = ref_sorted[:, rank - 1]

    # Simulate the (noise-only) CUT under the same M-frame averaging.
    cut_samples = rng.gamma(shape=shape_m, scale=1.0 / shape_m, size=num_trials)

    ratio = cut_samples / noise_estimate

    # alpha is the value the CUT/noise ratio exceeds with probability target_pfa under H0.
    return float(np.quantile(ratio, 1 - target_pfa))


# ---------------------------------------------------------------------------
# Theoretical Pd-vs-Pfa curves (no data, pure model; see module docstring)
# ---------------------------------------------------------------------------


def ca_cfar_roc(
    pfa_values: ArrayLike,
    num_training_total: int,
    num_frames: int,
    snr_linear: float,
    model: FluctuationModel | None = None,
    effective_looks_per_frame: float = 1.0,
    signal_looks_per_frame: float | None = None,
) -> FloatArray:
    """Pd vs Pfa sweep for CA-CFAR, at a fixed target-power ratio, under a fluctuation model.

    Closed form throughout for the default :class:`RayleighFluctuation`
    model (:func:`solve_ca_cfar_alpha` +
    :meth:`RayleighFluctuation.ca_cfar_pd`), so each point is independent
    and exact; no shared-simulation trick needed here, unlike :func:`os_cfar_roc` at
    ``num_frames > 1``. A model without a closed form for CA-CFAR would fall back to its own
    (independent, per-point) Monte Carlo instead.

    Parameters
    ----------
    pfa_values : ArrayLike
        Pfa operating points to evaluate Pd at, each in (0, 1).
    num_training_total : int
        Total number of reference cells (N).
    num_frames : int
        The number of frames incoherently integrated into this one look (M).
    snr_linear : float
        Target-power / noise-power ratio (linear). snr_linear = 10 ** (target_snr_db / 10).
    model : FluctuationModel, optional
        Target fluctuation model. Defaults to
        :class:`RayleighFluctuation`.
    effective_looks_per_frame : float, optional
        Independent looks integrated into each per-frame sample (K_n), by default 1.0. Applied
        to the threshold and the Pd alike, so the curve stays self-consistent. 1.0 is right only
        for single-bin square-law data: on broadband beamformer output it can misstate Pd by an
        order of magnitude. Measure it with
        :func:`~.metrics.estimate_effective_looks_per_frame`; see
        :doc:`/auto_examples/comparing_theoretical_and_empirical_pd`.
    signal_looks_per_frame : float or None, optional
        Looks the target occupies (K_s), by default ``None`` meaning K_s = K_n. Affects only
        Pd; H0 has no signal in it, so the threshold is untouched. See the
        module docstring above.

    Returns
    -------
    FloatArray
        Pd at each requested Pfa, same shape as ``pfa_values``.

    """
    model = model or RayleighFluctuation()
    return np.array(
        [
            model.ca_cfar_pd(
                solve_ca_cfar_alpha(
                    pfa, num_training_total, num_frames, effective_looks_per_frame
                ),
                num_training_total,
                num_frames,
                snr_linear,
                effective_looks_per_frame=effective_looks_per_frame,
                signal_looks_per_frame=signal_looks_per_frame,
            )
            for pfa in pfa_values
        ]
    )


def os_cfar_roc(
    pfa_values: ArrayLike,
    num_training_total: int,
    rank: int,
    num_frames: int,
    snr_linear: float,
    model: FluctuationModel | None = None,
    num_trials: int = 1_000_000,
    rng: np.random.Generator | None = None,
    effective_looks_per_frame: float = 1.0,
    signal_looks_per_frame: float | None = None,
) -> FloatArray:
    """Pd vs Pfa sweep for OS-CFAR, at a fixed target-power ratio, under a fluctuation model.

    At ``num_frames == 1``, this is closed form throughout for the default
    :class:`RayleighFluctuation` model: each point is evaluated
    independently via :func:`solve_os_cfar_alpha_single_look` +
    :meth:`model.os_cfar_pd() <FluctuationModel.os_cfar_pd>`. A model
    without a closed form at ``num_frames == 1`` (e.g.
    :class:`NonFluctuating`) still goes through this same per-point path,
    just falling back to its own independent Monte Carlo per point.

    Otherwise alpha for each Pfa point comes from :func:`solve_os_cfar_alpha`,
    the same deterministic calculation :class:`~.algorithms.OSCFARDetector` uses, so the curve
    describes the detector's actual threshold. Pd has no closed form for OS-CFAR in this case
    under any fluctuation model implemented here (see
    :meth:`RayleighFluctuation.os_cfar_pd`), so it is estimated by Monte
    Carlo. Reference-cell and H1 CUT samples are drawn ONCE and reused for every Pfa value,
    which is cheaper than simulating each point and gives a smoother curve (adjacent points
    share the same draws). The noise estimate's distribution does not depend on the hypothesis,
    so sharing draws across points introduces no bias.

    Precision: Pd is a Monte Carlo fraction with standard error ``sqrt(Pd * (1 - Pd) /
    num_trials)``; alpha itself carries no sampling error.

    Parameters
    ----------
    pfa_values : ArrayLike
        Pfa operating points to evaluate Pd at, each in (0, 1].
    num_training_total : int
        Total number of reference cells (N).
    rank : int
        Order-statistic rank used as the noise estimate (k, 1-indexed).
    num_frames : int
        The number of frames incoherently integrated into this one look (M).
    snr_linear : float
        Target-power / noise-power ratio (linear). snr_linear = 10 ** (target_snr_db / 10).
    model : FluctuationModel, optional
        Target fluctuation model. Defaults to
        :class:`RayleighFluctuation`.
    num_trials : int, optional
        Monte Carlo trial count, shared across every Pfa point when num_frames > 1 (and passed
        through to ``model.os_cfar_pd`` when num_frames == 1), by default 1_000_000.
    rng : np.random.Generator, optional
        Random number generator for reproducibility, by default None.
    effective_looks_per_frame : float, optional
        Independent looks integrated into each per-frame sample (K_n), by default 1.0. Applied
        to the threshold and the Pd alike, so the curve stays self-consistent. Match it to the
        detector being compared against. 1.0 is right only for single-bin square-law data: on
        broadband beamformer output it can misstate Pd by an order of magnitude. Measure it with
        :func:`~.metrics.estimate_effective_looks_per_frame`; see
        :doc:`/auto_examples/comparing_theoretical_and_empirical_pd`.
    signal_looks_per_frame : float or None, optional
        Looks the target occupies (K_s), by default ``None`` meaning K_s = K_n. Affects only
        Pd; H0 has no signal in it, so the threshold is untouched. Set it well below K_n for
        a narrowband target in a wide processing band. See this module's docstring above.

    Returns
    -------
    FloatArray
        Pd at each requested Pfa, same shape as ``pfa_values``.

    """
    model = model or RayleighFluctuation()
    single_look = effective_looks_per_frame == 1.0 and signal_looks_per_frame in (None, 1.0)

    if num_frames == 1 and single_look:
        return np.array(
            [
                model.os_cfar_pd(
                    solve_os_cfar_alpha_single_look(pfa, num_training_total, rank),
                    num_training_total,
                    rank,
                    num_frames,
                    snr_linear,
                    num_trials=num_trials,
                    rng=rng,
                )
                for pfa in pfa_values
            ]
        )

    rng = rng or np.random.default_rng()

    # Simulate M-frame-averaged noise-only training cells once; noise_estimate's distribution
    # is the same under H0 and H1 (see this module's docstring), so it's shared
    # across every Pfa point.
    ref = rng.gamma(
        effective_looks_per_frame,
        1.0 / effective_looks_per_frame,
        size=(num_trials, num_training_total, num_frames),
    ).mean(axis=2)
    ref.sort(axis=1)
    noise_estimate = ref[:, rank - 1]

    # H1 CUT (target present at snr_linear), used to evaluate Pd once alpha is fixed per Pfa
    # value. This is the only piece that depends on the fluctuation model.
    cut_h1 = model.cut_power_samples(
        num_trials,
        num_frames,
        snr_linear,
        rng,
        effective_looks_per_frame,
        signal_looks_per_frame,
    )

    # H0 statistics don't depend on the fluctuation model or on the target's bandwidth, so alpha
    # uses the noise look count alone.
    pds = [
        np.mean(
            cut_h1
            > solve_os_cfar_alpha(
                float(pfa), num_training_total, rank, num_frames, effective_looks_per_frame
            )
            * noise_estimate
        )
        for pfa in pfa_values
    ]
    return np.array(pds)



def _resolve_look_counts(
    effective_looks_per_frame: float,
    signal_looks_per_frame: float | None,
) -> tuple[float, float]:
    """Validate the look counts and default the signal count to the full band.

    Parameters
    ----------
    effective_looks_per_frame : float
        Total independent looks integrated per frame (K_n).
    signal_looks_per_frame : float or None
        Looks the target occupies (K_s). ``None`` means the target fills the band, K_s = K_n.

    Returns
    -------
    tuple[float, float]
        ``(K_n, K_s)`` as floats.

    Raises
    ------
    ValueError
        If either count is non-positive, or if K_s exceeds K_n.

    """
    noise_looks = float(effective_looks_per_frame)
    signal_looks = noise_looks if signal_looks_per_frame is None else float(signal_looks_per_frame)

    if noise_looks <= 0:
        raise ValueError(f"effective_looks_per_frame ({noise_looks}) must be positive")
    if signal_looks <= 0:
        raise ValueError(f"signal_looks_per_frame ({signal_looks}) must be positive")
    if signal_looks > noise_looks:
        raise ValueError(
            f"signal_looks_per_frame ({signal_looks}) cannot exceed "
            f"effective_looks_per_frame ({noise_looks}); the target cannot occupy more "
            "looks than the detector integrates"
        )
    return noise_looks, signal_looks


def _noise_only_looks(
    rng: np.random.Generator,
    noise_looks: float,
    signal_looks: float,
    size: tuple[int, int],
) -> FloatArray | float:
    """Power from the looks the target does NOT occupy, on a per-look unit-mean scale.

    Returns a scalar ``0.0`` when the target fills the band, so the caller can add it
    unconditionally.
    """
    remainder = noise_looks - signal_looks
    if remainder <= 0:
        return 0.0
    return rng.gamma(remainder, 1.0, size=size)


def _noise_only_cells(
    rng: np.random.Generator,
    num_trials: int,
    num_training_total: int,
    num_frames: int,
    noise_looks: float,
) -> FloatArray:
    """M-frame-averaged noise-only reference cells, unit mean, K_n looks per frame.

    Every look here is iid Gamma(K_n, 1/K_n), so their M-frame mean is exactly
    Gamma(M*K_n, 1/(M*K_n)) and the frame axis never has to be materialised. Drawing it
    would cost (num_trials, num_training_total, num_frames) float64, which runs to
    gigabytes for a window sized to a wide mainlobe and to tens of gigabytes for broadband
    STFT data. Note the same shortcut does NOT apply to a partially-occupied cell (see
    cut_power_samples): there each look is a sum of two Gammas of different scale, which is
    not itself Gamma, so those are still sampled per frame.
    """
    shape_m = noise_looks * num_frames
    return rng.gamma(shape_m, 1.0 / shape_m, size=(num_trials, num_training_total))


class FluctuationModel(ABC):
    """A target fluctuation model: H1 CUT statistics for CA-CFAR and OS-CFAR Pd.

    Each subclass determines internally whether a closed form exists for a given detector and
    ``num_frames`` combination; callers always receive a Pd value, regardless of whether it was
    obtained from an exact formula or from Monte Carlo estimation. ``num_trials``/``rng`` are
    accepted uniformly across the interface, even though a model with an exact closed form for a
    given call may disregard them entirely.
    """

    @abstractmethod
    def cut_power_samples(
        self,
        num_trials: int,
        num_frames: int,
        snr_linear: float,
        rng: np.random.Generator,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> FloatArray:
        """Monte Carlo samples of the M-frame-averaged CUT power under H1.

        This is the shared building block for every Monte Carlo Pd path in this module: both
        :meth:`ca_cfar_pd` and :meth:`os_cfar_pd` fall back to it when no closed form applies,
        and :func:`os_cfar_roc` reuses it directly for its shared-simulation sweep.

        Parameters
        ----------
        num_trials : int
            Monte Carlo trial count.
        num_frames : int
            Number of frames (M) integrated per detection cycle.
        snr_linear : float
            Target-power / noise-power ratio (linear). snr_linear = 10 ** (target_snr_db / 10).
            Defined on the integrated cell, so it already accounts for any dilution from
            integrating a wider band than the target occupies.
        rng : np.random.Generator
            Random number generator.
        effective_looks_per_frame : float, optional
            Independent looks integrated into each per-frame sample (K_n), by default 1.0,
            which is right only for single-bin data; see :func:`os_cfar_roc`.
        signal_looks_per_frame : float or None, optional
            Looks the target occupies (K_s), by default ``None`` meaning K_s = K_n (the target
            fills the processed band). See the module docstring for why this matters.

        Returns
        -------
        FloatArray
            M-frame-averaged CUT power under H1, shape (num_trials,), unit-mean-noise scaled
            so the noise-only case has mean 1 and the target-present case has mean
            ``1 + snr_linear``.

        """
        ...

    @abstractmethod
    def ca_cfar_pd(
        self,
        alpha: float,
        num_training_total: int,
        num_frames: int,
        snr_linear: float,
        num_trials: int = 500_000,
        rng: np.random.Generator | None = None,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> float:
        """Pd for CA-CFAR at target-power ratio snr_linear (linear), under this model.

        Parameters
        ----------
        alpha : float
            Threshold multiplier, as calibrated by :func:`solve_ca_cfar_alpha`.
            Pass the same ``effective_looks_per_frame`` to both, or the threshold and the Pd
            will describe different detectors.
        num_training_total : int
            Total number of reference cells (N).
        num_frames : int
            Number of frames (M) integrated per detection cycle.
        snr_linear : float
            Target-power / noise-power ratio (linear). snr_linear = 10 ** (target_snr_db / 10).
        num_trials : int, optional
            Monte Carlo trial count, by default 500_000. Ignored by models/paths with a closed
            form.
        rng : np.random.Generator, optional
            Random number generator for reproducibility, by default None. Ignored by models/paths
            with a closed form.
        effective_looks_per_frame : float, optional
            Independent looks integrated into each per-frame sample (K_n), by default 1.0,
            which is right only for single-bin data; see :func:`os_cfar_roc`.
        signal_looks_per_frame : float or None, optional
            Looks the target occupies (K_s), by default ``None`` meaning K_s = K_n.

        Returns
        -------
        float
            Pd at the given snr_linear.

        """
        ...

    @abstractmethod
    def os_cfar_pd(
        self,
        alpha: float,
        num_training_total: int,
        rank: int,
        num_frames: int,
        snr_linear: float,
        num_trials: int = 500_000,
        rng: np.random.Generator | None = None,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> float:
        """Pd for OS-CFAR at target-power ratio snr_linear (linear), under this model.

        Parameters
        ----------
        alpha : float
            Threshold multiplier, as calibrated by
            :func:`solve_os_cfar_alpha_single_look` (num_frames == 1) or
            :func:`solve_os_cfar_alpha` (num_frames > 1). Pass the same
            ``effective_looks_per_frame`` to both, or the threshold and the Pd will describe
            different detectors.
        num_training_total : int
            Total number of reference cells (N).
        rank : int
            Order-statistic rank used as the noise estimate (k, 1-indexed).
        num_frames : int
            Number of frames (M) integrated per detection cycle.
        snr_linear : float
            Target-power / noise-power ratio (linear). snr_linear = 10 ** (target_snr_db / 10).
        num_trials : int, optional
            Monte Carlo trial count, by default 500_000. Ignored by models/paths with a closed
            form.
        rng : np.random.Generator, optional
            Random number generator for reproducibility, by default None. Ignored by models/paths
            with a closed form.
        effective_looks_per_frame : float, optional
            Independent looks integrated into each per-frame sample (K_n), by default 1.0,
            which is right only for single-bin data; see :func:`os_cfar_roc`.
        signal_looks_per_frame : float or None, optional
            Looks the target occupies (K_s), by default ``None`` meaning K_s = K_n.

        Returns
        -------
        float
            Pd at the given snr_linear.

        """
        ...


class RayleighFluctuation(FluctuationModel):
    """Rayleigh-fading target (Swerling II-equivalent): fully random, decorrelated per look.

    Under H1, per-look received power is Exponential with mean (1 + snr_linear) relative to the
    noise-only mean of 1, independent across looks (target returns decorrelated across the M
    looks being non-coherently integrated), the same statistical family as noise, differing
    only by a scale factor.
    """

    def cut_power_samples(
        self,
        num_trials: int,
        num_frames: int,
        snr_linear: float,
        rng: np.random.Generator,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> FloatArray:
        """M-frame-averaged CUT power under H1, as a signal/noise-bandwidth mixture.

        Each frame integrates K_n unit-mean looks, of which K_s carry the target. The target's
        total power in the cell is ``snr_linear`` times the noise power, so spreading it over
        only K_s looks gives each signal-bearing look a mean of
        ``1 + snr_linear * K_n / K_s``. A Rayleigh-fading target is Exponential in power and
        decorrelated look to look, so those K_s looks sum to
        ``Gamma(K_s, 1 + snr_linear * K_n / K_s)`` and the untouched remainder sums to
        ``Gamma(K_n - K_s, 1)``. Dividing by K_n restores the unit-mean-noise scale, giving a
        cell of mean ``1 + snr_linear`` for any (K_n, K_s).

        At K_n = K_s = 1 this is Exponential(1 + snr_linear), the classic single-look model.
        At K_n = K_s > 1 it is Gamma(K_n, (1 + snr_linear) / K_n): the target fills the band,
        so signal and noise integrate identically and the whole cell simply scales by
        (1 + snr_linear).
        """
        noise_looks, signal_looks = _resolve_look_counts(
            effective_looks_per_frame, signal_looks_per_frame
        )
        size = (num_trials, num_frames)

        per_look_mean = 1.0 + snr_linear * noise_looks / signal_looks
        occupied = rng.gamma(signal_looks, per_look_mean, size=size)
        remainder = _noise_only_looks(rng, noise_looks, signal_looks, size)

        return ((occupied + remainder) / noise_looks).mean(axis=1)

    def ca_cfar_pd(
        self,
        alpha: float,
        num_training_total: int,
        num_frames: int,
        snr_linear: float,
        num_trials: int = 500_000,
        rng: np.random.Generator | None = None,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> float:
        """Exact Pd for CA-CFAR when the target fills the band; Monte Carlo when it doesn't.

        The cell-averaging formulation is Finn and Johnson's [CA1]_. The derivation here
        proceeds by writing CUT = (1 + snr_linear) * Z, where
        Z ~ Gamma(K_n*M, 1/(K_n*M)) (unit mean, M-frame averaged, K_n looks per frame), the
        same scale shared by the reference-cell sum. Then::

            Pd = P(CUT > alpha * mean_of_N_refs)
               = P(Z > [alpha / (1 + snr_linear)] * mean_of_N_refs)

        which is exactly the Pfa relationship with alpha replaced by
        alpha_eff = alpha / (1 + snr_linear), the pair given as equations (13)-(14) in
        [CA2]_. The Beta relationship established in
        :func:`solve_ca_cfar_alpha` (W = CUT / (CUT + sum_of_refs) ~
        Beta(K_n*M, N*K_n*M)) is reused directly::

            w0 = alpha_eff / (N + alpha_eff)
            Pd(snr_linear) = Beta_sf(w0; K_n*M, N*K_n*M)

        At snr_linear=0 this recovers the calibrated Pfa exactly (validated to floating-point
        precision); as snr_linear approaches infinity, Pd approaches 1.

        This factorisation needs CUT and the reference cells to share a scale, which holds only
        when the target fills the band (K_s = K_n) so that the whole cell scales uniformly by
        (1 + snr_linear). A narrowband target makes the CUT a mixture of two different scales
        (see :meth:`cut_power_samples`), which admits no such Beta form, so K_s < K_n falls
        back to Monte Carlo using ``num_trials``/``rng``.

        References
        ----------
        .. [CA1] Finn, H.M. and Johnson, R.S. "Adaptive detection mode with threshold control as a
               function of spatially sampled clutter level estimates." RCA Review, 29, 414-464,
               1968.
        .. [CA2] Gandhi, P.P. and Kassam, S.A. "Analysis of CFAR Processors in Nonhomogeneous
               Background." IEEE Transactions on Aerospace and Electronic Systems, 24(4), 427-445,
               1988. Equations (13)-(14) give this exact single-look Pd/Pfa pair for a Swerling I
               target, using a SUM-based threshold (Z = sum of N cells, threshold = T*Z) rather
               than this module's MEAN-based alpha (threshold = alpha*mean(N cells)). The two
               conventions reconcile exactly via alpha = N*T, validated numerically to 6 decimal
               places across several (N, T) pairs, not just algebraically.
        .. [CA3] Chalabi, I. "Application of CFAR detection to multiple pulses for gamma
               distributed clutter." Remote Sensing Letters, 13(10), 1011-1019, 2022.

        The multi-look (M > 1) generalisation implemented here, via the Gamma/Beta scale-sharing
        relationship, is independently confirmed by [CA3]_, which derives the same multi-look
        CA-CFAR Pfa (their eq. 8, general Gamma clutter shape; our exponential case is their shape
        parameter = 1) via direct numerical integration rather than the Beta-function
        simplification used here. Validated numerically: their formula and this module's exact
        closed form agree to 6 decimal places across several threshold values.

        """
        noise_looks, signal_looks = _resolve_look_counts(
            effective_looks_per_frame, signal_looks_per_frame
        )

        if signal_looks < noise_looks:
            rng = rng or np.random.default_rng()
            ref = _noise_only_cells(rng, num_trials, num_training_total, num_frames, noise_looks)
            cut = self.cut_power_samples(
                num_trials, num_frames, snr_linear, rng, noise_looks, signal_looks
            )
            return float(np.mean(cut > alpha * ref.mean(axis=1)))

        alpha_eff = alpha / (1 + snr_linear)
        a = noise_looks * num_frames
        b = num_training_total * noise_looks * num_frames
        w0 = alpha_eff / (num_training_total + alpha_eff)
        return beta_dist.sf(w0, a, b)  # pyright: ignore[reportReturnType]

    def os_cfar_pd(
        self,
        alpha: float,
        num_training_total: int,
        rank: int,
        num_frames: int,
        snr_linear: float,
        num_trials: int = 500_000,
        rng: np.random.Generator | None = None,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> float:
        """Pd for OS-CFAR: exact closed form at single-look num_frames == 1, Monte Carlo else.

        Single-look (num_frames == 1) derivation: the same alpha-rescaling technique employed in
        :meth:`ca_cfar_pd` applies equally here. Since the CUT is independent of the
        reference-cell order statistic X_(k), and the distribution of X_(k) does not depend on
        the target hypothesis::

            Pd = P(CUT > alpha * X_(k)) = E[exp(-alpha * X_(k) / (1 + snr_linear))]

        which is exactly :func:`_os_cfar_log_pfa`'s Pfa formula, equation (37)
        of [OS2]_, evaluated at
        alpha_eff = alpha / (1 + snr_linear). That form rests on Renyi's exponential
        order-statistic spacings [OS3]_, so it additionally requires single-LOOK reference cells
        (K_n == 1) and a band-filling target (K_s == K_n); band-integrated data breaks both and
        routes to Monte Carlo even at num_frames == 1.

        Multi-look (num_frames > 1): no closed form exists in the manner it does for CA-CFAR,
        which is bound up with the multi-target robustness OS-CFAR was introduced for [OS1]_.
        Order statistics of Gamma-distributed (M-frame-averaged) reference cells do not reduce
        to a tractable distribution the way a sum of Gammas does; Renyi's spacings
        representation, on which the num_frames == 1 case relies, is specific to exponential
        (single-frame) order statistics. Instead, Pd is estimated by simulation:

            1. Simulate num_trials realizations of N reference cells, each the mean of M iid
               Exponential(1) samples (noise-only, H0).
            2. Take the rank-th order statistic as the noise estimate, per trial.
            3. Simulate the CUT under H1 via :meth:`cut_power_samples`.
            4. Pd = fraction of trials where CUT > alpha * noise_estimate.

        Caveat: Monte Carlo estimates are sample-inefficient far in either tail (very low or
        very high Pd). Increasing num_trials yields a tighter estimate near the ends of a
        Pd-vs-SNR curve.

        References
        ----------
        .. [OS1] Rohling, H. "Radar CFAR Thresholding in Clutter and Multiple Target Situations."
               IEEE Transactions on Aerospace and Electronic Systems, AES-19(4), 608-621, 1983.
               Motivates OS-CFAR's use in multi-target situations, which is also why no closed
               form exists for Pd here the way it does for CA-CFAR.
        .. [OS2] Gandhi, P.P. and Kassam, S.A. "Analysis of CFAR Processors in Nonhomogeneous
               Background." IEEE Transactions on Aerospace and Electronic Systems, 24(4), 427-445,
               1988. Equation (37) gives the single-look closed form used here for a Swerling I
               target model, independently derived there; matches the rescaling derivation here
               and direct Monte Carlo simulation to within sampling noise (validated numerically).
        .. [OS3] Renyi, A. "On the theory of order statistics." Acta Mathematica Academiae
               Scientiarum Hungaricae, 4(3-4), 191-231, 1953.
        .. [OS4] Chalabi, I. "Application of CFAR detection to multiple pulses for gamma
               distributed clutter." Remote Sensing Letters, 13(10), 1011-1019, 2022.
               Independently derives the same multi-frame OS-CFAR problem (their eq. 15-20:
               order statistic across N reference cells, each an M-frame-integrated
               Gamma-distributed variable; general clutter shape, with our exponential case as
               their shape parameter = 1) and states explicitly that no closed form exists,
               matching the multi-look rationale above. Validated numerically: their eq. 20
               (shape=1) and this Monte Carlo path agree to within sampling noise across four
               threshold values. This Monte Carlo procedure independently reproduces the
               numerical approach [OS4]_ uses for the same problem, rather than being drawn from a
               single specific published method.

        """
        noise_looks, signal_looks = _resolve_look_counts(
            effective_looks_per_frame, signal_looks_per_frame
        )

        if num_frames == 1 and noise_looks == 1.0 and signal_looks == 1.0:
            alpha_eff = alpha / (1 + snr_linear)
            return float(np.exp(_os_cfar_log_pfa(alpha_eff, num_training_total, rank)))

        rng = rng or np.random.default_rng()
        ref = _noise_only_cells(rng, num_trials, num_training_total, num_frames, noise_looks)
        ref.sort(axis=1)
        noise_estimate = ref[:, rank - 1]
        cut = self.cut_power_samples(
            num_trials, num_frames, snr_linear, rng, noise_looks, signal_looks
        )
        return float(np.mean(cut > alpha * noise_estimate))


class NonFluctuating(FluctuationModel):
    """Non-fluctuating target (Swerling 0-equivalent): fixed amplitude every look.

    Under H1, the target has a fixed amplitude at every frame (only the ambient noise varies),
    in contrast to :class:`RayleighFluctuation`, where the target amplitude itself fluctuates
    randomly frame to frame. No closed form is implemented for either detector: combining a
    Gamma-distributed reference-cell sum (or its order statistics) with a noncentral-chi-squared
    CUT admits no closed form as convenient as the Rayleigh case's Beta-function relationship, so
    both :meth:`ca_cfar_pd` and :meth:`os_cfar_pd` rely on Monte Carlo estimation throughout. The
    CFAR Pfa side is unaffected either way (see :func:`solve_ca_cfar_alpha`: alpha
    calibration depends only on H0 statistics, which do not change with the target model).
    """

    def cut_power_samples(
        self,
        num_trials: int,
        num_frames: int,
        snr_linear: float,
        rng: np.random.Generator,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> FloatArray:
        """M-frame-averaged CUT power under H1 for a non-fluctuating target.

        Each signal-bearing look's complex voltage is a fixed signal (the phase is irrelevant,
        since only ``|signal + noise|**2`` is used, and is therefore fixed at 0 without loss of
        generality) plus independent unit-power circularly-symmetric complex Gaussian noise,
        the same physical noise model used elsewhere in this module, with a deterministic
        (rather than random) signal added on top. One such look has power
        ~ 0.5 * noncentral-chi-squared(df=2, nc=2*s) for per-look signal power s.

        Spreading the target's total power over K_s of the frame's K_n looks makes
        s = snr_linear * K_n / K_s, and summing K_s independent such looks adds both the
        degrees of freedom and the noncentralities::

            occupied looks ~ 0.5 * ncx2(df=2*K_s, nc=2*K_s*s) = 0.5 * ncx2(2*K_s, 2*snr*K_n)

        so the noncentrality depends only on K_n, the total signal energy is what it is,
        however few bins carry it. K_s controls only how many degrees of freedom that energy
        is spread across, which is exactly the fluctuation effect this parameter exists to
        capture. The remaining K_n - K_s looks contribute Gamma(K_n - K_s, 1) of pure noise,
        and dividing by K_n restores the unit-mean-noise scale.

        At K_n = K_s = 1 this reduces to 0.5 * ncx2(df=2, nc=2*snr_linear) per frame, the
        classic single-look model, validated numerically against scipy.stats.ncx2, with mean,
        variance, and quantiles all in agreement. At snr_linear=0 it reduces exactly to
        Gamma(K_n, 1/K_n), matching the noise-only model.

        """
        noise_looks, signal_looks = _resolve_look_counts(
            effective_looks_per_frame, signal_looks_per_frame
        )
        size = (num_trials, num_frames)

        occupied = 0.5 * rng.noncentral_chisquare(
            2.0 * signal_looks, 2.0 * snr_linear * noise_looks, size=size
        )
        remainder = _noise_only_looks(rng, noise_looks, signal_looks, size)

        return ((occupied + remainder) / noise_looks).mean(axis=1)

    def ca_cfar_pd(
        self,
        alpha: float,
        num_training_total: int,
        num_frames: int,
        snr_linear: float,
        num_trials: int = 500_000,
        rng: np.random.Generator | None = None,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> float:
        """Pd via Monte Carlo for CA-CFAR against a non-fluctuating target.

        See :meth:`cut_power_samples` for the full H1 derivation. No closed form is implemented
        for this fluctuation model.

        """
        noise_looks, signal_looks = _resolve_look_counts(
            effective_looks_per_frame, signal_looks_per_frame
        )
        rng = rng or np.random.default_rng()
        ref = _noise_only_cells(rng, num_trials, num_training_total, num_frames, noise_looks)
        noise_estimate = ref.mean(axis=1)
        cut = self.cut_power_samples(
            num_trials, num_frames, snr_linear, rng, noise_looks, signal_looks
        )
        return float(np.mean(cut > alpha * noise_estimate))

    def os_cfar_pd(
        self,
        alpha: float,
        num_training_total: int,
        rank: int,
        num_frames: int,
        snr_linear: float,
        num_trials: int = 500_000,
        rng: np.random.Generator | None = None,
        effective_looks_per_frame: float = 1.0,
        signal_looks_per_frame: float | None = None,
    ) -> float:
        """Pd via Monte Carlo for OS-CFAR against a non-fluctuating target.

        See :meth:`cut_power_samples` for the full H1 derivation. Unlike
        :class:`RayleighFluctuation` (which has an exact closed form at num_frames == 1), no
        closed form is implemented for this fluctuation model at any num_frames; Monte Carlo is
        used uniformly here.

        """
        noise_looks, signal_looks = _resolve_look_counts(
            effective_looks_per_frame, signal_looks_per_frame
        )
        rng = rng or np.random.default_rng()
        ref = _noise_only_cells(rng, num_trials, num_training_total, num_frames, noise_looks)
        ref.sort(axis=1)
        noise_estimate = ref[:, rank - 1]
        cut = self.cut_power_samples(
            num_trials, num_frames, snr_linear, rng, noise_looks, signal_looks
        )
        return float(np.mean(cut > alpha * noise_estimate))
