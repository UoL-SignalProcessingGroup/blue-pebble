"""Detection probability (Pd) under different target fluctuation models.

A :class:`FluctuationModel` determines the H1 (target-present) statistics of the CUT, and
therefore Pd, for a given detector, since Pd depends on how the target's amplitude fluctuates
from look to look, not merely on its mean SNR. Alpha/Pfa calibration
(:func:`~.algorithms.solve_ca_cfar_alpha`, :func:`~.algorithms.solve_os_cfar_alpha_single_look`,
:func:`~.algorithms.calibrate_os_cfar_alpha_mc`) resides in :mod:`.algorithms` instead, and
remains shared across every fluctuation model defined here: it depends only on H0 (noise-only)
statistics, which do not change with how the target fluctuates. Correspondingly,
``CACFARDetector``/``OSCFARDetector`` never call anything in this module; only :mod:`.metrics`'s
theoretical ROC curve functions do, via a ``model: FluctuationModel`` parameter rather than by
selecting a Pd function by name.

Fluctuation models
-------------------
- **Rayleigh-fading** (:class:`RayleighFluctuation`; Swerling II-equivalent): the target return
  is itself fully random, Rayleigh-amplitude / Exponential-power distributed, and decorrelated
  frame to frame -- the same statistical family as noise, scaled only by (1 + snr_linear). A
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
  correlates neighbouring bins). This is the same quantity
  :func:`~.algorithms.calibrate_os_cfar_alpha_mc` calibrates alpha against, and
  :func:`~.metrics.estimate_effective_looks_per_frame` measures it from simulator output.
- ``signal_looks_per_frame`` (K_s): how many of those looks the TARGET actually occupies.
  Defaults to K_n, i.e. a target whose energy fills the processed band the way the noise does.

Noise always fills the band, so H0 depends on K_n alone -- which is why alpha calibration needs
only that one number. H1 does not: a narrowband source in a wide processing band puts its energy
in a handful of bins while noise arrives in all of them, making the cell a MIXTURE of K_s
signal-bearing looks and K_n - K_s noise-only looks. Setting K_s = K_n when the target is
actually a tonal understates its look-to-look fluctuation badly, because a signal spread over
many looks self-averages (CV ~ 1/sqrt(K_s)) and a signal confined to two does not. The mean is
unaffected -- ``snr_linear`` is defined on the integrated cell either way -- so this is purely a
statement about variance, and therefore about the shape of the Pd curve rather than its
midpoint.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist

from .algorithms import _os_cfar_log_pfa

FloatArray = NDArray[np.float64]


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
    signal_looks = noise_looks if signal_looks_per_frame is None else float(
        signal_looks_per_frame
    )

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
    """M-frame-averaged noise-only reference cells, unit mean, K_n looks per frame."""
    return rng.gamma(
        noise_looks, 1.0 / noise_looks, size=(num_trials, num_training_total, num_frames)
    ).mean(axis=2)


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
        and :func:`~.metrics.os_cfar_roc` reuses it directly for its shared-simulation sweep.

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
            Independent looks integrated into each per-frame sample (K_n), by default 1.0.
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
            Threshold multiplier, as calibrated by :func:`~.algorithms.solve_ca_cfar_alpha`.
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
            Independent looks integrated into each per-frame sample (K_n), by default 1.0.
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
            :func:`~.algorithms.solve_os_cfar_alpha_single_look` (num_frames == 1) or
            :func:`~.algorithms.calibrate_os_cfar_alpha_mc` (num_frames > 1). Pass the same
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
            Independent looks integrated into each per-frame sample (K_n), by default 1.0.
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
    looks being non-coherently integrated) -- the same statistical family as noise, differing
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
        At K_n = K_s > 1 it is Gamma(K_n, (1 + snr_linear) / K_n) -- the target fills the band,
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

        The derivation proceeds by writing CUT = (1 + snr_linear) * Z, where
        Z ~ Gamma(K_n*M, 1/(K_n*M)) (unit mean, M-frame averaged, K_n looks per frame) -- the
        same scale shared by the reference-cell sum. Then::

            Pd = P(CUT > alpha * mean_of_N_refs)
               = P(Z > [alpha / (1 + snr_linear)] * mean_of_N_refs)

        which is exactly the Pfa relationship with alpha replaced by
        alpha_eff = alpha / (1 + snr_linear). The Beta relationship established in
        :func:`~.algorithms.solve_ca_cfar_alpha` (W = CUT / (CUT + sum_of_refs) ~
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
               conventions reconcile exactly via alpha = N*T -- validated numerically to 6 decimal
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
            ref = _noise_only_cells(
                rng, num_trials, num_training_total, num_frames, noise_looks
            )
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

        which is exactly :func:`~.algorithms._os_cfar_log_pfa`'s Pfa formula evaluated at
        alpha_eff = alpha / (1 + snr_linear). That form rests on Renyi's exponential
        order-statistic spacings, so it additionally requires single-LOOK reference cells
        (K_n == 1) and a band-filling target (K_s == K_n); band-integrated data breaks both and
        routes to Monte Carlo even at num_frames == 1.

        Multi-look (num_frames > 1): no closed form exists in the manner it does for CA-CFAR.
        Order statistics of Gamma-distributed (M-frame-averaged) reference cells do not reduce
        to a tractable distribution the way a sum of Gammas does; Renyi's spacings
        representation, on which the num_frames == 1 case relies, is specific to exponential
        (single-frame) order statistics. Instead, the same simulation already used for alpha
        calibration in :func:`~.algorithms.calibrate_os_cfar_alpha_mc` is extended:

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

    Under H1, the target has a fixed amplitude at every frame -- only the ambient noise varies --
    in contrast to :class:`RayleighFluctuation`, where the target amplitude itself fluctuates
    randomly frame to frame. No closed form is implemented for either detector: combining a
    Gamma-distributed reference-cell sum (or its order statistics) with a noncentral-chi-squared
    CUT admits no closed form as convenient as the Rayleigh case's Beta-function relationship, so
    both :meth:`ca_cfar_pd` and :meth:`os_cfar_pd` rely on Monte Carlo estimation throughout. The
    CFAR Pfa side is unaffected either way (see :func:`~.algorithms.solve_ca_cfar_alpha`: alpha
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
        generality) plus independent unit-power circularly-symmetric complex Gaussian noise --
        the same physical noise model used elsewhere in this module, with a deterministic
        (rather than random) signal added on top. One such look has power
        ~ 0.5 * noncentral-chi-squared(df=2, nc=2*s) for per-look signal power s.

        Spreading the target's total power over K_s of the frame's K_n looks makes
        s = snr_linear * K_n / K_s, and summing K_s independent such looks adds both the
        degrees of freedom and the noncentralities::

            occupied looks ~ 0.5 * ncx2(df=2*K_s, nc=2*K_s*s) = 0.5 * ncx2(2*K_s, 2*snr*K_n)

        so the noncentrality depends only on K_n -- the total signal energy is what it is,
        however few bins carry it. K_s controls only how many degrees of freedom that energy
        is spread across, which is exactly the fluctuation effect this parameter exists to
        capture. The remaining K_n - K_s looks contribute Gamma(K_n - K_s, 1) of pure noise,
        and dividing by K_n restores the unit-mean-noise scale.

        At K_n = K_s = 1 this reduces to 0.5 * ncx2(df=2, nc=2*snr_linear) per frame, the
        classic single-look model -- validated numerically against scipy.stats.ncx2, with mean,
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
