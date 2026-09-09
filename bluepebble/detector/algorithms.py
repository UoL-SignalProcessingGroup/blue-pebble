"""Signal detection algorithms for 1D time-series and beamformed data.

Assumptions
-----------
Every Pfa/alpha calibration in this module rests on three assumptions about the input data. All
three can be violated in practice. The consequence of each violation is noted so a wrong Pfa shows
up as a specific, checkable symptom rather than an unexplained discrepancy:

- **Noise statistics.** Reference cells and the CUT are modelled as i.i.d. Exponential(1) per look.
  The statistics of square-law-detected magnitude from complex Gaussian noise. Under impulsive or
  heavy-tailed interference (e.g. snapping shrimp), the true tail is heavier than exponential, so
  the achieved Pfa will run higher than the calibrated target_pfa.
- **Independence across looks.** The M looks/frames averaged into a single cell are assumed
  statistically independent. If the integration period is shorter than the clutter's decorrelation
  time, the effective M is smaller than the nominal frame count. The Gamma(M, ...) model
  understates the true variance, so the calibrated alpha is too low and the achieved Pfa runs
  higher than target.
- **Homogeneous reference window.** All N reference cells are assumed to share the CUT's underlying
  noise level, differing only by random fluctuation. Clutter edges or interfering targets in the
  window violate this; OS-CFAR's order-statistic censoring tolerates a minority of contaminated
  cells (see OSCFARDetector), but calibrated Pfa is exact only under full homogeneity.

Pd (as opposed to Pfa/alpha) additionally depends on how the target's amplitude fluctuates from
look to look -- that's a separate, independent assumption, and lives in a separate module:
:mod:`.fluctuation_models`. Nothing in this module (including the detector classes) calls a Pd
function; only :mod:`.metrics`'s theoretical ROC curve functions do.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.signal import find_peaks
from scipy.stats import beta as beta_dist
from stonesoup.base import Base, Property

IntArray: TypeAlias = NDArray[np.integer[Any]]
DetectionArray: TypeAlias = NDArray[np.float64]

# Shared floor to keep log10/division finite at zero power without biasing real signals.
_EPS = np.finfo(np.float64).eps


def _empty_detections() -> DetectionArray:
    """Return a standard empty detection matrix of shape ``(0, 2)``."""
    return np.empty((0, 2), dtype=np.float64)


def _stack_detections(indices: IntArray, data: ArrayLike) -> DetectionArray:
    """Create a ``(N, 2)`` matrix with detection indices and values."""
    data_array = np.asarray(data, dtype=np.float64)
    if indices.size == 0:
        return _empty_detections()
    return np.column_stack((indices, data_array[indices])).astype(np.float64, copy=False)


def _to_db(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Convert a power ratio to dB, floored against zero/negative inputs via ``_EPS``."""
    return 10 * np.log10((numerator + _EPS) / (denominator + _EPS))


def _directional_power(data: ArrayLike) -> np.ndarray:
    """Per-look power from raw beamformed data, regardless of what the beamformer emits.

    ``BeamformedData`` (see ``bluepebble.types.sensordata``) is deliberately either complex
    or real: ``DelayAndSumBeamformer`` in ``'time'``/``'frequency'`` domain returns complex
    amplitude, so power is ``|amplitude|**2``. ``MinimumVarianceDistortionlessResponseBeamformer``,
    and ``DelayAndSumBeamformer`` in ``'broadband_power'`` domain, already return real-valued
    power directly -- squaring that again would double-apply the power law and break the
    Exponential(1)-per-look Pfa/alpha calibration this module depends on (an MVDR-fed detector
    calibrated for Pfa=0.05 was empirically measured at Pfa~=0.09 before this branch existed).

    Parameters
    ----------
    data : ArrayLike
        Raw beamformed data, shape (num_beams, num_frames); complex amplitude or real power.

    Returns
    -------
    np.ndarray
        Per-look power, same shape as ``data``.

    """
    data_array = np.asarray(data)
    return np.abs(data_array) ** 2 if np.iscomplexobj(data_array) else data_array


# Parameters removed by the single-detector refactor, mapped to what replaces them.
_REMOVED_DETECTOR_KWARGS = {
    "threshold_factor": (
        "Thresholds are now calibrated from a false-alarm rate: pass target_pfa instead. "
        "There is no fixed conversion -- the equivalent alpha depends on num_training_cells "
        "and the number of frames integrated."
    ),
    "mode": (
        "Edge handling is set by the circular flag: circular=True wraps (a full 360-degree "
        "bearing sweep), circular=False pads at the edges."
    ),
}


def _as_beamformed_2d(data: ArrayLike) -> np.ndarray:
    """Validate raw beamformed input, naming the pre-refactor call pattern when it appears.

    Detectors consume raw ``(num_beams, num_frames)`` data and estimate their own noise
    floor. Before the single-detector refactor they were handed a precomputed 1-D SNR map
    instead, so a 1-D array here is nearly always a caller that has not migrated yet. Say
    so directly, rather than failing later on a tuple unpack or a numpy axis error.

    A frame is one look within a single timestep -- an STFT snapshot from that timestep's
    sample block -- not a successive timestep. Detectors are called once per timestep and
    return that timestep's detections; nothing is buffered across timesteps. The frames
    axis has to survive as far as the detector because incoherent averaging over M looks
    narrows the noise distribution, so the threshold multiplier achieving a given Pfa
    depends on M; see :meth:`_CFARDetectorBase._alpha_for`. One frame is a valid input:
    pass ``data[:, None]``.

    Parameters
    ----------
    data : ArrayLike
        Raw beamformed data, expected shape (num_beams, num_frames).

    Returns
    -------
    np.ndarray
        The input as an array, unchanged.

    Raises
    ------
    ValueError
        If ``data`` is not two-dimensional.

    """
    data_array = np.asarray(data)
    if data_array.ndim == 2:
        return data_array
    if data_array.ndim == 1:
        raise ValueError(
            f"Expected raw beamformed data with shape (num_beams, num_frames), got a 1-D "
            f"array of length {data_array.size}. CFAR detectors consume raw beamformed data "
            f"and estimate their own local noise floor -- they no longer take a precomputed "
            f"SNR map as the old detection chain did. Pass data[:, None] for a single frame."
        )
    raise ValueError(
        f"Expected raw beamformed data with shape (num_beams, num_frames), got a "
        f"{data_array.ndim}-D array of shape {data_array.shape}. For banded data, pass each "
        f"band's map separately (see MultibandPassiveSonarDetector)."
    )


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
    Rohling's own equation (14) in [1] -- k*C(N,k)*(k-1)!*(T+N-k)!/(T+N)! reduces exactly to the
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
        :func:`calibrate_os_cfar_alpha_mc` for how to estimate it. Need not be an integer.

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
           reconciles with this module's mean-based alpha via alpha = N*T -- see
           fluctuation_models.RayleighFluctuation.ca_cfar_pd).
    .. [3] Chalabi, I. "Application of CFAR detection to multiple pulses for gamma distributed
           clutter." Remote Sensing Letters, 13(10), 1011-1019, 2022. Independently derives the
           same multi-look CA-CFAR Pfa (their eq. 8) via direct numerical integration rather than
           this function's Beta-function closed form -- see
           fluctuation_models.RayleighFluctuation.ca_cfar_pd for the numerical cross-check
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


def solve_os_cfar_alpha_single_look(
    target_pfa: float, num_training_total: int, rank: int
) -> float:
    """Exact alpha for target_pfa when num_frames == 1 (CORRECTED -- see module note).

    Solves Pfa(alpha) = target_pfa for alpha, using the corrected closed form from
    _os_cfar_log_pfa::

        target_pfa = prod_{l=0}^{k-1} (N - l) / (alpha + N - l)

    Pfa is strictly decreasing in alpha, so a standard root-finder (Brent's method)
    converges reliably.

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

    def f(alpha: float) -> float:
        return _os_cfar_log_pfa(alpha, num_training_total, rank) - np.log(target_pfa)

    return brentq(f, 1e-9, 1e9)  # pyright: ignore[reportReturnType]


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
           it -- see fluctuation_models.RayleighFluctuation.os_cfar_pd for the numerical
           cross-check against their eq. 20 (shape=1). This function calibrates alpha for the
           same model fluctuation_models.RayleighFluctuation.os_cfar_pd evaluates Pd against.

    """
    rng = rng or np.random.default_rng()

    # Each per-frame look is Gamma(K, 1/K): unit mean, K effective independent exponential
    # looks integrated per frame. K = 1 is exactly the classic Exponential(1) per-look model.
    k = effective_looks_per_frame

    # Simulate M-frame-averaged noise-only training cells: mean of `num_frames` unit-mean
    # per-frame looks per cell, for every training cell, over all trials.
    # Only the frame-average is ever used, and the mean of M iid Gamma(k, 1/k) draws is
    # exactly Gamma(M*k, 1/(M*k)) -- so the frame axis is drawn in closed form rather than
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


class DetectionAlgorithm(Base, ABC):
    """Abstract base class for all detector types."""

    @abstractmethod
    def detect(self, data: ArrayLike) -> DetectionArray:
        """Detect signals in the data array.

        Parameters
        ----------
        data : ArrayLike
            Numeric data to process. Exact shape and dtype requirements are defined by the concrete
            implementation.

        Returns
        -------
        DetectionArray
            Detection matrix of shape ``(N, 2)`` with columns
            ``[detection_index, detection_value]``. Returns an empty ``(0, 2)`` matrix when no
            detections are found.

        """
        ...


class _CFARDetectorBase(DetectionAlgorithm, ABC):
    """Shared plumbing for CFAR-family detectors on beamformed data.

    Handles threshold application, dB conversion, and wrap-aware peak consolidation. Concrete
    subclasses supply only the noise-floor estimator and alpha calibration strategy.

    peak_distance vs peak_prominence
    --------------------------------
    These two parameters constitute independent filtering criteria addressing distinct failure
    modes in peak consolidation.

    ``peak_distance`` imposes a fixed lower bound on angular separation, determined by the array's
    resolution. Candidates separated by less than this value are attributed to a single physical
    source irrespective of the magnitude of any intervening minimum. This criterion is required
    because the null between a mainlobe and its adjacent sidelobe constitutes a genuine reduction
    in received power rather than measurement noise, and is therefore indistinguishable from an
    independent source on the basis of prominence alone.

    ``peak_prominence`` imposes an adaptive threshold on the magnitude of a candidate's local
    maximum relative to its surrounding minimum, thereby accommodating sidelobe leakage whose
    spatial extent varies with source strength.

    The two criteria should be applied jointly; neither is sufficient to address the failure mode
    associated with the other.

    Turning consolidation off
    -------------------------
    ``consolidate_peaks=False`` reports every cell above the threshold instead of one per
    source. This exists to make the Pfa calibration observable: consolidation merges adjacent
    crossings, so the number of reported detections falls below the requested rate as soon as
    crossings stop being sparse. Measured on noise-only data at 361 beams, reported detections
    hold at the requested Pfa up to about 0.05, then fall away -- roughly 78% of crossings at
    Pfa 0.2, 57% at 0.5, and 37% at 0.9. Unconsolidated output tracks the requested Pfa across
    that whole range, which is what makes it useful for verifying calibration, for ROC/PR
    sweeps that need the full false-positive range, and for comparison against theoretical
    curves.

    It is a diagnostic mode, not an operational one. Without consolidation a single source
    reports once per bearing bin its mainlobe and sidelobes cover, so anything downstream that
    assumes one detection per source -- a tracker's data associator above all -- will be
    swamped. Leave it True for detection; switch it off to measure.

    Attributes
    ----------
    peak_distance : int
        A fixed lower bound, in bearing bins, on the angular separation between distinct
        detections, determined by array resolution rather than source strength. See class
        docstring.
    peak_prominence : float, optional
        An adaptive threshold, in decibels, on the prominence a candidate cluster must exhibit to
        be classified as an independent detection. See class docstring.
    wrap_pad_width : int, optional
        Bearing-bin margin for circular wrap-padding; must cover the widest expected candidate
        cluster (worst-case leakage/sidelobe footprint) -- the same sizing consideration as
        num_guard_cells on CFAR subclasses. Deliberately decoupled from peak_distance so narrowing
        peak_distance to a resolution floor doesn't silently shrink wrap safety margin too.
        Defaults to peak_distance if unset, which is only safe if peak_distance itself is still
        sized for worst-case leakage rather than pure angular resolution.
    circular : bool
        Whether the bearing axis wraps (True for a full -180..180 sweep).
    num_guard_cells : int
        CFAR guard cells on each side of the CUT, in bearing bins. Size to the worst-case
        (strong-source) sidelobe leakage extent -- the same sizing consideration as
        wrap_pad_width above (peak consolidation side vs. CFAR side).
    num_training_cells : int
        CFAR reference cells on each side of the guard cells.
    target_pfa : float
        Desired probability of false alarm. Alpha is calibrated automatically per num_frames seen
        at each detect() call.

    """

    consolidate_peaks: bool = Property(
        default=True,
        doc="Whether to reduce CFAR-passing cells to one detection per source. Leave True for "
        "operational detection. False reports every cell above the threshold, which is a "
        "diagnostic mode -- see the class docstring.",
    )
    peak_distance: int = Property(
        default=1,
        doc="Minimum bearing-bin separation between consolidated detections; also the "
        "wrap-padding width.",
    )
    peak_prominence: float | None = Property(
        default=None,
        doc="Minimum prominence (dB) for a cluster to count as an independent detection. Prefer "
        "tuning this over peak_distance.",
    )
    wrap_pad_width: int | None = Property(
        default=None,
        doc="Bearing-bin margin for circular wrap-padding; must cover the widest expected "
        "candidate cluster. Decoupled from peak_distance so narrowing peak_distance to a "
        "resolution floor doesn't silently shrink wrap safety margin too. Defaults to "
        "peak_distance if unset -- see class docstring for when that's unsafe.",
    )
    circular: bool = Property(
        default=True,
        doc="Whether the bearing axis wraps (True for a full -180..180 sweep).",
    )
    num_guard_cells: int = Property(
        doc="CFAR guard cells on each side of the CUT, in bearing bins.",
    )
    num_training_cells: int = Property(
        doc="CFAR reference cells on each side of the guard cells.",
    )
    target_pfa: float = Property(
        doc="Desired probability of false alarm. Alpha is calibrated automatically per num_frames "
        "seen at each detect() call.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        # Caught before Stone Soup's Base sees them: an unknown kwarg there surfaces as
        # "missing a required argument: 'target_pfa'", which names the replacement but not
        # the actual mistake.
        for removed, guidance in _REMOVED_DETECTOR_KWARGS.items():
            if removed in kwargs:
                raise TypeError(
                    f"{removed} is no longer a parameter of {type(self).__name__}. {guidance}"
                )
        super().__init__(*args, **kwargs)
        if self.peak_distance < 1:
            raise ValueError(f"peak_distance ({self.peak_distance}) must be >= 1")
        if self.wrap_pad_width is not None and self.wrap_pad_width < 1:
            raise ValueError(f"wrap_pad_width ({self.wrap_pad_width}) must be >= 1")
        if self.num_guard_cells < 0:
            raise ValueError(f"num_guard_cells ({self.num_guard_cells}) must be >= 0")
        if self.num_training_cells < 1:
            raise ValueError(f"num_training_cells ({self.num_training_cells}) must be >= 1")
        if not 0 < self.target_pfa < 1:
            raise ValueError(f"target_pfa ({self.target_pfa}) must be in (0, 1)")
        self.num_training_total: int = 2 * self.num_training_cells
        # Alpha depends only on num_frames (not on the data itself), so it's cheap to memoize
        # across detect() calls that share a frame count. Avoids re-solving/re-simulating.
        self._alpha_cache: dict[int, float] = {}

    @property
    def _wrap_pad(self) -> int:
        """Effective wrap-padding width: wrap_pad_width if set, else peak_distance."""
        return max(self.wrap_pad_width or self.peak_distance, 1)

    @property
    def _pad_mode(self) -> str:
        """Edge-handling mode for noise-floor padding.

        ``"wrap"`` for a circular bearing axis; ``"edge"`` (replicate the boundary cell) otherwise,
        so non-circular data never estimates noise from physically unrelated cells at the opposite
        end of the sweep.
        """
        return "wrap" if self.circular else "edge"

    @abstractmethod
    def _alpha_for(self, num_frames: int) -> float:
        """Return the calibrated threshold multiplier for a given number of frames."""
        ...

    @abstractmethod
    def _local_noise_floor(self, directional_power: np.ndarray) -> np.ndarray:
        """Per-bearing local noise floor estimate."""
        ...

    def _power_and_noise(self, data: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        # Power per beam, averaged (incoherently) across looks/frames. See
        # _directional_power for why this isn't simply |data|**2.
        data_array = _as_beamformed_2d(data)
        directional_power = np.mean(_directional_power(data_array), axis=1)
        noise_estimate = self._local_noise_floor(directional_power)
        return directional_power, noise_estimate

    def snr_map(self, data: ArrayLike) -> np.ndarray:
        """Full per-beam SNR (dB) relative to the local noise floor."""
        directional_power, noise_estimate = self._power_and_noise(data)
        return _to_db(directional_power, noise_estimate)

    def _consolidate_peaks(
        self, snr_db: np.ndarray, candidate_mask: np.ndarray, num_beams: int
    ) -> IntArray:
        """Reduce CFAR-passing candidate cells to one index per physical source."""
        # Sparse candidate map: real SNR at CFAR-passing cells, -inf elsewhere, so find_peaks
        # only ever consolidates among candidates.
        sparse = np.where(candidate_mask, snr_db, -np.inf)

        if self.circular:
            # Pad both ends with wrapped data so a peak straddling the 0/num_beams seam is
            # still found as one peak instead of two truncated ones at the array edges.
            # Padding width is wrap_pad_width, NOT peak_distance -- see class docstring; only
            # safe to conflate the two if peak_distance is still sized for worst-case leakage
            # rather than pure angular resolution.
            pad = self._wrap_pad
            padded = np.pad(sparse, pad_width=pad, mode="wrap")
            padded_indices, _ = find_peaks(
                padded, distance=self.peak_distance, prominence=self.peak_prominence
            )
            # Map padded indices back to original bearing bins; modulo handles peaks that
            # were only visible inside the wrapped padding itself.
            return np.unique((padded_indices - pad) % num_beams)

        indices, _ = find_peaks(
            sparse, distance=self.peak_distance, prominence=self.peak_prominence
        )
        return indices

    def detect(self, data: ArrayLike) -> DetectionArray:
        """Detect and consolidate signals from raw beamformed data.

        Parameters
        ----------
        data : ArrayLike
            Raw beamformed data, shape (num_beams, num_frames).

        Returns
        -------
        DetectionArray
            Shape (N, 2), columns [bearing_index, snr_db_relative_to_noise].

        """
        data_array = _as_beamformed_2d(data)
        num_beams, num_frames = data_array.shape

        directional_power, noise_estimate = self._power_and_noise(data_array)
        alpha = self._alpha_for(num_frames)
        threshold = alpha * noise_estimate
        # Reported SNR is noise-relative (not threshold-relative) so it matches snr_map() and
        # reads as a physical quantity independent of the chosen Pfa/rank.
        snr_db = _to_db(directional_power, noise_estimate)

        candidate_mask = directional_power > threshold
        if not np.any(candidate_mask):
            return _stack_detections(np.array([], dtype=int), snr_db)

        if not self.consolidate_peaks:
            # Every crossing, unmerged: the count is then a direct estimate of the achieved
            # Pfa, which consolidation otherwise suppresses. Diagnostic only.
            return _stack_detections(np.flatnonzero(candidate_mask), snr_db)

        indices = self._consolidate_peaks(snr_db, candidate_mask, num_beams)
        return _stack_detections(indices, snr_db)


class CACFARDetector(_CFARDetectorBase):
    """CA-CFAR detector.

    Takes raw beamformed data of shape (num_beams, num_frames) and returns detections
    where the per-beam power exceeds the CFAR threshold.

    Not recommended over OSCFARDetector for environments with multiple contacts or
    impulsive interference -- CA-CFAR's mean-based noise estimate has no way to reject
    a contaminated training cell the way OS-CFAR's order statistic can. Kept for
    codebase consistency and as a benchmark; see OSCFARDetector as the default choice.

    See :class:`_CFARDetectorBase` for ``num_guard_cells``, ``num_training_cells``, and
    ``target_pfa`` -- CA-CFAR adds no attributes of its own.

    """

    def _alpha_for(self, num_frames: int) -> float:
        if num_frames not in self._alpha_cache:
            self._alpha_cache[num_frames] = solve_ca_cfar_alpha(
                self.target_pfa, self.num_training_total, num_frames
            )
        return self._alpha_cache[num_frames]

    def _local_noise_floor(self, directional_power: np.ndarray) -> np.ndarray:
        """Per-bearing local noise floor via cell-averaging sliding window."""
        one_sided_window = self.num_guard_cells + self.num_training_cells
        kernel_size = 2 * one_sided_window + 1
        # Ones everywhere, then zero out the guard band (+ CUT) so the convolution below sums
        # only training cells: layout per side is [training][guard], CUT sits at the centre.
        kernel = np.ones(kernel_size)
        start_gap = self.num_training_cells
        end_gap = start_gap + (2 * self.num_guard_cells) + 1
        kernel[start_gap:end_gap] = 0

        padded_power = np.pad(directional_power, pad_width=one_sided_window, mode=self._pad_mode)  # pyright: ignore[reportCallIssue]
        # "valid" convolution with a window this size reproduces a per-cell sliding sum over
        # the padded array, aligned so output[i] corresponds to input cell i.
        noise_sum = np.convolve(padded_power, kernel, mode="valid")
        return noise_sum / self.num_training_total


class OSCFARDetector(_CFARDetectorBase):
    """OS-CFAR detector with wrap-aware peak consolidation.

    Takes raw beamformed data of shape (num_beams, num_frames) and returns consolidated detections;
    one per physical source, even when sidelobe leakage causes multiple adjacent bearing bins to
    exceed the CFAR threshold.

    See :class:`_CFARDetectorBase` for ``num_guard_cells``, ``num_training_cells``, and
    ``target_pfa``.

    Attributes
    ----------
    rank : int
        k-th smallest training-cell value (1-indexed) used as the noise estimate. Avoid low values
        (e.g. 1) in multi-target / leakage-prone environments. A common starting point is
        ~0.75 * (2 * num_training_cells).
    mc_trials : int
        Monte Carlo trials for num_frames > 1 alpha calibration. Scale up for smaller target_pfa
        (see calibrate_os_cfar_alpha_mc docstring).
    rng : np.random.Generator, optional
        Random number generator for num_frames > 1 Monte Carlo alpha calibration. Pass a seeded
        Generator for reproducible calibration -- e.g. so a deep-copied detector (as used when
        sweeping a parameter, see metrics.py) recalibrates deterministically instead of drawing
        fresh randomness each time. Defaults to a fresh unseeded Generator per calibration when
        unset.

    Notes
    -----
    Alpha is memoised per ``num_frames``, so calibration is paid once per distinct frame count
    per detector instance rather than on every :meth:`detect` call. Subsequent calls at the same
    frame count cost microseconds.

    ``num_frames == 1`` with ``effective_looks_per_frame == 1`` uses the closed form and is
    effectively free. Every other case falls back to Monte Carlo, which draws arrays sized
    ``(mc_trials, 2 * num_training_cells, num_frames)``. At the default 200,000 trials that is a
    few tenths of a second and around 0.8 GiB peak at ``num_training_cells=16, num_frames=8``,
    growing linearly in all three factors -- wide training windows at high frame counts can want
    several GiB. Reduce ``mc_trials`` if that is too much, accepting a noisier alpha.

    :func:`~.metrics.sweep_detection_parameter` deep-copies the detector per swept value, so a
    sweep pays calibration once per point. Pass a seeded ``rng`` to make those reproducible.

    """

    rank: int = Property(
        default=1,
        doc="k-th smallest training-cell value (1-indexed). Avoid low values in multi-target / "
        "leakage-prone environments.",
    )
    mc_trials: int = Property(
        default=200_000,
        doc="Monte Carlo trials for num_frames > 1 alpha calibration.",
    )
    rng: np.random.Generator | None = Property(
        default=None,
        doc="Random number generator for num_frames > 1 Monte Carlo alpha calibration. Defaults "
        "to a fresh unseeded Generator per calibration when unset.",
    )
    effective_looks_per_frame: float = Property(
        default=1.0,
        doc="Effective number of independent Exponential(1) looks integrated into each "
        "per-frame power sample fed to this detector. Leave at 1.0 for narrowband/single-bin "
        "data; broadband STFT beam power (a sum over many frequency bins per frame) needs "
        "this set to the data's effective per-frame degrees of freedom or the calibrated "
        "Pfa runs far below target. See calibrate_os_cfar_alpha_mc for how to estimate it.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the OS-CFAR detector with the given parameters."""
        super().__init__(*args, **kwargs)

        if self.effective_looks_per_frame <= 0:
            raise ValueError(
                f"effective_looks_per_frame ({self.effective_looks_per_frame}) must be positive"
            )
        if not 1 <= self.rank <= self.num_training_total:
            raise ValueError(
                f"Rank ({self.rank}) must be between 1 and "
                f"2 * num_training_cells ({self.num_training_total})"
            )
        # A low rank picks the noise estimate from the small-value tail of the training cells,
        # which underestimates the true noise floor whenever some training cells are
        # contaminated by a second target or sidelobe leakage -- warn rather than fail, since
        # rank=1 (minimum statistic) is a legitimate choice in genuinely clean environments.
        if self.rank < 0.5 * self.num_training_total:
            warnings.warn(
                f"rank={self.rank} is low relative to num_training_total="
                f"{self.num_training_total}; likely underestimates the "
                "noise floor in multi-target or leakage-prone environments.",
                stacklevel=2,
            )

    def _alpha_for(self, num_frames: int) -> float:
        """Get the calibrated alpha for a given number of frames.

        Parameters
        ----------
        num_frames : int
            The number of independent frames (averages).

        Returns
        -------
        float
            The calibrated alpha value.

        """
        if num_frames not in self._alpha_cache:
            if num_frames == 1 and self.effective_looks_per_frame == 1.0:
                # Exact closed form only holds for single-frame (exponential) statistics.
                alpha = solve_os_cfar_alpha_single_look(
                    self.target_pfa, self.num_training_total, self.rank
                )
            else:
                # M-frame (and/or band-integrated) power is Gamma-distributed; fall back to
                # Monte Carlo calibration.
                alpha = calibrate_os_cfar_alpha_mc(
                    self.target_pfa,
                    self.num_training_total,
                    self.rank,
                    num_frames,
                    num_trials=self.mc_trials,
                    rng=self.rng,
                    effective_looks_per_frame=self.effective_looks_per_frame,
                )
            self._alpha_cache[num_frames] = alpha
        return self._alpha_cache[num_frames]

    def _local_noise_floor(self, directional_power: np.ndarray) -> np.ndarray:
        """Per-bearing local noise floor via order-statistic sliding window.

        Parameters
        ----------
        directional_power : np.ndarray
            The directional power array of shape (num_beams,).

        Returns
        -------
        np.ndarray
            The local noise floor array of shape (num_beams,).

        """
        one_sided_window = self.num_guard_cells + self.num_training_cells
        window_size = 2 * one_sided_window + 1

        padded_power = np.pad(directional_power, pad_width=one_sided_window, mode=self._pad_mode)  # pyright: ignore[reportCallIssue]
        # Each row of `windows` is one CUT's full window: [training][guard] CUT [guard][training].
        windows = sliding_window_view(padded_power, window_size)

        # Guard cells (nearest the CUT on each side) are excluded from the noise estimate.
        # Only take the outer num_training_cells columns on each side.
        leading_cells = windows[:, : self.num_training_cells]
        lagging_cells = windows[:, -self.num_training_cells :]
        training_cells = np.concatenate((leading_cells, lagging_cells), axis=1)
        training_cells.sort(axis=1)

        # rank is 1-indexed (rank=1 -> minimum training-cell value).
        return training_cells[:, self.rank - 1]
