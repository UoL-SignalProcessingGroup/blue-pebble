"""CFAR (Constant False Alarm Rate) detectors for raw 2-D beamformed data.

Alpha, the threshold multiplier applied to the local noise estimate, is either given directly
(``threshold_factor``) or calibrated empirically from noise-only scans (``target_pfa`` with
``noise_calibration``); see "Threshold modes" in :class:`_CFARDetectorBase`.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks, peak_prominences
from stonesoup.base import Base, Property

from .calibration import NoiseCalibration

IntArray: TypeAlias = NDArray[np.integer[Any]]
DetectionArray: TypeAlias = NDArray[np.float64]

# Shared floor to keep log10/division finite at zero power without biasing real signals.
_EPS = np.finfo(np.float64).eps


def _stack_detections(indices: IntArray, data: ArrayLike) -> DetectionArray:
    """Create a ``(N, 2)`` matrix with detection indices and values."""
    data_array = np.asarray(data, dtype=np.float64)
    if indices.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((indices, data_array[indices])).astype(np.float64, copy=False)


def _to_db(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Convert a power ratio to dB, floored against zero/negative inputs via ``_EPS``."""
    return 10 * np.log10((numerator + _EPS) / (denominator + _EPS))


def _directional_power(data: ArrayLike) -> np.ndarray:
    """Per-look power from raw beamformed data, regardless of what the beamformer emits.

    ``BeamformedData`` (see ``bluepebble.types.sensordata``) is deliberately either complex
    or real: ``DelayAndSumBeamformer`` in ``'time'``/``'frequency'`` domain returns complex
    amplitude, so power is ``|amplitude|**2``. ``MinimumVarianceDistortionlessResponseBeamformer``
    and ``DelayAndSumBeamformer`` in ``'broadband_power'`` domain already return real-valued
    power directly; squaring that again would double-apply the power law (an MVDR-fed detector
    calibrated for Pfa=0.05 was once measured at Pfa~=0.09 from exactly this bug).

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

    A frame is one look within a single timestep, an STFT snapshot from that timestep's
    sample block, not a successive timestep. Detectors are called once per timestep and
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
            f"and estimate their own local noise floor; they no longer take a precomputed "
            f"SNR map as the old detection chain did. Pass data[:, None] for a single frame."
        )
    raise ValueError(
        f"Expected raw beamformed data with shape (num_beams, num_frames), got a "
        f"{data_array.ndim}-D array of shape {data_array.shape}. For banded data, pass each "
        f"band's map separately (see MultibandPassiveSonarDetector)."
    )


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
    """Shared implementation of the CFAR detectors on beamformed data.

    A cell is detected when its power, averaged over the frames of one timestep, exceeds a
    multiple (alpha) of the local noise estimate from the training cells either side of it::

        power > alpha * noise_estimate

    Subclasses supply only the noise estimate: the training-cell mean for CA-CFAR, a ranked
    training cell (order statistic) for OS-CFAR. This class sets alpha, applies the threshold,
    reports SNR in dB relative to the noise estimate, and consolidates peaks.

    Attributes
    ----------
    consolidate_peaks : bool
        Whether to reduce threshold crossings to one detection per source. See Peak
        consolidation.
    peak_distance : int
        Minimum separation, in bearing bins, between distinct detections, set by array
        resolution rather than source strength. See Peak consolidation.
    peak_prominence : float, optional
        Minimum prominence, in dB, for a candidate to count as an independent detection. See
        Peak consolidation.
    wrap_pad_width : int, optional
        Bearing-bin margin for circular wrap-padding; must cover the widest expected candidate
        cluster (worst-case leakage/sidelobe footprint), the same sizing consideration as
        num_guard_cells. Deliberately decoupled from peak_distance so narrowing peak_distance to
        a resolution floor doesn't silently shrink the wrap safety margin too. Defaults to
        peak_distance if unset, which is only safe if peak_distance itself is still sized for
        worst-case leakage rather than pure angular resolution.
    circular : bool
        Whether the bearing axis wraps (True for a full -180..180 sweep).
    num_guard_cells : int
        CFAR guard cells on each side of the CUT (cell under test), in bearing bins. Size to the
        worst-case (strong-source) sidelobe leakage extent, the same sizing consideration as
        wrap_pad_width (peak consolidation side vs. CFAR side).
    num_training_cells : int
        CFAR reference cells on each side of the guard cells.
    threshold_factor : float, optional
        Alpha, given directly, for fixed mode. See Threshold modes.
    target_pfa : float, optional
        Desired noise-only probability of false alarm, for calibrated mode. See Threshold modes.
    noise_calibration : NoiseCalibration, optional
        Empirical calibration from noise-only scans, for calibrated mode. See Noise calibration.

    Threshold modes
    ---------------
    Alpha is set in one of two ways:

    - **Calibrated:** ``target_pfa`` with ``noise_calibration``, and ``threshold_factor`` left
      as ``None``. Alpha is read from the calibration for the requested Pfa at each frame count
      (see Noise calibration), so the noise-only false-alarm rate is the one requested, within
      the limits :mod:`.calibration` documents.
    - **Fixed:** ``threshold_factor`` alone, with ``target_pfa`` and ``noise_calibration`` left
      as ``None``. Alpha is ``threshold_factor`` at every frame count, and nothing ties it to a
      false-alarm rate: the same factor gives a different Pfa as ``num_frames``, the training
      window, ``rank`` or the noise spectrum changes. It suits data with no noise-only segment
      to calibrate on, or cases where alpha itself must stay fixed. Factors from releases
      before v0.3.0 are not directly comparable, since those thresholded a precomputed SNR map
      rather than frame-averaged power.

    Any other combination raises ``ValueError``. Construction rejects setting both or neither of
    ``threshold_factor`` and ``target_pfa``. A calibration is normally attached after
    construction, so every ``detect()`` checks the full combination again, which also catches
    properties changed after construction.
    :meth:`~.NoiseCalibrator.calibrate_from_noise` refuses a detector in fixed mode.

    Noise calibration
    -----------------
    Calibrated mode needs a :class:`~.NoiseCalibration` because no single look count makes an
    i.i.d.-Gamma model exact on beamformer output: correlation between beams and a
    heavier-than-Gamma tail remain regardless (see :mod:`._theory` for that model). Build one
    with ``NoiseCalibrator(detector).calibrate_from_noise(noise_scans)`` on noise-only scans
    produced exactly like the operational data; alpha for any ``target_pfa`` is then read from
    the measured distribution (generalised Pareto tail below the directly observable rate), so
    one calibration serves every ``target_pfa``. A calibration is specific to the detector's
    window, rank and edge handling and to the frame count; a mismatch raises at detection time.

    Sidelobe false alarms
    ---------------------
    Pfa, whether requested through ``target_pfa`` or implied by ``threshold_factor``, is the
    noise-only false-alarm rate: the probability that a cell containing only noise crosses the
    threshold, as in the CFAR literature's definition. It does not cover every false alarm in a
    scene: sidelobes, leakage and multipath from real sources cross the threshold at bearings
    where no source is present. They are false alarms, and are counted as false positives by
    :mod:`.metrics`, but they are not noise, so neither threshold mode controls them. Their rate
    depends on source strength and array design. With many looks a calibrated alpha approaches 1
    (0 dB), which is correct for noise, and every sidelobe of a strong source crosses. Reduce
    sidelobes at the beamformer (``DelayAndSumBeamformer(shading=...)``). ``peak_prominence``
    also rejects them, but as a fixed dB criterion outside the Pfa model.

    Peak consolidation
    ------------------
    Neighbouring cells that cross the threshold are reduced to one detection per source by two
    independent criteria. They address different failure modes, so apply them together; neither
    is sufficient for the other's.

    - ``peak_distance`` is a fixed lower bound on angular separation, set by the array's
      resolution. Candidates closer than this are attributed to one source whatever the depth
      of any minimum between them. It is needed because the null between a mainlobe and its
      adjacent sidelobe is a genuine drop in received power, not measurement noise, so
      prominence alone cannot tell a sidelobe from an independent source.
    - ``peak_prominence`` is an adaptive threshold on how far a candidate's local maximum rises
      above its surrounding minimum, which accommodates sidelobe leakage whose extent varies
      with source strength.

    ``consolidate_peaks=False`` reports every cell above the threshold instead of one per
    source. This exists to make the achieved Pfa observable: consolidation merges adjacent
    crossings, so the number of reported detections falls below the per-cell rate as soon as
    crossings stop being sparse. Measured on noise-only data at 361 beams, reported detections
    hold at the requested Pfa up to about 0.05, then fall away: roughly 78% of crossings at
    Pfa 0.2, 57% at 0.5, and 37% at 0.9. Unconsolidated output tracks the per-cell rate across
    that whole range, which makes it the way to verify a calibration, to measure the Pfa a
    fixed ``threshold_factor`` achieves, to run ROC/PR sweeps that need the full
    false-positive range, and to compare against theoretical curves.

    It is a diagnostic mode, not an operational one. Without consolidation a single source
    reports once per bearing bin its mainlobe and sidelobes cover, so anything downstream that
    assumes one detection per source (a tracker's data associator above all) will be
    swamped. Leave it True for detection; switch it off to measure.

    Assumptions
    -----------
    - The training cells see the same noise level as the CUT. Where the noise level changes
      within the window (near a strong interferer, or at the edges of a non-circular sweep),
      the noise estimate and hence the threshold are biased.
    - The guard cells cover the target's own spread. A target leaking into the training cells
      raises its noise estimate and can mask itself; OS-CFAR tolerates up to
      ``2 * num_training_cells - rank`` contaminated training cells, CA-CFAR none.
    - In calibrated mode, the operational noise matches the calibration scans in everything
      :mod:`.calibration` lists (array, shading, beamformer, band, ambient spectrum, scan
      length). The detector checks only the settings it can see: its window, rank, edge
      handling and the frame count.

    Relation to sonar noise normalisation
    -------------------------------------
    In sonar, estimating the local background and dividing each cell by it is called
    normalisation [NN2]_, and it is applied across bearing to beamformed towed-array data as well
    as across frequency [NN1]_. [NN2]_ (Sects. 8.6 and 9.3) names the sonar normalisers after the
    CFAR processors directly, as cell-averaging and order-statistic CFAR normalisers, and takes the
    auxiliary data for broadband energy detection from nearby beams (Sect. 9.3.1). These
    detectors perform that normalisation across bearing: the power-to-noise-estimate ratio they
    threshold is the normalised output, and alpha is the detection threshold applied to it.

    - CA-CFAR corresponds to a split-window normaliser: the noise estimate averages beams either
      side of the cell of interest, excluding a central gap (the guard cells), which [NN2]_
      (Fig. 8.23) places to account for signal spreading. The two-pass split-window normaliser
      of [NN1]_ also replaces cells above a "shearing threshold" with the local mean before
      averaging again, to keep strong signals out of the estimate; CA-CFAR has no such pass and
      relies on the guard cells alone.
    - OS-CFAR is the order-statistic normaliser of [NN2]_ (Sect. 8.6.3), which trades some
      performance in a benign background for robustness to interfering signals in the
      auxiliary data. [NN2]_ (Sect. 8.6.3.3) recommends a rank between 3/4 and 7/8 of the
      reference cells.

    [NN1]_ reports that strong signals near endfire leak into neighbouring beams and bias
    split-window noise estimates, and that the few beams in a broadband bearing record make the
    edge beams hard to normalise, which bears on ``circular`` and on sidelobe-induced false
    alarms. Its shearing threshold is derived assuming Rayleigh-distributed envelope noise
    averaged over a known number of statistically independent beams, the same kind of
    assumption that ``noise_calibration`` removes, and it does not set the detection threshold
    for a stated false-alarm rate. [NN2]_ does, with the same closed forms used here
    (Eqs. (8.336) and (8.372) for single-look CA and OS), assuming independent exponentially
    distributed auxiliary data that is also independent of the test cell.

    References
    ----------
    .. [NN1] Stergiopoulos, S. "Noise normalization technique for beamformed towed array data."
             Journal of the Acoustical Society of America, 97(4), 2334-2345, 1995.
             doi:10.1121/1.411958
    .. [NN2] Abraham, D. A. "Underwater Acoustic Signal Processing: Modeling, Detection, and
             Estimation." Springer, Cham, 2019. doi:10.1007/978-3-319-92983-5

    """

    consolidate_peaks: bool = Property(
        default=True,
        doc="Whether to reduce CFAR-passing cells to one detection per source. Leave True for "
        "operational detection. False reports every cell above the threshold, which is a "
        "diagnostic mode; see the class docstring.",
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
        "peak_distance if unset; see class docstring for when that's unsafe.",
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
    threshold_factor: float | None = Property(
        default=None,
        doc="Linear threshold multiplier (alpha) on the local noise estimate, used as given with "
        "no false-alarm guarantee. Set alone, never with target_pfa or noise_calibration.",
    )
    target_pfa: float | None = Property(
        default=None,
        doc="Desired probability of false alarm. Alpha is calibrated automatically per num_frames "
        "seen at each detect() call. Requires noise_calibration; never set with "
        "threshold_factor.",
    )
    noise_calibration: NoiseCalibration | None = Property(
        default=None,
        doc="Empirical calibration from noise-only scans (see calibration.NoiseCalibrator). "
        "Must match this detector's window, rank, edge handling and the data's frame count. "
        "Required with target_pfa; must be None when threshold_factor is set.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        # Caught before Stone Soup's Base sees them: an unknown kwarg there surfaces as
        # "unexpected keyword argument", which names the mistake but not its replacement.
        for removed, guidance in _REMOVED_DETECTOR_KWARGS.items():
            if removed in kwargs:
                raise TypeError(
                    f"{removed} is no longer a parameter of {type(self).__name__}. {guidance}"
                )

        super().__init__(*args, **kwargs)

        # A missing calibration is not checked here: NoiseCalibrator normally attaches it after
        # construction. _alpha_for repeats the full check before every detection.
        self._check_threshold_mode(require_calibration=False)
        if self.peak_distance < 1:
            raise ValueError(f"peak_distance ({self.peak_distance}) must be >= 1")
        if self.wrap_pad_width is not None and self.wrap_pad_width < 1:
            raise ValueError(f"wrap_pad_width ({self.wrap_pad_width}) must be >= 1")
        if self.num_guard_cells < 0:
            raise ValueError(f"num_guard_cells ({self.num_guard_cells}) must be >= 0")
        if self.num_training_cells < 1:
            raise ValueError(f"num_training_cells ({self.num_training_cells}) must be >= 1")
        if self.threshold_factor is not None:
            if not self.threshold_factor > 0:
                raise ValueError(f"threshold_factor ({self.threshold_factor}) must be > 0")
            # Stored as float so an integer factor (e.g. 2) isn't treated as an integer
            # parameter by sweep_detection_parameter, which would reject fractional values.
            self.threshold_factor = float(self.threshold_factor)
        if self.target_pfa is not None:
            if not 0 < self.target_pfa < 1:
                raise ValueError(f"target_pfa ({self.target_pfa}) must be in (0, 1)")

        # Alpha depends only on num_frames and the calibration parameters (not on the data
        # itself), so it's cheap to memoize across detect() calls that share a frame count.
        # Avoids re-solving/re-simulating. The cache is keyed by num_frames and invalidated
        # whenever _calibration_signature() changes; see _alpha_for.
        self._alpha_cache: dict[int, float] = {}
        self._alpha_cache_signature: tuple = self._calibration_signature()

    @property
    def num_training_total(self) -> int:
        """Total reference cells, both sides of the CUT.

        Derived on access rather than stored at construction, so it can never disagree with
        ``num_training_cells`` if that property is changed later (e.g. by a parameter sweep).
        """
        return 2 * self.num_training_cells

    def _calibration_signature(self) -> tuple:
        """Every parameter alpha depends on, besides num_frames."""
        # The calibration object itself, not id(): NoiseCalibration compares by identity, and
        # holding the reference here keeps it alive, so a freed calibration's id can never be
        # reused by a replacement and mistaken for it.
        return (
            self.target_pfa,
            self.num_training_cells,
            self.noise_calibration,
        )

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

    def _check_threshold_mode(self, require_calibration: bool = True) -> None:
        """Raise unless threshold_factor xor (target_pfa and noise_calibration) is set."""
        name = type(self).__name__
        if self.threshold_factor is not None and self.target_pfa is not None:
            raise ValueError(
                f"{name} has both threshold_factor and target_pfa set. Use threshold_factor "
                "alone for a fixed threshold, or target_pfa with a noise_calibration for a "
                "calibrated one."
            )
        if self.threshold_factor is not None and self.noise_calibration is not None:
            raise ValueError(
                f"{name} has threshold_factor set and a noise_calibration attached; a "
                "calibration only applies with target_pfa. Remove the calibration, or set "
                "threshold_factor to None and target_pfa instead."
            )
        if self.threshold_factor is None and self.target_pfa is None:
            raise ValueError(
                f"{name} needs a threshold: set threshold_factor for a fixed one, or target_pfa "
                "with a noise_calibration for a calibrated one."
            )
        if require_calibration and self.target_pfa is not None and self.noise_calibration is None:
            raise ValueError(
                f"{name} has target_pfa set but no noise_calibration. Build one with "
                "calibration.NoiseCalibrator(detector).calibrate_from_noise(noise_scans) before "
                "calling detect(), or set threshold_factor (and target_pfa to None) for a fixed "
                "threshold."
            )

    def _alpha_for(self, num_frames: int) -> float:
        """Return the threshold multiplier for a given number of frames.

        ``threshold_factor`` is returned as given when set. Otherwise alpha is read from
        ``noise_calibration`` and memoised per num_frames. Stone Soup properties are ordinary
        mutable attributes, so the cache is discarded if any calibration parameter has changed
        since it was filled; otherwise a detector reconfigured after first use (directly, or on a
        deep copy during a sweep) would silently keep thresholding with the old alpha.
        """
        # Checked on every call, not just on a cache miss: a calibrated detector with a cached
        # alpha that later has threshold_factor set must raise, not detect in a mixed mode.
        self._check_threshold_mode()
        if self.threshold_factor is not None:
            return float(self.threshold_factor)

        signature = self._calibration_signature()
        if signature != self._alpha_cache_signature:
            self._alpha_cache = {}
            self._alpha_cache_signature = signature
        if num_frames not in self._alpha_cache:
            # Calibrated mode: _check_threshold_mode guarantees both are set.
            calibration = cast(NoiseCalibration, self.noise_calibration)
            mismatch = calibration.matches(self, num_frames)
            if mismatch is not None:
                raise ValueError(
                    f"noise_calibration does not apply to this {type(self).__name__}: "
                    f"{mismatch}. Recalibrate with NoiseCalibrator for these settings."
                )
            self._alpha_cache[num_frames] = calibration.alpha(
                cast(float, self.target_pfa), num_frames
            )
        return self._alpha_cache[num_frames]

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

    def detection_snr_map(self, data: ArrayLike) -> np.ndarray:
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
            # Padding width is wrap_pad_width, NOT peak_distance; see class docstring, only
            # safe to conflate the two if peak_distance is still sized for worst-case leakage
            # rather than pure angular resolution.
            pad = self._wrap_pad
            padded = np.pad(sparse, pad_width=pad, mode="wrap")
            padded_indices, _ = find_peaks(padded, distance=self.peak_distance)
            # Map padded indices back to original bearing bins; modulo handles peaks that
            # were only visible inside the wrapped padding itself.
            indices = np.unique((padded_indices - pad) % num_beams)
        else:
            indices, _ = find_peaks(sparse, distance=self.peak_distance)

        return self._filter_by_prominence(snr_db, indices, num_beams)

    def _filter_by_prominence(
        self, snr_db: np.ndarray, indices: IntArray, num_beams: int
    ) -> IntArray:
        """Keep consolidated peaks whose prominence in the full SNR map meets peak_prominence.

        Prominence is measured on the SNR map itself, not the sparse candidate map used for
        consolidation: there, every non-candidate cell is -inf, so any candidate isolated by
        sub-threshold cells (e.g. a sidelobe separated from its mainlobe by a null) would
        have infinite prominence and could never be rejected.

        A candidate that is not a local maximum of the SNR map (a shoulder of a higher,
        sub-threshold cell) has zero prominence and is dropped.
        """
        if self.peak_prominence is None or len(indices) == 0:
            return indices

        if self.circular:
            # Three copies give every peak the full circle on both sides to search for its
            # bases, which is what prominence on a circular axis means.
            extended = np.concatenate([snr_db, snr_db, snr_db])
            positions = indices + num_beams
        else:
            extended = snr_db
            positions = indices

        with warnings.catch_warnings():
            # scipy flags zero-prominence (non-maximum) positions with PeakPropertyWarning, a
            # RuntimeWarning subclass that is not exported publicly; those are dropped below.
            warnings.simplefilter("ignore", RuntimeWarning)
            prominences, _, _ = peak_prominences(extended, positions)
        return indices[prominences >= self.peak_prominence]

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
        # Reported SNR is noise-relative rather than threshold-relative, so it matches
        # detection_snr_map() and reads as a physical quantity independent of the chosen
        # Pfa or rank.
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
    impulsive interference: CA-CFAR's mean-based noise estimate has no way to reject
    a contaminated training cell the way OS-CFAR's order statistic can. Kept for
    codebase consistency and as a benchmark; see OSCFARDetector as the default choice.

    See :class:`~.algorithms._CFARDetectorBase` for ``num_guard_cells``, ``num_training_cells``,
    ``threshold_factor`` and ``target_pfa``; CA-CFAR adds no attributes of its own.

    """

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

    See :class:`~.algorithms._CFARDetectorBase` for ``num_guard_cells``, ``num_training_cells``,
    ``threshold_factor`` and ``target_pfa``.

    Attributes
    ----------
    rank : int
        k-th smallest training-cell value (1-indexed) used as the noise estimate. Avoid low values
        (e.g. 1) in multi-target / leakage-prone environments. A common starting point is
        ~0.75 * (2 * num_training_cells).
    mc_trials : int
        Deprecated and ignored: alpha is calibrated empirically now (see ``noise_calibration``),
        not by any model, Monte Carlo or otherwise.
    rng : np.random.Generator, optional
        Deprecated and ignored: alpha is calibrated empirically now (see ``noise_calibration``),
        not by any model, Monte Carlo or otherwise.

    Notes
    -----
    In calibrated mode, alpha is memoised per ``num_frames``, so calibration is paid once per
    distinct frame count per detector instance rather than on every :meth:`detect` call.
    Subsequent calls at the same frame count cost microseconds. Changing a calibration parameter
    (``target_pfa``, ``num_training_cells`` or ``rank``) discards the memoised values. In fixed
    mode alpha is ``threshold_factor`` itself and nothing is memoised.

    :func:`~.metrics.sweep_detection_parameter` deep-copies the detector per swept value, so a
    sweep pays calibration once per point.

    """

    rank: int = Property(
        default=1,
        doc="k-th smallest training-cell value (1-indexed). Avoid low values in multi-target / "
        "leakage-prone environments.",
    )
    mc_trials: int = Property(
        default=200_000,
        doc="Deprecated and ignored: alpha is calibrated empirically now (see noise_calibration), "
        "not by any model, Monte Carlo or otherwise.",
    )
    rng: np.random.Generator | None = Property(
        default=None,
        doc="Deprecated and ignored: alpha is calibrated empirically now (see noise_calibration), "
        "not by any model, Monte Carlo or otherwise.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the OS-CFAR detector with the given parameters."""
        super().__init__(*args, **kwargs)

        if not 1 <= self.rank <= self.num_training_total:
            raise ValueError(
                f"Rank ({self.rank}) must be between 1 and "
                f"2 * num_training_cells ({self.num_training_total})"
            )
        # A low rank picks the noise estimate from the small-value tail of the training cells,
        # which underestimates the true noise floor whenever some training cells are
        # contaminated by a second target or sidelobe leakage; warn rather than fail, since
        # rank=1 (minimum statistic) is a legitimate choice in genuinely clean environments.
        if self.rank < 0.5 * self.num_training_total:
            warnings.warn(
                f"rank={self.rank} is low relative to num_training_total="
                f"{self.num_training_total}; likely underestimates the "
                "noise floor in multi-target or leakage-prone environments.",
                stacklevel=2,
            )
        if self.mc_trials != 200_000 or self.rng is not None:
            warnings.warn(
                "OSCFARDetector mc_trials and rng are deprecated and ignored: alpha is "
                "calibrated empirically now (see noise_calibration), not by any model.",
                DeprecationWarning,
                stacklevel=2,
            )

    def _calibration_signature(self) -> tuple:
        """Return the base calibration parameters plus rank."""
        return (*super()._calibration_signature(), self.rank)

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
