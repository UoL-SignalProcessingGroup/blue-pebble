"""Detection performance metrics for passive sonar systems.

This module has two complementary halves:

- **Empirical**: evaluate one or more CFAR-family detectors by sweeping a scalar
  parameter (e.g. ``target_pfa``) over raw beamformed data and comparing the resulting
  detections against ground-truth bearings (:class:`SweepSpec`, :func:`sweep_detection_parameter`,
  :func:`sweep_detection_parameter_multiband`).
- **Theoretical**: compute the Pd-vs-Pfa curve directly from the closed-form/Monte Carlo
  model in :mod:`.algorithms`, at an assumed target-power ratio and no data at all
  (:func:`ca_cfar_roc`, :func:`os_cfar_roc`).
  :func:`snr_linear_from_ground_truth_bearing` bridges the two by estimating the
  snr_linear actually present in a simulated scenario, so a theoretical curve can be
  parameterised to match the empirical one it's being compared against.

Both halves produce (or can produce) a :class:`SweepResult`, so a theoretical curve and
an empirical sweep can be plotted together via :func:`~bluepebble.plotter.plot_roc` with
no glue code. See :func:`sweep_detection_parameter` for a full runnable example of the
empirical path.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .algorithms import (
    _directional_power,
    solve_ca_cfar_alpha,
    solve_os_cfar_alpha_single_look,
)
from .fluctuation_models import FluctuationModel, RayleighFluctuation

if TYPE_CHECKING:
    from .algorithms import DetectionAlgorithm

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
DetectionArray: TypeAlias = NDArray[np.float64]

_DetectorState = TypeVar("_DetectorState")


class _BearingStateLike(Protocol):
    """Protocol for states carrying a bearing in ``state_vector``."""

    state_vector: FloatArray


class _GroundTruthPathLike(Protocol):
    """Protocol for Stone Soup-like ground-truth paths."""

    states: Sequence[_BearingStateLike]


@dataclass
class SweepSpec:
    """Specification for a single detector parameter sweep.

    Attributes
    ----------
    detector : DetectionAlgorithm
        The base detector (e.g. a configured :class:`~.CACFARDetector`).  Deep-copied
        for each parameter value so the original is never mutated.
    param_name : str
        Attribute name to sweep (e.g. ``"target_pfa"``).
    param_values : ArrayLike
        Sequence of values to evaluate, ordered from most permissive to most
        strict so ROC / PR curves trace in the canonical direction.
    label : str | None
        Human-readable name used in plot legends.  Defaults to
        ``"<param_name> sweep"`` when ``None``.

    """

    detector: DetectionAlgorithm
    param_name: str
    param_values: ArrayLike
    label: str | None = None


@dataclass
class _TimestepMetrics:
    """Raw confusion-matrix counts for a single timestep."""

    tp: int
    fp: int
    fn: int
    tn: int


def _compute_timestep_metrics(
    detected_bearings_rad: ArrayLike,
    ground_truth_bearings_rad: ArrayLike,
    association_threshold_rad: float,
    num_beam_cells: int,
) -> _TimestepMetrics:
    """Compute confusion-matrix counts for a single timestep.

    Greedy nearest-neighbour assignment is used: detections and ground-truth
    bearings are sorted by pairwise angular distance, and each is claimed by
    the closest partner not already matched.  Bearings are treated as circular
    (wrap-around handled via ``arctan2``).

    Parameters
    ----------
    detected_bearings_rad : ArrayLike
        Detected bearing angles in radians, shape ``(N_det,)``.
    ground_truth_bearings_rad : ArrayLike
        Ground-truth bearing angles in radians, shape ``(N_gt,)``.
    association_threshold_rad : float
        Maximum angular distance (rad) for a detection to be counted as a TP.
    num_beam_cells : int
        Total number of beam cells (used to estimate TN).

    Returns
    -------
    _TimestepMetrics
        Aggregated TP, FP, FN, TN counts.

    """
    detected_bearings = np.asarray(detected_bearings_rad, dtype=np.float64)
    ground_truth_bearings = np.asarray(ground_truth_bearings_rad, dtype=np.float64)
    n_det = len(detected_bearings)
    n_gt = len(ground_truth_bearings)

    if n_gt == 0:
        # No targets present — every detection is a false alarm
        return _TimestepMetrics(tp=0, fp=n_det, fn=0, tn=num_beam_cells)

    if n_det == 0:
        # No detections — every target is missed
        return _TimestepMetrics(tp=0, fp=0, fn=n_gt, tn=num_beam_cells - n_gt)

    # Pairwise circular angular distance matrix: shape (N_det, N_gt)
    diff = detected_bearings[:, np.newaxis] - ground_truth_bearings[np.newaxis, :]
    dist_matrix = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))

    # Greedy matching: claim closest unmatched pairs within threshold
    matched_det = np.zeros(n_det, dtype=bool)
    matched_gt = np.zeros(n_gt, dtype=bool)

    for flat_idx in np.argsort(dist_matrix.ravel()):
        det_i, gt_j = np.unravel_index(flat_idx, dist_matrix.shape)
        if dist_matrix[det_i, gt_j] > association_threshold_rad:
            break  # All remaining distances exceed the threshold
        if not matched_det[det_i] and not matched_gt[gt_j]:
            matched_det[det_i] = True
            matched_gt[gt_j] = True

    tp = int(np.sum(matched_gt))
    fp = int(np.sum(~matched_det))
    fn = int(np.sum(~matched_gt))
    # TN: non-target beam cells that were *not* detected as false alarms
    tn = max((num_beam_cells - n_gt) - fp, 0)

    return _TimestepMetrics(tp=tp, fp=fp, fn=fn, tn=tn)


def _safe_ratio(numerator: ArrayLike, denominator: ArrayLike, default: float) -> FloatArray:
    """Return ``numerator / denominator`` with a default where denominator is zero."""
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.full(denominator_array.shape, default, dtype=np.float64)
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array > 0)
    return result


@dataclass
class SweepResult:
    """Aggregated detection metrics from a parameter sweep.

    Constructed either from confusion-matrix counts (the empirical path -- see
    :func:`sweep_detection_parameter`) or directly from a theoretical Pd-vs-Pfa curve
    via :meth:`from_theoretical_roc`. Exactly one of the two must be supplied.

    Attributes
    ----------
    param_values : FloatArray
        The swept parameter values, shape ``(P,)``.
    tp : IntArray, optional
        Total true positives across all timesteps for each parameter value. ``None``
        for a result built via :meth:`from_theoretical_roc`.
    fp : IntArray, optional
        Total false positives across all timesteps for each parameter value. ``None``
        for a result built via :meth:`from_theoretical_roc`.
    fn : IntArray, optional
        Total false negatives across all timesteps for each parameter value. ``None``
        for a result built via :meth:`from_theoretical_roc`.
    tn : IntArray, optional
        Total true negatives across all timesteps for each parameter value. ``None``
        for a result built via :meth:`from_theoretical_roc`.
    label : str | None
        Human-readable name carried through from :class:`SweepSpec`, used in
        plot legends.

    """

    param_values: FloatArray
    tp: IntArray | None = None
    fp: IntArray | None = None
    fn: IntArray | None = None
    tn: IntArray | None = None
    label: str | None = None
    _fpr: FloatArray | None = None
    _tpr: FloatArray | None = None

    def __post_init__(self) -> None:
        """Enforce that exactly one of the two construction modes was supplied."""
        has_counts = self.tp is not None
        has_theoretical = self._fpr is not None
        if has_counts == has_theoretical:
            raise ValueError(
                "SweepResult needs exactly one of confusion-matrix counts "
                "(tp/fp/fn/tn) or a theoretical curve (via from_theoretical_roc) -- got "
                f"{'both' if has_counts else 'neither'}."
            )

    @classmethod
    def from_theoretical_roc(
        cls, pfa: ArrayLike, pd: ArrayLike, label: str | None = None
    ) -> SweepResult:
        """Build a SweepResult directly from a theoretical Pd-vs-Pfa curve.

        Unlike :func:`sweep_detection_parameter`, there are no confusion-matrix counts
        here: ``pfa`` IS the false-positive rate and ``pd`` IS the true-positive rate by
        construction (alpha is calibrated exactly to each requested Pfa -- see
        :func:`ca_cfar_roc`, :func:`os_cfar_roc`).

        ``.precision``, ``.f1``, and ``.auc_pr`` raise on the result this returns: a PR
        curve needs an assumed target prevalence that pure Pfa/Pd theory doesn't have.
        ``.fpr``, ``.tpr``, ``.auc_roc``, ``.best_param``, and ``.param_at_fpr`` all work
        normally, since they depend only on Pfa/Pd, not on raw counts.

        Parameters
        ----------
        pfa : ArrayLike
            The swept Pfa values -- doubles as ``param_values`` and as the false-positive
            rate, since a Pfa/Pd curve is defined by Pfa directly.
        pd : ArrayLike
            Pd (true-positive rate) at each ``pfa`` entry.
        label : str, optional
            Human-readable name used in plot legends.

        Returns
        -------
        SweepResult
            A result exposing ``.fpr``/``.tpr``/``.auc_roc``/``.best_param``/``.param_at_fpr``
            only.

        """
        pfa_array = np.asarray(pfa, dtype=np.float64)
        return cls(
            param_values=pfa_array,
            _fpr=pfa_array,
            _tpr=np.asarray(pd, dtype=np.float64),
            label=label,
        )

    def _require_counts(self, prop_name: str) -> None:
        if self.tp is None:
            raise ValueError(
                f"{prop_name} needs confusion-matrix counts (tp/fp/fn/tn), which this "
                "SweepResult doesn't have -- it was built via from_theoretical_roc() from "
                "a pure Pfa/Pd curve. A PR curve needs an assumed target prevalence that "
                "Pfa/Pd theory alone doesn't provide; use .fpr/.tpr/.auc_roc instead."
            )

    @property
    def precision(self) -> FloatArray:
        """Positive predictive value: TP / (TP + FP).

        Defaults to 1 where TP + FP = 0 (no detections issued).
        """
        self._require_counts("precision")
        denom = self.tp + self.fp
        return _safe_ratio(self.tp, denom, default=1.0)

    @property
    def recall(self) -> FloatArray:
        """Sensitivity / true positive rate: TP / (TP + FN).

        Defaults to 0 where TP + FN = 0 (no positives present). For a theoretical
        result (see :meth:`from_theoretical_roc`) this is simply the Pd curve it was
        built from.
        """
        if self._tpr is not None:
            return self._tpr
        denom = self.tp + self.fn
        return _safe_ratio(self.tp, denom, default=0.0)

    @property
    def tpr(self) -> FloatArray:
        """True positive rate (alias for :attr:`recall`)."""
        return self.recall

    @property
    def fpr(self) -> FloatArray:
        """False positive rate: FP / (FP + TN).

        Defaults to 0 where FP + TN = 0. For a theoretical result (see
        :meth:`from_theoretical_roc`) this is simply the Pfa curve it was built from.
        """
        if self._fpr is not None:
            return self._fpr
        denom = self.fp + self.tn
        return _safe_ratio(self.fp, denom, default=0.0)

    @property
    def f1(self) -> FloatArray:
        """Harmonic mean of precision and recall."""
        self._require_counts("f1")
        p, r = self.precision, self.recall
        denom = p + r
        return _safe_ratio(2.0 * p * r, denom, default=0.0)

    @property
    def auc_roc(self) -> float:
        """Area under the ROC curve over the full ``[0, 1]`` false-positive range.

        Every detector's ROC passes through ``(0, 0)`` -- the threshold so strict it rejects
        everything -- and ``(1, 1)`` -- the threshold so permissive it accepts everything.
        Neither is an assumption about this detector; both are degenerate operating points any
        detector has. A parameter sweep, though, rarely reaches either end, so those two points
        are added before integrating.

        Without them the area is computed over only the FPR range the sweep happened to cover,
        which silently rescales the result: a detector whose achieved Pfa saturates at 0.13 can
        score no higher than 0.13 no matter how perfectly it separates target from noise, making
        an excellent detector look worse than random. That failure mode is easy to hit on real
        beamformed data, where peak consolidation caps achieved Pfa well below 1 (see
        :func:`sweep_detection_parameter`).

        Returns
        -------
        float
            Area under the ROC curve, in ``[0, 1]``, directly comparable across sweeps
            regardless of the FPR range each one covered.

        Notes
        -----
        Between the last measured operating point and ``(1, 1)`` the curve is interpolated
        linearly, which corresponds to randomly mixing that operating point with the
        accept-everything detector -- an achievable strategy, so the result is a valid lower
        bound rather than an optimistic guess. It does mean that when a sweep covers only a
        small part of the FPR range, most of the area comes from that interpolated segment
        rather than from measured points. Check the span of :attr:`fpr` before reading much
        into a comparison between two sweeps that cover very different ranges.

        """
        order = np.argsort(self.fpr)
        fpr = self.fpr[order]
        tpr = self.tpr[order]

        if fpr.size == 0:
            return 0.0
        if fpr[0] > 0.0:
            fpr = np.concatenate(([0.0], fpr))
            tpr = np.concatenate(([0.0], tpr))
        if fpr[-1] < 1.0:
            fpr = np.concatenate((fpr, [1.0]))
            tpr = np.concatenate((tpr, [1.0]))

        return float(np.trapezoid(tpr, fpr))

    @property
    def auc_pr(self) -> float:
        """Area under the precision-recall curve, over the recall range the sweep covered.

        Unlike :attr:`auc_roc`, this is NOT extended to a fixed ``[0, 1]`` box. A PR curve has
        no equivalent of ROC's two degenerate corner points: precision at zero recall is
        undefined, and precision as recall approaches 1 depends on class balance rather than on
        the detector, so there is no endpoint that could be added without inventing information.

        The truncation caveat therefore applies here in full: a sweep whose recall spans only
        part of ``[0, 1]`` yields an area over that partial span, and two sweeps covering
        different recall ranges are not directly comparable. Compare the :attr:`recall` spans
        before comparing two values of this property.

        """
        self._require_counts("auc_pr")
        order = np.argsort(self.recall)
        return float(np.trapezoid(self.precision[order], self.recall[order]))

    @property
    def best_param(self) -> float:
        """Parameter value that maximises Youden's J statistic (TPR − FPR).

        Youden's J is the vertical distance above the random-classifier diagonal on the
        ROC curve.  The operating point with the highest J gives the best trade-off
        between sensitivity and specificity for this sweep.

        Returns
        -------
        float
            The swept parameter value achieving ``max(TPR − FPR)``.

        """
        j = self.tpr - self.fpr
        return float(self.param_values[np.argmax(j)])

    def param_at_fpr(self, target_fpr: float = 0.05) -> float:
        """Parameter value whose operating point is nearest to a given FPR.

        Selects the sweep index where ``|FPR − target_fpr|`` is minimised, breaking
        ties by preferring the index with the higher TPR.

        Parameters
        ----------
        target_fpr : float
            Desired false positive rate.  Defaults to ``0.05``.

        Returns
        -------
        float
            The swept parameter value whose FPR is closest to ``target_fpr``.

        """
        dist = np.abs(self.fpr - target_fpr)
        # Among all indices tied for minimum distance, prefer highest TPR
        min_dist = dist.min()
        candidates = np.where(dist == min_dist)[0]
        best = candidates[np.argmax(self.tpr[candidates])]
        return float(self.param_values[best])


def _clone_detector_with_param(spec: SweepSpec, value: float) -> DetectionAlgorithm:
    """Deep-copy a spec's detector with its swept parameter set to ``value``."""
    detector_copy = copy.deepcopy(spec.detector)
    setattr(detector_copy, spec.param_name, float(value))
    # A CFAR detector's alpha is memoized per num_frames (see _CFARDetectorBase). Deep-copying
    # a detector that was already used to detect() carries that stale cache along with it, so
    # the clone would keep reusing an alpha calibrated for the OLD parameter value instead of
    # recalibrating for `value`. Not every DetectionAlgorithm has this cache, hence the guard.
    if hasattr(detector_copy, "_alpha_cache"):
        detector_copy._alpha_cache = {}
    return detector_copy


def _sweep_confusion_counts(
    param_values: FloatArray,
    detector_factory: Callable[[float], _DetectorState],
    detected_beams_at_t: Callable[[_DetectorState, int], IntArray],
    gt_bearings_per_t: list[FloatArray],
    steering_azimuths: FloatArray,
    association_threshold_rad: float,
    num_timesteps: int,
    num_beams: int,
) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    """Accumulate TP/FP/FN/TN counts across timesteps for each swept parameter value.

    Shared by :func:`sweep_detection_parameter` and
    :func:`sweep_detection_parameter_multiband`; the two differ only in how a parameter
    value becomes detector state (``detector_factory``) and how that state and a timestep
    become detected beam indices (``detected_beams_at_t``).

    Parameters
    ----------
    param_values : FloatArray
        Swept parameter values, shape ``(P,)``.
    detector_factory : Callable[[float], _DetectorState]
        Builds whatever detector state ``detected_beams_at_t`` needs for one parameter
        value.  Called once per parameter value, not once per timestep.
    detected_beams_at_t : Callable[[_DetectorState, int], IntArray]
        Returns the detected beam indices for a timestep, given the detector state built
        for the current parameter value.
    gt_bearings_per_t : list[FloatArray]
        Ground-truth bearings per timestep, as returned by
        :func:`_bearings_from_ground_truth_paths`.
    steering_azimuths : FloatArray
        Beam steering angles in radians, shape ``(num_beams,)``.
    association_threshold_rad : float
        Maximum angular distance (rad) for a detection to count as a true positive.
    num_timesteps : int
        Number of timesteps to accumulate over.
    num_beams : int
        Total number of beam cells, used to estimate true negatives.

    Returns
    -------
    tuple of (IntArray, IntArray, IntArray, IntArray)
        ``(tp, fp, fn, tn)`` arrays, one entry per parameter value.

    """
    tp_arr: IntArray = np.zeros(len(param_values), dtype=np.int64)
    fp_arr: IntArray = np.zeros(len(param_values), dtype=np.int64)
    fn_arr: IntArray = np.zeros(len(param_values), dtype=np.int64)
    tn_arr: IntArray = np.zeros(len(param_values), dtype=np.int64)

    for p_idx, p_val in enumerate(param_values):
        detector_state = detector_factory(float(p_val))

        for t_idx in range(num_timesteps):
            beam_idx = detected_beams_at_t(detector_state, t_idx)
            det_bearings = (
                steering_azimuths[beam_idx] if beam_idx.size > 0 else np.empty(0, dtype=np.float64)
            )

            m = _compute_timestep_metrics(
                detected_bearings_rad=det_bearings,
                ground_truth_bearings_rad=gt_bearings_per_t[t_idx],
                association_threshold_rad=association_threshold_rad,
                num_beam_cells=num_beams,
            )

            tp_arr[p_idx] += m.tp
            fp_arr[p_idx] += m.fp
            fn_arr[p_idx] += m.fn
            tn_arr[p_idx] += m.tn

    return tp_arr, fp_arr, fn_arr, tn_arr


def sweep_detection_parameter(
    beamformed_data: Sequence[ArrayLike],
    sweep_specs: Sequence[SweepSpec],
    ground_truth_paths: Sequence[_GroundTruthPathLike],
    steering_azimuths_rad: ArrayLike,
    association_threshold_rad: float,
    bearing_state_index: int = 0,
) -> list[SweepResult]:
    """Sweep parameters for one or more detectors and compute ROC / PR metrics.

    For each :class:`SweepSpec` and each value in its ``param_values`` the function:

    1. Deep-copies the detector (the original is never mutated).
    2. Sets ``detector.<param_name>`` to the current sweep value.
    3. Runs the modified detector's ``detect()`` over every timestep's raw beamformed
       data, letting it compute its own noise floor and threshold from scratch.
    4. Associates detections with ground-truth bearings via greedy nearest-neighbour
       matching and accumulates TP / FP / FN / TN counts.

    The returned :class:`SweepResult` objects expose precision, recall, TPR, FPR,
    F1 and AUC values as properties.  Pass the list directly to
    :func:`~bluepebble.plotter.plot_roc_pr` to compare all detectors in one figure.

    Parameters
    ----------
    beamformed_data : Sequence[ArrayLike]
        Raw beamformed data, one entry per timestep, each of shape
        ``(num_beams, num_frames)`` — the same input a detector's own ``detect()``
        expects.  Since :class:`~.PassiveSonarDetector` doesn't retain raw frames after
        detecting, collect this separately, e.g. by saving
        ``sensor_data.beamformed_data`` while iterating ``simulator.sensor_data_gen()``.
    sweep_specs : Sequence[SweepSpec]
        One :class:`SweepSpec` per detector to evaluate.  Each spec carries its own
        detector, parameter name, sweep range, and optional legend label.
    ground_truth_paths : Sequence[_GroundTruthPathLike]
        Stone Soup ground-truth paths whose state vectors carry bearing values —
        typically ``relative_bearing_ground_truths`` from the simulation workflow.
        The number of timesteps is inferred from ``beamformed_data``.
    steering_azimuths_rad : ArrayLike
        Beam steering angles in radians, shape ``(N_beams,)``.  Maps a detection
        index back to a physical bearing.
    association_threshold_rad : float
        Maximum angular distance (rad) between a detection and a ground-truth
        bearing for the detection to count as a true positive.
    bearing_state_index : int
        Row index within each ground-truth state vector that holds the bearing
        in radians.  Defaults to ``0``.

    Returns
    -------
    list[SweepResult]
        One result per :class:`SweepSpec`, in the same order.

    Examples
    --------
    >>> import numpy as np
    >>> from bluepebble.detector.algorithms import CACFARDetector, OSCFARDetector
    >>> from bluepebble.detector.metrics import SweepSpec, sweep_detection_parameter
    >>> from bluepebble.plotter import plot_roc_pr
    >>>
    >>> beamformed_data = [
    ...     sensor_data.beamformed_data
    ...     for _, sensor_data_set in simulator.sensor_data_gen()
    ...     for sensor_data in sensor_data_set
    ... ]
    >>>
    >>> specs = [
    ...     SweepSpec(
    ...         detector=CACFARDetector(
    ...             num_guard_cells=2, num_training_cells=10, target_pfa=1e-2, peak_distance=3
    ...         ),
    ...         param_name="target_pfa",
    ...         param_values=np.geomspace(1e-1, 1e-5, 80),
    ...         label="CA-CFAR",
    ...     ),
    ...     SweepSpec(
    ...         detector=OSCFARDetector(
    ...             num_guard_cells=2, num_training_cells=10, target_pfa=1e-2, peak_distance=3
    ...         ),
    ...         param_name="target_pfa",
    ...         param_values=np.geomspace(1e-1, 1e-5, 80),
    ...         label="OS-CFAR",
    ...     ),
    ... ]
    >>> results = sweep_detection_parameter(
    ...     beamformed_data=beamformed_data,
    ...     sweep_specs=specs,
    ...     ground_truth_paths=relative_bearing_ground_truths,
    ...     steering_azimuths_rad=BF_PARAMS["steering_azimuths_rad"],
    ...     association_threshold_rad=np.deg2rad(3.0),
    ... )
    >>> plot_roc_pr(results).show()

    """
    steering_azimuths = np.asarray(steering_azimuths_rad, dtype=np.float64)
    num_beams = steering_azimuths.shape[0]
    num_timesteps = len(beamformed_data)
    frames = [np.asarray(d) for d in beamformed_data]
    gt_bearings_per_t = _bearings_from_ground_truth_paths(
        ground_truth_paths, num_timesteps, bearing_state_index
    )

    results: list[SweepResult] = []

    for spec in sweep_specs:
        param_values = np.asarray(spec.param_values, dtype=np.float64)

        def _detected_beams_at_t(
            detector: DetectionAlgorithm,
            t_idx: int,
            _frames: list[FloatArray] = frames,
        ) -> IntArray:
            raw_dets = detector.detect(_frames[t_idx])
            return (
                raw_dets[:, 0].astype(np.intp) if raw_dets.size > 0 else np.empty(0, dtype=np.intp)
            )

        tp_arr, fp_arr, fn_arr, tn_arr = _sweep_confusion_counts(
            param_values=param_values,
            detector_factory=lambda p_val, _spec=spec: _clone_detector_with_param(_spec, p_val),
            detected_beams_at_t=_detected_beams_at_t,
            gt_bearings_per_t=gt_bearings_per_t,
            steering_azimuths=steering_azimuths,
            association_threshold_rad=association_threshold_rad,
            num_timesteps=num_timesteps,
            num_beams=num_beams,
        )

        results.append(
            SweepResult(
                param_values=param_values,
                tp=tp_arr,
                fp=fp_arr,
                fn=fn_arr,
                tn=tn_arr,
                label=spec.label or f"{spec.param_name} sweep",
            )
        )

    return results


def _validate_multiband_sweep_inputs(
    beamformed_data: dict[str, Sequence[ArrayLike]],
    sweep_specs: dict[str, SweepSpec],
) -> tuple[FloatArray, int]:
    """Check that per-band data and specs describe one coherent sweep.

    Parameters
    ----------
    beamformed_data : dict[str, Sequence[ArrayLike]]
        Per-band raw beamformed data keyed by band label.
    sweep_specs : dict[str, SweepSpec]
        Per-band sweep specifications keyed by band label.

    Returns
    -------
    tuple of (FloatArray, int)
        The shared parameter values and the timestep count.

    Raises
    ------
    ValueError
        If no bands are given, if the band labels of ``beamformed_data`` and
        ``sweep_specs`` disagree, if the bands cover different numbers of
        timesteps, or if the specs do not all sweep the same parameter values.

    """
    if not beamformed_data:
        raise ValueError("beamformed_data must contain at least one band")

    if set(beamformed_data) != set(sweep_specs):
        missing = sorted(set(beamformed_data) - set(sweep_specs))
        extra = sorted(set(sweep_specs) - set(beamformed_data))
        raise ValueError(
            "beamformed_data and sweep_specs must cover the same bands; "
            f"bands with no spec: {missing}, specs with no band: {extra}"
        )

    timestep_counts = {label: len(d) for label, d in beamformed_data.items()}
    if len(set(timestep_counts.values())) != 1:
        raise ValueError(
            f"All bands must share the same number of timesteps, got {timestep_counts}"
        )

    reference_label = next(iter(sweep_specs))
    param_values = np.asarray(sweep_specs[reference_label].param_values, dtype=np.float64)
    for label, spec in sweep_specs.items():
        band_values = np.asarray(spec.param_values, dtype=np.float64)
        if band_values.shape != param_values.shape or not np.array_equal(
            band_values, param_values
        ):
            raise ValueError(
                f"Every band must sweep identical param_values so the bands can be "
                f"combined at a common operating point, but band {label!r} differs from "
                f"{reference_label!r}."
            )

    return param_values, timestep_counts[reference_label]


def sweep_detection_parameter_multiband(
    beamformed_data: dict[str, Sequence[ArrayLike]],
    sweep_specs: dict[str, SweepSpec],
    ground_truth_paths: Sequence[_GroundTruthPathLike],
    steering_azimuths_rad: ArrayLike,
    association_threshold_rad: float,
    bearing_state_index: int = 0,
    result_label: str | None = None,
) -> SweepResult:
    """Sweep a detection parameter across several bands, scoring their combined output.

    Each band is detected independently with its own detector, then the detections from
    every band are combined into a single set per timestep and scored once. This is the
    "squashed" multiband configuration: the output a single tracker would consume.

    Bands are combined by taking the **union of detected beam indices**, so a bearing found
    in several bands counts once rather than as one detection plus several false alarms.
    The combined output therefore lives in the same beam space as a single-band sweep, and
    the false-positive rate uses the same ``num_beams`` denominator, which makes results
    directly comparable across band counts. Detections on *adjacent* beams from different
    bands are not merged; resolving those is a tracker's job.

    Per-band detector geometry (CFAR guard and training cells, peak separation) belongs in
    each band's own detector, since it should scale with the band's frequency. Only the
    swept parameter is shared, and it must be, since combining bands at different operating
    points would be meaningless.

    Parameters
    ----------
    beamformed_data : dict[str, Sequence[ArrayLike]]
        Per-band raw beamformed data keyed by band label, one entry per timestep, each of
        shape ``(num_beams, num_frames)``. Bands may have different ``num_frames`` (e.g.
        different STFT settings), but must share ``num_beams`` and timestep count.
    sweep_specs : dict[str, SweepSpec]
        One :class:`SweepSpec` per band, keyed by the same labels as ``beamformed_data``.
        Every spec must sweep the same ``param_values``.
    ground_truth_paths : Sequence
        Bearing ground-truth paths used to score detections.
    steering_azimuths_rad : ArrayLike
        Beam azimuths in radians, of length ``num_beams``.
    association_threshold_rad : float
        Maximum bearing error for a detection to count as a true positive.
    bearing_state_index : int, optional
        Index of the bearing element within each ground-truth state vector.
    result_label : str | None, optional
        Human-readable name for plot legends. Defaults to a summary of the band count.

    Returns
    -------
    SweepResult
        Metrics for the combined output, one entry per swept parameter value.

    """
    param_values, num_timesteps = _validate_multiband_sweep_inputs(beamformed_data, sweep_specs)
    steering_azimuths = np.asarray(steering_azimuths_rad, dtype=np.float64)
    num_beams = steering_azimuths.shape[0]
    gt_bearings_per_t = _bearings_from_ground_truth_paths(
        ground_truth_paths, num_timesteps, bearing_state_index
    )
    band_labels = list(sweep_specs)
    band_frames = {
        band_label: [np.asarray(d) for d in beamformed_data[band_label]]
        for band_label in band_labels
    }

    def _detector_factory(p_val: float) -> dict[str, DetectionAlgorithm]:
        # Clone each band's detector once per parameter value, not once per timestep.
        return {
            band_label: _clone_detector_with_param(sweep_specs[band_label], p_val)
            for band_label in band_labels
        }

    def _detected_beams_at_t(
        detectors: dict[str, DetectionAlgorithm],
        t_idx: int,
    ) -> IntArray:
        detected_beams: set[int] = set()
        for band_label in band_labels:
            raw_dets = detectors[band_label].detect(band_frames[band_label][t_idx])
            if raw_dets.size > 0:
                detected_beams.update(raw_dets[:, 0].astype(int).tolist())
        return np.array(sorted(detected_beams), dtype=np.intp)

    tp_arr, fp_arr, fn_arr, tn_arr = _sweep_confusion_counts(
        param_values=param_values,
        detector_factory=_detector_factory,
        detected_beams_at_t=_detected_beams_at_t,
        gt_bearings_per_t=gt_bearings_per_t,
        steering_azimuths=steering_azimuths,
        association_threshold_rad=association_threshold_rad,
        num_timesteps=num_timesteps,
        num_beams=num_beams,
    )

    reference_spec = sweep_specs[band_labels[0]]
    return SweepResult(
        param_values=param_values,
        tp=tp_arr,
        fp=fp_arr,
        fn=fn_arr,
        tn=tn_arr,
        label=result_label or f"{len(band_labels)} band(s), {reference_spec.param_name} sweep",
    )


# ---------------------------------------------------------------------------
# Ground-truth helper (internal)
# ---------------------------------------------------------------------------


def _bearings_from_ground_truth_paths(
    ground_truth_paths: Sequence[_GroundTruthPathLike],
    num_timesteps: int,
    bearing_state_index: int = 0,
) -> list[FloatArray]:
    """Convert a list of Stone Soup ``GroundTruthPath`` objects to a per-timestep bearing array.

    Parameters
    ----------
    ground_truth_paths : Sequence[_GroundTruthPathLike]
        Stone Soup-like ground-truth paths whose state vectors carry bearing values.
    num_timesteps : int
        Number of timesteps to extract.
    bearing_state_index : int
        Row index within each state vector that holds the bearing in radians.

    Returns
    -------
    list[FloatArray]
        Length-``num_timesteps`` list; element ``t`` is a 1-D array of bearings in radians.

    """
    return [
        np.array(
            [path.states[t].state_vector[bearing_state_index] for path in ground_truth_paths],
            dtype=float,
        )
        for t in range(num_timesteps)
    ]


# ---------------------------------------------------------------------------
# Theoretical Pd-vs-Pfa curves (no data, pure model -- see module docstring)
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

    Closed form throughout for the default :class:`~.fluctuation_models.RayleighFluctuation`
    model (:func:`~.algorithms.solve_ca_cfar_alpha` +
    :meth:`~.fluctuation_models.RayleighFluctuation.ca_cfar_pd`), so each point is independent
    and exact -- no shared-simulation trick needed here, unlike :func:`os_cfar_roc` at
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
        :class:`~.fluctuation_models.RayleighFluctuation`.
    effective_looks_per_frame : float, optional
        Independent looks integrated into each per-frame sample (K_n), by default 1.0. Applied
        to the threshold and the Pd alike, so the curve stays self-consistent.
    signal_looks_per_frame : float or None, optional
        Looks the target occupies (K_s), by default ``None`` meaning K_s = K_n. Affects only
        Pd -- H0 has no signal in it, so the threshold is untouched. See the
        :mod:`~.fluctuation_models` module docstring.

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
    :class:`~.fluctuation_models.RayleighFluctuation` model -- each point is evaluated
    independently via :func:`~.algorithms.solve_os_cfar_alpha_single_look` +
    :meth:`model.os_cfar_pd() <.fluctuation_models.FluctuationModel.os_cfar_pd>`. A model
    without a closed form at ``num_frames == 1`` (e.g.
    :class:`~.fluctuation_models.NonFluctuating`) still goes through this same per-point path,
    just falling back to its own independent Monte Carlo per point.

    At ``num_frames > 1``, no closed form exists for OS-CFAR under any fluctuation model
    implemented here (see :meth:`~.fluctuation_models.RayleighFluctuation.os_cfar_pd`).
    Calling :func:`~.algorithms.calibrate_os_cfar_alpha_mc` and ``model.cut_power_samples``
    once per Pfa point would run a fresh simulation for every point in the sweep; this instead
    draws the reference-cell and CUT samples ONCE and reuses them for every Pfa value, which is
    both cheaper and produces a smoother curve (adjacent points share the same underlying draws
    instead of independent noise).

    Validity of reusing the same noise_estimate draws for both alpha calibration and Pd
    evaluation: noise_estimate's distribution doesn't depend on which hypothesis (H0/H1) the
    CUT is under, so pairing the same reference-cell draws with an independent H1 CUT sample
    introduces no bias. It's the same "common random numbers" idea used to reduce variance
    when comparing scenarios, not a shortcut that changes what's being estimated. Cross-checked
    numerically against independently-simulated per-point
    :func:`~.algorithms.calibrate_os_cfar_alpha_mc` /
    :meth:`~.fluctuation_models.RayleighFluctuation.os_cfar_pd` calls: both agree to within
    Monte Carlo noise at 1e6 trials.

    Precision floor: per calibrate_os_cfar_alpha_mc's rule of thumb, a stable quantile estimate
    needs num_trials >= ~100/pfa. With the default 1e6 trials, don't trust points below roughly
    pfa=1e-4. The quantile (and therefore Pd) estimate gets noisy below that without a much
    larger (and much more memory-hungry) num_trials.

    Parameters
    ----------
    pfa_values : ArrayLike
        Pfa operating points to evaluate Pd at, each in (0, 1). See precision floor above.
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
        :class:`~.fluctuation_models.RayleighFluctuation`.
    num_trials : int, optional
        Monte Carlo trial count, shared across every Pfa point when num_frames > 1 (and passed
        through to ``model.os_cfar_pd`` when num_frames == 1), by default 1_000_000.
    rng : np.random.Generator, optional
        Random number generator for reproducibility, by default None.
    effective_looks_per_frame : float, optional
        Independent looks integrated into each per-frame sample (K_n), by default 1.0. Applied
        to the threshold and the Pd alike, so the curve stays self-consistent. Match it to the
        detector being compared against -- see
        :func:`estimate_effective_looks_per_frame`.
    signal_looks_per_frame : float or None, optional
        Looks the target occupies (K_s), by default ``None`` meaning K_s = K_n. Affects only
        Pd -- H0 has no signal in it, so the threshold is untouched. Set it well below K_n for
        a narrowband target in a wide processing band. See the :mod:`~.fluctuation_models`
        module docstring.

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
    # is the same under H0 and H1 (see fluctuation_models module docstring), so it's shared
    # across every Pfa point.
    ref = rng.gamma(
        effective_looks_per_frame,
        1.0 / effective_looks_per_frame,
        size=(num_trials, num_training_total, num_frames),
    ).mean(axis=2)
    ref.sort(axis=1)
    noise_estimate = ref[:, rank - 1]

    # H0 CUT, used only to calibrate alpha per Pfa value via a quantile of this ratio. H0
    # statistics don't depend on the fluctuation model or on the target's bandwidth, so this
    # is always plain noise filling the band.
    cut_h0 = rng.gamma(
        effective_looks_per_frame,
        1.0 / effective_looks_per_frame,
        size=(num_trials, num_frames),
    ).mean(axis=1)
    ratio_h0 = cut_h0 / noise_estimate

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

    pds = [np.mean(cut_h1 > np.quantile(ratio_h0, 1 - pfa) * noise_estimate) for pfa in pfa_values]
    return np.array(pds)


# ---------------------------------------------------------------------------
# Bridging simulation ground truth to the theoretical model
# ---------------------------------------------------------------------------


def snr_linear_from_ground_truth_bearing(
    beamformed_data: ArrayLike,
    true_bearing_rad: float,
    steering_azimuths_rad: np.ndarray,
    validation_guard_bins: int,
    noise_floor_percentile: float = 50.0,
) -> tuple[float, int]:
    """Estimate SNR at a single timestep by reading from the beam nearest the ground truth bearing.

    snr_linear_hat = power_at_true_bearing / noise_estimate - 1, where noise_estimate is computed
    over all OTHER bins (excluding a wide zone around the true bearing), deliberately decoupled
    from the operational detector's own num_guard_cells. See Failure modes below for why.

    Parameters
    ----------
    beamformed_data : ArrayLike
        Raw beamformed data for one timestep, shape (num_beams, num_frames).
    true_bearing_rad : float
        Ground-truth target bearing, radians, from the simulator (not detector output).
    steering_azimuths_rad : np.ndarray
        Steering azimuths for each beam, radians. Assumed circular (spans a full -pi..pi sweep).
    validation_guard_bins : int
        Bins excluded on each side of the nearest-beam index when estimating the noise floor.
        Should be set wider than the operational detector's num_guard_cells. See Failure modes,
        item 2.
    noise_floor_percentile : float, optional
        Percentile of the (excluded-region-free) directional power used as the noise floor, by
        default 50 (median). A simple, robust choice for this one-off diagnostic reading,
        deliberately not tied to the operational detector's own rank/order-statistic convention,
        since consistency with the operational detector isn't the goal here.

    Returns
    -------
    tuple[float, int]
        (snr_linear_hat, cut_index). The estimated snr_linear, and the beam index it was read from,
        so the caller can cross-check which physical bin was actually used.

    Failure modes (read before trusting a single snr_linear_hat)
    --------------------------------------------------------------
    1. Single noisy realisation, not the true injected value. The CUT's own power is itself random
       under H1 (Gamma-distributed, per the Rayleigh-fading target model documented in
       fluctuation_models.RayleighFluctuation.ca_cfar_pd). One reading has real variance, not just
       measurement error. If this target persists across
       several scans, average snr_linear_hat across its associated readings for a stable estimate
       rather than trusting any single timestep.
    2. Contaminated noise estimate from the target's OWN leakage. This is the same "trench"
       mechanism that affects detect()/detection_snr_map(). Reading the noise floor from a
       WIDE exclusion zone (validation_guard_bins, wider than the operational
       num_guard_cells) mitigates this but
       doesn't eliminate it: for a very strong source, leakage can still extend past a
       generously-sized exclusion zone. Size validation_guard_bins from the same worst-case
       leakage characterization used for num_guard_cells, not the operational value itself.
    3. Beam straddle loss is NOT corrected here. If the true bearing falls between beam centers,
       cut_index is the nearest beam, and snr_linear_hat reflects whatever roll-off that beam's
       response has at the true offset. This is "received SNR at the nearest beam," not "target SNR
       corrected for geometry." No automatic correction is applied: for MVDR there's no fixed
       analytic straddle-loss curve to apply, unlike conventional beamforming.
    4. Multi-target bin overlap. If another contact's true bearing is at or near this target's true
       bearing at this instant, snr_linear_hat reflects their COMBINED power, not this target's
       alone. Check other targets' true bearings at the same timestep before trusting an isolated
       snr_linear_hat, particularly near a crossing.

    """
    data_array = np.asarray(beamformed_data)
    num_beams = data_array.shape[0]
    directional_power = np.mean(_directional_power(data_array), axis=1)

    # Nearest beam to true bearing, wrap-aware (bearing spans a full -180..180 circle).
    angular_diff = np.angle(np.exp(1j * (steering_azimuths_rad - true_bearing_rad)))
    cut_index = int(np.argmin(np.abs(angular_diff)))

    excluded = np.zeros(num_beams, dtype=bool)
    for offset in range(-validation_guard_bins, validation_guard_bins + 1):
        excluded[(cut_index + offset) % num_beams] = True

    if excluded.all():
        raise ValueError(
            f"validation_guard_bins ({validation_guard_bins}) excludes the entire "
            f"array of {num_beams} beams -- reduce it."
        )

    noise_estimate = np.percentile(directional_power[~excluded], noise_floor_percentile)
    snr_linear_hat = directional_power[cut_index] / noise_estimate - 1
    return float(snr_linear_hat), cut_index


def estimate_effective_looks_per_frame(
    beamformed_data: Sequence[ArrayLike],
    noise_only_mask: ArrayLike | None = None,
    min_scans_per_beam: int = 10,
) -> float:
    """Estimate a detector's ``effective_looks_per_frame`` from noise-only simulated scans.

    :func:`~.algorithms.calibrate_os_cfar_alpha_mc` models each per-frame power sample as
    Gamma(K, 1/K) -- K independent unit-mean Exponential looks integrated per frame. K is 1
    for classic narrowband square-law data, but a broadband STFT beamformer sums power over
    every active frequency bin before the detector sees it, making K far larger. This reads
    K back off the simulator's own output instead of assuming it, so alpha calibration
    describes the data the detector will actually be fed.

    The estimator is the moment relation ``CV = 1 / sqrt(K * num_frames)`` for a unit-mean
    Gamma, inverted per beam across scans and combined with a median::

        K = 1 / median_over_beams(CV_beam ** 2) / num_frames

    Using the spread of one beam ACROSS scans (rather than across beams within a scan)
    keeps the estimate free of the beam-to-beam variation in mean noise level that array
    geometry imposes -- that variation is real structure, not the per-look fluctuation K
    describes. The median across beams then rejects the minority of beams contaminated by
    target leakage. Both choices matter more the less homogeneous the scene is.

    Parameters
    ----------
    beamformed_data : Sequence[ArrayLike]
        One raw beamformed array per scan, each of shape (num_beams, num_frames), as
        collected from ``simulator.sensor_data_gen()``. Every scan must share a shape.
    noise_only_mask : ArrayLike, optional
        Boolean array of shape (num_scans, num_beams), True where a cell is believed
        target-free. Defaults to using every cell, which is only appropriate when the run
        genuinely contains no target -- otherwise mask out the target's bearing and a
        generous guard around it, as ``validation_guard_bins`` does elsewhere in this module.
    min_scans_per_beam : int, optional
        Beams with fewer than this many unmasked scans are dropped, by default 10, since a
        CV from a handful of samples is too noisy to contribute. Raise it for a tighter
        estimate when scans are plentiful.

    Returns
    -------
    float
        Estimated effective independent looks per frame (K), to pass as
        ``effective_looks_per_frame`` to :class:`~.algorithms.OSCFARDetector` or
        :func:`~.algorithms.calibrate_os_cfar_alpha_mc`. Not generally an integer, and
        typically well below the nominal in-band bin count, since window spectral leakage
        correlates neighbouring frequency bins.

    Raises
    ------
    ValueError
        If ``beamformed_data`` is empty, scan shapes disagree, ``noise_only_mask`` doesn't
        match the stacked data shape, or no beam clears ``min_scans_per_beam``.

    Notes
    -----
    This measures fluctuation ACROSS scans, whereas a CFAR detector estimates its noise
    floor from neighbouring beams WITHIN one scan. The two coincide only when the noise
    field is spatially homogeneous. Where adjacent beams are correlated (finite beamwidth
    always correlates them to some degree), the noise-floor estimate carries extra variance
    that this K does not describe, so achieved Pfa will track target Pfa less tightly than
    a K calibrated on independent cells would suggest.

    """
    if len(beamformed_data) == 0:
        raise ValueError("beamformed_data is empty; at least one scan is required")

    per_scan_power = [np.mean(_directional_power(scan), axis=1) for scan in beamformed_data]
    shapes = {scan.shape for scan in per_scan_power}
    if len(shapes) != 1:
        raise ValueError(f"All scans must share a shape; got beam counts {sorted(shapes)}")

    cell_power = np.asarray(per_scan_power)  # (num_scans, num_beams)
    num_frames = np.asarray(beamformed_data[0]).shape[1]

    if noise_only_mask is None:
        mask = np.ones_like(cell_power, dtype=bool)
    else:
        mask = np.asarray(noise_only_mask, dtype=bool)
        if mask.shape != cell_power.shape:
            raise ValueError(
                f"noise_only_mask shape {mask.shape} does not match the stacked "
                f"(num_scans, num_beams) data shape {cell_power.shape}"
            )

    usable = mask.sum(axis=0) >= min_scans_per_beam
    if not usable.any():
        raise ValueError(
            f"No beam has at least min_scans_per_beam ({min_scans_per_beam}) noise-only "
            f"scans; the most any beam has is {int(mask.sum(axis=0).max())}"
        )

    # Per-beam mean and variance across that beam's noise-only scans only. NaN-masking keeps
    # each beam's sample count independent, since a beam is target-free in its own subset of
    # scans as the target tracks across the array.
    masked = np.where(mask, cell_power, np.nan)[:, usable]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        beam_mean = np.nanmean(masked, axis=0)
        beam_std = np.nanstd(masked, axis=0, ddof=1)

    cv_squared = (beam_std / beam_mean) ** 2
    cv_squared = cv_squared[np.isfinite(cv_squared) & (cv_squared > 0)]
    if cv_squared.size == 0:
        raise ValueError(
            "Could not form a finite, positive CV for any beam; check that the input is "
            "real power data with a non-zero mean"
        )

    return float(1.0 / np.median(cv_squared) / num_frames)
