"""Detection performance metrics for passive sonar systems.

This module provides tools to evaluate one or more CFAR-family detectors by sweeping
a scalar parameter (e.g. ``target_pfa``) over raw beamformed data and comparing the
resulting detections against ground-truth bearings.

Each detector to evaluate is described by a :class:`SweepSpec`; passing a list of them
to :func:`sweep_detection_parameter` produces a parallel list of :class:`SweepResult`
objects that can be plotted together for comparison. See :func:`sweep_detection_parameter`
for a full runnable example.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

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

    Attributes
    ----------
    param_values : FloatArray
        The swept parameter values, shape ``(P,)``.
    tp : IntArray
        Total true positives across all timesteps for each parameter value.
    fp : IntArray
        Total false positives across all timesteps for each parameter value.
    fn : IntArray
        Total false negatives across all timesteps for each parameter value.
    tn : IntArray
        Total true negatives across all timesteps for each parameter value.
    label : str | None
        Human-readable name carried through from :class:`SweepSpec`, used in
        plot legends.

    """

    param_values: FloatArray
    tp: IntArray
    fp: IntArray
    fn: IntArray
    tn: IntArray
    label: str | None = None

    @property
    def precision(self) -> FloatArray:
        """Positive predictive value: TP / (TP + FP).

        Defaults to 1 where TP + FP = 0 (no detections issued).
        """
        denom = self.tp + self.fp
        return _safe_ratio(self.tp, denom, default=1.0)

    @property
    def recall(self) -> FloatArray:
        """Sensitivity / true positive rate: TP / (TP + FN).

        Defaults to 0 where TP + FN = 0 (no positives present).
        """
        denom = self.tp + self.fn
        return _safe_ratio(self.tp, denom, default=0.0)

    @property
    def tpr(self) -> FloatArray:
        """True positive rate (alias for :attr:`recall`)."""
        return self.recall

    @property
    def fpr(self) -> FloatArray:
        """False positive rate: FP / (FP + TN).

        Defaults to 0 where FP + TN = 0.
        """
        denom = self.fp + self.tn
        return _safe_ratio(self.fp, denom, default=0.0)

    @property
    def f1(self) -> FloatArray:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        denom = p + r
        return _safe_ratio(2.0 * p * r, denom, default=0.0)

    @property
    def auc_roc(self) -> float:
        """Area under the ROC curve, computed via trapezoidal integration."""
        order = np.argsort(self.fpr)
        return float(np.trapezoid(self.tpr[order], self.fpr[order]))

    @property
    def auc_pr(self) -> float:
        """Area under the precision-recall curve, computed via trapezoidal integration."""
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
