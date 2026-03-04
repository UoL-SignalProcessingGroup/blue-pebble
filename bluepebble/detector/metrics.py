"""Detection performance metrics for passive sonar systems.

This module provides tools to evaluate one or more detection chains by sweeping
a scalar parameter (e.g. CFAR ``threshold_factor``) over a pre-computed SNR map and
comparing the resulting detections against ground-truth bearings.

Each chain to evaluate is described by a :class:`SweepSpec`; passing a list of them
to :func:`sweep_detection_parameter` produces a parallel list of :class:`SweepResult`
objects that can be plotted together for comparison.

Typical usage::

    import numpy as np
    from bluepebble.detector.algorithms import CACFARDetector, OSCFARDetector, PeakDetector
    from bluepebble.detector.metrics import SweepSpec, sweep_detection_parameter
    from bluepebble.plotter import plot_roc_pr

    specs = [
        SweepSpec(
            detection_chain=[
                CACFARDetector(num_guard_cells=2, num_training_cells=10, threshold_factor=1.1),
                PeakDetector(distance=3),
            ],
            algorithm_index=0,
            param_name="threshold_factor",
            param_values=np.linspace(0.7, 2.5, 80),
            label="CA-CFAR",
        ),
        SweepSpec(
            detection_chain=[
                OSCFARDetector(num_guard_cells=2, num_training_cells=10, threshold_factor=1.1),
                PeakDetector(distance=3),
            ],
            algorithm_index=0,
            param_name="threshold_factor",
            param_values=np.linspace(0.7, 2.5, 80),
            label="OS-CFAR",
        ),
    ]

    results = sweep_detection_parameter(
        snr_map=detector.snr_history,
        sweep_specs=specs,
        ground_truth_paths=relative_bearing_ground_truths,
        steering_azimuths_rad=BF_PARAMS["steering_azimuths_rad"],
        association_threshold_rad=np.deg2rad(3.0),
    )

    for r in results:
        print(f"{r.label}: AUC-ROC={r.auc_roc:.4f}  AUC-PR={r.auc_pr:.4f}")

    plot_roc_pr(results).show()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .algorithms import DetectionAlgorithm


@dataclass
class SweepSpec:
    """Specification for a single detection-chain parameter sweep.

    Attributes
    ----------
    detection_chain : list[DetectionAlgorithm]
        The base detection chain.  Deep-copied for each parameter value so the
        original is never mutated.
    algorithm_index : int
        Index into ``detection_chain`` selecting the algorithm whose parameter
        is swept (0-based).
    param_name : str
        Attribute name to sweep (e.g. ``"threshold_factor"``).
    param_values : np.ndarray | list[float]
        Sequence of values to evaluate, ordered from most permissive to most
        strict so ROC / PR curves trace in the canonical direction.
    label : str | None
        Human-readable name used in plot legends.  Defaults to
        ``"<param_name> sweep"`` when ``None``.

    """

    detection_chain: list[DetectionAlgorithm]
    algorithm_index: int
    param_name: str
    param_values: np.ndarray | list[float] | list[int]
    label: str | None = None


@dataclass
class _TimestepMetrics:
    """Raw confusion-matrix counts for a single timestep."""

    tp: int
    fp: int
    fn: int
    tn: int


def _run_detection_chain(
    detection_chain: list[DetectionAlgorithm],
    snr_vector: np.ndarray,
) -> np.ndarray:
    """Run a detection chain on a single SNR vector without a detector instance.

    Mirrors ``PassiveSonarDetector._run_detection_chain`` but operates as a
    standalone function so the sweep loop does not require a full detector object.

    Parameters
    ----------
    detection_chain : list[DetectionAlgorithm]
        Ordered list of detection algorithms to apply sequentially.
    snr_vector : np.ndarray
        1-D SNR data vector (dB) of shape ``(N_beams,)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(N_det, 2)`` with columns ``[beam_index, snr_dB]``, or
        an empty ``(0, 2)`` array when no detections pass the full chain.

    """
    if not detection_chain:
        return np.empty((0, 2))

    input_data = snr_vector.copy()
    final_detections = np.empty((0, 2))

    for algorithm in detection_chain:
        current_detections = algorithm.detect(input_data)

        if current_detections.size == 0:
            return np.empty((0, 2))

        final_detections = current_detections

        # Build a sparse input for the next stage: set non-detected cells to −∞
        input_data = np.full(len(snr_vector), -np.inf)
        input_data[final_detections[:, 0].astype(int)] = final_detections[:, 1]

    return final_detections


def _compute_timestep_metrics(
    detected_bearings_rad: np.ndarray,
    ground_truth_bearings_rad: np.ndarray,
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
    detected_bearings_rad : np.ndarray
        Detected bearing angles in radians, shape ``(N_det,)``.
    ground_truth_bearings_rad : np.ndarray
        Ground-truth bearing angles in radians, shape ``(N_gt,)``.
    association_threshold_rad : float
        Maximum angular distance (rad) for a detection to be counted as a TP.
    num_beam_cells : int
        Total number of beam cells in the SNR vector (used to estimate TN).

    Returns
    -------
    _TimestepMetrics
        Aggregated TP, FP, FN, TN counts.

    """
    n_det = len(detected_bearings_rad)
    n_gt = len(ground_truth_bearings_rad)

    if n_gt == 0:
        # No targets present — every detection is a false alarm
        return _TimestepMetrics(tp=0, fp=n_det, fn=0, tn=num_beam_cells)

    if n_det == 0:
        # No detections — every target is missed
        return _TimestepMetrics(tp=0, fp=0, fn=n_gt, tn=num_beam_cells - n_gt)

    # Pairwise circular angular distance matrix: shape (N_det, N_gt)
    diff = detected_bearings_rad[:, np.newaxis] - ground_truth_bearings_rad[np.newaxis, :]
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


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, default: float) -> np.ndarray:
    """Return ``numerator / denominator`` with a default where denominator is zero."""
    result = np.full(np.shape(denominator), default, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


@dataclass
class SweepResult:
    """Aggregated detection metrics from a parameter sweep.

    Attributes
    ----------
    param_values : np.ndarray
        The swept parameter values, shape ``(P,)``.
    tp : np.ndarray
        Total true positives across all timesteps for each parameter value.
    fp : np.ndarray
        Total false positives across all timesteps for each parameter value.
    fn : np.ndarray
        Total false negatives across all timesteps for each parameter value.
    tn : np.ndarray
        Total true negatives across all timesteps for each parameter value.
    label : str | None
        Human-readable name carried through from :class:`SweepSpec`, used in
        plot legends.

    """

    param_values: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    tn: np.ndarray
    label: str | None = None

    @property
    def precision(self) -> np.ndarray:
        """Positive predictive value: TP / (TP + FP).

        Defaults to 1 where TP + FP = 0 (no detections issued).
        """
        denom = self.tp + self.fp
        return _safe_ratio(self.tp, denom, default=1.0)

    @property
    def recall(self) -> np.ndarray:
        """Sensitivity / true positive rate: TP / (TP + FN).

        Defaults to 0 where TP + FN = 0 (no positives present).
        """
        denom = self.tp + self.fn
        return _safe_ratio(self.tp, denom, default=0.0)

    @property
    def tpr(self) -> np.ndarray:
        """True positive rate (alias for :attr:`recall`)."""
        return self.recall

    @property
    def fpr(self) -> np.ndarray:
        """False positive rate: FP / (FP + TN).

        Defaults to 0 where FP + TN = 0.
        """
        denom = self.fp + self.tn
        return _safe_ratio(self.fp, denom, default=0.0)

    @property
    def f1(self) -> np.ndarray:
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


def sweep_detection_parameter(
    snr_map: np.ndarray,
    sweep_specs: list[SweepSpec],
    ground_truth_paths: list,
    steering_azimuths_rad: np.ndarray,
    association_threshold_rad: float,
    bearing_state_index: int = 0,
) -> list[SweepResult]:
    """Sweep parameters for one or more detection chains and compute ROC / PR metrics.

    For each :class:`SweepSpec` and each value in its ``param_values`` the function:

    1. Deep-copies the detection chain (the originals are never mutated).
    2. Sets ``chain[algorithm_index].<param_name>`` to the current sweep value.
    3. Runs the modified chain over every row of ``snr_map``.
    4. Associates detections with ground-truth bearings via greedy nearest-neighbour
       matching and accumulates TP / FP / FN / TN counts.

    The returned :class:`SweepResult` objects expose precision, recall, TPR, FPR,
    F1 and AUC values as properties.  Pass the list directly to
    :func:`~bluepebble.plotter.plot_roc_pr` to compare all chains in one figure.

    Parameters
    ----------
    snr_map : np.ndarray
        Pre-computed SNR map of shape ``(T, N_beams)`` — typically
        ``PassiveSonarDetector.snr_history`` after running a simulation.
    sweep_specs : list[SweepSpec]
        One :class:`SweepSpec` per detection chain to evaluate.  Each spec
        carries its own chain, algorithm index, parameter name, sweep range,
        and optional legend label.
    ground_truth_paths : list of GroundTruthPath
        Stone Soup ground-truth paths whose state vectors carry bearing values —
        typically ``relative_bearing_ground_truths`` from the simulation workflow.
        The number of timesteps is inferred from ``snr_map``.
    steering_azimuths_rad : np.ndarray
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
    >>> from bluepebble.detector.algorithms import CACFARDetector, OSCFARDetector, PeakDetector
    >>> from bluepebble.detector.metrics import SweepSpec, sweep_detection_parameter
    >>> from bluepebble.plotter import plot_roc_pr
    >>>
    >>> specs = [
    ...     SweepSpec(
    ...         detection_chain=[
    ...             CACFARDetector(num_guard_cells=2, num_training_cells=10, threshold_factor=1.1),
    ...             PeakDetector(distance=3),
    ...         ],
    ...         algorithm_index=0,
    ...         param_name="threshold_factor",
    ...         param_values=np.linspace(0.7, 2.5, 80),
    ...         label="CA-CFAR",
    ...     ),
    ...     SweepSpec(
    ...         detection_chain=[
    ...             OSCFARDetector(num_guard_cells=2, num_training_cells=10, threshold_factor=1.1),
    ...             PeakDetector(distance=3),
    ...         ],
    ...         algorithm_index=0,
    ...         param_name="threshold_factor",
    ...         param_values=np.linspace(0.7, 2.5, 80),
    ...         label="OS-CFAR",
    ...     ),
    ... ]
    >>> results = sweep_detection_parameter(
    ...     snr_map=detector.snr_history,
    ...     sweep_specs=specs,
    ...     ground_truth_paths=relative_bearing_ground_truths,
    ...     steering_azimuths_rad=BF_PARAMS["steering_azimuths_rad"],
    ...     association_threshold_rad=np.deg2rad(3.0),
    ... )
    >>> plot_roc_pr(results).show()

    """
    num_timesteps, num_beams = snr_map.shape
    gt_bearings_per_t = _bearings_from_ground_truth_paths(
        ground_truth_paths, num_timesteps, bearing_state_index
    )

    results: list[SweepResult] = []

    for spec in sweep_specs:
        param_values = np.asarray(spec.param_values, dtype=float)

        tp_arr = np.zeros(len(param_values), dtype=np.int64)
        fp_arr = np.zeros(len(param_values), dtype=np.int64)
        fn_arr = np.zeros(len(param_values), dtype=np.int64)
        tn_arr = np.zeros(len(param_values), dtype=np.int64)

        for p_idx, p_val in enumerate(param_values):
            # Clone the chain and inject the new parameter value
            chain_copy = copy.deepcopy(spec.detection_chain)
            setattr(chain_copy[spec.algorithm_index], spec.param_name, float(p_val))

            for t_idx, snr_vector in enumerate(snr_map):
                gt_bearings = gt_bearings_per_t[t_idx]

                raw_dets = _run_detection_chain(chain_copy, snr_vector)

                det_bearings = (
                    steering_azimuths_rad[raw_dets[:, 0].astype(int)]
                    if raw_dets.size > 0
                    else np.empty(0)
                )

                m = _compute_timestep_metrics(
                    detected_bearings_rad=det_bearings,
                    ground_truth_bearings_rad=gt_bearings,
                    association_threshold_rad=association_threshold_rad,
                    num_beam_cells=num_beams,
                )

                tp_arr[p_idx] += m.tp
                fp_arr[p_idx] += m.fp
                fn_arr[p_idx] += m.fn
                tn_arr[p_idx] += m.tn

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


# ---------------------------------------------------------------------------
# Ground-truth helper (internal)
# ---------------------------------------------------------------------------


def _bearings_from_ground_truth_paths(
    ground_truth_paths: list,
    num_timesteps: int,
    bearing_state_index: int = 0,
) -> list[np.ndarray]:
    """Convert a list of Stone Soup ``GroundTruthPath`` objects to a per-timestep bearing array.

    Parameters
    ----------
    ground_truth_paths : list of GroundTruthPath
        Stone Soup ground-truth paths whose state vectors carry bearing values.
    num_timesteps : int
        Number of timesteps to extract.
    bearing_state_index : int
        Row index within each state vector that holds the bearing in radians.

    Returns
    -------
    list[np.ndarray]
        Length-``num_timesteps`` list; element ``t`` is a 1-D array of bearings in radians.

    """
    return [
        np.array(
            [path.states[t].state_vector[bearing_state_index] for path in ground_truth_paths],
            dtype=float,
        )
        for t in range(num_timesteps)
    ]
