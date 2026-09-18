"""Calibrate CFAR detectors against noise-only data so ``target_pfa`` is achieved.

The closed-form and numerical calibrations in :mod:`._theory` derive alpha from a model:
reference cells and the cell under test (CUT) are i.i.d. Gamma with a known number of looks.
Beamformer output violates that model in ways no single look count repairs:

- Neighbouring beams share noise, so the CUT and its reference cells are correlated.
  Low-frequency energy is common to every beam and cancels between them. Pink ambient noise
  therefore gives a much tighter cell-to-noise ratio than white noise at the same frame count,
  and the nominal alpha produces far fewer false alarms than requested. This matches the
  analysis of CA-CFAR in spatially correlated Rayleigh clutter in [5]_, where the threshold
  derived for independent samples is too high and the actual false-alarm probability falls
  by up to several orders of magnitude as the correlation between cells increases.
- The ratio's upper tail is heavier than any Gamma/F model fitted to its body. A model tuned
  to be right at Pfa 1e-2 under-predicts false alarms at 1e-3 to 1e-4 by roughly 1.5x (white)
  to 6x (pink).

:func:`calibrate_from_noise` therefore measures the quantity the detector actually thresholds,
``frame-averaged power / local noise estimate``, on noise-only scans processed exactly as the
detector processes them, and reads alpha off its distribution:

- For ``pfa >= tail_pfa``, alpha is the empirical ``1 - pfa`` quantile of the pooled ratios.
- Below that, where too few scans exceed the threshold to measure it directly, the tail is
  extrapolated with a generalised Pareto distribution fitted to the exceedances over the
  ``1 - tail_pfa`` quantile (peaks over threshold [1]_, [2]_, [3]_). With ``tail_pfa`` the
  measured fraction of cells above the tail threshold ``u``, and fitted shape ``xi`` and scale
  ``sigma``, alpha is ``u + sigma / xi * ((pfa / tail_pfa) ** -xi - 1)``: the threshold
  estimate of [4]_, Eq. (15).

Assign the result to a detector's ``noise_calibration`` and keep controlling detection with
``target_pfa``. What becomes accurate is the noise-only false-alarm rate per cell, before peak
consolidation. False alarms induced by real sources (sidelobes, leakage, multipath) add to it and
are not controlled by any noise calibration.

:func:`~.metrics.estimate_effective_looks_per_frame` remains the model-based alternative:
cheaper, but only as accurate as the Gamma model for the data at hand.

Calibration is specific to everything that shapes the ratio distribution: array, shading,
beamformer domain and band, ambient spectrum, scan length (num_frames), and the detector's
guard/training window, rank and edge handling. The detector checks the parts it can see
(num_frames and its own window settings) and raises on a mismatch; the rest is the caller's
responsibility.

Frame counts need not match exactly. Continuous simulators emit a final scan that runs to the
end of the padded synthesis buffer, up to a whole STFT frame longer than the others, and a
calibration may be reused for slightly different scan lengths. Integration over more frames
tightens the ratio about 1 in proportion to ``1 / sqrt(frames)``, so ratios are referred to the
calibration's median frame count as ``1 + (ratio - 1) * sqrt(frames / reference)`` when pooled,
and alpha is referred back the same way for the frame count being detected on. Measured on
noise-only DAS output, applying a calibration 20% away in frame count without this left
achieved Pfa at 0.2-0.5x target; with it, 0.74-0.96x. Frame counts further than
``frame_count_tolerance`` (default 25%) from the calibration range are rejected.

Pooling every cell estimates the per-cell false-alarm rate averaged over bearing, which is
what ``target_pfa`` means, including edge cells of non-circular data. Cells are correlated,
so uncertainty estimates derived from the pooled count or the tail fit would be optimistic;
none is reported.

The ``1 / sqrt(frames)`` referral between frame counts is not taken from the literature: it is
the central-limit scaling of an average of independent looks, applied approximately to
correlated beamformer output and justified only by the measurements quoted above.

Estimating detection thresholds from a generalised Pareto fit to the upper tail of observed
detector output is established: [4]_ motivates it the same way, since thresholds derived from
invalid distribution assumptions do not hold on real sensor data. [4]_ applies it to each data
set's own detector output, fitting the upper 5-10% of scores, and adds a Kolmogorov-Smirnov
test to detect and remove target samples, which otherwise bias the tail; its experiments use
hyperspectral imagery. This module instead fits the CFAR detector's power-to-noise-estimate
ratio on separate noise-only scans from the same processing chain, so no target screening is
needed, and uses a 1% tail by default, which suits the large pooled cell counts of simulated
scans. Like [4]_, the likelihood fit treats exceedances as independent, which correlated cells
are not.

Sonar noise normalisers are the sonar counterpart of these CFAR detectors. The towed-array
normaliser of [6]_ sets its internal shearing threshold by assuming Rayleigh-distributed envelope
noise averaged over a known number of statistically independent beams, and does not set the
detection threshold for a stated false-alarm rate. The textbook treatment in [7]_ (Sect. 8.6)
does derive cell-averaging and order-statistic thresholds for a stated false-alarm rate, the same
closed forms as :mod:`._theory`, but for independent exponentially distributed auxiliary data.
It handles correlated (oversampled) auxiliary data through an equivalent number of independent
samples, ``tr(S)**2 / tr(S**2)`` for auxiliary-data covariance ``S`` (Sect. 8.6.1.2), or
characteristic functions (Sect. 9.3.4), in both cases keeping the test cell independent of its
auxiliary data. It notes that auxiliary data shared between overlapping windows, including
neighbouring beams, introduces correlation that complicates the analysis (Sect. 9.3.1), and that
signal processing, finite-sample normalisation and non-stationarity within the normaliser window
can all make the normalised output heavier tailed than the Rayleigh model (Sect. 7.4.3), which
raises the false-alarm rate at a threshold set under that model. Calibrating against noise-only
scans replaces these independence and distribution assumptions, which do not hold for
neighbouring beams.

References
----------
.. [1] Balkema, A. A. and de Haan, L. "Residual life time at great age." Annals of Probability,
       2(5), 792-804, 1974. doi:10.1214/aop/1176996548
.. [2] Pickands, J. "Statistical inference using extreme order statistics." Annals of
       Statistics, 3(1), 119-131, 1975. doi:10.1214/aos/1176343003
.. [3] Scarrott, C. and MacDonald, A. "A review of extreme value threshold estimation and
       uncertainty quantification." REVSTAT - Statistical Journal, 10(1), 33-60, 2012.
.. [4] Broadwater, J. B. and Chellappa, R. "Adaptive threshold estimation via extreme value
       theory." IEEE Transactions on Signal Processing, 58(2), 490-500, 2010.
       doi:10.1109/TSP.2009.2031285
.. [5] Himonas, S. D. and Barkat, M. "An adaptive CFAR signal detector for spatially correlated
       noise samples." Proceedings of the 29th IEEE Conference on Decision and Control,
       3540-3545, 1990. doi:10.1109/CDC.1990.203482
.. [6] Stergiopoulos, S. "Noise normalization technique for beamformed towed array data."
       Journal of the Acoustical Society of America, 97(4), 2334-2345, 1995.
       doi:10.1121/1.411958
.. [7] Abraham, D. A. "Underwater Acoustic Signal Processing: Modeling, Detection, and
       Estimation." Springer, Cham, 2019. doi:10.1007/978-3-319-92983-5

"""

import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import genpareto

# Tail shapes this close to zero use the exponential limit of the GPD quantile, avoiding a
# 0/0 in the general formula.
_EXPONENTIAL_TAIL_SHAPE = 1e-6

# Warn once the requested Pfa is this many times below what the pooled cell count resolves.
_EXTRAPOLATION_WARNING_FACTOR = 100.0

# Default relative spread in frame count tolerated within calibration scans, and between the
# calibration and the data it is applied to.
DEFAULT_FRAME_COUNT_TOLERANCE = 0.25


def detector_signature(detector: Any) -> tuple:
    """Detector settings that shape the cell-to-noise ratio distribution.

    ``target_pfa``, ``effective_looks_per_frame`` and consolidation settings are deliberately
    absent: they do not change the ratios a calibration is built from.

    Parameters
    ----------
    detector : _CFARDetectorBase
        A CFAR detector.

    Returns
    -------
    tuple
        ``(detector type, num_guard_cells, num_training_cells, rank, circular)``, with
        ``rank`` ``None`` for detectors that have none.

    """
    return (
        type(detector).__name__,
        int(detector.num_guard_cells),
        int(detector.num_training_cells),
        None if getattr(detector, "rank", None) is None else int(detector.rank),
        bool(detector.circular),
    )


_SIGNATURE_FIELDS = ("detector type", "num_guard_cells", "num_training_cells", "rank", "circular")


def _ratios_and_frame_counts(
    detector: Any,
    beamformed_scans: Iterable[ArrayLike],
    frame_count_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if frame_count_tolerance < 0:
        raise ValueError(f"frame_count_tolerance ({frame_count_tolerance}) must be >= 0")
    ratios: list[NDArray[np.float64]] = []
    frame_counts: list[int] = []
    for index, scan in enumerate(beamformed_scans):
        data = np.asarray(scan)
        power, noise_estimate = detector._power_and_noise(data)
        frame_counts.append(int(data.shape[1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            scan_ratios = power / noise_estimate
        if not np.all(np.isfinite(scan_ratios)):
            raise ValueError(
                f"Scan {index} gives non-finite power/noise ratios (zero noise estimate); "
                "noise-only scans must contain noise in every beam."
            )
        ratios.append(scan_ratios)
    if not ratios:
        raise ValueError("beamformed_scans is empty; at least one noise-only scan is required")
    counts = np.asarray(frame_counts)
    if counts.max() > counts.min() * (1.0 + frame_count_tolerance):
        raise ValueError(
            f"Scan frame counts range from {counts.min()} to {counts.max()}, more than "
            f"frame_count_tolerance={frame_count_tolerance:g} apart; alpha depends on the frame "
            "count, so calibrate each integration time separately."
        )
    reference = float(np.median(counts))
    referred = [
        _refer_ratios(scan_ratios, frames, reference)
        for scan_ratios, frames in zip(ratios, counts, strict=True)
    ]
    return np.concatenate(referred), counts


def _refer_ratios(ratios: ArrayLike, from_frames: float, to_frames: float) -> NDArray[np.float64]:
    """Refer power/noise ratios between frame counts: deviation from 1 scales as 1/sqrt(frames)."""
    return 1.0 + (np.asarray(ratios, dtype=float) - 1.0) * np.sqrt(from_frames / to_frames)


@dataclass(frozen=True, eq=False)
class NoiseCalibration:
    """Empirical alpha for any target Pfa, measured on noise-only scans.

    Build with :func:`calibrate_from_noise` rather than directly.

    Attributes
    ----------
    num_frames : int
        Reference (median) frame count of the calibration scans; ``sorted_ratios`` and
        the tail refer to it.
    min_num_frames, max_num_frames : int
        Frame-count range of the calibration scans.
    frame_count_tolerance : float
        Relative margin beyond that range within which the calibration is applied.
    detector_signature : tuple
        :func:`detector_signature` of the detector used to compute the ratios.
    sorted_ratios : numpy.ndarray
        Every pooled cell-to-noise ratio, ascending.
    tail_pfa : float
        Fraction of cells above ``tail_threshold``; below this Pfa the GPD tail is used.
    tail_threshold : float
        Ratio above which the tail is modelled (u).
    tail_shape : float
        Fitted GPD shape (xi). Positive is heavy-tailed, negative bounded.
    tail_scale : float
        Fitted GPD scale (sigma).

    """

    num_frames: int
    min_num_frames: int
    max_num_frames: int
    frame_count_tolerance: float
    detector_signature: tuple
    sorted_ratios: NDArray[np.float64] = field(repr=False)
    tail_pfa: float
    tail_threshold: float
    tail_shape: float
    tail_scale: float
    _warned_pfas: set = field(default_factory=set, repr=False, init=False)

    @property
    def num_cells(self) -> int:
        """Number of pooled cells the calibration was measured on."""
        return int(self.sorted_ratios.size)

    def extrapolation_factor(self, pfa: float) -> float:
        """How far ``pfa`` lies below the smallest rate the pooled cells resolve.

        Values above 1 mean alpha comes from the fitted tail beyond the observed data.
        """
        return (1.0 / self.num_cells) / pfa

    def alpha(self, pfa: float, num_frames: int | None = None) -> float:
        """Threshold multiplier achieving ``pfa`` on data like the calibration scans.

        Parameters
        ----------
        pfa : float
            Target probability of false alarm, in (0, 1).
        num_frames : int, optional
            Frame count of the data being detected on. Alpha is referred from the reference
            frame count to this one; ``None`` uses the reference.

        Returns
        -------
        float
            Alpha such that a fraction ``pfa`` of noise-only cells exceed
            ``alpha * noise_estimate``.

        """
        if not 0 < pfa < 1:
            raise ValueError(f"pfa ({pfa}) must be in (0, 1)")

        if pfa >= self.tail_pfa:
            reference_alpha = float(np.quantile(self.sorted_ratios, 1.0 - pfa))
            return self._refer_alpha(reference_alpha, num_frames)

        factor = self.extrapolation_factor(pfa)
        if factor > _EXTRAPOLATION_WARNING_FACTOR and pfa not in self._warned_pfas:
            self._warned_pfas.add(pfa)
            warnings.warn(
                f"pfa={pfa:g} is {factor:.0f}x below the smallest rate {self.num_cells} "
                "calibration cells resolve; alpha rests on the extrapolated tail. Calibrate "
                "with more noise-only scans for a better-supported threshold.",
                stacklevel=2,
            )

        relative_rate = pfa / self.tail_pfa
        if abs(self.tail_shape) < _EXPONENTIAL_TAIL_SHAPE:
            excess = -self.tail_scale * np.log(relative_rate)
        else:
            excess = self.tail_scale / self.tail_shape * (relative_rate**-self.tail_shape - 1.0)
        return self._refer_alpha(float(self.tail_threshold + excess), num_frames)

    def _refer_alpha(self, reference_alpha: float, num_frames: int | None) -> float:
        if num_frames is None or num_frames == self.num_frames:
            return reference_alpha
        return float(_refer_ratios(reference_alpha, self.num_frames, num_frames))

    def matches(self, detector: Any, num_frames: int) -> str | None:
        """Describe why this calibration does not fit ``detector`` at ``num_frames``, if not.

        Parameters
        ----------
        detector : _CFARDetectorBase
            The detector this calibration would be applied to.
        num_frames : int
            Frame count of the data being detected on.

        Returns
        -------
        str or None
            A message naming the first mismatched setting, or ``None`` when it applies.

        """
        margin = 1.0 + self.frame_count_tolerance
        lowest = self.min_num_frames / margin
        highest = self.max_num_frames * margin
        if not lowest <= num_frames <= highest:
            return (
                f"calibrated for {self.min_num_frames}-{self.max_num_frames} frames "
                f"(accepting {lowest:.0f}-{highest:.0f}) but the data has {num_frames}"
            )
        current = detector_signature(detector)
        for name, calibrated, actual in zip(
            _SIGNATURE_FIELDS, self.detector_signature, current, strict=True
        ):
            if calibrated != actual:
                return (
                    f"calibrated with {name}={calibrated!r} but the detector has {name}={actual!r}"
                )
        return None


class NoiseCalibrator:
    """Measures a detector's noise-only ratio distribution and builds a calibration for it.

    Construct around the detector to be calibrated. Its structural settings (window, rank, edge
    handling) and its own noise estimator (``_power_and_noise``) are read from it; ``target_pfa``
    is not, since one calibration serves every Pfa. The detector need not have a
    ``noise_calibration`` yet.
    """

    def __init__(self, detector: Any) -> None:
        """Store the detector to be calibrated."""
        self.detector = detector

    def cell_noise_ratios(
        self,
        beamformed_scans: Iterable[ArrayLike],
        frame_count_tolerance: float = DEFAULT_FRAME_COUNT_TOLERANCE,
    ) -> NDArray[np.float64]:
        """Pooled ``frame-averaged power / local noise estimate`` for every cell of every scan.

        Ratios from scans with differing frame counts are referred to the median frame count (see
        the module docstring).

        It is the quantity :meth:`~.algorithms._CFARDetectorBase.detect` compares with alpha,
        computed with the detector's own noise estimator and edge handling.

        Parameters
        ----------
        beamformed_scans : Iterable[ArrayLike]
            Noise-only beamformer output, one ``(num_beams, num_frames)`` array per scan.
        frame_count_tolerance : float, optional
            Largest relative spread of frame counts across scans, by default 0.25.

        Returns
        -------
        numpy.ndarray
            One ratio per cell, shape ``(num_scans * num_beams,)``.

        Raises
        ------
        ValueError
            If there are no scans, frame counts spread beyond the tolerance, or a noise estimate is
            zero.

        """
        return _ratios_and_frame_counts(self.detector, beamformed_scans, frame_count_tolerance)[0]

    def calibrate_from_noise(
        self,
        beamformed_scans: Iterable[ArrayLike],
        tail_pfa: float = 1e-2,
        min_tail_exceedances: int = 200,
        frame_count_tolerance: float = DEFAULT_FRAME_COUNT_TOLERANCE,
    ) -> NoiseCalibration:
        """Measure the detector's cell-to-noise ratio distribution and assign it.

        Its window, rank and edge handling are recorded; ``target_pfa`` is not used, so one
        calibration serves every Pfa. The result is assigned to ``self.detector.noise_calibration``
        as well as being returned, so calibrating and attaching is a single call.

        Parameters
        ----------
        beamformed_scans : Iterable[ArrayLike]
            Noise-only beamformer output, one ``(num_beams, num_frames)`` array per scan, all with
            the same frame count and produced exactly as operational data will be (same array,
            shading, beamformer, band and scan length). A simulator built with no
            ``ground_truth_paths`` provides these; see :func:`beamformed_scans_from_sensor_data`.
        tail_pfa : float, optional
            Exceedance rate at which the generalised Pareto tail takes over from the empirical
            quantile, by default 1e-2. Broadwater and Chellappa (2010; see the module references)
            suggest 0.05-0.1 for detector outputs with fewer samples; the 1% default relies on the
            many cells pooled from noise-only scans. Lower it only when there are enough cells for
            the empirical quantile to be reliable there. Any fixed tail fraction is a rule of
            thumb: choosing it trades bias in the tail approximation against variance from fewer
            exceedances, and fixed-fraction rules lack theoretical support (Scarrott and MacDonald,
            2012, Sects. 1 and 3; see the module references).
        min_tail_exceedances : int, optional
            Minimum number of cells above the tail threshold for the tail fit, by default 200.
        frame_count_tolerance : float, optional
            Largest relative spread of frame counts accepted across the calibration scans, and the
            margin beyond their range within which the calibration is later applied, by default
            0.25.

        Returns
        -------
        NoiseCalibration
            Also assigned to ``self.detector.noise_calibration``.

        Raises
        ------
        ValueError
            If ``tail_pfa`` is not in (0, 1), the scans are empty or inconsistent, or they contain
            too few cells to fit the tail.

        Notes
        -----
        The tail fit needs roughly ``min_tail_exceedances / tail_pfa`` pooled cells (about 20,000
        at the defaults). Each scan contributes ``num_beams`` cells, so divide that total by
        ``num_beams`` for the number of noise-only time steps needed: roughly 560 scans for a
        36-beam detector, roughly 110 for a 180-beam one. This only applies when the requested
        ``target_pfa`` is below ``tail_pfa``, since only then is the fitted tail used.

        Examples
        --------
        >>> calibration = NoiseCalibrator(detector).calibrate_from_noise(noise_scans)
        ... # doctest: +SKIP
        >>> detector.target_pfa = 1e-4  # doctest: +SKIP

        """
        if not 0 < tail_pfa < 1:
            raise ValueError(f"tail_pfa ({tail_pfa}) must be in (0, 1)")

        ratios, frame_counts = _ratios_and_frame_counts(
            self.detector, beamformed_scans, frame_count_tolerance
        )
        sorted_ratios = np.sort(ratios)
        num_cells = sorted_ratios.size

        tail_threshold = float(np.quantile(sorted_ratios, 1.0 - tail_pfa))
        exceedances = sorted_ratios[sorted_ratios > tail_threshold] - tail_threshold
        if exceedances.size < min_tail_exceedances:
            needed = int(np.ceil(min_tail_exceedances / tail_pfa))
            raise ValueError(
                f"Only {exceedances.size} of {num_cells} cells exceed the tail threshold at "
                f"tail_pfa={tail_pfa:g}; at least {min_tail_exceedances} are needed to fit the "
                f"tail. Provide noise-only scans totalling about {needed} cells."
            )

        tail_shape, _, tail_scale = genpareto.fit(exceedances, floc=0.0)
        calibration = NoiseCalibration(
            num_frames=int(np.median(frame_counts)),
            min_num_frames=int(frame_counts.min()),
            max_num_frames=int(frame_counts.max()),
            frame_count_tolerance=frame_count_tolerance,
            detector_signature=detector_signature(self.detector),
            sorted_ratios=sorted_ratios,
            tail_pfa=exceedances.size / num_cells,
            tail_threshold=tail_threshold,
            tail_shape=float(tail_shape),
            tail_scale=float(tail_scale),
        )
        self.detector.noise_calibration = calibration
        return calibration


def beamformed_scans_from_sensor_data(
    sensor_data_gen: Iterable[Any],
    band_label: str | None = None,
    progress_bar: bool = False,
    total: int | None = None,
) -> Iterator[NDArray]:
    """Yield raw beamformer output from a simulator's ``sensor_data_gen()``.

    For calibration, build the simulator exactly as for the operational run but with no
    ``ground_truth_paths``, so every scan is ambient noise only.

    Parameters
    ----------
    sensor_data_gen : Iterable
        ``(timestamp, set of PassiveSonarSensorData)`` pairs.
    band_label : str, optional
        For multiband output (three-dimensional ``beamformed_data`` with ``band_labels``), the
        band whose ``(num_beams, num_frames)`` slice to yield. Calibrate each band's detector
        separately. Must be ``None`` for single-band output.
    progress_bar : bool, optional
        If True, show a progress bar over ``sensor_data_gen``, by default False. The bar only
        appears once ``sensor_data_gen`` starts yielding, so it appears after (not during) a
        simulator's own ``sensor_data_gen(progress_bar=True)`` target-propagation phase.
    total : int, optional
        Total number of timesteps for the progress bar.

    Yields
    ------
    numpy.ndarray
        Each sensor data item's ``(num_beams, num_frames)`` beamformed data, skipping empty
        ones.

    Raises
    ------
    ValueError
        If multiband output is encountered without ``band_label``, ``band_label`` is given for
        single-band output, or the label is not one of the data's bands.

    """
    if progress_bar:
        # Deferred to break a module cycle: .passive imports from .algorithms, which imports
        # NoiseCalibration from this module.
        from .passive import _lazy_progress_bar

        sensor_data_gen = _lazy_progress_bar(sensor_data_gen, "Noise calibration", total)

    for _, sensor_data_set in sensor_data_gen:
        for sensor_data in sensor_data_set:
            data = sensor_data.beamformed_data
            if data is None or np.asarray(data).size == 0:
                continue
            labels = getattr(sensor_data, "band_labels", None)
            if labels is None:
                if band_label is not None:
                    raise ValueError(
                        f"band_label={band_label!r} was given but the beamformed data is "
                        "single-band"
                    )
                yield data
                continue
            if band_label is None:
                raise ValueError(
                    f"Beamformed data has bands {list(labels)}; pass band_label to calibrate "
                    "one band's detector at a time."
                )
            if band_label not in labels:
                raise ValueError(f"band_label={band_label!r} is not one of {list(labels)}")
            yield np.asarray(data)[list(labels).index(band_label)]
