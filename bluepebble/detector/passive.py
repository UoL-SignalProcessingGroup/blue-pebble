"""Defines a passive sonar detector that processes beamformed sensor data."""

from collections import defaultdict, deque
from collections.abc import Generator, Iterable
from datetime import datetime
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader
from stonesoup.types.detection import Detection
from tqdm import tqdm

from ..types.sensordata import PassiveSonarSensorData
from .algorithms import _CFARDetectorBase, _directional_power

FloatArray: TypeAlias = NDArray[np.float64]
DetectionArray: TypeAlias = NDArray[np.float64]
SensorDataStep: TypeAlias = tuple[datetime, Iterable["PassiveSonarSensorData"]]
DetectionBatch: TypeAlias = tuple[datetime, set[Detection]]
_BandedStep: TypeAlias = tuple[datetime, dict[str, set[Detection]]]


def snr_from_beamformed_data(
    beamformed_data: ArrayLike,
    output_type: str = "snr_percentile",
    percentile: int = 10,
) -> FloatArray:
    """Reduce a beamformed power map to a per-beam map, using a single global normalisation.

    Not used internally by :class:`PassiveSonarDetector` or :class:`BandDetector` -- those now
    get per-beam SNR from their CFAR detector's own local (per-bearing) noise-floor estimate via
    ``detector.snr_map()``, which is more representative of what actually drove each detection
    decision. Kept as a standalone utility for cases that want a quick global-normalised view
    (e.g. debugging or plotting) without needing a configured CFAR detector.

    Parameters
    ----------
    beamformed_data : ArrayLike
        Beamformer output with shape ``(num_beams, num_frames)``.
    output_type : str, optional
        Normalisation to apply. One of ``"snr_percentile"`` (default), ``"median_power"``,
        ``"log_power"``, or ``"power"``.
    percentile : int, optional
        Percentile of directional power used as the noise estimate when ``output_type`` is
        ``"snr_percentile"`` (default is 10).

    Returns
    -------
    FloatArray
        Per-beam map with shape ``(num_beams,)``. In dB for every ``output_type``
        except ``"power"``, which is linear.

    Raises
    ------
    ValueError
        If ``output_type`` is not one of the supported options.

    """
    data = np.asarray(beamformed_data)

    if output_type == "power":
        return np.mean(_directional_power(data), axis=1)

    if output_type == "log_power":
        return 10 * np.log10(np.mean(_directional_power(data), axis=1))

    if output_type in ("snr_percentile", "median_power"):
        directional_power = np.mean(_directional_power(data), axis=1)
        if output_type == "snr_percentile":
            noise_power_estimate = np.percentile(directional_power, percentile)
        else:
            noise_power_estimate = np.median(directional_power)
        epsilon = np.finfo(float).eps
        return 10 * np.log10((directional_power + epsilon) / (noise_power_estimate + epsilon))

    raise ValueError(f"Unsupported beamformer_output_type: {output_type}")


class PassiveSonarDetector(DetectionReader):
    """A passive sonar detector that processes beamformed sensor data.

    This detector takes ``PassiveSonarSensorData`` as input and runs a single CFAR-family
    ``detector`` directly against each frame's raw beamformed power map. Frame integration,
    local noise-floor estimation, and wrap-aware peak consolidation are all handled internally
    by the detector (see :mod:`.algorithms`). Detections are produced with bearing values
    derived from the provided steering azimuths.

    Attributes
    ----------
    detector : _CFARDetectorBase
        The CFAR detector (e.g. :class:`~.algorithms.OSCFARDetector`) applied to each frame's
        raw beamformed data.
    sensor_data_gen : Generator[SensorDataStep, None, None]
        Generator yielding sensor-data batches.
    steering_azimuths_rad : FloatArray
        An array of steering azimuth angles in radians corresponding to the beams.

    """

    detector: _CFARDetectorBase = Property(
        doc="CFAR detector applied to each frame's raw beamformed data.",
    )
    sensor_data_gen: Generator[SensorDataStep, None, None] = Property(
        doc="Generator that yields PassiveSonarSensorData objects",
    )
    steering_azimuths_rad: FloatArray = Property(
        doc="Array of steering azimuth angles in radians.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the passive sonar detector."""
        super().__init__(*args, **kwargs)
        self._snr_history: list[FloatArray] = []

    @property
    def snr_history(self) -> FloatArray:
        """Recorded SNR history.

        Returns
        -------
        FloatArray
            Array of shape (num_timesteps, num_beams) containing SNR values. If no history is
            available an empty array is returned.

        """
        if not self._snr_history:
            return np.array([], dtype=np.float64)
        return np.asarray(self._snr_history, dtype=np.float64)

    @BufferedGenerator.generator_method
    def detections_gen(
        self,
        progress_bar: bool = False,
        total_timesteps: int | None = None,
    ) -> Generator[DetectionBatch, None, None]:
        """Generate detections from sensor data.

        Iterates through ``sensor_data_gen`` and runs ``detector`` directly against each
        frame's raw beamformed data, yielding Stone Soup ``Detection`` objects (bearing-only
        measurements).

        Parameters
        ----------
        progress_bar : bool, optional
            If True, wrap the input generator with a progress bar (default is False).
        total_timesteps : int, optional
            Total number of timesteps for the progress bar. Required if `progress_bar` is True.

        Yields
        ------
        tuple
            A tuple of ``(timestamp, set[Detection])`` for each processed timestep.

        """
        sensor_data_iterator: Iterable[SensorDataStep] = self.sensor_data_gen
        if progress_bar:
            sensor_data_iterator = tqdm(
                sensor_data_iterator, desc="Generating Detections", total=total_timesteps
            )

        for timestamp, sensor_data_set in sensor_data_iterator:
            detections: set[Detection] = set()
            snr: FloatArray = np.array([], dtype=np.float64)

            for sensor_data in sensor_data_set:
                beamformed_data = sensor_data.beamformed_data

                if beamformed_data is None or beamformed_data.size == 0:
                    continue

                # snr_map() and detect() each estimate the noise floor independently -- a
                # modest redundant computation in exchange for keeping "report the full
                # picture" and "decide detections" as separate concerns. Worth revisiting if
                # this shows up in profiling.
                snr = self.detector.snr_map(beamformed_data)
                raw_detections: DetectionArray = self.detector.detect(beamformed_data)

                if raw_detections.size > 0:
                    for raw_det in raw_detections:
                        detection_index = int(raw_det[0])
                        bearing_rad = self.steering_azimuths_rad[detection_index]

                        detections.add(
                            Detection(
                                state_vector=[[bearing_rad]],
                                timestamp=sensor_data.timestamp,
                                metadata={"snr_db": raw_det[1]},
                            )
                        )

            self._snr_history.append(snr)

            yield timestamp, detections


class BandDetector(Base):
    """CFAR detector for one frequency band.

    Each band of a multiband beamformer gets its own instance, so bands can differ in
    sensitivity (guard/training cell sizing, target Pfa, etc). Bands are matched to detectors
    by label; see :class:`MultibandPassiveSonarDetector`.
    """

    detector: _CFARDetectorBase = Property(
        doc="CFAR detector applied to this band's raw beamformed data.",
    )

    def detect(self, beamformed_data: ArrayLike) -> tuple[FloatArray, DetectionArray]:
        """Run this band's detector against its raw beamformed data.

        Parameters
        ----------
        beamformed_data : ArrayLike
            This band's beamformer output, with shape ``(num_beams, num_frames)``.

        Returns
        -------
        tuple[FloatArray, DetectionArray]
            The band's full per-beam SNR map, and the detections found in it.

        """
        snr = self.detector.snr_map(beamformed_data)
        detections = self.detector.detect(beamformed_data)
        return snr, detections


class _SensorDataPump:
    """Step a source generator once and fan each result out to every subscriber.

    A sensor-data generator can only be consumed once, but a multiband detector and its
    per-band readers all need the same stream. Each subscriber gets its own queue; whichever
    one runs dry first advances the shared source and appends the result to every queue.
    That keeps K band readers on one simulation pass.
    """

    def __init__(self, source: "Generator[_BandedStep, None, None]") -> None:
        """Wrap a source generator that yields per-band detection batches."""
        self._source = source
        self._queues: list[deque[_BandedStep]] = []
        self._started = False

    def subscribe(self) -> int:
        """Register a new subscriber and return its identifier.

        Returns
        -------
        int
            Queue index for the new subscriber.

        Raises
        ------
        RuntimeError
            If the source has already been advanced, since the new subscriber would
            silently miss everything consumed so far.

        """
        if self._started:
            raise RuntimeError(
                "Cannot subscribe after iteration has started: this reader would miss "
                "every timestep already consumed. Create all band readers before "
                "iterating any of them."
            )
        self._queues.append(deque())
        return len(self._queues) - 1

    def stream(self, subscriber_id: int) -> "Generator[_BandedStep, None, None]":
        """Yield every step for one subscriber, advancing the source only when needed.

        Parameters
        ----------
        subscriber_id : int
            Identifier returned by :meth:`subscribe`.

        Yields
        ------
        _BandedStep
            The next timestep for this subscriber.

        """
        queue = self._queues[subscriber_id]
        while True:
            while not queue:
                self._started = True
                try:
                    item = next(self._source)
                except StopIteration:
                    return
                for each in self._queues:
                    each.append(item)
            yield queue.popleft()


class MultibandPassiveSonarDetector(DetectionReader):
    """Detect independently in each band of a multiband beamformer's output.

    This detector consumes the three-dimensional ``beamformed_data`` produced by a
    beamformer configured with :class:`~.FrequencyBand` objects, and applies a separate
    :class:`BandDetector` to each band's power map. Bands are matched to detectors by the
    labels carried on the sensor data, so a band's ``label`` must appear in
    ``band_detectors`` unless a ``default_detector`` is set.

    Separate or combined tracking
    -----------------------------
    Iterating this detector yields the **union** of all bands' detections per timestep, so it
    is a drop-in :class:`~stonesoup.reader.base.DetectionReader` for a single tracker. Every
    detection carries its originating band in ``metadata['band']``, so provenance survives
    into a track's associated detections.

    For per-band tracking, :meth:`band_reader` returns a reader restricted to one band,
    suitable for feeding one tracker per band. All readers share a single pass over the
    sensor data, so K trackers cost one simulation. Create every band reader before
    iterating any of them.

    Each reader may be iterated once, as the underlying sensor-data generator is consumed.
    """

    band_detectors: dict[str, BandDetector] = Property(
        doc="Detector per band, keyed by the band's label.",
    )
    sensor_data_gen: Generator[SensorDataStep, None, None] = Property(
        doc="Generator that yields PassiveSonarSensorData objects",
    )
    steering_azimuths_rad: FloatArray = Property(
        doc="Array of steering azimuth angles in radians.",
    )
    default_detector: BandDetector | None = Property(
        default=None,
        doc="Detector for bands with no entry in 'band_detectors'. When None, an "
        "unrecognised band label is an error.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the multiband detector."""
        super().__init__(*args, **kwargs)
        self._snr_history: dict[str, list[FloatArray]] = defaultdict(list)
        self._pump = _SensorDataPump(self._banded_steps())
        self._subscriber_id: int | None = None

    @property
    def snr_history(self) -> dict[str, FloatArray]:
        """Recorded SNR history per band.

        Returns
        -------
        dict of str to FloatArray
            Maps each band label to an array of shape ``(num_timesteps, num_beams)``.
            Bands are present only once they have been processed, so this is empty before
            iteration begins.

        """
        return {
            label: np.asarray(rows, dtype=np.float64) for label, rows in self._snr_history.items()
        }

    def band_reader(self, band_label: str) -> "_BandDetectionReader":
        """Return a detection reader restricted to a single band.

        Parameters
        ----------
        band_label : str
            Label of the band to read, matching a :class:`~.FrequencyBand` label.

        Returns
        -------
        _BandDetectionReader
            A reader yielding only this band's detections, sharing the parent's single pass
            over the sensor data.

        Raises
        ------
        RuntimeError
            If iteration has already started (see :meth:`_SensorDataPump.subscribe`).

        """
        return _BandDetectionReader(
            parent=self,
            band_label=band_label,
            subscriber_id=self._pump.subscribe(),
        )

    def _detector_for(self, band_label: str) -> BandDetector:
        """Return the detector configured for a band.

        Parameters
        ----------
        band_label : str
            Band label taken from the sensor data.

        Returns
        -------
        BandDetector
            The band's own detector, or ``default_detector`` when it has no entry.

        Raises
        ------
        KeyError
            If the band has no detector and no ``default_detector`` is configured.

        """
        detector = self.band_detectors.get(band_label, self.default_detector)
        if detector is None:
            raise KeyError(
                f"No detector for band {band_label!r} and no default_detector is set. "
                f"Configured bands: {sorted(self.band_detectors)}"
            )
        return detector

    def _banded_steps(self) -> "Generator[_BandedStep, None, None]":
        """Process each sensor-data step into per-band detection sets.

        This runs exactly once per timestep no matter how many readers are attached, and is
        where SNR history is recorded.

        Yields
        ------
        tuple of (datetime, dict)
            Timestamp and a mapping of band label to that band's detections.

        Raises
        ------
        ValueError
            If the beamformed data is not three-dimensional with band labels, which means
            the beamformer was not configured with bands.

        """
        for timestamp, sensor_data_set in self.sensor_data_gen:
            detections_by_band: dict[str, set[Detection]] = defaultdict(set)

            for sensor_data in sensor_data_set:
                beamformed_data = sensor_data.beamformed_data
                if beamformed_data is None or beamformed_data.size == 0:
                    continue

                data = np.asarray(beamformed_data)
                band_labels = sensor_data.band_labels
                if band_labels is None or data.ndim != 3:
                    raise ValueError(
                        "MultibandPassiveSonarDetector requires beamformed data with a "
                        f"band axis, but got {data.ndim}D data with band_labels="
                        f"{band_labels!r}. Configure the beamformer with 'bands', or use "
                        "PassiveSonarDetector for single-band output."
                    )

                for band_idx, band_label in enumerate(band_labels):
                    band_detector = self._detector_for(band_label)
                    snr, raw_detections = band_detector.detect(data[band_idx])
                    self._snr_history[band_label].append(snr)

                    for raw_det in raw_detections:
                        detections_by_band[band_label].add(
                            Detection(
                                state_vector=[[self.steering_azimuths_rad[int(raw_det[0])]]],
                                timestamp=sensor_data.timestamp,
                                metadata={"band": band_label, "snr_db": float(raw_det[1])},
                            )
                        )

            yield timestamp, dict(detections_by_band)

    @BufferedGenerator.generator_method
    def detections_gen(
        self,
        progress_bar: bool = False,
        total_timesteps: int | None = None,
    ) -> Generator[DetectionBatch, None, None]:
        """Generate the union of every band's detections for each timestep.

        Parameters
        ----------
        progress_bar : bool, optional
            If True, wrap iteration with a progress bar (default is False).
        total_timesteps : int, optional
            Total number of timesteps for the progress bar.

        Yields
        ------
        tuple
            A tuple of ``(timestamp, set[Detection])`` combining all bands, where each
            detection records its band in ``metadata['band']``.

        """
        if self._subscriber_id is None:
            self._subscriber_id = self._pump.subscribe()

        steps: Iterable[_BandedStep] = self._pump.stream(self._subscriber_id)
        if progress_bar:
            steps = tqdm(steps, desc="Generating Detections", total=total_timesteps)

        for timestamp, detections_by_band in steps:
            combined: set[Detection] = set()
            for band_detections in detections_by_band.values():
                combined |= band_detections
            yield timestamp, combined


class _BandDetectionReader(DetectionReader):
    """Detections for one band of a parent :class:`MultibandPassiveSonarDetector`."""

    parent: MultibandPassiveSonarDetector = Property(doc="Detector producing the bands.")
    band_label: str = Property(doc="Label of the band this reader is restricted to.")
    subscriber_id: int = Property(doc="Identifier of this reader's queue on the parent pump.")

    @BufferedGenerator.generator_method
    def detections_gen(self) -> Generator[DetectionBatch, None, None]:
        """Generate this band's detections for each timestep.

        Yields
        ------
        tuple
            A tuple of ``(timestamp, set[Detection])`` for this band alone. The set is
            empty for timesteps where the band produced no detections.

        """
        for timestamp, detections_by_band in self.parent._pump.stream(self.subscriber_id):
            yield timestamp, detections_by_band.get(self.band_label, set())
