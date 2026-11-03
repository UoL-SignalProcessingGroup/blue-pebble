"""Active sonar detection: matched filtering then CFAR (Constant False Alarm Rate) along range."""

from collections.abc import Generator, Iterable, Iterator
from datetime import datetime
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader
from stonesoup.types.angle import Bearing
from stonesoup.types.detection import Detection

from ..sigproc.matchedfilter import MatchedFilter
from ..sigproc.utils import parabolic_peak_offset
from ..types.sensordata import ActiveSonarSensorData
from .algorithms import _CFARDetectorBase

FloatArray: TypeAlias = NDArray[np.float64]
DetectionBatch: TypeAlias = tuple[datetime, set[Detection]]

# Transmitter and receiver closer than this are treated as co-located (monostatic).
_MONOSTATIC_TOLERANCE_M = 1e-6


class ActiveSonarDetector(DetectionReader):
    r"""Detect echoes in active sonar data by matched filtering each beam and running CFAR.

    For each ping, each beam of ``beamformed_data`` (or the single channel of
    ``received_waveform`` when there is no beamformer) is matched-filtered against the
    transmit pulse, and ``detector`` runs along range on the filter's output power. Each
    detected cell :math:`k` gives a round-trip delay :math:`\tau = (k + \delta)/f_s`, where
    :math:`\delta` is the parabolic sub-sample offset of the peak, and from it a range:

    - monostatic (transmitter at the receiving array's reference point):
      :math:`r = c\tau/2`, the slant range;
    - bistatic: :math:`r = c\tau`, the range sum :math:`R_\text{tx} + R_\text{rx}`, which
      places the target on an ellipse with the transmitter and receiver at its foci.

    Each detection's state vector is ``[bearing, range]`` when beamformed, with the bearing
    as a Stone Soup :class:`~stonesoup.types.angle.Bearing`, otherwise ``[range]``. No
    measurement model is attached; converting a bistatic range sum to a position is left to
    the tracker's measurement model.

    Assumptions
    -----------
    - The sound speed is constant along both legs, so delay converts to range linearly.
    - Sample 0 of the waveform is the transmit instant, as
      :class:`~bluepebble.simulator.ActiveSonarSimulator` produces.
    - Range cells are independent of each other for CFAR, so the detector must not wrap
      (``circular=False``).
    """

    detector: _CFARDetectorBase = Property(
        doc="CFAR detector run along range on each beam's matched-filter output power",
    )
    sensor_data_gen: Iterable[tuple[datetime, set[ActiveSonarSensorData]]] = Property(
        doc="Generator yielding (timestamp, set of ActiveSonarSensorData)",
    )
    sampling_rate_hz: float = Property(doc="Sampling rate of the received waveform, in Hz")
    matched_filter: MatchedFilter | None = Property(
        default=None,
        doc="Matched filter applied to each beam. None uses an unwindowed MatchedFilter.",
    )
    sound_speed_ms: float = Property(
        default=1500.0, doc="Sound speed converting delay to range, in m/s"
    )
    min_range_m: float = Property(
        default=0.0, doc="Detections closer than this range are dropped, in metres"
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the detector and check its settings.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``DetectionReader``.
        **kwargs : object
            Keyword arguments forwarded to ``DetectionReader``.

        Raises
        ------
        ValueError
            If ``detector`` wraps around (``circular=True``), or ``sampling_rate_hz`` or
            ``sound_speed_ms`` is not positive.

        """
        super().__init__(*args, **kwargs)
        if self.matched_filter is None:
            self.matched_filter = MatchedFilter()
        if self.detector.circular:
            raise ValueError(
                "Range does not wrap around, so the detector needs circular=False."
            )
        if not self.sampling_rate_hz > 0:
            raise ValueError(f"sampling_rate_hz ({self.sampling_rate_hz}) must be positive.")
        if not self.sound_speed_ms > 0:
            raise ValueError(f"sound_speed_ms ({self.sound_speed_ms}) must be positive.")

    @BufferedGenerator.generator_method
    def detections_gen(self) -> Generator[DetectionBatch, None, None]:
        """Yield each ping's detections.

        Yields
        ------
        tuple of (datetime, set of Detection)
            The ping's timestamp and its detections.

        Raises
        ------
        ValueError
            If the sensor data has several channels and no beamformed data, or its
            beamformed data has a different length from the received waveform.

        """
        for timestamp, sensor_data_set in self.sensor_data_gen:
            detections: set[Detection] = set()
            for sensor_data in sensor_data_set:
                detections |= self._detect(sensor_data)
            yield timestamp, detections

    def _detect(self, sensor_data: ActiveSonarSensorData) -> set[Detection]:
        """Return one ping's detections across every beam."""
        power = matched_filter_power(sensor_data, self.matched_filter)
        bearings = sensor_data.steering_bearings_rad
        bistatic = _is_bistatic(sensor_data)
        # A monostatic echo travels the range twice.
        metres_per_second = self.sound_speed_ms if bistatic else self.sound_speed_ms / 2

        detections = set()
        for beam, beam_power in enumerate(power):
            amplitude = np.sqrt(beam_power)
            for cell, snr_db in self.detector.detect(beam_power[:, np.newaxis]):
                k = int(cell)
                delay_s = (k + _peak_offset(amplitude, k)) / self.sampling_rate_hz
                range_m = metres_per_second * delay_s
                if range_m < self.min_range_m:
                    continue
                state_vector = (
                    [[range_m]] if bearings is None else [[Bearing(bearings[beam])], [range_m]]
                )
                detections.add(
                    Detection(
                        state_vector=state_vector,
                        timestamp=sensor_data.timestamp,
                        metadata={
                            "delay_s": delay_s,
                            "snr_db": float(snr_db),
                            "bistatic": bistatic,
                            "transmitter_position": sensor_data.transmitter_position,
                            "receiver_reference_position": (
                                sensor_data.receiver_reference_position
                            ),
                        },
                    )
                )
        return detections


def matched_filter_power(
    sensor_data: ActiveSonarSensorData, matched_filter: MatchedFilter | None = None
) -> FloatArray:
    """Return the matched-filter output power of each beam, shape ``(num_beams, num_samples)``.

    Uses ``beamformed_data`` when present, otherwise the single channel of
    ``received_waveform``.

    Parameters
    ----------
    sensor_data : ActiveSonarSensorData
        One ping's sensor data.
    matched_filter : MatchedFilter, optional
        Filter to apply; None uses an unwindowed :class:`~bluepebble.sigproc.MatchedFilter`.

    Returns
    -------
    FloatArray
        Squared matched-filter magnitude, the power CFAR compares against its threshold.

    Raises
    ------
    ValueError
        If there are several channels and no beamformed data, or the beamformed data's length
        differs from the received waveform's.

    """
    received = np.asarray(sensor_data.received_waveform)
    if sensor_data.beamformed_data is not None:
        channels = np.asarray(sensor_data.beamformed_data)
        if channels.ndim != 2 or channels.shape[1] != received.shape[1]:
            raise ValueError(
                f"beamformed_data has shape {channels.shape}, but it must keep the received "
                f"waveform's {received.shape[1]} samples so that sample k is still delay k/fs; "
                "use a time-preserving beamformer such as DelayAndSumBeamformer with "
                "domain='frequency'."
            )
    elif received.shape[0] == 1:
        channels = received
    else:
        raise ValueError(
            f"The received waveform has {received.shape[0]} channels and no beamformed data; "
            "configure the simulator's beamformer and steering_calculator."
        )
    magnitude = (matched_filter or MatchedFilter()).filter(channels, sensor_data.transmit_pulse)
    return np.square(magnitude)


def matched_filter_scans_from_sensor_data(
    sensor_data_gen: Iterable[tuple[datetime, set[ActiveSonarSensorData]]],
    matched_filter: MatchedFilter | None = None,
) -> Iterator[NDArray[Any]]:
    """Yield each beam's matched-filter power as a noise scan for calibration.

    Pass the scans to :meth:`~bluepebble.detector.NoiseCalibrator.calibrate_from_noise`. For
    calibration, build the simulator exactly as for the operational run but with no
    ``ground_truth_paths``, so every ping is noise only.

    Parameters
    ----------
    sensor_data_gen : Iterable
        ``(timestamp, set of ActiveSonarSensorData)`` pairs.
    matched_filter : MatchedFilter, optional
        The same filter the detector uses; None uses an unwindowed one.

    Yields
    ------
    numpy.ndarray
        One ``(num_samples, 1)`` power scan per beam per ping: range cells by one frame, the
        shape :class:`ActiveSonarDetector` passes to its CFAR detector.

    """
    for _, sensor_data_set in sensor_data_gen:
        for sensor_data in sensor_data_set:
            for beam_power in matched_filter_power(sensor_data, matched_filter):
                yield beam_power[:, np.newaxis]


def _peak_offset(amplitude: FloatArray, k: int) -> float:
    """Return the parabolic sub-sample offset at ``k``, or 0 if ``k`` is not a local maximum.

    A CFAR cell need not be a peak of the matched-filter output, and on a slope the parabola's
    vertex can lie many samples away.
    """
    if 0 < k < len(amplitude) - 1 and amplitude[k] >= max(amplitude[k - 1], amplitude[k + 1]):
        return parabolic_peak_offset(amplitude, k)
    return 0.0


def _is_bistatic(sensor_data: ActiveSonarSensorData) -> bool:
    """Return whether the transmitter is away from the receiving array's reference point."""
    transmitter = sensor_data.transmitter_position
    receiver = sensor_data.receiver_reference_position
    if transmitter is None or receiver is None:
        return False
    separation = np.linalg.norm(np.ravel(transmitter) - np.ravel(receiver))
    return bool(separation > _MONOSTATIC_TOLERANCE_M)
