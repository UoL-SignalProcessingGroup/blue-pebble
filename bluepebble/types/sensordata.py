"""Acoustic sensor data module."""

from datetime import datetime
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.types.sensordata import SensorData

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.float64]
BeamformedData: TypeAlias = NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]]


class PassiveSonarSensorData(SensorData):
    """Custom sensor data for passive sonar arrays.

    This class extends Stone Soup's ``SensorData`` to include data specific to passive sonar
    simulation. It holds the raw time-series signals from each sensor, the final beamformed power
    map, and the timestamp of the data snapshot.

    ``steering_bearings_rad`` gives the bearing each beam was steered at for this scan. It
    travels with the data because an array-frame steering grid turns with the array, so the
    bearings can differ from one scan to the next.

    Multiband output
    ----------------
    A beamformer configured with frequency bands produces one power map per band, so
    ``beamformed_data`` is three-dimensional with a leading band axis and ``band_labels``
    names each slice of it. Single-band output leaves ``beamformed_data`` two-dimensional
    and ``band_labels`` as ``None``.
    """

    raw_signals: ComplexArray = Property(doc="Raw acoustic signals from sensor array")
    beamformed_data: BeamformedData | None = Property(
        default=None,
        doc="Processed beamformed output",
    )
    timestamp: datetime = Property(doc="Timestamp of the sensor data")
    band_labels: list[str] | None = Property(
        default=None,
        doc="Band label for each slice of a three-dimensional 'beamformed_data' band axis. "
        "None for two-dimensional single-band output.",
    )
    steering_bearings_rad: FloatArray | None = Property(
        default=None,
        doc="Bearing of each beam for this scan, in radians anticlockwise from +x, in the "
        "order of the beam axis of 'beamformed_data': its first axis, or its second for "
        "three-dimensional multiband data.",
    )
    demodulation_frequency_hz: float = Property(
        default=0.0,
        doc="Physical frequency of 0 Hz in ``raw_signals``, in Hz: the carrier for complex "
        "baseband data, 0 for passband data. The passive simulators produce passband data.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the sensor data and check band labels and bearings against its shape.

        Raises
        ------
        ValueError
            If ``band_labels`` is set but ``beamformed_data`` is missing, is not
            three-dimensional, or has a band axis of a different length; or if
            ``steering_bearings_rad`` is set but ``beamformed_data`` is missing, or the
            bearings are not one per beam.

        """
        super().__init__(*args, **kwargs)
        self._check_band_labels()
        self._check_steering_bearings()

    def _check_band_labels(self) -> None:
        """Raise if ``band_labels`` does not match the band axis of ``beamformed_data``."""
        if self.band_labels is None:
            return

        if self.beamformed_data is None:
            raise ValueError("band_labels was given without any beamformed_data to label")

        data = np.asarray(self.beamformed_data)
        if data.ndim != 3:
            raise ValueError(
                f"band_labels requires 3D beamformed_data with a leading band axis, got "
                f"{data.ndim}D data with shape {data.shape}"
            )
        if data.shape[0] != len(self.band_labels):
            raise ValueError(
                f"Got {len(self.band_labels)} band labels but beamformed_data has "
                f"{data.shape[0]} bands"
            )

    def _check_steering_bearings(self) -> None:
        """Raise unless ``steering_bearings_rad`` holds one bearing per beam."""
        if self.steering_bearings_rad is None:
            return

        if self.beamformed_data is None:
            raise ValueError(
                "steering_bearings_rad was given without any beamformed_data to label"
            )

        bearings = np.asarray(self.steering_bearings_rad)
        data = np.asarray(self.beamformed_data)
        beam_axis = 1 if data.ndim == 3 else 0
        if bearings.ndim != 1 or bearings.shape[0] != data.shape[beam_axis]:
            raise ValueError(
                f"steering_bearings_rad must hold one bearing per beam: got shape "
                f"{bearings.shape} for beamformed_data of shape {data.shape}, whose beam "
                f"axis is axis {beam_axis}"
            )


class ActiveSonarSensorData(SensorData):
    """Sensor data produced by one active sonar ping.

    ``received_waveform`` always has one row per receiving sensor, so a single hydrophone is a
    one-row array rather than a special case.

    Raises
    ------
    ValueError
        If ``received_waveform`` is not two-dimensional, or ``transmit_pulse`` is not
        one-dimensional.

    """

    received_waveform: ComplexArray = Property(
        doc="Complex analytic received waveform, shape (num_sensors, num_samples): every "
        "delayed, scaled copy of the transmit pulse returned from every target, plus noise. "
        "Its complex envelope about ``demodulation_frequency_hz`` when that is non-zero."
    )
    transmit_pulse: ComplexArray = Property(
        doc="Transmitted pulse, shape (num_pulse_samples,), kept for matched filtering. At "
        "baseband or passband, as ``received_waveform`` is."
    )
    timestamp: datetime = Property(doc="Time at which the ping was transmitted")
    beamformed_data: ComplexArray | None = Property(
        default=None,
        doc="Beamformed waveform, shape (num_beams, num_samples), if a beamformer was "
        "configured",
    )
    steering_bearings_rad: FloatArray | None = Property(
        default=None,
        doc="World-frame bearing of each beam in ``beamformed_data``, in radians",
    )
    transmitter_position: FloatArray | None = Property(
        default=None,
        doc="World-frame (x, y, z) of the transmitter's reference point at the ping",
    )
    receiver_reference_position: FloatArray | None = Property(
        default=None,
        doc="World-frame (x, y, z) of the receiving array's reference point at the ping",
    )
    demodulation_frequency_hz: float = Property(
        default=0.0,
        doc="Physical frequency of 0 Hz in the waveforms, in Hz: the carrier when they are "
        "complex envelopes (baseband), 0 when they are at passband. A beamformer needs it as "
        "its ``f0`` to steer baseband data.",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the sensor data and check the waveform shapes.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass.
        **kwargs : dict
            Keyword arguments passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        received = np.asarray(self.received_waveform)
        if received.ndim != 2:
            raise ValueError(
                "received_waveform must have shape (num_sensors, num_samples), got shape "
                f"{received.shape}; give a single sensor's waveform as one row."
            )
        pulse = np.asarray(self.transmit_pulse)
        if pulse.ndim != 1:
            raise ValueError(f"transmit_pulse must be one-dimensional, got shape {pulse.shape}")

    @property
    def num_sensors(self) -> int:
        """Number of receiving sensors (rows of ``received_waveform``)."""
        return int(np.asarray(self.received_waveform).shape[0])
