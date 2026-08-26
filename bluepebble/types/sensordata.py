"""Acoustic sensor data module."""

from datetime import datetime
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.types.sensordata import SensorData

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
BeamformedData: TypeAlias = NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]]


class ActiveSonarSensorData(SensorData):
    """Sensor data produced by one active sonar ping.

    Attributes
    ----------
    received_waveform : ComplexArray
        Complex analytic received waveform of shape ``(n_receive_samples,)``.
        Contains the sum of all eigenray-weighted, delayed copies of the
        transmit pulse returned from all targets.
    transmit_pulse : ComplexArray
        The transmitted pulse of shape ``(n_pulse_samples,)``, retained for
        matched filtering downstream.
    timestamp : datetime
        Time at which the ping was transmitted.

    """

    received_waveform: ComplexArray = Property(doc="Complex received echo waveform")
    transmit_pulse: ComplexArray = Property(doc="Transmitted pulse waveform")
    timestamp: datetime = Property(doc="Ping transmission timestamp")


class PassiveSonarSensorData(SensorData):
    """Custom sensor data for passive sonar arrays.

    This class extends Stone Soup's ``SensorData`` to include data specific to passive sonar
    simulation. It holds the raw time-series signals from each sensor, the final beamformed power
    map, and the timestamp of the data snapshot.
    """

    raw_signals: ComplexArray = Property(doc="Raw acoustic signals from sensor array")
    beamformed_data: BeamformedData | None = Property(
        default=None,
        doc="Processed beamformed output",
    )
    timestamp: datetime = Property(doc="Timestamp of the sensor data")
