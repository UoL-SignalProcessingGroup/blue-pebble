"""Acoustic sensor data module."""

from datetime import datetime
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from stonesoup.base import Property
from stonesoup.types.sensordata import SensorData

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
BeamformedData: TypeAlias = NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]]


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
