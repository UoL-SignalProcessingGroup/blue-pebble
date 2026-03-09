"""Acoustic sensor data module."""

from datetime import datetime

import numpy as np
from stonesoup.base import Property
from stonesoup.types.sensordata import SensorData


class PassiveSonarSensorData(SensorData):
    """Custom sensor data for passive sonar arrays.

    This class extends Stone Soup's ``SensorData`` to include data specific to passive sonar
    simulation. It holds the raw time-series signals from each sensor, the final beamformed power
    map, and the timestamp of the data snapshot.
    """

    raw_signals = Property(np.ndarray, doc="Raw acoustic signals from sensor array")
    beamformed_data = Property(np.ndarray, doc="Processed beamformed output")
    timestamp = Property(datetime, doc="Timestamp of the sensor data")