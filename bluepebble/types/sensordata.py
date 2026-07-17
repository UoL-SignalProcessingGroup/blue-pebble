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

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the sensor data and check any band labels against the data shape.

        Raises
        ------
        ValueError
            If ``band_labels`` is set but ``beamformed_data`` is missing, is not
            three-dimensional, or has a band axis of a different length.

        """
        super().__init__(*args, **kwargs)

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
