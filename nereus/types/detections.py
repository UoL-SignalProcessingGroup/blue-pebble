"""Defines the detection classes to represent different types of sensor detections.

This module is a derivative work of Stone Soup, available at
https://github.com/dstl/StoneSoup.

The original work is licensed under the MIT License.
© Crown Copyright 2017-2025 Defence Science and Technology Laboratory UK
© Crown Copyright 2018-2025 Defence Research and Development Canada / Recherche et développement pour la défense Canada
© Copyright 2018-2025 University of Liverpool UK
© Copyright 2020-2025 Fraunhofer FKIE
© Copyright 2020-2025 John Hiles
© Copyright 2020-2025 Riskaware Ltd
© Copyright 2021-2025 Roke Manor Research Ltd UK
© Copyright 2023-2025 Loughborough University UK
© Copyright 2025 Joshua J. Wakefield.
"""  # noqa: E501

from __future__ import annotations

from datetime import datetime

from nereus.models.measurement import LinearGaussianMeasurementModel
from nereus.types.arrays import StateVector


class Detection:
    """Abstract base class for a detection.

    A detection represents a measurement received from a sensor at a specific
    point in time. This class should not be instantiated directly.

    Attributes:
        state_vector (StateVector): The measurement vector.
        timestamp (datetime): The time at which the measurement was made.
        measurement_model (LinearGaussianMeasurementModel | None): The model
            that maps the system state to the measurement space.
        probability (float | None): The probability associated with the
            detection, if applicable.

    """

    def __init__(
        self,
        state_vector: StateVector | None = None,
        timestamp: datetime | None = None,
        measurement_model: LinearGaussianMeasurementModel | None = None,
        probability: float | None = None,
    ) -> None:
        """Initialise the Detection.

        Args:
            state_vector (StateVector): The measurement vector.
            timestamp (datetime): The time at which the measurement was made.
            measurement_model (LinearGaussianMeasurementModel | None, optional):
                The model that maps the system state to the measurement space.
                Defaults to None.
            probability (float | None, optional): The probability associated with
                the detection, if applicable. Defaults to None.

        """
        self.state_vector = state_vector
        self.measurement_model = measurement_model
        self.timestamp = timestamp
        self.probability = probability


class TrueDetection(Detection):
    """Represents a true detection originating from a target."""


class MissedDetection(Detection):
    """Represents a missed detection.

    This type of detection occurs when a true target is not detected.
    """


class Clutter(Detection):
    """Represents a false alarm or clutter in the measurement space.

    This type of detection does not originate from any true target.
    """
