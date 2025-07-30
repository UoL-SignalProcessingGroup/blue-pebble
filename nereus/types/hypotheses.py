"""Defines classes for representing hypotheses about measurement-to-track associations.

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

from nereus.types.detections import Detection, MissedDetection
from nereus.types.states import State


class Hypothesis:
    """Hypothesis associating a prediction with a measurement."""

    def __init__(
        self,
        prediction: State,
        measurement: Detection,
        measurement_prediction: State | None = None,
    ) -> None:
        """Initialise the Hypothesis.

        Args:
            prediction (State): The predicted state of the target.
            measurement (Detection): The measurement to associate.
            measurement_prediction (State | None, optional): The predicted
                measurement in the sensor's frame. Defaults to None.

        """
        self.prediction = prediction
        self.measurement = measurement
        self.measurement_prediction = measurement_prediction

    def __bool__(self) -> bool:
        """Return `True` if the hypothesis represents a valid measurement.

        A hypothesis is considered "true" if its associated measurement is
        not `None` and is a valid detection (i.e., not a `MissedDetection` or
        `Clutter`).
        """
        return (not isinstance(self.measurement, MissedDetection)) and (
            self.measurement is not None
        )


class ProbabilityHypothesis(Hypothesis):
    """A hypothesis that includes a probability or weight.

    This class extends the standard `Hypothesis` to include a probability,
    which is essential for probabilistic data association algorithms like JPDA.

    Attributes:
        probability (float): The probability of this hypothesis being correct.

    """

    def __init__(
        self,
        prediction: State,
        measurement: Detection,
        probability: float,
        measurement_prediction: State | None = None,
    ) -> None:
        """Initialise the ProbabilityHypothesis.

        Args:
            prediction (State): The predicted state of the target.
            measurement (Detection): The measurement to associate.
            probability (float): The probability of this hypothesis.
            measurement_prediction (State | None, optional): The predicted
                measurement in the sensor's frame. Defaults to None.

        """
        super().__init__(prediction, measurement, measurement_prediction)
        self.probability = probability
