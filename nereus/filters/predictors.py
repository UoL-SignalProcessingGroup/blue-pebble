"""Provide predictor implementations for state estimation.

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
© Copyright 2025 Joshua J. Wakefield
"""  # noqa: E501

from datetime import timedelta

from nereus.models.transition import TransitionModel
from nereus.types.states import GaussianState, ParticleState


class Predictor:
    """A base class for predictors.

    Predictors are used to project a system's state forward in time according
    to a given transition model.
    """


class KalmanPredictor(Predictor):
    r"""A predictor based on the Kalman filter formulation.

    This predictor uses a linear Gaussian transition model to predict the next
    :class:`~.GaussianState` of a target. The prediction is performed by
    applying the transition model to the state mean and calculating the new
    state covariance according to the following equations:

    .. math::
        \mathbf{x}_{k|k-1} = F_{k-1} \mathbf{x}_{k-1|k-1}

    .. math::
        P_{k|k-1} = F_{k-1} P_{k-1|k-1} F_{k-1}^T + Q_{k-1}

    Where $F$ is the state transition matrix and $Q$ is the process noise covariance.
    """

    transition_model: TransitionModel

    def __init__(self, transition_model: TransitionModel):
        """Initialise the Kalman predictor.

        Args:
            transition_model: The linear transition model to be used for
                prediction. It must define the matrices F and Q.

        """
        self.transition_model = transition_model

    def predict(self, state: GaussianState, time_interval: timedelta) -> GaussianState:
        """Predicts the next state.

        Args:
            state (GaussianState): The current state.
            time_interval (timedelta): The time interval over which to predict.

        Returns:
            GaussianState: The predicted state.

        """
        # Predict the state mean using the transition model function
        pred_mean = self.transition_model.function(state, noise=False)

        # Predict the state covariance using the Kalman prediction equation
        F = self.transition_model.F
        Q = self.transition_model.Q
        pred_covar = F @ state.covar @ F.T + Q

        return GaussianState(
            mean=pred_mean,
            covar=pred_covar,
            timestamp=state.timestamp + time_interval,
        )


class ParticlePredictor(Predictor):
    """A predictor that uses a particle filter approach.

    This predictor applies a :class:`~.TransitionModel` to a
    :class:`~.ParticleState` to propagate its particles forward in time.
    It is suitable for both linear and non-linear transition models.
    """

    transition_model: TransitionModel

    def __init__(self, transition_model: TransitionModel):
        """Initialise the particle predictor.

        Args:
            transition_model: The transition model used to propagate particles.

        """
        self.transition_model = transition_model

    def predict(self, state: ParticleState, time_interval: timedelta) -> ParticleState:
        """Predicts the next particle state.

        Args:
            state (ParticleState): The current particle state.
            time_interval (timedelta): The time interval for the prediction.

        Returns:
            ParticleState: The predicted particle state with updated particle
            positions and timestamp. The particle weights are unchanged.

        """
        # Propagate each particle's state vector using the transition model
        new_state_vector = self.transition_model.function(state, noise=True)

        # Create a new ParticleState with the new particles and the original weights
        return ParticleState(
            state_vector=new_state_vector,
            weights=state.weights,
            timestamp=state.timestamp + time_interval,
        )
