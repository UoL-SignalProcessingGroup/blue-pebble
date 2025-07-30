"""Provide implementations of resampling methods for particle filters.

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

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from nereus.types.states import ParticleState


class Resampler(ABC):
    """Abstract base class for resamplers."""

    @abstractmethod
    def resample(self, particle_state: ParticleState) -> ParticleState:
        """Resample the particles in a particle state.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError


class SystematicResampler(Resampler):
    """A systematic resampler for particle filters."""

    def __init__(
        self, regularisation: Callable[[np.ndarray], np.ndarray] | None = None
    ):
        """Initialise the SystematicResampler.

        Args:
            regularisation (Callable[[np.ndarray], np.ndarray], optional): A
                function to apply to the state vectors after resampling to
                mitigate sample impoverishment. Defaults to None.

        """
        self.regularisation = regularisation

    def resample(self, particle_state: ParticleState) -> ParticleState:
        """Resample the particles if the effective sample size is too low.

        This method uses a systematic resampling algorithm. Resampling is only
        performed if the effective sample size (ESS) drops below 50% of the
        total number of particles.

        Args:
            particle_state (ParticleState): The particle state to be resampled.

        Returns:
            ParticleState: The resampled particle state.

        """
        num_particles = particle_state.num_particles
        weights = particle_state.weights

        # Calculate the effective sample size (ESS)
        ess = 1 / np.sum(weights**2)

        # Only resample if ESS is below a 50% threshold to avoid impoverishment
        if ess >= (num_particles / 2):
            return particle_state

        # Normalise weights to form a probability distribution
        weights /= np.sum(weights)

        # Calculate the cumulative distribution function (CDF)
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0  # Ensure the CDF ends at exactly 1.0

        # Generate systematically spaced points with a random offset
        positions = (np.arange(num_particles) + np.random.uniform()) / num_particles

        # Find the indices of the particles to be resampled
        index = np.searchsorted(cdf, positions)
        state_vectors = particle_state.state_vector[:, index]

        # Apply regularisation if provided
        if self.regularisation is not None:
            state_vectors = self.regularisation(state_vectors)

        # Create new weights, which are uniform after resampling
        new_weights = np.full(num_particles, 1.0 / num_particles)

        # Create and return the new particle state
        return ParticleState(
            state_vector=state_vectors,
            weights=new_weights,
            timestamp=particle_state.timestamp,
        )
