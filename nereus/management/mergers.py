"""Provide implementations of track merging strategies.

© Copyright 2025 Joshua J. Wakefield
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from nereus.filters.resamplers import Resampler
from nereus.types.states import Particle, ParticleState, Track


class Merger(ABC):
    """Abstract base class for track mergers."""

    @abstractmethod
    def merge(self, tracks: set[Track]) -> set[Track]:
        """Merge a set of tracks based on a defined strategy.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError


class DistanceBasedMerger(Merger):
    """A merger that combines tracks that are closer than a given threshold."""

    def __init__(
        self,
        metric: Callable[[ParticleState, ParticleState], float],
        resampler: Resampler,
        threshold: float = 1.0,
    ):
        """Initialise the distance-based merger.

        Args:
            metric (Callable[[ParticleState, ParticleState], float]): A function
                that takes two particle states and returns a scalar distance.
            resampler (Resampler): The resampler used to regularise the number
                of particles after a merge.
            threshold (float, optional): The distance threshold below which
                tracks will be merged. Defaults to 1.0.

        """
        self.metric = metric
        self.resampler = resampler
        self.threshold = threshold

    def merge(self, tracks: set[Track]) -> set[Track]:
        """Merge tracks using a greedy clustering approach.

        This method iterates through tracks, and for each one, finds all other
        tracks within the distance threshold. It merges this entire cluster
        into a single track and continues until no tracks are left to process.

        Args:
            tracks (set[Track]): A set of tracks to be merged.

        Returns:
            set[Track]: The set of merged tracks.

        """
        unmerged_tracks = set(tracks)
        merged_tracks = set()

        while unmerged_tracks:
            # Pop a track to serve as the base for a potential cluster
            base_track = unmerged_tracks.pop()

            # Find all other tracks within the distance threshold of the base track
            cluster = {
                other_track
                for other_track in unmerged_tracks
                if self.metric(base_track[-1], other_track[-1]) < self.threshold
            }

            # If no other tracks are close, the base track is final
            if not cluster:
                merged_tracks.add(base_track)
                continue

            # Remove the clustered tracks from the pool of unmerged tracks
            unmerged_tracks -= cluster
            cluster.add(base_track)  # Add the base track to the cluster for merging

            # Combine all particles from the tracks in the cluster
            all_particles = [
                particle for track in cluster for particle in track[-1].particles
            ]

            # Use NumPy for efficient weight normalization
            state_vectors = np.array([p.state_vector.flatten() for p in all_particles])
            weights = np.array([p.weight for p in all_particles])
            weights /= np.sum(weights)

            # Create new particles for the merged state
            merged_particles = [
                Particle(state, weight) for state, weight in zip(state_vectors, weights)
            ]
            timestamp = base_track[-1].timestamp
            num_particles = base_track[-1].num_particles

            merged_state = ParticleState(merged_particles, timestamp=timestamp)

            # Resample to maintain a consistent number of particles
            resampled_state = self.resampler.resample(merged_state, num_particles)

            # Replace the last state of the base track with the new merged state
            base_track[-1] = resampled_state
            merged_tracks.add(base_track)

        return merged_tracks
