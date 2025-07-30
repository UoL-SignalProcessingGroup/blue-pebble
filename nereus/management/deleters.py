"""Provide implementations for track deletion.

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

import numpy as np

from nereus.types.states import Track


class Deleter(ABC):
    """Abstract base class for track deleters."""

    @abstractmethod
    def delete_tracks(self, tracks: set[Track]) -> set[Track]:
        """Identify tracks for deletion from a given set.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError


class CovarianceBasedDeleter(Deleter):
    """Deletes tracks if their state uncertainty grows too large.

    This deleter checks if the trace of a track's state covariance matrix
    exceeds a defined threshold.
    """

    def __init__(self, covar_trace_thresh: float):
        """Initialise the deleter.

        Args:
            covar_trace_thresh (float): The threshold for the covariance trace
                above which tracks are deleted.

        """
        self.covar_trace_thresh = covar_trace_thresh

    def delete_tracks(self, tracks: set[Track]) -> set[Track]:
        """Identify tracks to delete based on their covariance trace.

        Args:
            tracks (set[Track]): A set of tracks to evaluate for deletion.

        Returns:
            set[Track]: A subset of the input tracks marked for deletion.


        """
        tracks_to_delete = set()

        for track in tracks:
            # The trace of the covariance matrix is a measure of total uncertainty
            covar_trace = np.trace(track[-1].covar)

            # Mark track for deletion if the uncertainty exceeds the threshold
            if covar_trace > self.covar_trace_thresh:
                tracks_to_delete.add(track)

        return tracks_to_delete
