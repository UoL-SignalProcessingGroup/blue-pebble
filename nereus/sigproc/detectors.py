"""Defines signal detectors for processing time series data.

This module provides classes for detecting signals in one-dimensional data arrays,
such as time series or beamformed output. It includes a simple threshold-based
detector, a peak detector that uses Scipy's `find_peaks`, and a constant false alarm
rate (CFAR) detector.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from scipy.signal import find_peaks


class Detector(ABC):
    """Abstract base class for all detector types."""

    @abstractmethod
    def detect(self: Self, data: np.ndarray) -> np.ndarray:
        """Detect signals in the data array.

        Args:
            data (np.ndarray): A 1D NumPy array of numerical data to process.

        Returns:
            np.ndarray: A 2D NumPy array where each row contains two elements:
            the index of a detection and its corresponding value. Returns an
            empty array if no detections are found.

        """
        raise NotImplementedError


class ThresholdDetector(Detector):
    """Detects data points that exceed a predefined scalar threshold.

    This detector performs a simple comparison, identifying all indices in an
    array where the value is greater than the specified threshold.

    Attributes:
        threshold (float): The value that data points must exceed to be
            considered a detection.

    """

    def __init__(self: Self, threshold: float) -> None:
        """Initialise the ThresholdDetector.

        Args:
            threshold (float): The value that data points must exceed to be
                considered a detection.

        """
        self.threshold = threshold

    def detect(self: Self, data: np.ndarray) -> np.ndarray:
        """Detect values in the data array that are above the threshold."""
        indices = np.where(data > self.threshold)[0]
        if indices.size == 0:
            return np.array([])
        return np.column_stack((indices, data[indices]))


class PeakDetector(Detector):
    """Finds local maxima (peaks) in a 1D data array.

    This class is a wrapper around the `scipy.signal.find_peaks` function,
    providing a simple interface for peak detection. It identifies peaks that
    meet a minimum threshold and are separated by a specified minimum distance.

    Attributes:
        threshold (float | None): The minimum required value for a data point
            to be considered a peak.
        distance (int): The minimum required horizontal distance (in number of
            samples) between neighbouring peaks.

    """

    def __init__(self: Self, distance: int = 1) -> None:
        """Initialise the PeakDetector.

        Args:
            distance (int): The minimum required horizontal distance (in number
                of samples) between neighbouring peaks.
            distance (int): The minimum required horizontal distance (in number
                of samples) between neighbouring peaks. Defaults to 1.

        """
        self.distance = distance

    def detect(self: Self, data: np.ndarray) -> np.ndarray:
        """Find all peaks in the data array."""
        indices, _ = find_peaks(data, distance=self.distance)
        if indices.size == 0:
            return np.array([])
        return np.column_stack((indices, data[indices]))


class CFARDetector(Detector):
    """Detects signals using a Constant False Alarm Rate (CFAR) algorithm.

    This detector adapts its threshold by estimating the noise level from
    surrounding data cells. For each Cell Under Test (CUT), it calculates the
    mean of the training cells and multiplies it by a threshold factor to set
    the detection threshold.

    This implementation is a Cell-Averaging CFAR (CA-CFAR).

    Attributes:
        num_guard_cells (int): The number of cells to ignore on each side of
            the Cell Under Test (CUT). These cells are ignored to prevent
            signal leakage from the CUT into the noise estimate.
        num_training_cells (int): The number of cells to use for noise
            estimation on each side of the guard cells.
        threshold_factor (float): A scaling factor (alpha) used to set the
            detection threshold above the estimated noise floor.

    """

    def __init__(
        self: Self,
        num_guard_cells: int,
        num_training_cells: int,
        threshold_factor: float,
        mode: str = "valid",
    ) -> None:
        """Initialise the CFARDetector.

        Args:
            num_guard_cells (int): The number of guard cells on each side of
                the CUT.
            num_training_cells (int): The number of training cells on each side
                of the CUT.
            threshold_factor (float): The scaling factor to apply to the noise
                estimate.
            mode (str): The convolution mode to use. Defaults to "valid". Other
                options include "same" and "wrap". "valid" means the output is
                only where the kernel fully overlaps with the input data, while
                "same" returns an output of the same size as the input, and
                "wrap" handles edge cases by wrapping around the data.

        """
        self.num_guard_cells = num_guard_cells
        self.num_training_cells = num_training_cells
        self.threshold_factor = threshold_factor
        self.mode = mode

    def detect(self: Self, data: np.ndarray) -> np.ndarray:
        """Detect signals in the data array using the CFAR algorithm.

        Note: This method assumes the input data is in decibels (dB) and
        converts it to linear power for processing, as CFAR averaging is
        performed on power, not dB values.

        Args:
            data (np.ndarray): A 1D NumPy array of signal data (e.g., SNR) in
                decibels.

        Returns:
            np.ndarray: A 2D NumPy array where each row contains two elements:
            the index of a detection and its corresponding value in dB.
            Returns an empty array if no detections are found.

        """
        # Convert dB data to linear power for processing
        power = 10 ** (data / 10)

        # Total number of cells on one side of the CUT
        one_sided_window = self.num_guard_cells + self.num_training_cells

        # Use convolution to efficiently calculate the sum in the training window
        # The kernel represents the training cells (1s) and guard/CUT cells (0s)
        kernel_size = 2 * one_sided_window + 1
        kernel = np.ones(kernel_size)
        start_gap = self.num_training_cells
        end_gap = start_gap + (2 * self.num_guard_cells) + 1
        kernel[start_gap:end_gap] = 0

        # Handle boundary conditions based on the specified mode
        if self.mode == "wrap":
            # Pad the power array by wrapping the ends for circular convolution
            padded_power = np.pad(power, pad_width=one_sided_window, mode="wrap")
            noise_sum = np.convolve(padded_power, kernel, mode="valid")
            num_training_cells_total = 2 * self.num_training_cells
            noise_estimate = noise_sum / num_training_cells_total
        else:  # Default 'valid' mode
            # Calculate the sum of training cells for each CUT
            noise_sum = np.convolve(power, kernel, mode="same")

            # Calculate the adaptive threshold for each cell
            # For edge cases, the number of training cells is smaller
            edge_kernel = np.ones_like(power)
            effective_num_training = np.convolve(edge_kernel, kernel, mode="same")
            with np.errstate(divide="ignore", invalid="ignore"):
                noise_estimate = noise_sum / effective_num_training

        # The threshold is the noise estimate scaled by the threshold factor
        threshold = self.threshold_factor * noise_estimate

        # Find indices where the signal power exceeds the adaptive threshold
        indices = np.where(power > threshold)[0]

        if indices.size == 0:
            return np.array([])

        return np.column_stack((indices, data[indices]))
