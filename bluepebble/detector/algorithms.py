"""Signal detection algorithms for 1D time-series and beamformed data."""

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks
from stonesoup.base import Base, Property

IntArray: TypeAlias = NDArray[np.integer[Any]]
DetectionArray: TypeAlias = NDArray[np.float64]


def _empty_detections() -> DetectionArray:
    """Return a standard empty detection matrix of shape ``(0, 2)``."""
    return np.empty((0, 2), dtype=np.float64)


def _stack_detections(indices: IntArray, data: ArrayLike) -> DetectionArray:
    """Create a ``(N, 2)`` matrix with detection indices and values."""
    data_array = np.asarray(data, dtype=np.float64)
    if indices.size == 0:
        return _empty_detections()
    return np.column_stack((indices, data_array[indices])).astype(np.float64, copy=False)


class DetectionAlgorithm(Base, ABC):
    """Abstract base class for all detector types."""

    @abstractmethod
    def detect(self, data: ArrayLike) -> DetectionArray:
        """Detect signals in the data array.

        Parameters
        ----------
        data : ArrayLike
            One-dimensional numeric data to process.

        Returns
        -------
        DetectionArray
            Detection matrix of shape ``(N, 2)`` with columns
            ``[detection_index, detection_value]``. Returns an empty
            ``(0, 2)`` matrix when no detections are found.

        """
        ...


class ThresholdDetector(DetectionAlgorithm):
    """Detects data points that exceed a predefined scalar threshold.

    This detector performs a simple comparison, identifying all indices in an array where the value
    is greater than the specified threshold.

    Attributes
    ----------
    threshold : float
        The value that data points must exceed to be considered a detection.

    """

    threshold: float = Property(
        float, doc="The value that data points must exceed to be considered a detection"
    )

    def detect(self, data: ArrayLike) -> DetectionArray:
        """Detect values in the data array that are above the threshold."""
        data_array = np.asarray(data)
        indices = np.where(data_array > self.threshold)[0]
        return _stack_detections(indices, data_array)


class PeakDetector(DetectionAlgorithm):
    """Finds local maxima (peaks) in a 1D data array.

    This class is a wrapper around the `scipy.signal.find_peaks` function, providing a simple
    interface for peak detection. It identifies peaks that are separated by a specified minimum
    distance.

    Attributes
    ----------
    distance : int
        The minimum required horizontal distance (in number of samples) between neighbouring peaks.

    """

    distance: int = Property(
        default=1,
        doc="The minimum required horizontal distance (in number of samples) between neighbouring "
        "peaks",
    )

    def detect(self, data: ArrayLike) -> DetectionArray:
        """Find all peaks in the data array."""
        data_array = np.asarray(data)
        indices, _ = find_peaks(data_array, distance=self.distance)
        return _stack_detections(indices, data_array)


class CACFARDetector(DetectionAlgorithm):
    """Detects signals using a Constant False Alarm Rate (CFAR) algorithm.

    This detector adapts its threshold by estimating the noise level from surrounding data cells.
    For each Cell Under Test (CUT), it calculates the mean of the training cells and multiplies it
    by a threshold factor to set the detection threshold.

    This implementation is a Cell-Averaging CFAR (CA-CFAR).

    Attributes
    ----------
    num_guard_cells : int
        The number of cells to ignore on each side of the Cell Under Test (CUT). These cells are
        ignored to prevent signal leakage from the CUT into the noise estimate.
    num_training_cells : int
        The number of cells to use for noise estimation on each side of the guard cells.
    threshold_factor : float
        A scaling factor (alpha) used to set the detection threshold above the estimated noise
        floor.
    mode : str
        The convolution mode for boundary handling. Can be 'valid', 'same', or 'wrap'. Defaults to
        'wrap'.

    """

    num_guard_cells: int = Property(
        int,
        doc="The number of cells to ignore on each side of the Cell Under Test (CUT). "
        "These cells are ignored to prevent signal leakage from the CUT into the noise estimate",
    )
    num_training_cells: int = Property(
        int,
        doc="The number of cells to use for noise estimation on each side of the guard cells",
    )
    threshold_factor: float = Property(
        float,
        doc="A scaling factor (alpha) used to set the detection threshold above the estimated"
        "noise floor",
    )
    mode: str = Property(
        str,
        default="wrap",
        doc="The convolution mode for boundary handling ('valid', 'same', or 'wrap')",
    )

    def detect(self, data: ArrayLike) -> DetectionArray:
        """Detect signals in the data array using the CFAR algorithm.

        This method applies the Cell-Averaging CFAR (CA-CFAR) algorithm. It assumes the input data
        is in decibels (dB) and converts it to linear power for processing, as the averaging is
        performed on power values.

        Parameters
        ----------
        data : ArrayLike
            One-dimensional signal data (for example SNR) in decibels.

        Returns
        -------
        DetectionArray
            Detection matrix of shape ``(N, 2)`` with columns
            ``[detection_index, detection_value_db]``. Returns an empty
            ``(0, 2)`` matrix when no detections are found.

        """
        # Convert dB to linear power, as CFAR averaging is done on power.
        data_array = np.asarray(data)
        power = 10 ** (data_array / 10)

        # Total number of cells on one side of the CUT
        one_sided_window = self.num_guard_cells + self.num_training_cells

        # Create a convolution kernel to sum the training cells.
        # The kernel has 1s for training cells and 0s for guard/CUT cells.
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
        else:  # Non-circular modes use zero-padded edge handling
            padded_power = np.pad(power, pad_width=one_sided_window, mode="constant")
            noise_sum = np.convolve(padded_power, kernel, mode="valid")
            # For edge cases, the number of training cells is smaller
            edge_kernel = np.pad(
                np.ones_like(power),
                pad_width=one_sided_window,
                mode="constant",
            )
            effective_num_training = np.convolve(edge_kernel, kernel, mode="valid")
            with np.errstate(divide="ignore", invalid="ignore"):
                noise_estimate = noise_sum / effective_num_training

        # The adaptive threshold is the noise estimate scaled by the factor.
        threshold = self.threshold_factor * noise_estimate

        # Find indices where the signal power exceeds the adaptive threshold
        indices = np.where(power > threshold)[0]
        return _stack_detections(indices, data_array)


class OSCFARDetector(DetectionAlgorithm):
    """Detects signals using an Ordered-Statistic (OS) CFAR algorithm.

    This detector is more robust to multi-target situations than CA-CFAR.
    It estimates the noise level by sorting the values in the training
    cells and selecting the k-th smallest value (the 'rank'). This
    value is then scaled by the threshold factor to set the detection
    threshold.

    This implementation uses a 'wrap' mode for boundary handling, consistent
    with the 'wrap' mode in the CA-CFAR detector.

    Attributes:
        num_guard_cells (int): The number of cells to ignore on each side of
            the Cell Under Test (CUT).
        num_training_cells (int): The number of cells to use for noise
            estimation on each side of the guard cells.
        rank (int): The k-th smallest value (1-indexed) to select from the
            sorted training cells. Must be between 1 and
            (2 * num_training_cells).
        threshold_factor (float): A scaling factor (alpha) used to set the
            detection threshold above the estimated noise floor.

    """

    num_guard_cells: int = Property(
        int,
        doc="The number of cells to ignore on each side of the Cell Under Test (CUT).",
    )
    num_training_cells: int = Property(
        int,
        doc="The number of cells to use for noise estimation on each side of the guard cells.",
    )
    rank: int = Property(
        int,
        default=1,
        doc="The k-th smallest value (1-indexed) to select from the sorted "
        "training cells. Must be between 1 and (2 * num_training_cells).",
    )
    threshold_factor: float = Property(
        float,
        default=1.0,
        doc="A scaling factor (alpha) to apply to the k-th rank value.",
    )

    def __init__(self, *args: object, **kwargs: object):
        """Initialise the OS-CFAR detector and validate parameters."""
        super().__init__(*args, **kwargs)

        # Validate the rank parameter
        self.num_training_total: int = 2 * self.num_training_cells
        if not 1 <= self.rank <= self.num_training_total:
            raise ValueError(
                f"Rank ({self.rank}) must be between 1 and "
                f"2 * num_training_cells ({self.num_training_total})"
            )

    def detect(self, data: ArrayLike) -> DetectionArray:
        """Detect signals in the data array using the OS-CFAR algorithm.

        This method applies the OS-CFAR algorithm. It assumes the
        input data is in decibels (dB) and converts it to linear
        power for processing, as the sorting is performed on power values.

        Parameters
        ----------
        data : ArrayLike
            One-dimensional signal data (for example SNR) in decibels.

        Returns
        -------
        DetectionArray
            Detection matrix of shape ``(N, 2)`` with columns
            ``[detection_index, detection_value_db]``. Returns an empty
            ``(0, 2)`` matrix when no detections are found.

        """
        # Convert dB to linear power, as CFAR processing is done on power.
        data_array = np.asarray(data)
        power = 10 ** (data_array / 10)

        # Total number of cells on one side of the CUT
        one_sided_window = self.num_guard_cells + self.num_training_cells
        window_size = 2 * one_sided_window + 1

        # Pad the power array by wrapping the ends for circular processing
        # This creates a 'wrap' mode, consistent with the CA-CFAR
        padded_power = np.pad(power, pad_width=one_sided_window, mode="wrap")

        # Create a sliding window view over the padded data.
        # This creates a 2D array where each row is a window.
        # Shape will be (len(data), window_size)
        windows = sliding_window_view(padded_power, window_size)

        # Extract the leading and lagging training cells from all windows
        # at once (vectorized).
        leading_cells = windows[:, : self.num_training_cells]
        lagging_cells = windows[:, -self.num_training_cells :]

        # Concatenate into a single array of training cells for each CUT
        # Shape: (len(data), 2 * num_training_cells)
        training_cells = np.concatenate((leading_cells, lagging_cells), axis=1)

        # Sort the training cells for each row
        training_cells.sort(axis=1)

        # Select the k-th rank value as the noise estimate.
        # We use `self.rank - 1` for 0-based indexing.
        # Cast to int to guard against float injection from parameter sweeps.
        noise_estimate = training_cells[:, int(self.rank) - 1]

        # The adaptive threshold is the noise estimate scaled by the factor.
        threshold = self.threshold_factor * noise_estimate

        # Find indices where the original signal power exceeds the threshold
        indices = np.where(power > threshold)[0]
        return _stack_detections(indices, data_array)
