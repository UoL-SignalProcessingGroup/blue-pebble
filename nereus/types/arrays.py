"""Defines data structures.

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

from collections.abc import Sequence

import numpy as np


class Matrix(np.ndarray):
    """General Matrix wrapper for numpy arrays."""

    def __new__(cls, *args, **kwargs):
        """Construct the Matrix object from an array-like input."""
        array = np.asarray(*args, **kwargs)
        return array.view(cls)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Ensure NumPy ufuncs return an instance of the correct class."""
        plain_inputs = [
            np.asarray(input_) if isinstance(input_, Matrix) else input_
            for input_ in inputs
        ]
        result = super().__array_ufunc__(ufunc, method, *plain_inputs, **kwargs)

        if result is NotImplemented:
            return NotImplemented

        # Check the shape of the result to determine the return type
        if isinstance(result, np.ndarray) and result.ndim == 2:
            if result.shape[1] == 1:
                # If result is a single column, return a StateVector
                return StateVector(result)
            else:
                # If result has multiple columns, return a StateVectors
                return StateVectors(result)

        # For scalars or 1D arrays, we cannot reliably wrap them,
        # so return the raw result.
        return result


class StateVector(Matrix):
    """State vector wrapper ensuring a column vector shape of (N, 1)."""

    def __new__(cls, input_array):
        """Construct a StateVector, enforcing a column-vector shape.

        Args:
            input_array (array-like): The data to be wrapped.

        Returns:
            StateVector: The new StateVector object.

        Raises:
            ValueError: If the input cannot be shaped into a column vector.

        """
        # Ensure input_array is converted to an array
        array = np.asarray(input_array)
        if array.ndim == 1:
            # If it's a 1D array, reshape it to a column vector
            array = array[:, None]
        # elif array.ndim != 2 or array.shape[1] != 1:
        #     raise ValueError("StateVector data must be a column vector (shape N, 1).")
        return array.view(cls)

    def mean(self, axis=None, **kwargs):
        """Compute the mean of the vector.

        Args:
            axis (int, optional): The axis to compute the mean along.
                Defaults to None.
            **kwargs: Other keyword arguments for np.mean.

        Returns:
            StateVector: A new StateVector containing the result.

        """
        result = np.mean(self, axis=axis, **kwargs)
        # Return a new StateVector for consistency
        return StateVector(result)

    def __getitem__(self, item):
        """Get an element from the vector with a convenience for integer indexing."""
        # Direct access for scalars
        if isinstance(item, int):
            item = (item, 0)
        return super().__getitem__(item)

    def flatten(self, *args, **kwargs):
        """Return a copy of the array collapsed into one dimension."""
        return np.ndarray.flatten(self, *args, **kwargs)

    def ravel(self, *args, **kwargs):
        """Return a contiguous flattened array."""
        return np.ndarray.ravel(self, *args, **kwargs)


class CovarianceMatrix(Matrix):
    """Covariance matrix wrapper ensuring a square shape of (N, N)."""

    def __new__(cls, *args, **kwargs):
        """Construct a CovarianceMatrix, ensuring it is square.

        Args:
            *args: Arguments passed to np.asarray.
            **kwargs: Keyword arguments passed to np.asarray.

        Returns:
            CovarianceMatrix: The new CovarianceMatrix object.

        Raises:
            ValueError: If the resulting matrix is not square.

        """
        array = np.asarray(*args, **kwargs)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("Covariance matrix must be square (NxN)")
        return array.view(cls)


class StateVectors(Matrix):
    """A matrix representing a collection of state vectors."""

    def __new__(cls, states, *args, **kwargs):
        """Construct a StateVectors object from various inputs.

        This can be initialised from a sequence of StateVector objects or any
        2D array-like data.

        Args:
            states (array-like or Sequence[StateVector]): The input data.
            *args: Additional arguments for np.asarray.
            **kwargs: Additional keyword arguments for np.asarray.

        Returns:
            StateVectors: The new StateVectors object.

        """
        if isinstance(states, Sequence) and not isinstance(states, np.ndarray):
            if states and isinstance(states[0], StateVector):
                return np.hstack(states).view(cls)
        array = np.asarray(states, *args, **kwargs)
        return array.view(cls)

    def __iter__(self):
        """Iterate through the matrix columns, yielding each as a StateVector.

        Yields:
            StateVector: The next column vector in the collection.

        """
        # Ensure each column is treated as a 2D column vector
        for col in np.array(self).T:
            if col.ndim == 1:  # Ensure it has at least 2 dimensions
                col = col[:, None]  # Make it a column vector
            yield StateVector(col)

    def mean(self, axis=None, **kwargs):
        """Compute the mean across all vectors in the collection.

        Args:
            axis (int, optional): The axis to compute the mean along.
                Defaults to None.
            **kwargs: Other keyword arguments for np.mean.

        Returns:
            StateVector: A StateVector representing the mean.

        """
        return StateVector(np.mean(np.asarray(self), axis=axis, **kwargs))

    def average(self, axis=None, weights=None):
        """Compute the weighted average of the vectors.

        Args:
            axis (int, optional): The axis to compute the average along.
                Defaults to None.
            weights (array-like, optional): Weights for averaging. If None,
                equal weights are used.

        Returns:
            StateVector: A StateVector representing the weighted average.

        """
        return StateVector(np.average(self, axis=axis, weights=weights))

    def cov(self, bias=False):
        """Compute the covariance of the state vectors.

        Args:
            bias (bool, optional): Determines if the result is normalized by
                (N-1) or N. Defaults to False.

        Returns:
            CovarianceMatrix: The resulting covariance matrix.

        """
        return CovarianceMatrix(np.cov(self, rowvar=True, bias=bias))
