"""Hydrophone models for converting acoustic pressure to voltage."""

from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]


class FrequencyResponse(ABC, Base):
    """Abstract base class for hydrophone frequency response models.

    A frequency response model describes how a hydrophone's sensitivity
    varies with frequency. Subclasses implement :meth:`evaluate` to return
    a complex transfer function at a given set of frequencies.
    """

    @abstractmethod
    def evaluate(self, frequencies_hz: ArrayLike) -> ComplexArray:
        """Evaluate the frequency response at the given frequencies.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the response.

        Returns
        -------
        ComplexArray
            Complex frequency response values, shape ``(num_frequencies,)``.

        """
        ...


class FlatFrequencyResponse(FrequencyResponse):
    """Flat (unity) frequency response model.

    Models an ideal hydrophone with no frequency-dependent variation in
    response. This is the default when no measured response is available.
    """

    def evaluate(self, frequencies_hz: ArrayLike) -> ComplexArray:
        """Return unity response at all frequencies.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the response.

        Returns
        -------
        ComplexArray
            Array of ones with shape ``(num_frequencies,)``.

        """
        frequencies = np.asarray(frequencies_hz, dtype=float)
        return np.ones(frequencies.shape, dtype=np.complex128)


class TabulatedFrequencyResponse(FrequencyResponse):
    """Frequency response model interpolated from tabulated measurements.

    Attributes
    ----------
    frequencies_hz : ArrayLike
        Frequencies in Hz at which the response is measured.
    response : ArrayLike
        Complex response values at each frequency in ``frequencies_hz``.

    """

    frequencies_hz: ArrayLike = Property(
        doc="Frequencies in Hz at which the response is measured."
    )
    response: ArrayLike = Property(
        doc="Complex response values at each tabulated frequency."
    )

    def evaluate(self, frequencies_hz: ArrayLike) -> ComplexArray:
        """Interpolate the response at the given frequencies.

        Real and imaginary parts are interpolated independently. Frequencies
        outside the tabulated range are extrapolated using the nearest endpoint
        value.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the response.

        Returns
        -------
        ComplexArray
            Interpolated complex response, shape ``(num_frequencies,)``.

        """
        f_query = np.asarray(frequencies_hz, dtype=float)
        f_table = np.asarray(self.frequencies_hz, dtype=float)
        r_table = np.asarray(self.response, dtype=np.complex128)

        real = np.interp(f_query, f_table, r_table.real)
        imag = np.interp(f_query, f_table, r_table.imag)
        return (real + 1j * imag).astype(np.complex128)


class HydrophoneModel(Base):
    """Model of a single hydrophone element.

    Combines a scalar sensitivity with a frequency response to produce
    a complex transfer function that maps received acoustic pressure
    (in μPa) to voltage (in V).

    Attributes
    ----------
    sensitivity_db : float
        Hydrophone sensitivity in dB re 1 V/μPa.
    frequency_response : FrequencyResponse
        Frequency-dependent response model. Defaults to
        :class:`FlatFrequencyResponse`.

    """

    sensitivity_db: float = Property(doc="Hydrophone sensitivity in dB re 1 V/μPa.")
    frequency_response: FrequencyResponse = Property(
        default=None,
        doc="Frequency-dependent response model. Defaults to FlatFrequencyResponse.",
    )

    def __post_init__(self, *args, **kwargs) -> None:
        """Initialise default frequency response if not provided."""
        super().__init__(*args, **kwargs)
        if self.frequency_response is None:
            self.frequency_response = FlatFrequencyResponse()

    def transfer_function(self, frequencies_hz: ArrayLike) -> ComplexArray:
        r"""Compute the combined complex transfer function at the given frequencies.

        The transfer function combines the scalar sensitivity with the
        frequency response:

        .. math::

            H(f) = S \cdot R(f)

        where :math:`S` is the linear sensitivity (V/μPa) and :math:`R(f)`
        is the complex frequency response from :attr:`frequency_response`.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the transfer function.

        Returns
        -------
        ComplexArray
            Complex transfer function, shape ``(num_frequencies,)``.

        """
        sensitivity_linear = 10 ** (self.sensitivity_db / 20.0)
        response = self.frequency_response.evaluate(frequencies_hz)
        return (sensitivity_linear * response).astype(np.complex128)
