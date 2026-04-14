r"""Hydrophone models for converting acoustic pressure to voltage.

A hydrophone is modelled as a linear time-invariant (LTI) system whose
transfer function maps received acoustic pressure (in uPa) to output
voltage (in V).  The combined transfer function is:

.. math::

    H_{\text{hydrophone}}(f) = S \cdot R(f)

where *S* is the scalar sensitivity (V/uPa, converted from dB re 1 V/uPa)
and *R(f)* is a frequency-dependent complex response.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property

ComplexArray: TypeAlias = NDArray[np.complexfloating[Any, Any]]
FloatArray: TypeAlias = NDArray[np.floating[Any]]


class FrequencyResponse(ABC, Base):
    """Abstract base class for hydrophone frequency response models.

    A frequency response model describes how a hydrophone's sensitivity
    varies with frequency.  Subclasses implement :meth:`evaluate` to return
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
    response.  This is the default when no measured response is available.
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
    """Frequency response interpolated from tabulated measurements.

    Accepts magnitude in dB and optional phase in degrees, matching the
    format commonly found on hydrophone datasheets.  Interpolation is
    performed independently in the dB and degree domains before conversion
    to a complex transfer function.

    Frequencies outside the tabulated range are extrapolated using the
    nearest endpoint value (``np.interp`` default behaviour).

    For query frequencies that are negative (e.g. from ``np.fft.fftfreq``),
    the absolute value is used under the assumption that the hydrophone
    response is symmetric about zero frequency.
    """

    frequencies_hz: NDArray = Property(
        doc="Tabulated frequencies in Hz (non-negative, monotonically increasing).",
    )
    magnitude_db: NDArray = Property(
        doc="Magnitude in dB at each tabulated frequency.",
    )
    phase_deg: NDArray | None = Property(
        default=None,
        doc="Phase in degrees at each tabulated frequency.  ``None`` implies zero phase.",
    )

    def evaluate(self, frequencies_hz: ArrayLike) -> ComplexArray:
        """Interpolate the response at the given frequencies.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the response.

        Returns
        -------
        ComplexArray
            Interpolated complex response, shape ``(num_frequencies,)``.

        """
        f_query = np.abs(np.asarray(frequencies_hz, dtype=float))
        f_table = np.asarray(self.frequencies_hz, dtype=float)
        mag_table = np.asarray(self.magnitude_db, dtype=float)

        mag_db_interp = np.interp(f_query, f_table, mag_table)
        mag_linear = 10.0 ** (mag_db_interp / 20.0)

        if self.phase_deg is not None:
            phase_table = np.asarray(self.phase_deg, dtype=float)
            phase_interp_deg = np.interp(f_query, f_table, phase_table)
            phase_rad = np.deg2rad(phase_interp_deg)
        else:
            phase_rad = np.zeros_like(f_query)

        return (mag_linear * np.exp(1j * phase_rad)).astype(np.complex128)


class HydrophoneModel(Base):
    """Model of a single hydrophone element as an LTI system.

    Combines a scalar sensitivity with a frequency response to produce
    a complex transfer function that maps received acoustic pressure
    (in uPa) to voltage (in V).

    Parameters
    ----------
    sensitivity_db : float
        Hydrophone sensitivity in dB re 1 V/uPa.
    frequency_response : FrequencyResponse or None
        Frequency-dependent response model.  Defaults to
        :class:`FlatFrequencyResponse` when ``None``.

    """

    sensitivity_db: float = Property(doc="Hydrophone sensitivity in dB re 1 V/uPa.")
    frequency_response: FrequencyResponse | None = Property(
        default=None,
        doc="Frequency-dependent response model.  Defaults to FlatFrequencyResponse.",
    )

    def transfer_function(self, frequencies_hz: ArrayLike) -> ComplexArray:
        r"""Compute the combined complex transfer function.

        .. math::

            H(f) = S \cdot R(f)

        where :math:`S = 10^{(\text{sensitivity\_db} / 20)}` is the linear
        sensitivity and :math:`R(f)` is the frequency response.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the transfer function.

        Returns
        -------
        ComplexArray
            Complex transfer function, shape ``(num_frequencies,)``.

        """
        response = self.frequency_response or FlatFrequencyResponse()
        sensitivity_linear = 10.0 ** (self.sensitivity_db / 20.0)
        return (sensitivity_linear * response.evaluate(frequencies_hz)).astype(np.complex128)


def evaluate_hydrophone_transfer_functions(
    models: "HydrophoneModel | Sequence[HydrophoneModel]",
    num_sensors: int,
    frequencies_hz: ArrayLike,
) -> ComplexArray:
    """Evaluate hydrophone transfer functions for all sensors.

    Parameters
    ----------
    models : HydrophoneModel or Sequence[HydrophoneModel]
        A single model (applied uniformly to every sensor) or a list of
        per-sensor models.
    num_sensors : int
        Number of sensor elements in the array.
    frequencies_hz : ArrayLike
        Frequencies in Hz at which to evaluate.

    Returns
    -------
    ComplexArray
        Complex transfer functions, shape ``(num_sensors, num_frequencies)``.

    Raises
    ------
    ValueError
        If a list of models is provided whose length does not match
        *num_sensors*.

    """
    if isinstance(models, HydrophoneModel):
        h = models.transfer_function(frequencies_hz)
        return np.tile(h, (num_sensors, 1))

    models_list = list(models)
    if len(models_list) != num_sensors:
        msg = (
            f"Number of hydrophone models ({len(models_list)}) must match "
            f"number of sensors ({num_sensors})"
        )
        raise ValueError(msg)

    return np.stack(
        [m.transfer_function(frequencies_hz) for m in models_list],
        axis=0,
    )
