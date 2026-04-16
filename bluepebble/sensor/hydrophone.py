r"""Hydrophone models for converting acoustic pressure to voltage.

A hydrophone is modelled as a linear time-invariant (LTI) system whose
transfer function maps received acoustic pressure (in uPa) to output
voltage (in V).  The combined transfer function is:

.. math::

    H_{\text{hydrophone}}(f) = S \cdot R(f) \cdot e^{j\varphi}

where *S* is the scalar sensitivity (V/uPa, converted from dB re 1 V/uPa),
*R(f)* is a frequency-dependent complex response, and :math:`\varphi` is a
constant phase offset in radians.

Classes
-------
FrequencyResponse
    Abstract base class for frequency response models.
FlatFrequencyResponse
    Unity response at all frequencies.
TabulatedFrequencyResponse
    Response interpolated from datasheet measurements.
FirstOrderHighPassResponse
    First-order high-pass roll-off model.
FirstOrderLowPassResponse
    First-order low-pass roll-off model.
HydrophoneResponse
    Combined LTI electro-acoustic response model.
Hydrophone
    A single physical hydrophone element: response model and position state.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from stonesoup.base import Base, Property
from stonesoup.types.state import State

if TYPE_CHECKING:
    pass

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


class FirstOrderHighPassResponse(FrequencyResponse):
    r"""First-order high-pass frequency response.

    Models the low-frequency acoustic coupling cutoff of a hydrophone.
    Below the cutoff the response rolls off at +20 dB/decade; above it
    the response is flat (unity magnitude, zero phase).

    .. math::

        H(f) = \frac{j f / f_c}{1 + j f / f_c}

    Parameters
    ----------
    cutoff_hz : float
        -3 dB cutoff frequency in Hz.

    """

    cutoff_hz: float = Property(doc="-3 dB cutoff frequency in Hz.")

    def evaluate(self, frequencies_hz: ArrayLike) -> ComplexArray:
        """Evaluate the high-pass response at the given frequencies.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the response.

        Returns
        -------
        ComplexArray
            Complex frequency response, shape ``(num_frequencies,)``.

        """
        f = np.asarray(frequencies_hz, dtype=float)
        ratio = 1j * f / self.cutoff_hz
        return (ratio / (1.0 + ratio)).astype(np.complex128)


class FirstOrderLowPassResponse(FrequencyResponse):
    r"""First-order low-pass frequency response.

    Models the high-frequency roll-off of a hydrophone.  Below the cutoff
    the response is flat (unity magnitude, zero phase); above it the
    response rolls off at -20 dB/decade.

    .. math::

        H(f) = \frac{1}{1 + j f / f_c}

    Parameters
    ----------
    cutoff_hz : float
        -3 dB cutoff frequency in Hz.

    """

    cutoff_hz: float = Property(doc="-3 dB cutoff frequency in Hz.")

    def evaluate(self, frequencies_hz: ArrayLike) -> ComplexArray:
        """Evaluate the low-pass response at the given frequencies.

        Parameters
        ----------
        frequencies_hz : ArrayLike
            Frequencies in Hz at which to evaluate the response.

        Returns
        -------
        ComplexArray
            Complex frequency response, shape ``(num_frequencies,)``.

        """
        f = np.asarray(frequencies_hz, dtype=float)
        return (1.0 / (1.0 + 1j * f / self.cutoff_hz)).astype(np.complex128)


class HydrophoneResponse(Base):
    """Electro-acoustic response model of a single hydrophone transducer.

    Combines a scalar sensitivity with a frequency response to produce
    a complex transfer function that maps received acoustic pressure
    (in uPa) to voltage (in V).

    Parameters
    ----------
    sensitivity_db : float, optional
        Hydrophone sensitivity in dB re 1 V/uPa.  Defaults to ``0.0``, which
        applies no additional scaling and is equivalent to running a simulation
        without an explicit hydrophone model.
    frequency_response : FrequencyResponse or None
        Frequency-dependent response model.  Defaults to
        :class:`FlatFrequencyResponse` when ``None``.
    phase_offset_deg : float
        Constant phase offset in degrees applied uniformly across all
        frequencies.  Use this to model inter-element phase mismatch from
        sources such as cable length variation or connector tolerances.
        Defaults to 0.

    """

    sensitivity_db: float = Property(
        default=0.0,
        doc=(
            "Hydrophone sensitivity in dB re 1 V/uPa. "
            "Defaults to 0.0 dB, which applies no additional scaling and produces "
            "output equivalent to a simulation without an explicit hydrophone model."
        ),
    )
    frequency_response: FrequencyResponse | None = Property(
        default=None,
        doc="Frequency-dependent response model.  Defaults to FlatFrequencyResponse.",
    )
    phase_offset_deg: float = Property(
        default=0.0,
        doc=(
            "Constant phase offset in degrees applied across all frequencies. "
            "Models inter-element phase mismatch from cable or connector variation."
        ),
    )

    def transfer_function(self, frequencies_hz: ArrayLike) -> ComplexArray:
        r"""Compute the combined complex transfer function.

        .. math::

            H(f) = S \cdot R(f) \cdot e^{j\varphi}

        where :math:`S = 10^{(\text{sensitivity\_db} / 20)}` is the linear
        sensitivity, :math:`R(f)` is the frequency response, and
        :math:`\varphi` is the constant phase offset in radians.

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
        phase_phasor = np.exp(1j * np.deg2rad(self.phase_offset_deg))
        return (sensitivity_linear * phase_phasor * response.evaluate(frequencies_hz)).astype(
            np.complex128
        )


class Hydrophone(Base):
    """A single physical hydrophone element.

    Pairs an electro-acoustic response model with a dynamic position state.
    Position state is managed externally by :class:`LinearHydrophoneArray`
    via repeated calls to its :meth:`~LinearHydrophoneArray.move` method.

    Parameters
    ----------
    response : HydrophoneResponse
        Electro-acoustic transducer model for this element.

    Notes
    -----
    Each ``Hydrophone`` instance must be a distinct object.  Sharing one
    instance across multiple elements in an array will cause all elements
    to accumulate states on the same list.

    """

    response: HydrophoneResponse = Property(doc="Electro-acoustic transducer model.")

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialise the hydrophone with an empty state history.

        Parameters
        ----------
        *args : object
            Positional arguments forwarded to ``Base``.
        **kwargs : object
            Keyword arguments forwarded to ``Base``.

        """
        super().__init__(*args, **kwargs)
        self.states: list[State] = []

    @property
    def state(self) -> State | None:
        """Return the most recent position state, or ``None`` if uninitialised.

        Returns
        -------
        State or None
            The last appended state, or ``None`` before the first
            :meth:`~LinearHydrophoneArray.move` call.

        """
        return self.states[-1] if self.states else None
