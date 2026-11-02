"""Matched filtering (replica correlation) of one or many channels."""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve, get_window
from stonesoup.base import Base, Property

FloatArray: TypeAlias = NDArray[np.float64]


class MatchedFilter(Base):
    r"""Cross-correlate data with a replica, returning the correlation magnitude.

    For data :math:`x`, replica :math:`r` of length :math:`M` and taper :math:`w`, output
    sample :math:`k` is the match with the replica starting at input sample :math:`k`:

    .. math::

        y[k] = \left| \sum_{m=0}^{M-1} x[k + m] \, w[m] \, r^*[m] \right|.

    Single-channel data ``(n_samples,)`` and multi-channel data ``(n_channels, n_samples)``,
    such as one row per beam, are both accepted; each row is correlated along the last axis.
    The output is not normalised, so levels can be compared across channels. For white noise
    the matched filter maximises the output SNR (signal-to-noise ratio), by the pulse's
    time-bandwidth product for an LFM (linear frequency modulated) pulse [MF1]_.

    Assumptions
    -----------
    - The replica matches the received pulse in shape. A Doppler shift from a moving target is
      not compensated, so the peak loses gain as the shift approaches the pulse's resolution.
    - Data beyond the last sample is zero: the final ``n_replica - 1`` outputs correlate the
      replica with a partial pulse.

    References
    ----------
    .. [MF1] Van Trees, H. L. "Detection, Estimation, and Modulation Theory, Part I." Wiley, New
           York, 1968. ISBN 978-0-471-89955-0.

    """

    window: str | tuple[str, float] | None = Property(
        default=None,
        doc="Taper applied to the replica before correlating, as accepted by "
        "scipy.signal.get_window (e.g. 'hann' or ('tukey', 0.25)). Lowers the range "
        "sidelobes at the cost of a wider main lobe and lower peak. None applies no taper.",
    )

    def filter(self, data: ArrayLike, replica: ArrayLike) -> FloatArray:
        """Correlate ``data`` with ``replica``.

        Parameters
        ----------
        data : ArrayLike
            Received data, shape ``(n_samples,)`` or ``(n_channels, n_samples)``. Real or
            complex.
        replica : ArrayLike
            Replica, such as the transmit pulse, shape ``(n_replica,)``.

        Returns
        -------
        FloatArray
            Correlation magnitude, the same shape as ``data``.

        Raises
        ------
        ValueError
            If ``replica`` is not one-dimensional, ``data`` is not one- or two-dimensional,
            or the replica is longer than the data.

        """
        data_array = np.asarray(data)
        replica_array = np.asarray(replica)
        if replica_array.ndim != 1:
            raise ValueError(f"replica must be one-dimensional, got shape {replica_array.shape}.")
        if data_array.ndim not in (1, 2):
            raise ValueError(
                f"data must be (n_samples,) or (n_channels, n_samples), got shape "
                f"{data_array.shape}."
            )
        n_samples = data_array.shape[-1]
        n_replica = len(replica_array)
        if n_replica > n_samples:
            raise ValueError(
                f"replica ({n_replica} samples) is longer than the data ({n_samples} samples)."
            )
        if self.window is not None:
            replica_array = replica_array * get_window(self.window, n_replica)

        # Correlation is convolution with the time-reversed conjugate replica; the full
        # convolution's sample n_replica - 1 + k is output sample k.
        kernel = np.conj(replica_array[::-1]).reshape((1,) * (data_array.ndim - 1) + (-1,))
        full = fftconvolve(data_array, kernel, mode="full", axes=-1)
        return np.asarray(
            np.abs(full[..., n_replica - 1 : n_replica - 1 + n_samples]), dtype=np.float64
        )
