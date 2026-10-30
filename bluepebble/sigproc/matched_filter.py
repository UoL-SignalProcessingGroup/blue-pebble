"""Matched filtering (replica correlation) of one or many channels."""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window
from stonesoup.base import Base, Property

FloatArray: TypeAlias = NDArray[np.float64]


class MatchedFilter(Base):
    """Cross-correlate data with a replica, returning the correlation magnitude.

    Works on a single channel ``(n_samples,)`` or on many at once ``(n_channels,
    n_samples)`` (e.g. one row per beam), correlating along the last axis. Output sample
    ``k`` is the match with the replica starting at input sample ``k``. The output is not
    normalised, so levels can be compared across channels.
    """

    window: str | tuple[str, float] | None = Property(
        default=None,
        doc="Taper applied to the replica before correlating, as accepted by "
        "scipy.signal.get_window (e.g. 'hann' or ('tukey', 0.25)). Lowers the range "
        "sidelobes at the cost of a wider main lobe. None applies no taper.",
    )

    def filter(self, data: ArrayLike, replica: ArrayLike) -> FloatArray:
        """Correlate ``data`` with ``replica``.

        Parameters
        ----------
        data : ArrayLike
            Received data, shape ``(n_samples,)`` or ``(n_channels, n_samples)``. Real or
            complex.
        replica : ArrayLike
            Replica (e.g. the transmit pulse), shape ``(n_replica,)``.

        Returns
        -------
        FloatArray
            Correlation magnitude, same shape as ``data``.

        """
        data_array = np.asarray(data)
        replica_array = np.asarray(replica)
        if self.window is not None:
            replica_array = replica_array * get_window(self.window, len(replica_array))

        n_samples = data_array.shape[-1]
        n_fft = n_samples + len(replica_array) - 1
        replica_f = np.conj(np.fft.fft(replica_array, n_fft))
        corr = np.fft.ifft(np.fft.fft(data_array, n_fft, axis=-1) * replica_f, axis=-1)
        return np.asarray(np.abs(corr[..., :n_samples]), dtype=np.float64)
