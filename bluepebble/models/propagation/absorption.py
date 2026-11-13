"""Seawater volume absorption."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def thorp_db_per_km(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    r"""Return the seawater absorption coefficient of Thorp's formula, in dB/km.

    .. math::

        \alpha = 3.3 \times 10^{-3} + \frac{0.11 f^2}{1 + f^2} + \frac{44 f^2}{4100 + f^2}
            + 3.0 \times 10^{-4} f^2

    with :math:`f` in kHz. The second and third terms are the boric acid and magnesium
    sulphate relaxations. This is the form ``bellhopcxx`` uses for its Thorp option (``'T'``).

    Parameters
    ----------
    frequency_hz : ArrayLike
        Frequencies in Hz. Only :math:`|f|` matters, so negative FFT bins get the absorption
        of their positive counterparts.

    Returns
    -------
    NDArray[np.float64]
        Absorption in dB/km, with the shape of ``frequency_hz``.

    Assumptions
    -----------
    - Absorption depends on frequency only: not on temperature, salinity, depth or pH.

    References
    ----------
    Jensen, F. B., Kuperman, W. A., Porter, M. B. and Schmidt, H. (2011). Computational Ocean
    Acoustics, 2nd ed., Springer, Eq. 1.34.

    """
    f2 = (np.asarray(frequency_hz, dtype=np.float64) / 1000.0) ** 2
    return np.asarray(
        3.3e-3 + 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 3.0e-4 * f2, dtype=np.float64
    )
