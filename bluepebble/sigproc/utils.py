"""Small signal-processing helpers."""

import numpy as np
from numpy.typing import ArrayLike


def parabolic_peak_offset(y: ArrayLike, k: int) -> float:
    r"""Return the sub-sample offset of a peak from a three-point parabola fit at index ``k``.

    The parabola through :math:`y_-=y[k-1]`, :math:`y_0=y[k]` and :math:`y_+=y[k+1]` has its
    vertex at

    .. math::

        \delta = \frac{1}{2} \frac{y_- - y_+}{y_- - 2 y_0 + y_+}

    samples from ``k``, within :math:`(-1/2, 1/2)` when ``k`` is a local maximum. Add
    :math:`\delta` times the grid step to the peak's position to refine it, as for a range or
    bearing estimate.

    Parameters
    ----------
    y : ArrayLike
        One-dimensional data, such as a beam response or a correlation, ideally in dB.
    k : int
        Index of the peak sample.

    Returns
    -------
    float
        Offset from ``k`` in samples. 0 when ``k`` is at either end of ``y`` or the three
        points are collinear.

    Assumptions
    -----------
    - The peak is locally quadratic across the three samples. A mainlobe in dB is close to
      this; on linear data, or a mainlobe sampled at only a sample or two, the offset is
      biased towards the nearer sample.

    """
    values = np.asarray(y, dtype=float)
    if k <= 0 or k >= len(values) - 1:
        return 0.0
    denom = values[k - 1] - 2.0 * values[k] + values[k + 1]
    if denom == 0.0:
        return 0.0
    return float(0.5 * (values[k - 1] - values[k + 1]) / denom)
