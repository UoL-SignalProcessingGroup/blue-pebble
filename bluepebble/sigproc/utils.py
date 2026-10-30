"""Small signal-processing helpers."""

import numpy as np
from numpy.typing import ArrayLike


def parabolic_peak_offset(y: ArrayLike, k: int) -> float:
    """Sub-sample offset of a peak from a 3-point parabola fit at index ``k``.

    Fits a parabola through ``y[k - 1]``, ``y[k]`` and ``y[k + 1]`` and returns where its
    vertex sits relative to ``k``, in samples (within ``(-0.5, 0.5)`` when ``k`` is a local
    maximum). Add it to ``k`` (scaled by the grid step) to refine a peak position.

    Parameters
    ----------
    y : ArrayLike
        One-dimensional data, e.g. a beam response or a correlation, ideally in dB.
    k : int
        Index of the peak sample.

    Returns
    -------
    float
        Offset from ``k`` in samples. 0 when ``k`` is at either end of ``y`` or the
        three points are collinear.

    """
    values = np.asarray(y, dtype=float)
    if k <= 0 or k >= len(values) - 1:
        return 0.0
    denom = values[k - 1] - 2.0 * values[k] + values[k + 1]
    if denom == 0.0:
        return 0.0
    return float(0.5 * (values[k - 1] - values[k + 1]) / denom)
