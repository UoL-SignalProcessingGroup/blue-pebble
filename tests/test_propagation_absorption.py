"""Tests for the seawater absorption formulas."""

from __future__ import annotations

import numpy as np
import pytest

from .support import isolate_real_bluepebble_imports, restore_real_bluepebble_imports


@pytest.fixture
def thorp_db_per_km():
    """Import the real function without leaking cached modules."""
    snapshot = isolate_real_bluepebble_imports()
    from bluepebble.models.propagation import thorp_db_per_km

    yield thorp_db_per_km
    restore_real_bluepebble_imports(snapshot)


@pytest.mark.parametrize(
    ("frequency_hz", "terms"),
    [
        # Each term of Jensen et al. Eq. 1.34 worked by hand, f in kHz.
        (0.0, (3.3e-3, 0.0, 0.0, 0.0)),
        (400.0, (3.3e-3, 0.0151724, 0.0017170, 0.0000480)),
        (1000.0, (3.3e-3, 0.0550000, 0.0107291, 0.0003000)),
        (5400.0, (3.3e-3, 0.1063528, 0.3107266, 0.0087480)),
    ],
)
def test_thorp_matches_the_published_formula(thorp_db_per_km, frequency_hz, terms) -> None:
    """Thorp's absorption is the sum of the four terms of Eq. 1.34."""
    assert thorp_db_per_km(frequency_hz) == pytest.approx(sum(terms), abs=2e-7)


def test_thorp_takes_arrays_and_is_even_in_frequency(thorp_db_per_km) -> None:
    """Negative FFT bins get the absorption of their positive counterparts."""
    grid = np.fft.fftfreq(16, 1 / 12800.0).reshape(4, 4)

    alpha = thorp_db_per_km(grid)

    assert alpha.shape == grid.shape
    np.testing.assert_array_equal(alpha, thorp_db_per_km(np.abs(grid)))
    assert np.all(np.diff(thorp_db_per_km(np.linspace(0.0, 20e3, 200))) > 0)
